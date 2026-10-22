#!/usr/bin/env python3
"""Shared machinery for the H + O2 quantum-geometry pilot.

The production protocol follows one symmetry-equivalent perpendicular
component of the two lowest doublet Pi-derived states of linear HO2 in the
C2v Abelian subgroup, with equal-weight SA2-CASSCF(13,9).  The physical response is
extracted for the lower adiabatic member of that transported two-state
manifold.  Signed fields are always launched independently from the same
zero-field frame.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


VERSION = "1.1.0"
BASIS = "aug-cc-pvtz"
NCORE = 2
NCAS = 9
NELECAS = (7, 6)
NSTATE = 2
NRETAIN = NCORE + NCAS
FULL_NELECAS = (NELECAS[0] + NCORE, NELECAS[1] + NCORE)
SPIN = 1
SS_TARGET = 0.75
SPIN_PENALTY_SHIFT = 1.0
SYMMETRY_GROUP = "C2v"
ACTIVE_IRREPS = {"A1": 5, "B1": 2, "B2": 2}
CORE_IRREPS = {"A1": 2}
STATE_IRREP = "B1"
DEFAULT_FIELDS = (5.0e-5, 1.0e-4, 2.5e-4, 5.0e-4)
SEARCH_EXTRAS = (2, 4, 8)
HOMING_ANCHOR_SV_MIN = 0.50
HOMING_LOCAL_SV_MIN = 0.50
HOMING_EXPAND_SV = 0.75
CAS_FOLLOW_SV_MIN = 0.50


def qc_modules():
    try:
        import h5py
        from pyscf import fci, gto, lib, mcscf, scf, symm
        from pyscf.fci import addons as fci_addons
        from pyscf.fci import cistring, spin_op
    except ImportError as exc:
        raise SystemExit(
            "This command requires PySCF and h5py in the active environment."
        ) from exc
    return h5py, fci, gto, lib, mcscf, scf, symm, fci_addons, cistring, spin_op


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_h5(path: Path, arrays: dict[str, Any], attrs: dict[str, Any]) -> None:
    h5py, *_ = qc_modules()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        for key, value in attrs.items():
            if value is None:
                handle.attrs[key] = "null"
            elif isinstance(value, (dict, list, tuple)):
                handle.attrs[key] = json.dumps(value)
            else:
                handle.attrs[key] = value
        for key, value in arrays.items():
            handle.create_dataset(key, data=np.asarray(value))
    os.replace(tmp, path)


def load_h5(path: Path) -> dict[str, Any]:
    h5py, *_ = qc_modules()
    out: dict[str, Any] = {"attrs": {}}
    with h5py.File(path, "r") as handle:
        for key, value in handle.attrs.items():
            if isinstance(value, bytes):
                value = value.decode()
            if isinstance(value, str) and value[:1] in "[{":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            out["attrs"][key] = value
        for key in handle:
            out[key] = np.asarray(handle[key])
    return out


def parse_indices(text: str, size: int) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = [int(x) if x else None for x in token.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid slice {token!r}")
            start, stop = parts[:2]
            step = parts[2] if len(parts) == 3 else 1
            if stop is not None:
                stop += 1  # inclusive user-facing stop
            values.extend(range(*slice(start, stop, step).indices(size)))
        else:
            values.append(int(token))
    answer = sorted(set(values))
    if not answer or min(answer) < 0 or max(answer) >= size:
        raise ValueError(f"Indices must lie in [0,{size - 1}]")
    return answer


def parse_floats(text: str) -> list[float]:
    values = sorted(set(float(x) for x in text.split(",") if x.strip()))
    if not values or any(x <= 0.0 for x in values):
        raise ValueError("Field magnitudes must be positive")
    return values


def coords_from_record(record: dict[str, Any]) -> np.ndarray:
    value = record["coords_json"]
    if isinstance(value, str):
        value = json.loads(value)
    coords = np.asarray(value, dtype=float)
    if coords.shape != (3, 3):
        raise ValueError(f"Expected 3x3 O-O-H coordinates, got {coords.shape}")
    return coords


def geometry_label(source_index: int) -> str:
    return f"irc_{source_index:03d}"


def field_label(value: float) -> str:
    sign = "p" if value > 0 else "m"
    return f"F_{sign}{abs(value):.1e}"


def build_molecule(
    record: dict[str, Any], *, basis: str = BASIS, verbose: int = 4,
    memory_mb: int = 30000, symmetry: bool = True,
):
    _, _, gto, _, _, _, _, _, _, _ = qc_modules()
    coords = coords_from_record(record)
    atoms = [("O", coords[0]), ("O", coords[1]), ("H", coords[2])]
    return gto.M(
        atom=atoms,
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=SPIN,
        symmetry=SYMMETRY_GROUP if symmetry else False,
        verbose=verbose,
        max_memory=memory_mb,
    )


def tag_symmetry_orbitals(mol, mo: np.ndarray) -> np.ndarray:
    _, _, _, lib, _, _, symm, _, _, _ = qc_modules()
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"))
    labels = symm.label_orb_symm(
        mol, mol.irrep_name, mol.symm_orb, np.asarray(mo), s=overlap, check=True
    )
    ids = [
        symm.irrep_name2id(mol.groupname, str(label))
        if isinstance(label, (str, np.str_)) else int(label)
        for label in labels
    ]
    return lib.tag_array(np.asarray(mo), orbsym=np.asarray(ids, dtype=int))


def verify_required_irreps(mol) -> None:
    names = set(map(str, mol.irrep_name))
    required = set(ACTIVE_IRREPS)
    if not required.issubset(names):
        raise RuntimeError(
            f"{SYMMETRY_GROUP} irreps are {sorted(names)}, expected at least "
            f"{sorted(required)}"
        )


def metric_orthonormalize(coeff: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    metric = coeff.T @ overlap @ coeff
    eig, vec = np.linalg.eigh(0.5 * (metric + metric.T))
    if eig.min() < 1.0e-10:
        raise RuntimeError(
            f"Projected orbital metric is singular: min eigenvalue={eig.min():.3e}"
        )
    return coeff @ ((vec / np.sqrt(eig)) @ vec.T)


def project_mos(old_mol, new_mol, old_mo: np.ndarray, orbital_basis_mf) -> tuple[np.ndarray, dict[str, float]]:
    """Directly continue the complete CASSCF orbital frame to a new geometry.

    ``project_init_guess`` performs symmetry-resolved sequential SVDs.  Giving
    the active block first ensures that the transported CAS is deformed as
    little as possible; the core and virtual complements are then made
    orthogonal to it.  The converged ROHF orbitals are used only as a complete
    orthonormal basis at the new geometry--they never select or replace the
    transported core or active orbitals.
    """
    _, _, gto, _, mcscf, _, _, _, _, _ = qc_modules()
    if orbital_basis_mf.mol is not new_mol:
        raise ValueError("orbital_basis_mf must belong to new_mol")
    probe = mcscf.CASSCF(orbital_basis_mf, NCAS, NELECAS, ncore=NCORE)
    projected = mcscf.project_init_guess(
        probe,
        np.asarray(old_mo),
        prev_mol=old_mol,
        priority="active",
        use_hf_core=False,
    )
    s_new = np.asarray(new_mol.intor_symmetric("int1e_ovlp"))
    cross = np.asarray(gto.intor_cross("int1e_ovlp", new_mol, old_mol))
    projected = tag_symmetry_orbitals(new_mol, projected)
    error = float(np.max(np.abs(projected.T @ s_new @ projected - np.eye(projected.shape[1]))))
    core = projected[:, :NCORE].T @ cross @ old_mo[:, :NCORE]
    active = projected[:, NCORE:NRETAIN].T @ cross @ old_mo[:, NCORE:NRETAIN]
    retained = projected[:, :NRETAIN].T @ cross @ old_mo[:, :NRETAIN]
    core_sv = np.linalg.svd(core, compute_uv=False)
    active_sv = np.linalg.svd(active, compute_uv=False)
    retained_sv = np.linalg.svd(retained, compute_uv=False)
    return projected, {
        "method": "pyscf.mcscf.project_init_guess",
        "priority": "active",
        "use_hf_core": False,
        "orthonormality_error": error,
        "core_projection_sv_min": float(core_sv.min()),
        "active_projection_sv_min": float(active_sv.min()),
        "retained_projection_sv_min": float(retained_sv.min()),
    }


def transport_active_ci(old_mol, new_mol, old_mo: np.ndarray,
                        new_mo: np.ndarray, old_ci: Any):
    """Express adjacent-geometry CI vectors in the directly followed CAS frame.

    The polar factor of the old/new active-orbital cross overlap removes the
    arbitrary unitary rotation within the transported active space.  Without
    this step, comparing raw CI coefficient arrays can spuriously home to high
    roots even when the many-electron subspace is physically continuous.
    """
    _, _, gto, _, _, _, _, fci_addons, _, _ = qc_modules()
    cross = np.asarray(gto.intor_cross("int1e_ovlp", old_mol, new_mol))
    old_active = np.asarray(old_mo)[:, NCORE:NRETAIN]
    new_active = np.asarray(new_mo)[:, NCORE:NRETAIN]
    active_overlap = old_active.T @ cross @ new_active
    left, singular_values, right_h = np.linalg.svd(active_overlap)
    rotation = left @ right_h
    transported = []
    for root in ci_roots(old_ci):
        value = np.asarray(
            fci_addons.transform_ci_for_orbital_rotation(
                root, NCAS, NELECAS, rotation
            )
        )
        value /= np.linalg.norm(value)
        transported.append(value)
    return transported, {
        "active_cross_sv": singular_values.tolist(),
        "active_cross_sv_min": float(singular_values.min()),
        "active_polar_residual": float(
            np.linalg.norm(active_overlap - rotation, ord="fro")
        ),
    }


def ci_roots(ci: Any) -> list[np.ndarray]:
    if isinstance(ci, (list, tuple)):
        roots = [np.asarray(x) for x in ci]
    else:
        array = np.asarray(ci)
        if array.ndim < 3 or array.shape[0] != NSTATE:
            raise ValueError(f"Expected {NSTATE} CI roots, got {array.shape}")
        roots = [array[index] for index in range(NSTATE)]
    if len(roots) != NSTATE:
        raise ValueError(f"Expected {NSTATE} CI roots, got {len(roots)}")
    return [np.array(x, copy=True) for x in roots]


def copy_ci(ci: Any) -> list[np.ndarray]:
    return ci_roots(ci)


class RootHomingError(RuntimeError):
    pass


class SubspaceHomingFCISolver:
    """Select a locally continuous two-root CI subspace from a larger window."""

    __name_mixin__ = "HO2SubspaceHoming"
    _keys = {
        "homing_desired_roots", "homing_anchor", "homing_anchor_sv_min",
        "homing_local_sv_min", "homing_last_anchor_sv",
        "homing_last_local_sv", "homing_min_anchor_sv_seen",
        "homing_min_local_sv_seen", "homing_last_candidate_count",
    }

    @staticmethod
    def _decode(ci0, roots: int, norb: int, nelec) -> list[np.ndarray]:
        if isinstance(nelec, (tuple, list, np.ndarray)):
            na, nb = int(nelec[0]), int(nelec[1])
        else:
            total = int(nelec)
            na = (total + SPIN) // 2
            nb = total - na
        shape = (math.comb(int(norb), na), math.comb(int(norb), nb))
        size = int(np.prod(shape))
        try:
            packed = np.asarray(ci0)
        except ValueError:
            packed = np.asarray(ci0, dtype=object)
        if packed.dtype != object and packed.size == roots * size:
            packed = packed.reshape((roots,) + shape)
            return [np.asarray(packed[i]) for i in range(roots)]
        if isinstance(ci0, (list, tuple)) and len(ci0) == roots:
            answer = []
            for item in ci0:
                array = np.asarray(item)
                if array.size != size:
                    raise RootHomingError(
                        f"CI reference has {array.size} entries, expected {size}"
                    )
                answer.append(array.reshape(shape))
            return answer
        raise RootHomingError(
            f"Cannot decode {roots} CI roots from shape={packed.shape}, size={packed.size}"
        )

    def kernel(self, h1, h2, norb, nelec, ci0=None, nroots=None, **kwargs):
        desired = int(self.homing_desired_roots)
        anchor_raw = getattr(self, "homing_anchor", None)
        if ci0 is None and anchor_raw is None:
            return super().kernel(h1, h2, norb, nelec, nroots=desired, **kwargs)
        incoming = self._decode(anchor_raw if ci0 is None else ci0, desired, norb, nelec)
        anchors = self._decode(incoming if anchor_raw is None else anchor_raw, desired, norb, nelec)
        incoming_matrix = np.stack([x.ravel() / np.linalg.norm(x) for x in incoming])
        anchor_matrix = np.stack([x.ravel() / np.linalg.norm(x) for x in anchors])
        best = None
        fallback = None
        for extra in SEARCH_EXTRAS:
            search_roots = desired + extra
            energies, candidates = super().kernel(
                h1, h2, norb, nelec, ci0=incoming,
                nroots=search_roots, **kwargs
            )
            energies = np.atleast_1d(np.asarray(energies, dtype=float))
            candidates = list(candidates) if search_roots > 1 else [candidates]
            raw_converged = getattr(self, "converged", True)
            if isinstance(raw_converged, (list, tuple, np.ndarray)):
                candidate_converged = [bool(x) for x in raw_converged]
            else:
                candidate_converged = [bool(raw_converged)] * len(candidates)
            if len(candidate_converged) != len(candidates):
                candidate_converged = [bool(np.all(candidate_converged))] * len(candidates)
            candidate_matrix = np.stack([
                np.asarray(x).ravel() / np.linalg.norm(x) for x in candidates
            ])
            oa = anchor_matrix.conj() @ candidate_matrix.T
            ol = incoming_matrix.conj() @ candidate_matrix.T
            for columns in itertools.combinations(range(len(candidates)), desired):
                sva = np.linalg.svd(oa[:, columns], compute_uv=False)
                svl = np.linalg.svd(ol[:, columns], compute_uv=False)
                record = {
                    "continuity_score": (
                        float(svl.min()), float(sva.min()),
                        float(np.sum(svl**2 + sva**2)),
                    ),
                    "selection_score": (
                        -float(np.mean(energies[list(columns)])),
                        min(float(svl.min()), float(sva.min())),
                        float(np.sum(svl**2 + sva**2)),
                    ),
                    "anchor": float(sva.min()), "local": float(svl.min()),
                    "columns": list(columns), "energies": energies,
                    "candidates": candidates,
                    "converged": candidate_converged,
                }
                if (
                    fallback is None
                    or record["continuity_score"] > fallback["continuity_score"]
                ):
                    fallback = record
                if (
                    record["anchor"] >= float(self.homing_anchor_sv_min)
                    and record["local"] >= float(self.homing_local_sv_min)
                    and (
                        best is None
                        or record["selection_score"] > best["selection_score"]
                    )
                ):
                    best = record
            if (
                best is not None
                and best["local"] >= HOMING_EXPAND_SV
                and all(best["converged"][i] for i in best["columns"])
            ):
                break
        if best is None:
            assert fallback is not None
            raise RootHomingError(
                "No two-state CI subspace passed the continuity gates: "
                f"anchor={fallback['anchor']:.6f}, local={fallback['local']:.6f}"
            )
        selected = best["columns"]
        e = np.asarray(best["energies"])[selected]
        c = [best["candidates"][i] for i in selected]
        self.eci, self.ci = e, c
        self.homing_last_anchor_sv = best["anchor"]
        self.homing_last_local_sv = best["local"]
        self.homing_min_anchor_sv_seen = min(self.homing_min_anchor_sv_seen, best["anchor"])
        self.homing_min_local_sv_seen = min(self.homing_min_local_sv_seen, best["local"])
        self.homing_last_candidate_count = len(best["candidates"])
        self.converged = [bool(best["converged"][i]) for i in selected]
        return e, c


def install_homing(solver, anchor_ci: Sequence[np.ndarray] | None):
    _, _, _, lib, _, _, _, _, _, _ = qc_modules()
    solver.nroots = NSTATE
    solver.homing_desired_roots = NSTATE
    solver.homing_anchor = None if anchor_ci is None else np.asarray(anchor_ci)
    solver.homing_anchor_sv_min = HOMING_ANCHOR_SV_MIN
    solver.homing_local_sv_min = HOMING_LOCAL_SV_MIN
    solver.homing_last_anchor_sv = 1.0
    solver.homing_last_local_sv = 1.0
    solver.homing_min_anchor_sv_seen = 1.0
    solver.homing_min_local_sv_seen = 1.0
    solver.homing_last_candidate_count = NSTATE
    return lib.set_class(solver, (SubspaceHomingFCISolver, solver.__class__))


def build_rohf_seed(mol):
    _, _, _, _, _, scf, _, _, _, _ = qc_modules()
    mf = scf.ROHF(mol)
    mf.conv_tol = 1.0e-11
    mf.max_cycle = 300
    mf.level_shift = 0.20
    mf.diis_space = 12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("ROHF seed did not converge")
    return mf


def average_rdm1(ci: Sequence[np.ndarray]) -> np.ndarray:
    _, fci, _, _, _, _, _, _, _, _ = qc_modules()
    return np.mean([
        fci.direct_spin1.make_rdm1(x, NCAS, NELECAS) for x in ci
    ], axis=0)


def reference_density(mo: np.ndarray, ci: Sequence[np.ndarray]) -> np.ndarray:
    core = np.asarray(mo[:, :NCORE])
    active = np.asarray(mo[:, NCORE:NRETAIN])
    return 2.0 * core @ core.T + active @ average_rdm1(ci) @ active.T


def build_reference_mf(mol, mo: np.ndarray, ci: Sequence[np.ndarray], field_x: float = 0.0):
    _, _, _, _, _, scf, _, _, _, _ = qc_modules()
    mf = scf.ROHF(mol)
    mf.conv_tol = 1.0e-11
    h0 = np.asarray(mf.get_hcore())
    r_ao = np.asarray(mol.intor_symmetric("int1e_r", comp=3))
    hfield = h0 + float(field_x) * r_ao[0]
    mf.get_hcore = lambda *args, **kwargs: hfield
    mf.max_cycle = 0
    mf.mo_coeff = tag_symmetry_orbitals(mol, np.asarray(mo))
    mf.mo_occ = np.zeros(mo.shape[1])
    nalpha, nbeta = mol.nelec
    mf.mo_occ[:nbeta] = 2.0
    mf.mo_occ[nbeta:nalpha] = 1.0
    mf.mo_energy = np.zeros(mo.shape[1])
    mf.e_tot = 0.0
    mf.converged = False
    return mf


def initial_active_mos(mf) -> np.ndarray:
    _, _, _, _, mcscf, _, _, _, _, _ = qc_modules()
    verify_required_irreps(mf.mol)
    probe = mcscf.CASSCF(mf, NCAS, NELECAS, ncore=NCORE)
    try:
        mo = mcscf.sort_mo_by_irrep(
            probe, mf.mo_coeff, ACTIVE_IRREPS, CORE_IRREPS
        )
    except Exception as exc:
        raise RuntimeError(
            "Automatic CAS(13,9) irrep selection failed. Expected "
            "5 A1 + 2 B1 + 2 B2 active orbitals and two frozen A1 O(1s) cores."
        ) from exc
    return tag_symmetry_orbitals(mf.mol, np.asarray(mo))


def build_sa2(mf, anchor_ci: Sequence[np.ndarray] | None, *, newton: bool,
              grad_tol: float, max_macro: int, max_step: float,
              use_homing: bool = True):
    _, fci, _, _, mcscf, _, _, fci_addons, _, _ = qc_modules()
    mc = mcscf.CASSCF(mf, NCAS, NELECAS, ncore=NCORE)
    solver = fci.direct_spin1_symm.FCI(mc.mol)
    solver.spin = SPIN
    solver.wfnsym = STATE_IRREP
    solver.conv_tol = 1.0e-11
    solver.max_cycle = 240
    solver = fci_addons.fix_spin_(solver, shift=SPIN_PENALTY_SHIFT, ss=SS_TARGET)
    if use_homing:
        solver = install_homing(solver, anchor_ci)
    else:
        solver.nroots = NSTATE
    mc.fcisolver = solver
    mc.conv_tol = 1.0e-10
    mc.conv_tol_grad = grad_tol
    mc.max_cycle_macro = max_macro
    mc.max_cycle_micro = 12 if not newton else 20
    mc.max_stepsize = max_step
    mc.ah_level_shift = 0.5 if not newton else 1.0
    mc.ah_grad_trust_region = 1.0
    mc.canonicalization = False
    mc = mc.state_average_([0.5, 0.5])
    if newton:
        mc = mc.newton()
        mc.conv_tol_grad = grad_tol
        mc.max_cycle_macro = max_macro
        mc.max_cycle_micro = 20
        mc.max_stepsize = max_step
        mc.ah_level_shift = 1.0
        mc.ah_grad_trust_region = 1.0
        mc.canonicalization = False
    return mc


def optimize_sa2(mf, mo0: np.ndarray, ci0: Sequence[np.ndarray] | None,
                 *, anchor_ci: Sequence[np.ndarray] | None = None,
                 use_homing: bool = True):
    stages = (
        ("settle", False, 2.0e-5, 100, 0.015),
        ("tight_newton", True, 1.0e-6, 160, 0.006),
        ("floor_newton", True, 1.0e-6, 240, 0.002),
    )
    mo = np.asarray(mo0)
    ci = None if ci0 is None else copy_ci(ci0)
    reports = []
    converged = False
    mc = None
    for name, newton, grad_tol, max_macro, max_step in stages:
        mc = build_sa2(
            mf, (anchor_ci if anchor_ci is not None else ci) if use_homing else None,
            newton=newton, grad_tol=grad_tol,
            max_macro=max_macro, max_step=max_step, use_homing=use_homing,
        )
        started = time.time()
        try:
            mc.kernel(tag_symmetry_orbitals(mf.mol, mo), ci0=ci)
            error = None
        except RootHomingError as exc:
            error = str(exc)
            mc.converged = False
        reports.append({
            "name": name,
            "converged": bool(mc.converged),
            "wall_seconds": time.time() - started,
            "root_homing_error": error,
        })
        if getattr(mc, "mo_coeff", None) is not None:
            mo = np.asarray(mc.mo_coeff)
        if getattr(mc, "ci", None) is not None:
            ci = copy_ci(mc.ci)
        if newton and bool(mc.converged):
            converged = True
            break
    if mc is None or ci is None or not converged:
        raise RuntimeError(f"SA2-CASSCF did not converge: {json.dumps(reports)}")
    return mc, mo, ci, reports


def spin_and_rdm(ci: Sequence[np.ndarray]):
    _, fci, _, _, _, _, _, _, _, spin_op = qc_modules()
    spins = np.asarray([
        spin_op.spin_square0(x, NCAS, NELECAS)[0] for x in ci
    ])
    rdm1 = np.stack([
        fci.direct_spin1.make_rdm1(x, NCAS, NELECAS) for x in ci
    ])
    noons = np.stack([
        np.linalg.eigvalsh(0.5 * (dm + dm.T))[::-1] for dm in rdm1
    ])
    return spins, rdm1, noons


def physical_energies(mc, ci: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    penalized = np.asarray(mc.e_states, dtype=float)
    spins, _, _ = spin_and_rdm(ci)
    return penalized - SPIN_PENALTY_SHIFT * (spins - SS_TARGET), penalized


def embed_active_ci(ci: np.ndarray) -> np.ndarray:
    *_, cistring, _ = qc_modules()
    a_active = cistring.make_strings(range(NCAS), NELECAS[0])
    b_active = cistring.make_strings(range(NCAS), NELECAS[1])
    a_full = cistring.make_strings(range(NRETAIN), FULL_NELECAS[0])
    b_full = cistring.make_strings(range(NRETAIN), FULL_NELECAS[1])
    a_addr = {int(value): i for i, value in enumerate(a_full)}
    b_addr = {int(value): i for i, value in enumerate(b_full)}
    core = (1 << NCORE) - 1
    ia = [a_addr[(int(value) << NCORE) | core] for value in a_active]
    ib = [b_addr[(int(value) << NCORE) | core] for value in b_active]
    out = np.zeros((len(a_full), len(b_full)), dtype=np.asarray(ci).dtype)
    out[np.ix_(ia, ib)] = np.asarray(ci).reshape(len(a_active), len(b_active))
    return out


def state_overlap(
    left: dict[str, Any], right: dict[str, Any], s_cross: np.ndarray
) -> np.ndarray:
    *_, fci_addons, _, _ = qc_modules()[0:10]
    retained = (
        np.asarray(left["mo_coeff"])[:, :NRETAIN].T
        @ np.asarray(s_cross)
        @ np.asarray(right["mo_coeff"])[:, :NRETAIN]
    )
    left_ci = [embed_active_ci(x) for x in ci_roots(left["ci_vectors"])]
    right_ci = [embed_active_ci(x) for x in ci_roots(right["ci_vectors"])]
    out = np.empty((len(left_ci), len(right_ci)))
    for i, bra in enumerate(left_ci):
        for j, ket in enumerate(right_ci):
            out[i, j] = float(
                fci_addons.overlap(
                    bra, ket, NRETAIN, FULL_NELECAS, s=retained
                ).real
            )
    return out


def same_geometry_overlap(left: dict[str, Any], right: dict[str, Any], mol) -> np.ndarray:
    s = np.asarray(mol.intor_symmetric("int1e_ovlp"))
    return state_overlap(left, right, s)


def cross_geometry_overlap(left: dict[str, Any], right: dict[str, Any],
                           left_mol, right_mol) -> np.ndarray:
    _, _, gto, _, _, _, _, _, _, _ = qc_modules()
    cross = np.asarray(gto.intor_cross("int1e_ovlp", left_mol, right_mol))
    return state_overlap(left, right, cross)


def best_permutation(overlap: np.ndarray) -> tuple[list[int], float]:
    overlap = np.asarray(overlap)
    best = None
    for perm in itertools.permutations(range(overlap.shape[1]), overlap.shape[0]):
        score = float(sum(abs(overlap[i, perm[i]]) ** 2 for i in range(overlap.shape[0])))
        if best is None or score > best[0]:
            best = (score, list(perm))
    assert best is not None
    return best[1], best[0]


def fit_even_limit(fields: Sequence[float], values: Sequence[float]) -> tuple[float, float, float]:
    f = np.asarray(fields, dtype=float)
    y = np.asarray(values, dtype=float)
    x = f**2
    degree = 2 if len(f) >= 4 else 1
    coeff = np.polynomial.polynomial.polyfit(x, y, degree)
    fitted = np.polynomial.polynomial.polyval(x, coeff)
    limit = float(coeff[0])
    scale = max(float(np.max(np.abs(y))), 1.0)
    residual = float(np.max(np.abs(y - fitted)) / scale)
    small_coeff = np.polynomial.polynomial.polyfit(x[:-1], y[:-1], min(degree, len(f) - 2))
    stability = abs(limit - float(small_coeff[0])) / max(abs(limit), 1.0)
    return limit, residual, stability


def load_grid(workdir: Path) -> list[dict[str, str]]:
    return read_csv(workdir / "geometry_grid.csv")


def grid_by_source_index(workdir: Path) -> dict[int, dict[str, str]]:
    return {int(row["source_index"]): row for row in load_grid(workdir)}


def zero_path(workdir: Path, source_index: int) -> Path:
    return workdir / "zero" / f"{geometry_label(source_index)}.h5"


def field_path(workdir: Path, source_index: int, signed_field: float) -> Path:
    return (
        workdir / "fields" / geometry_label(source_index)
        / f"{field_label(signed_field)}.h5"
    )


def pass_h5(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(load_h5(path)["attrs"].get("converged", False))
    except Exception:
        return False
