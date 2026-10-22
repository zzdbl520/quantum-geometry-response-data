#!/usr/bin/env python3
"""Global signed-field response for the transported C2 SA6 model.

Input: a PASS global C2 SA6 transport directory.
Method: equal-weight SA6-CASSCF(8,8)/aug-cc-pVTZ at each signed field,
initialized independently from the same certified zero-field frame.  Every
field direction uses a direction-adapted C2v subgroup whose principal axis is
the field axis.  Fixed subgroup root counts continue the prescribed six-state
manifold instead of replacing it by the six lowest C1 roots.

The primary observable is the gauge-invariant rank-three target manifold
P={X 1Sigma_g+, Delta_g(x2-y2), Delta_g(xy)}.  Individual root labels are
reported only as diagnostics; all crossing conclusions use subspace polar
transport, effective curvature matrices, and projector metrics.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


BASIS = "aug-cc-pvtz"
NCORE = 2
NCAS = 8
NELECAS = (4, 4)
NSTATE = 6
NRETAIN = NCORE + NCAS
FULL_NELECAS = (NELECAS[0] + NCORE, NELECAS[1] + NCORE)
SPIN_PENALTY_SHIFT = 1.0
HOMING_ANCHOR_SV_MIN = 0.50
HOMING_LOCAL_SV_EXPAND = 0.70

TARGET_NAMES = ("X_1Sigma_g+", "Delta_g_x2-y2", "Delta_g_xy")
DEFAULT_R = (1.650000000, 1.652500000, 1.652734375, 1.652812500, 1.655000000)
DEFAULT_XY_AUDIT_R = (1.20, 1.65, 1.75, 2.20, 3.00)
DEFAULT_FIELDS = (5.0e-5, 1.0e-4, 2.5e-4, 5.0e-4)
EH_TO_EV = 27.211386245988

GATES = {
    "zero_target_subspace_sv_min": 0.98,
    "plus_minus_target_subspace_sv_min": 0.95,
    "active_subspace_sv_min": 0.995,
    "retained_subspace_sv_min": 0.995,
    "energy_parity_abs_Eh_max": 5.0e-8,
    "ensemble_noon_abs_max": 0.10,
    "spin_square_abs_max": 1.0e-6,
    "fit_relative_spread_max": 0.05,
    "xy_relative_error_max": 0.02,
    "common_frame_zero_offset_abs_Eh_max": 5.0e-8,
}

# The zero-field H5 files store CI roots in the D2h solver order
#   3 Ag, B1g, B2u, B3u.
# A field leaves a C2v subgroup when its direction is chosen as the subgroup
# z axis.  The entries below give (subgroup irrep, zero-field root indices).
# Keeping these counts fixed prevents unrelated C1 roots from entering the
# state-averaged orbital functional.
FIELD_STATE_BLOCKS = {
    "z": (("A1", (0, 1, 2)), ("A2", (3,)), ("B2", (4,)), ("B1", (5,))),
    "x": (("A1", (0, 1, 2, 5)), ("B2", (3, 4))),
    "y": (("A1", (0, 1, 2, 4)), ("B2", (3, 5))),
}


def subgroup_axes(axis: str) -> np.ndarray:
    """Return right-handed C2v axes with subgroup z along the field.

    Rows are the subgroup x, y, z axes expressed in the fixed molecular-frame
    Cartesian coordinates used by the AO integrals.  The nuclei remain on the
    original z axis; only the symmetry coordinate frame changes.
    """
    axes = {
        "z": np.eye(3),
        "x": np.asarray(((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0))),
        "y": np.asarray(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
    }
    try:
        result = axes[axis]
    except KeyError as exc:
        raise ValueError(f"Unknown field axis {axis!r}") from exc
    if not np.allclose(result @ result.T, np.eye(3), atol=1.0e-14):
        raise RuntimeError(f"Nonorthogonal subgroup axes for {axis}")
    if np.linalg.det(result) < 1.0 - 1.0e-14:
        raise RuntimeError(f"Left-handed subgroup axes for {axis}")
    return result


def subgroup_root_order(axis: str) -> list[int]:
    return [index for _, indices in FIELD_STATE_BLOCKS[axis] for index in indices]


def qc_modules():
    try:
        import h5py
        from pyscf import fci, gto, mcscf, scf
        from pyscf.fci import addons as fci_addons
        from pyscf.fci import cistring, spin_op
    except ImportError as exc:
        raise SystemExit(
            "This command requires h5py, SciPy-compatible NumPy, and PySCF."
        ) from exc
    return h5py, fci, gto, mcscf, scf, fci_addons, cistring, spin_op


def r_label(r: float) -> str:
    return f"R_{r:.9f}".rstrip("0").rstrip(".").replace(".", "p")


def f_label(value: float) -> str:
    sign = "p" if value > 0 else "m"
    return f"F_{sign}{abs(value):.1e}"


def task_is_excluded(record: dict[str, Any]) -> bool:
    exclusion = record.get("exclusion")
    return (
        isinstance(exclusion, dict)
        and str(exclusion.get("status", "")) == "EXCLUDED"
    )


def parse_task_id_spec(spec: str) -> list[int]:
    """Parse comma-separated task ids and inclusive ranges such as 8-15."""
    selected: set[int] = set()
    for token in (item.strip() for item in spec.split(",")):
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if start < 0 or stop < start:
                raise ValueError(f"Invalid task-id range {token!r}")
            selected.update(range(start, stop + 1))
        else:
            value = int(token)
            if value < 0:
                raise ValueError(f"Negative task id {value}")
            selected.add(value)
    if not selected:
        raise ValueError("No task ids were selected")
    return sorted(selected)


def partition_manifest_exclusions(
    records: Sequence[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[float, str], str],
]:
    """Return active tasks, excluded tasks, and fully excluded axis blocks.

    Exclusions are allowed only for a complete (R, axis) signed-field ladder.
    This prevents a missing field magnitude or one sign from being silently
    interpreted as a valid zero-field limit.
    """
    axis_groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        axis_groups[(float(record["R_angstrom"]), str(record["axis"]))].append(record)

    excluded_axes: dict[tuple[float, str], str] = {}
    for key, group in axis_groups.items():
        flags = [task_is_excluded(record) for record in group]
        if any(flags) and not all(flags):
            selected = [
                int(record["task_id"])
                for record, flag in zip(group, flags)
                if flag
            ]
            raise ValueError(
                f"Partial exclusion at R={key[0]:.10g}, axis={key[1]}: "
                f"tasks {selected}. Exclude the complete signed-field axis block."
            )
        if all(flags):
            reasons = sorted({
                str(record["exclusion"].get("reason", "")).strip()
                for record in group
            })
            if not reasons or any(not reason for reason in reasons):
                raise ValueError(
                    f"Excluded axis block R={key[0]:.10g}, axis={key[1]} "
                    "has no documented reason"
                )
            excluded_axes[key] = " | ".join(reasons)

    active = [record for record in records if not task_is_excluded(record)]
    excluded = [record for record in records if task_is_excluded(record)]
    return active, excluded, excluded_axes


def molecule(r: float, *, verbose: int = 4, memory_mb: int = 120000):
    _, _, gto, _, _, _, _, _ = qc_modules()
    return gto.M(
        atom=f"C 0 0 {-r/2:.14f}; C 0 0 {r/2:.14f}",
        unit="Angstrom",
        basis=BASIS,
        charge=0,
        spin=0,
        symmetry=False,
        verbose=verbose,
        max_memory=memory_mb,
    )


def field_molecule(
    r: float, axis: str, *, verbose: int = 4, memory_mb: int = 120000
):
    """Build C2 in a direction-adapted C2v symmetry coordinate frame.

    AO ordering and nuclear coordinates are unchanged from :func:`molecule`.
    Only the symmetry axes and SALCs are replaced, so zero-field MO and CI
    frames remain valid initial guesses without a spatial orbital rotation.
    """
    from pyscf import symm

    mol = molecule(r, verbose=verbose, memory_mb=memory_mb)
    axes = subgroup_axes(axis)
    origin = np.zeros(3)
    symm_orb, irrep_id = symm.symm_adapted_basis(
        mol, "C2v", origin, axes
    )
    mol.symmetry = True
    mol.topgroup = "D2h"
    mol.groupname = "C2v"
    mol._symm_orig = origin
    mol._symm_axes = axes
    mol.symm_orb = symm_orb
    mol.irrep_id = list(map(int, irrep_id))
    mol.irrep_name = [
        symm.irrep_id2name("C2v", value) for value in mol.irrep_id
    ]
    return mol


def tag_symmetry_orbitals(mol, mo: np.ndarray) -> np.ndarray:
    from pyscf import lib, symm

    s = np.asarray(mol.intor_symmetric("int1e_ovlp"))
    labels = symm.label_orb_symm(
        mol, mol.irrep_name, mol.symm_orb, np.asarray(mo), s=s, check=True
    )
    ids = [
        symm.irrep_name2id(mol.groupname, str(label))
        if isinstance(label, (str, np.str_)) else int(label)
        for label in labels
    ]
    return lib.tag_array(np.asarray(mo), orbsym=np.asarray(ids, dtype=int))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(text: str) -> list[float]:
    values = sorted(set(float(x) for x in text.split(",") if x.strip()))
    if not values:
        raise ValueError("Empty numeric list")
    return values


def grid_field_points(path: Path) -> list[float]:
    rows = read_csv(path)
    values = [
        float(row["R_angstrom"])
        for row in rows
        if str(row.get("field_use", "0")).strip().lower() in {"1", "true", "yes"}
    ]
    if not values:
        raise ValueError(f"No field_use points in {path}")
    return sorted(set(values))


def point_index(zero_root: Path) -> dict[float, Path]:
    result: dict[float, Path] = {}
    for summary_path in (zero_root / "points").glob("R_*/summary.json"):
        data = json.loads(summary_path.read_text())
        if data.get("status") != "PASS":
            continue
        result[float(data["R_angstrom"])] = summary_path.parent
    return result


def closest_point(index: dict[float, Path], target: float, tolerance: float = 1.0e-10) -> tuple[float, Path]:
    r = min(index, key=lambda x: abs(x - target))
    if abs(r - target) > tolerance:
        raise FileNotFoundError(f"No certified zero point at R={target:.12f} A")
    return r, index[r]


def identity_index(zero_root: Path) -> dict[float, dict[str, int]]:
    output: dict[float, dict[str, int]] = defaultdict(dict)
    for row in read_csv(zero_root / "analysis" / "state_identity.csv"):
        output[float(row["R_angstrom"])][row["identity"]] = int(row["solver_order"])
    return dict(output)


def zero_projected_model(zero_root: Path, r_values: Sequence[float], output: Path) -> list[dict[str, Any]]:
    """Evaluate the restricted six-state Van Vleck alpha/metric matrices."""
    identities = identity_index(zero_root)
    points = point_index(zero_root)
    rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    for requested_r in r_values:
        r, pdir = closest_point(points, requested_r)
        point = load_h5(pdir / "point.h5")
        energies = np.asarray(point["e_states_Eh"], dtype=float)
        target = [identities[r][name] for name in TARGET_NAMES]
        outside = [i for i in range(NSTATE) if i not in target]
        for axis_index, axis in enumerate("xyz"):
            if "mu_P_xyz" in point:
                mu = np.asarray(point["mu_P_xyz"][axis_index], dtype=float)
            else:
                mu = np.loadtxt(pdir / f"mu_P_{axis}.csv", delimiter=",")
            p_mu_p = mu[np.ix_(target, target)]
            alpha = np.zeros((3, 3))
            metric = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    for n in outside:
                        di = energies[n] - energies[target[i]]
                        dj = energies[n] - energies[target[j]]
                        prod = mu[target[i], n] * mu[n, target[j]]
                        alpha[i, j] += prod * (1.0 / di + 1.0 / dj)
                        metric[i, j] += 0.5 * prod * (1.0 / di**2 + 1.0 / dj**2)
            alpha = 0.5 * (alpha + alpha.T)
            metric = 0.5 * (metric + metric.T)
            ae, av = np.linalg.eigh(alpha)
            ge = np.linalg.eigvalsh(metric)
            key = f"{r_label(r)}_{axis}"
            matrices[f"{key}_alpha"] = alpha
            matrices[f"{key}_alpha_eigenvectors"] = av
            matrices[f"{key}_metric"] = metric
            rows.append({
                "R_angstrom": r,
                "axis": axis,
                "X_minus_Delta_mean_Eh": energies[target[0]] - np.mean(energies[target[1:]]),
                "max_abs_P_mu_P_au": float(np.max(np.abs(p_mu_p))),
                "alpha_P_eig_0": float(ae[0]),
                "alpha_P_eig_1": float(ae[1]),
                "alpha_P_eig_2": float(ae[2]),
                "metric_P_eig_0": float(ge[0]),
                "metric_P_eig_1": float(ge[1]),
                "metric_P_eig_2": float(ge[2]),
            })
    write_csv(output / "zero_projected_model.csv", rows)
    np.savez(output / "zero_projected_matrices.npz", **matrices)
    return rows


def prepare(args: argparse.Namespace) -> None:
    zero_root = Path(args.zero_root).resolve()
    workdir = Path(args.workdir).resolve()
    production = json.loads((zero_root / "production_summary.json").read_text())
    transport = production.get("transport_summary", {})
    if production.get("status") != "PASS" or transport.get("status") != "PASS":
        raise SystemExit("Signed fields are locked: zero-field transport is not PASS")
    if args.r_values:
        r_values = parse_float_list(args.r_values)
    else:
        grid_path = Path(args.grid_csv).resolve() if args.grid_csv else zero_root / "geometry_grid.csv"
        r_values = grid_field_points(grid_path)
    xy_audit_r = parse_float_list(args.xy_audit_r)
    fields = parse_float_list(args.fields)
    if any(f <= 0 for f in fields):
        raise ValueError("Field magnitudes must be positive")
    points = point_index(zero_root)
    workdir.mkdir(parents=True, exist_ok=True)
    model_rows = zero_projected_model(zero_root, r_values, workdir / "zero_model")

    records: list[dict[str, Any]] = []
    task_id = 0
    for requested_r in r_values:
        r, pdir = closest_point(points, requested_r)
        axes = ["x", "z"]
        if any(abs(r - x) < 1.0e-10 for x in xy_audit_r):
            axes.append("y")
        for axis in axes:
            unit = np.zeros(3)
            unit["xyz".index(axis)] = 1.0
            for magnitude in fields:
                for sign in (-1, 1):
                    signed = sign * magnitude
                    output = workdir / "fields" / r_label(r) / axis / f"{f_label(signed)}.h5"
                    records.append({
                        "task_id": task_id,
                        "R_angstrom": r,
                        "axis": axis,
                        "field_magnitude_au": magnitude,
                        "field_sign": sign,
                        "field_vector_au": (signed * unit).tolist(),
                        "residual_group": "C2v",
                        "subgroup_axes": subgroup_axes(axis).tolist(),
                        "state_blocks": [
                            {"irrep": irrep, "zero_root_indices": list(indices)}
                            for irrep, indices in FIELD_STATE_BLOCKS[axis]
                        ],
                        "zero_h5": str(pdir / "point.h5"),
                        "output_h5": str(output),
                    })
                    task_id += 1
    manifest = workdir / "field_manifest.jsonl"
    manifest.write_text("".join(json.dumps(x) + "\n" for x in records))
    protocol = {
        "status": "READY",
        "method": (
            "equal-weight SA6-CASSCF(8,8)/aug-cc-pVTZ signed-field "
            "rank-three transport with direction-adapted C2v root blocks"
        ),
        "field_state_policy": (
            "fixed residual-symmetry root counts plus locally continuous "
            "CI-subspace homing under an immutable zero-field identity gate; "
            "never the six lowest C1 roots"
        ),
        "root_homing_anchor_sv_min": HOMING_ANCHOR_SV_MIN,
        "root_homing_local_expand_sv": HOMING_LOCAL_SV_EXPAND,
        "state_blocks_by_axis": {
            axis: [
                {"irrep": irrep, "zero_root_indices": list(indices)}
                for irrep, indices in blocks
            ]
            for axis, blocks in FIELD_STATE_BLOCKS.items()
        },
        "target_manifold": list(TARGET_NAMES),
        "zero_transport_summary": transport,
        "r_values_angstrom": r_values,
        "xy_audit_r_values_angstrom": xy_audit_r,
        "field_magnitudes_au": fields,
        "task_count": len(records),
        "gates": GATES,
        "zero_model_max_abs_P_mu_P_au": max(x["max_abs_P_mu_P_au"] for x in model_rows),
        "field_convention": "H(F)=H0+F.r_electron",
        "downstream_observables": [
            "alpha_perpendicular", "alpha_parallel", "alpha_isotropic",
            "Tr_gFF_target", "Tr_A_target", "Delta_geom_target",
        ],
    }
    (workdir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))
    print(f"Manifest: {manifest}")


def load_h5(path: Path) -> dict[str, Any]:
    h5py, *_ = qc_modules()
    with h5py.File(path, "r") as h5:
        out = {key: np.asarray(h5[key]) for key in h5.keys()}
        out["attrs"] = {key: h5.attrs[key] for key in h5.attrs.keys()}
    return out


def save_h5(path: Path, arrays: dict[str, Any], attrs: dict[str, Any]) -> None:
    h5py, *_ = qc_modules()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as h5:
        for key, value in attrs.items():
            h5.attrs[key] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
        for key, value in arrays.items():
            h5.create_dataset(key, data=value)
    os.replace(tmp, path)


def build_field_mf(mol, field: np.ndarray, mo0: np.ndarray):
    _, _, _, _, scf, _, _, _ = qc_modules()
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-11
    mf.max_cycle = 0
    h0 = np.asarray(mf.get_hcore())
    r_ao = np.asarray(mol.intor_symmetric("int1e_r", comp=3))
    hfield = h0 + np.einsum("x,xij->ij", field, r_ao)
    offdiag = []
    for i, left in enumerate(mol.symm_orb):
        for right in mol.symm_orb[i + 1:]:
            offdiag.append(float(np.max(np.abs(left.T @ hfield @ right))))
    symmetry_offdiag_max = max(offdiag, default=0.0)
    if symmetry_offdiag_max > 1.0e-8:
        raise RuntimeError(
            "Field Hamiltonian violates the requested residual C2v symmetry: "
            f"maximum inter-irrep block={symmetry_offdiag_max:.3e} Eh"
        )
    mf.get_hcore = lambda *a, **k: hfield
    s = np.asarray(mf.get_ovlp())
    error = float(np.max(np.abs(mo0.T @ s @ mo0 - np.eye(mo0.shape[1]))))
    if error > 1.0e-8:
        raise RuntimeError(f"Zero-field MO/AO-order mismatch: orthogonality error={error:.3e}")
    occ = np.zeros(mo0.shape[1])
    occ[: mol.nelectron // 2] = 2.0
    dm0 = (mo0 * occ) @ mo0.T
    mf.kernel(dm0=dm0)
    mf.mo_coeff = tag_symmetry_orbitals(mol, mo0)
    mf.mo_occ = occ
    mf._field_symmetry_offdiag_max = symmetry_offdiag_max
    return mf


def ci_roots(ci: Any) -> list[np.ndarray]:
    if isinstance(ci, (list, tuple)):
        return [np.array(x, copy=True) for x in ci]
    arr = np.asarray(ci)
    if arr.shape[0] != NSTATE:
        raise ValueError(f"Expected {NSTATE} CI roots, got shape {arr.shape}")
    return [np.array(x, copy=True) for x in arr]


def project_singlet_ci(vector: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Project an M_S=0 determinant CI matrix onto strict spin singlet form.

    PySCF's direct_spin0 solver requires C(alpha,beta)=C(beta,alpha) to
    numerical precision.  Symmetry-resolved direct_spin1_symm roots can have
    tiny antisymmetric contamination even when their reported <S^2> is zero.
    """
    raw = np.asarray(vector)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError(f"Singlet CI root must be a square matrix, got {raw.shape}")
    raw_norm = float(np.linalg.norm(raw))
    if raw_norm <= 1.0e-14:
        raise ValueError("Cannot project a zero-norm CI root")
    transpose_overlap = float(np.vdot(raw, raw.T).real / raw_norm**2)
    projected = 0.5 * (raw + raw.T)
    removed_norm = float(np.linalg.norm(raw - projected) / raw_norm)
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm <= 1.0e-14:
        raise ValueError("CI root has no spin-singlet component")
    projected = np.asarray(projected / projected_norm)
    diagnostics = {
        "raw_norm": raw_norm,
        "raw_transpose_overlap": transpose_overlap,
        "antisymmetric_removed_relative_norm": removed_norm,
        "projected_transpose_error": float(np.max(np.abs(projected - projected.T))),
        "projected_norm_error": abs(float(np.linalg.norm(projected)) - 1.0),
    }
    return projected, diagnostics


def project_singlet_roots(ci: Any) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    projected, diagnostics = [], []
    for root in ci_roots(ci):
        clean, report = project_singlet_ci(root)
        projected.append(clean)
        diagnostics.append(report)
    return projected, diagnostics


def copy_ci(ci: Any) -> list[np.ndarray]:
    projected, _ = project_singlet_roots(ci)
    return projected


class SubspaceHomingFCISolver:
    """Mixin combining locally smooth and globally anchored CI homing."""

    __name_mixin__ = "SubspaceHoming"
    _keys = {
        "homing_desired_roots", "homing_irrep", "homing_min_sv_seen",
        "homing_last_min_sv", "homing_last_candidate_count",
        "homing_anchor_references", "homing_min_local_sv_seen",
        "homing_last_local_sv",
    }

    @staticmethod
    def _references(ci0, desired_roots: int, norb: int, nelec) -> list[np.ndarray]:
        """Normalize PySCF's several CI0 layouts into determinant matrices.

        ``state_average_mix_`` passes multi-root blocks as a list/stack but can
        pass a singleton symmetry block as one flattened FCI vector.  In the
        present CAS(8,8), that singleton has length 4900 rather than shape
        (70,70).  Treating the flat coefficients as a sequence of roots makes
        root homing see 4900 references.  Determine the exact determinant
        dimensions from ``norb`` and ``nelec`` and reshape without changing
        coefficients or normalization.
        """
        if isinstance(nelec, (tuple, list, np.ndarray)):
            neleca, nelecb = (int(nelec[0]), int(nelec[1]))
        else:
            total = int(nelec)
            neleca = total // 2
            nelecb = total - neleca
        root_shape = (
            math.comb(int(norb), neleca),
            math.comb(int(norb), nelecb),
        )
        root_size = root_shape[0] * root_shape[1]

        try:
            packed = np.asarray(ci0)
        except ValueError:
            packed = np.asarray(ci0, dtype=object)
        expected_size = desired_roots * root_size
        if packed.dtype != object and packed.size == expected_size:
            packed = packed.reshape((desired_roots,) + root_shape)
            return [np.array(packed[index], copy=False) for index in range(desired_roots)]

        if isinstance(ci0, (list, tuple)) and len(ci0) == desired_roots:
            references = []
            for vector in ci0:
                array = np.asarray(vector)
                if array.size != root_size:
                    raise RuntimeError(
                        "Root-homing CI reference has "
                        f"{array.size} coefficients; expected {root_size}"
                    )
                references.append(array.reshape(root_shape))
            return references

        raise RuntimeError(
            "Root-homing could not decode CI references: "
            f"desired_roots={desired_roots}, received_shape={packed.shape}, "
            f"received_size={packed.size}, root_shape={root_shape}"
        )

    def kernel(
        self, h1, h2, norb, nelec, ci0=None, nroots=None, **kwargs
    ):
        desired_roots = int(self.homing_desired_roots)
        irrep = str(self.homing_irrep)
        anchor_ci = getattr(self, "homing_anchor_references", None)
        if ci0 is None and anchor_ci is None:
            return super().kernel(
                h1, h2, norb, nelec, ci0=ci0,
                nroots=desired_roots, **kwargs
            )
        # Incoming CI vectors serve both as Davidson guesses and as the local
        # continuity reference.  The immutable zero-field vectors are a hard
        # identity gate.  Selecting locally among globally admissible
        # subspaces prevents both accumulated branch drift (local-only homing)
        # and sawtooth switching between near-equivalent anchored subsets
        # (anchor-only homing).
        incoming = self._references(
            anchor_ci if ci0 is None else ci0,
            desired_roots, norb, nelec,
        )
        anchors = self._references(
            incoming if anchor_ci is None else anchor_ci,
            desired_roots, norb, nelec,
        )
        if len(anchors) != desired_roots:
            raise RuntimeError(
                f"Root-homing {irrep} expected {desired_roots} anchors, "
                f"received {len(anchors)}"
            )
        anchor_matrix = np.stack([
            np.asarray(vector).ravel() / np.linalg.norm(vector)
            for vector in anchors
        ])
        incoming_matrix = np.stack([
            np.asarray(vector).ravel() / np.linalg.norm(vector)
            for vector in incoming
        ])

        best = None
        best_anchor_fallback = None
        for extra in (4, 10):
            search_roots = desired_roots + extra
            energies, candidates = super().kernel(
                h1, h2, norb, nelec, ci0=incoming,
                nroots=search_roots, **kwargs
            )
            if search_roots == 1:
                energies = np.asarray([energies], dtype=float)
                candidates = [candidates]
            else:
                energies = np.asarray(energies, dtype=float)
                candidates = list(candidates)
            candidate_converged = np.atleast_1d(
                getattr(self, "converged", True)
            ).copy()
            candidate_matrix = np.stack([
                np.asarray(vector).ravel() / np.linalg.norm(vector)
                for vector in candidates
            ])
            anchor_overlap = anchor_matrix.conj() @ candidate_matrix.T
            local_overlap = incoming_matrix.conj() @ candidate_matrix.T
            for columns in itertools.combinations(
                range(len(candidates)), desired_roots
            ):
                anchor_singular = np.linalg.svd(
                    anchor_overlap[:, columns], compute_uv=False
                )
                local_singular = np.linalg.svd(
                    local_overlap[:, columns], compute_uv=False
                )
                anchor_score = (
                    float(anchor_singular.min()),
                    float(np.sum(anchor_singular**2)),
                )
                local_score = (
                    float(local_singular.min()),
                    float(np.sum(local_singular**2)),
                )
                record = {
                    "score": local_score + anchor_score,
                    "anchor_score": anchor_score,
                    "local_score": local_score,
                    "columns": list(columns),
                    "energies": energies,
                    "candidates": candidates,
                    "candidate_count": len(candidates),
                    "converged": candidate_converged,
                }
                if (
                    best_anchor_fallback is None
                    or anchor_score > best_anchor_fallback["anchor_score"]
                ):
                    best_anchor_fallback = record
                # The zero-field anchor is a hard identity gate.  Among all
                # admissible subspaces, select by overlap to the immediately
                # preceding CI frame so the CASSCF objective remains locally
                # smooth instead of switching between near-equivalent anchor
                # matches.
                if anchor_score[0] < HOMING_ANCHOR_SV_MIN:
                    continue
                if best is None or record["score"] > best["score"]:
                    best = record
            if (
                best is not None
                and best["local_score"][0] >= HOMING_LOCAL_SV_EXPAND
            ):
                break

        if best is None:
            assert best_anchor_fallback is not None
            anchor_min = float(best_anchor_fallback["anchor_score"][0])
            raise RuntimeError(
                f"Root homing lost the {irrep} anchored subspace: "
                f"anchor min SV={anchor_min:.6f} within "
                f"{best_anchor_fallback['candidate_count']} candidate roots"
            )
        anchor_min_sv = float(best["anchor_score"][0])
        local_min_sv = float(best["local_score"][0])
        selected = best["columns"]
        selected_energies = np.asarray(best["energies"])[selected]
        selected_ci = [best["candidates"][index] for index in selected]
        self.homing_last_min_sv = anchor_min_sv
        self.homing_min_sv_seen = min(
            float(self.homing_min_sv_seen), anchor_min_sv
        )
        self.homing_last_local_sv = local_min_sv
        self.homing_min_local_sv_seen = min(
            float(self.homing_min_local_sv_seen), local_min_sv
        )
        self.homing_last_candidate_count = int(best["candidate_count"])
        self.eci = selected_energies
        self.ci = selected_ci
        base_converged = np.atleast_1d(best["converged"])
        if len(base_converged) >= max(selected) + 1:
            selected_converged = np.asarray(base_converged)[selected]
            self.converged = (
                bool(selected_converged[0])
                if desired_roots == 1 else selected_converged.tolist()
            )
        if desired_roots == 1:
            self.eci = float(selected_energies[0])
            self.ci = selected_ci[0]
            return self.eci, self.ci
        return selected_energies, selected_ci


def install_subspace_homing(
    solver, desired_roots: int, irrep: str,
    anchor_references: Sequence[np.ndarray],
):
    """Attach robust state homing without shadowing the mixed-SA kernel."""
    from pyscf import lib

    solver.nroots = desired_roots
    solver.homing_desired_roots = desired_roots
    solver.homing_irrep = irrep
    solver.homing_min_sv_seen = 1.0
    solver.homing_last_min_sv = 1.0
    solver.homing_min_local_sv_seen = 1.0
    solver.homing_last_local_sv = 1.0
    solver.homing_last_candidate_count = desired_roots
    solver.homing_anchor_references = np.asarray(anchor_references)
    return lib.set_class(
        solver,
        (SubspaceHomingFCISolver, solver.__class__),
    )


def build_optimizer(
    mf, *, axis: str, optimizer: str, grad_tol: float,
    max_macro: int, max_step: float, level_shift: float,
    anchor_ci: Sequence[np.ndarray],
):
    _, fci, _, mcscf, _, fci_addons, _, _ = qc_modules()
    mc = mcscf.CASSCF(mf, NCAS, NELECAS, ncore=NCORE)
    # Preserve the exact residual symmetry and the zero-field state count in
    # each subgroup irrep.  A single C1 solver would instead optimize the six
    # lowest roots, which is a different manifold once external roots intrude.
    solvers = []
    anchor_offset = 0
    for irrep, zero_indices in FIELD_STATE_BLOCKS[axis]:
        block_size = len(zero_indices)
        block_anchor = anchor_ci[anchor_offset:anchor_offset + block_size]
        if len(block_anchor) != block_size:
            raise RuntimeError(
                f"Missing zero-field anchors for {axis}/{irrep}: "
                f"expected {block_size}, received {len(block_anchor)}"
            )
        anchor_offset += block_size
        solver = fci.direct_spin1_symm.FCI(mc.mol)
        solver.spin = 0
        solver.wfnsym = irrep
        solver.nroots = len(zero_indices)
        solver.conv_tol = 1.0e-11
        solver.max_cycle = 240
        solver = fci_addons.fix_spin_(
            solver, shift=SPIN_PENALTY_SHIFT, ss=0.0
        )
        solver = install_subspace_homing(
            solver, block_size, irrep, block_anchor
        )
        solvers.append(solver)
    if anchor_offset != NSTATE:
        raise RuntimeError(
            f"Field anchor partition covered {anchor_offset} roots, expected {NSTATE}"
        )
    mc.conv_tol = 1.0e-10
    mc.conv_tol_grad = grad_tol
    mc.max_cycle_macro = max_macro
    mc.max_cycle_micro = 8 if optimizer == "first_order" else 20
    mc.max_stepsize = max_step
    mc.ah_level_shift = level_shift
    mc.ah_grad_trust_region = 1.5
    mc.kf_interval = 2
    mc.canonicalization = False
    mc = mcscf.state_average_mix_(
        mc, solvers, [1.0 / NSTATE] * NSTATE
    )
    if optimizer == "newton":
        mc = mc.newton()
        mc.max_cycle_macro = max_macro
        mc.max_cycle_micro = 20
        mc.max_stepsize = max_step
        mc.ah_level_shift = level_shift
        mc.ah_grad_trust_region = 1.5
        mc.kf_interval = 2
        mc.canonicalization = False
    return mc


def optimize_stage(mc, mo0: np.ndarray, ci0: list[np.ndarray], name: str):
    best = {"gradient": float("inf"), "mo": np.array(mo0, copy=True), "ci": copy_ci(ci0)}

    def callback(envs):
        gradient = envs.get("norm_gall", envs.get("norm_gorb"))
        mo = envs.get("mo")
        ci = envs.get("fcivec")
        if gradient is None or mo is None or ci is None:
            return
        value = float(gradient)
        if np.isfinite(value) and value < best["gradient"]:
            best["gradient"] = value
            best["mo"] = np.array(mo, copy=True)
            best["ci"] = copy_ci(ci)

    mc.callback = callback
    started = time.time()
    mc.kernel(tag_symmetry_orbitals(mc.mol, mo0), ci0=ci0)
    report = {
        "name": name,
        "converged": bool(mc.converged),
        "wall_seconds": time.time() - started,
        "best_gradient": best["gradient"],
        "root_homing": [
            {
                "irrep": str(getattr(solver, "homing_irrep", "unknown")),
                "minimum_anchor_sv_seen": float(
                    getattr(solver, "homing_min_sv_seen", float("nan"))
                ),
                "minimum_local_sv_seen": float(
                    getattr(
                        solver, "homing_min_local_sv_seen", float("nan")
                    )
                ),
                "last_candidate_count": int(
                    getattr(
                        solver, "homing_last_candidate_count", solver.nroots
                    )
                ),
            }
            for solver in mc.fcisolver.fcisolvers
        ],
    }
    print(json.dumps(report), flush=True)
    if mc.converged:
        return np.asarray(mc.mo_coeff), copy_ci(mc.ci), report
    return best["mo"], best["ci"], report


def run_task(args: argparse.Namespace) -> None:
    records = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    if not 0 <= args.index < len(records):
        raise SystemExit(f"Task index must lie in [0,{len(records)-1}]")
    record = records[args.index]
    if task_is_excluded(record):
        reason = str(record["exclusion"].get("reason", "unspecified"))
        print(f"SKIP EXCLUDED task {args.index}: {reason}")
        return
    if record.get("residual_group") != "C2v":
        raise SystemExit(
            "Refusing a legacy C1 manifest. Run prepare into a new field work "
            "directory with this package version."
        )
    output = Path(record["output_h5"])
    if output.exists() and not args.force:
        try:
            existing = load_h5(output)
            if (
                bool(existing["attrs"].get("converged", False))
                and str(existing["attrs"].get("residual_group", "")) == "C2v"
            ):
                print(f"SKIP completed task {args.index}: {output}")
                return
        except Exception:
            pass
        raise SystemExit(f"Refusing to overwrite incomplete {output}; use --force")

    zero = load_h5(Path(record["zero_h5"]))
    initial_frame_path = (
        Path(args.restart_h5).resolve()
        if args.restart_h5 else Path(record["zero_h5"])
    )
    initial_frame = load_h5(initial_frame_path)
    r = float(record["R_angstrom"])
    axis = str(record["axis"])
    field = np.asarray(record["field_vector_au"], dtype=float)
    if np.linalg.norm(field) <= 0.0 or not np.allclose(
        abs(float(field @ subgroup_axes(axis)[2])),
        float(np.linalg.norm(field)),
        atol=1.0e-14,
    ):
        raise RuntimeError(
            f"Field vector {field.tolist()} is not parallel to subgroup z for axis {axis}"
        )
    mol = field_molecule(r, axis, memory_mb=args.memory_mb)
    mf = build_field_mf(mol, field, np.asarray(initial_frame["mo_coeff"]))
    mo = np.asarray(initial_frame["mo_coeff"])
    ci, initial_ci_projection = project_singlet_roots(
        initial_frame["ci_vectors"]
    )
    zero_ci, zero_anchor_projection = project_singlet_roots(
        zero["ci_vectors"]
    )
    zero_anchor_order = subgroup_root_order(axis)
    anchor_ci = [zero_ci[index] for index in zero_anchor_order]
    source_attrs = initial_frame.get("attrs", {})
    source_is_same_subgroup = (
        str(source_attrs.get("residual_group", "")) == "C2v"
        and str(source_attrs.get("field_axis", "")) == axis
    )
    if not source_is_same_subgroup:
        order = subgroup_root_order(axis)
        ci = [ci[index] for index in order]
    else:
        order = list(range(NSTATE))
    print(json.dumps({
        "initial_ci_singlet_projection": initial_ci_projection,
        "maximum_antisymmetric_removed_relative_norm": max(
            x["antisymmetric_removed_relative_norm"] for x in initial_ci_projection
        ),
    }), flush=True)
    if args.rescue:
        stages = (
            ("rescue_settle", "first_order", 2.0e-5, 240, 0.0020, 1.00),
            ("rescue_newton", "newton", 1.0e-6, 240, 0.0010, 1.00),
            ("rescue_ultratight_settle", "first_order", 2.0e-6, 320, 0.0005, 2.00),
            ("rescue_ultratight_newton", "newton", 1.0e-6, 240, 0.0005, 2.00),
        )
    else:
        stages = (
            ("settle", "first_order", 1.0e-4, 60, 0.010, 0.50),
            ("damped_newton", "newton", 1.0e-6, 120, 0.005, 0.50),
            ("tight_settle", "first_order", 5.0e-7, 180, 0.002, 1.00),
            ("tight_newton", "newton", 1.0e-6, 120, 0.002, 1.00),
        )
    reports = []
    mc = None
    for name, optimizer, grad, macro, step, shift in stages:
        mc = build_optimizer(
            mf, axis=axis, optimizer=optimizer, grad_tol=grad, max_macro=macro,
            max_step=step, level_shift=shift, anchor_ci=anchor_ci,
        )
        mo, ci, report = optimize_stage(mc, mo, ci, name)
        reports.append(report)
        if optimizer == "newton" and mc.converged:
            break
    if mc is None or not mc.converged:
        raise RuntimeError(f"SA6 field task {args.index} did not converge")

    _, fci, _, _, _, _, _, spin_op = qc_modules()
    penalized_energies = np.asarray(mc.e_states, dtype=float)
    raw_ci = ci_roots(mc.ci)
    spins = np.asarray([spin_op.spin_square0(x, NCAS, NELECAS)[0] for x in raw_ci])
    if float(np.max(np.abs(spins))) > GATES["spin_square_abs_max"]:
        raise RuntimeError(
            "Spin-penalty SA6 returned a contaminated root: "
            f"max |<S^2>|={np.max(np.abs(spins)):.3e}"
        )
    # fix_spin_ diagonalizes H + shift*S^2 for ss=0. Remove its residual
    # numerical penalty from the reported physical energies. For an exact
    # singlet this correction is identically zero.
    energies = penalized_energies - SPIN_PENALTY_SHIFT * spins
    ci, final_ci_projection = project_singlet_roots(raw_ci)
    mo = np.asarray(mc.mo_coeff)
    rdm1 = np.stack([fci.direct_spin1.make_rdm1(x, NCAS, NELECAS) for x in ci])
    noons = np.stack([np.linalg.eigvalsh(0.5 * (d + d.T))[::-1] for d in rdm1])
    homing_reports = [
        {
            "irrep": str(getattr(solver, "homing_irrep", "unknown")),
            "minimum_anchor_sv_seen": float(
                getattr(solver, "homing_min_sv_seen", float("nan"))
            ),
            "minimum_local_sv_seen": float(
                getattr(solver, "homing_min_local_sv_seen", float("nan"))
            ),
            "last_anchor_sv": float(
                getattr(solver, "homing_last_min_sv", float("nan"))
            ),
            "last_local_sv": float(
                getattr(solver, "homing_last_local_sv", float("nan"))
            ),
            "last_candidate_count": int(
                getattr(solver, "homing_last_candidate_count", solver.nroots)
            ),
        }
        for solver in mc.fcisolver.fcisolvers
    ]
    attrs = {
        "kind": "C2_SA6_signed_field",
        "task_id": int(record["task_id"]),
        "R_angstrom": r,
        "axis": record["axis"],
        "field_magnitude_au": float(record["field_magnitude_au"]),
        "field_sign": int(record["field_sign"]),
        "field_vector_json": field.tolist(),
        "zero_h5": str(record["zero_h5"]),
        "initial_frame_h5": str(initial_frame_path),
        "rescue_mode": bool(args.rescue),
        "method": (
            "equal-weight SA6-CASSCF(8,8)/aug-cc-pVTZ in a "
            "direction-adapted C2v subgroup"
        ),
        "field_axis": axis,
        "residual_group": "C2v",
        "subgroup_axes_json": subgroup_axes(axis).tolist(),
        "state_blocks_json": [
            {"irrep": irrep, "zero_root_indices": list(indices)}
            for irrep, indices in FIELD_STATE_BLOCKS[axis]
        ],
        "initial_ci_reorder_json": order,
        "zero_anchor_ci_reorder_json": zero_anchor_order,
        "initial_frame_already_in_same_subgroup_order": source_is_same_subgroup,
        "field_hcore_inter_irrep_abs_max_Eh": float(
            mf._field_symmetry_offdiag_max
        ),
        "ci_solver": "pyscf.fci.direct_spin1_symm + fix_spin_(ss=0)",
        "root_homing": (
            "maximum local CI-subspace overlap within enlarged irrep root "
            "windows, subject to a fixed zero-field anchor min-SV gate of "
            f"{HOMING_ANCHOR_SV_MIN:.2f}"
        ),
        "root_homing_reports_json": homing_reports,
        "spin_penalty_shift_Eh": SPIN_PENALTY_SHIFT,
        "initial_ci_singlet_projection_json": initial_ci_projection,
        "zero_anchor_ci_singlet_projection_json": zero_anchor_projection,
        "maximum_initial_ci_antisymmetric_removed_relative_norm": max(
            x["antisymmetric_removed_relative_norm"] for x in initial_ci_projection
        ),
        "final_ci_singlet_projection_json": final_ci_projection,
        "maximum_final_ci_antisymmetric_removed_relative_norm": max(
            x["antisymmetric_removed_relative_norm"] for x in final_ci_projection
        ),
        "energies_corrected_for_residual_spin_penalty": True,
        "converged": True,
        "stage_reports_json": reports,
    }
    save_h5(output, {
        "e_states_Eh": energies,
        "e_states_spin_penalized_Eh": penalized_energies,
        "mo_coeff": mo,
        "ci_vectors": np.asarray(ci),
        "state_rdm1": rdm1,
        "state_noons": noons,
        "state_spin_square": spins,
    }, attrs)
    print(f"PASS task {args.index}: {output}")


def embed_active_ci(ci: np.ndarray) -> np.ndarray:
    _, _, _, _, _, _, cistring, _ = qc_modules()
    active_strings = cistring.make_strings(range(NCAS), NELECAS[0])
    full_strings = cistring.make_strings(range(NRETAIN), FULL_NELECAS[0])
    addresses = {int(s): i for i, s in enumerate(full_strings)}
    selected = [addresses[(int(s) << NCORE) | ((1 << NCORE) - 1)] for s in active_strings]
    out = np.zeros((len(full_strings), len(full_strings)), dtype=np.asarray(ci).dtype)
    out[np.ix_(selected, selected)] = np.asarray(ci).reshape(len(active_strings), len(active_strings))
    return out


def state_overlap(left: dict[str, Any], right: dict[str, Any], s_ao: np.ndarray) -> np.ndarray:
    _, _, _, _, _, fci_addons, _, _ = qc_modules()
    retained = left["mo_coeff"][:, :NRETAIN].T @ s_ao @ right["mo_coeff"][:, :NRETAIN]
    left_ci = [embed_active_ci(x) for x in left["ci_vectors"]]
    right_ci = [embed_active_ci(x) for x in right["ci_vectors"]]
    transformed = [
        fci_addons.transform_ci_for_orbital_rotation(x, NRETAIN, FULL_NELECAS, retained)
        for x in left_ci
    ]
    out = np.empty((len(left_ci), len(right_ci)))
    for i, bra in enumerate(transformed):
        for j, ket in enumerate(right_ci):
            out[i, j] = float(np.vdot(bra.ravel(), ket.ravel()).real)
    return out


def polar(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, s, vh = np.linalg.svd(matrix)
    return u @ vh, s


def select_target(overlap_zero_field: np.ndarray, target: Sequence[int]) -> tuple[list[int], np.ndarray, np.ndarray]:
    block = overlap_zero_field[np.asarray(target), :]
    best = None
    for columns in itertools.combinations(range(NSTATE), len(target)):
        candidate = block[:, columns]
        unitary, singular = polar(candidate)
        score = (float(singular.min()), float(np.sum(singular**2)))
        if best is None or score > best[0]:
            best = (score, list(columns), unitary, singular)
    assert best is not None
    return best[1], best[2], best[3]


def subspace_sv(left_mo: np.ndarray, right_mo: np.ndarray, s: np.ndarray, start: int, stop: int) -> float:
    block = left_mo[:, start:stop].T @ s @ right_mo[:, start:stop]
    return float(np.linalg.svd(block, compute_uv=False).min())


def orthogonal_ao_density(mo: np.ndarray, gamma: np.ndarray, s: np.ndarray) -> np.ndarray:
    c = mo[:, NCORE:NCORE + NCAS]
    d = c @ gamma @ c.T
    eig, vec = np.linalg.eigh(0.5 * (s + s.T))
    shalf = (vec * np.sqrt(eig)) @ vec.T
    return shalf @ d @ shalf


def scaled_spread(values: Sequence[float], fitted: Sequence[float]) -> float:
    """Residual on a relative scale with a 1 a.u. absolute floor.

    The floor is essential for the nominally dark eigenchannel and the
    longitudinal six-state diagnostic, whose exact values can be zero.
    """
    y = np.asarray(values)
    return float(np.max(np.abs(y - fitted)) / max(float(np.max(np.abs(y))), 1.0))


def fit_limit(fields: Sequence[float], values: Sequence[float]) -> tuple[float, float, float]:
    x = np.asarray(fields) ** 2
    y = np.asarray(values)
    all_fit = np.polyfit(x, y, 1)
    fitted = np.polyval(all_fit, x)
    small_fit = np.polyfit(x[:3], y[:3], 1)
    stability = abs(float(all_fit[1] - small_fit[1])) / max(abs(float(all_fit[1])), 1.0)
    return float(all_fit[1]), scaled_spread(y, fitted), stability


def fit_matrix_limit(
    fields: Sequence[float], matrices: Sequence[np.ndarray]
) -> tuple[np.ndarray, float, float]:
    """Fit a common-frame response matrix elementwise against F^2."""
    x = np.asarray(fields) ** 2
    y = np.asarray(matrices)
    flat = y.reshape(len(y), -1)
    design = np.column_stack((x, np.ones_like(x)))
    coefficients = np.linalg.lstsq(design, flat, rcond=None)[0]
    limit = coefficients[1].reshape(y.shape[1:])
    fitted = (design @ coefficients).reshape(y.shape)

    small_design = design[:3]
    small_coefficients = np.linalg.lstsq(small_design, flat[:3], rcond=None)[0]
    small_limit = small_coefficients[1].reshape(y.shape[1:])
    scale = max(float(np.max(np.linalg.norm(y, axis=(1, 2)))), 1.0)
    spread = float(np.max(np.linalg.norm(y - fitted, axis=(1, 2))) / scale)
    stability = float(np.linalg.norm(limit - small_limit) / max(np.linalg.norm(limit), 1.0))
    return 0.5 * (limit + limit.T), spread, stability


def fit_even_hamiltonian_limit(
    fields: Sequence[float], matrices: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Extract zero-field curvature with a free common-frame intercept.

    Fit H_even(F)=H_subgroup(0)-A*F^2/2+B*F^4. A field-independent
    difference between the D2h zero archive and the relaxed residual-symmetry
    frame belongs to the intercept and must not be divided by F^2.
    """
    f = np.asarray(fields, dtype=float)
    y = np.asarray(matrices)
    if len(f) < 4:
        raise ValueError("Even-Hamiltonian curvature requires at least four fields")
    fmax2 = float(np.max(f) ** 2)
    x = f**2 / fmax2
    flat = y.reshape(len(y), -1)

    coefficients = np.polynomial.polynomial.polyfit(x, flat, 2)
    h_c1_zero = coefficients[0].reshape(y.shape[1:])
    alpha = (-2.0 * coefficients[1] / fmax2).reshape(y.shape[1:])
    alpha = 0.5 * (alpha + alpha.T)
    fitted = np.polynomial.polynomial.polyval(x, coefficients).T.reshape(y.shape)
    response_scale = max(0.5 * float(np.linalg.norm(alpha)) * fmax2, 1.0e-14)
    spread = float(np.max(np.linalg.norm(y - fitted, axis=(1, 2))) / response_scale)

    linear = np.polynomial.polynomial.polyfit(x, flat, 1)
    alpha_linear = (-2.0 * linear[1] / fmax2).reshape(y.shape[1:])
    alpha_linear = 0.5 * (alpha_linear + alpha_linear.T)
    omit_smallest = np.polynomial.polynomial.polyfit(x[1:], flat[1:], 2)
    alpha_omit_smallest = (-2.0 * omit_smallest[1] / fmax2).reshape(y.shape[1:])
    alpha_omit_smallest = 0.5 * (alpha_omit_smallest + alpha_omit_smallest.T)
    stability = max(
        float(np.linalg.norm(alpha - alpha_linear)),
        float(np.linalg.norm(alpha - alpha_omit_smallest)),
    ) / max(float(np.linalg.norm(alpha)), 1.0)
    return 0.5 * (h_c1_zero + h_c1_zero.T), alpha, spread, stability


def task_status(args: argparse.Namespace) -> None:
    """Report completed and retryable task indices without changing results."""
    manifest = Path(args.manifest).resolve()
    records = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    rows = []
    retry = []
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        path = Path(record["output_h5"])
        state = "MISSING"
        detail = ""
        if task_is_excluded(record):
            state = "EXCLUDED"
            detail = str(record["exclusion"].get("reason", ""))
        elif path.exists():
            try:
                attrs = load_h5(path)["attrs"]
                state = (
                    "PASS"
                    if bool(attrs.get("converged", False))
                    and str(attrs.get("residual_group", "")) == "C2v"
                    else "INCOMPATIBLE_OR_INCOMPLETE"
                )
            except Exception as exc:
                state, detail = "CORRUPT", f"{type(exc).__name__}: {exc}"
        counts[state] += 1
        if state not in {"PASS", "EXCLUDED"}:
            retry.append(int(record["task_id"]))
        rows.append({
            "task_id": int(record["task_id"]), "status": state,
            "R_angstrom": record["R_angstrom"], "axis": record["axis"],
            "field_magnitude_au": record["field_magnitude_au"],
            "field_sign": record["field_sign"], "output_h5": str(path),
            "detail": detail,
        })
    output = manifest.parent / "task_status.csv"
    write_csv(output, rows)
    if args.retry_file:
        Path(args.retry_file).write_text("\n".join(map(str, retry)) + ("\n" if retry else ""))
    summary = {
        "task_count": len(records), "counts": dict(sorted(counts.items())),
        "retry_count": len(retry), "retry_indices": retry,
        "status_csv": str(output),
    }
    print(json.dumps(summary, indent=2))


def exclude_tasks(args: argparse.Namespace) -> None:
    """Document a scientifically justified complete-axis task exclusion."""
    manifest = Path(args.manifest).resolve()
    records = [
        json.loads(line)
        for line in manifest.read_text().splitlines()
        if line.strip()
    ]
    requested = set(parse_task_id_spec(args.task_ids))
    available = {int(record["task_id"]) for record in records}
    unknown = sorted(requested - available)
    if unknown:
        raise SystemExit(f"Unknown task ids: {unknown}")
    reason = args.reason.strip()
    if not reason:
        raise SystemExit("An exclusion reason is required")

    chosen_groups = {
        (float(record["R_angstrom"]), str(record["axis"]))
        for record in records
        if int(record["task_id"]) in requested
    }
    for key in chosen_groups:
        complete = {
            int(record["task_id"])
            for record in records
            if (float(record["R_angstrom"]), str(record["axis"])) == key
        }
        if not complete <= requested:
            missing = sorted(complete - requested)
            raise SystemExit(
                f"Refusing a partial exclusion at R={key[0]:.10g}, "
                f"axis={key[1]}; also select tasks {missing}"
            )

    timestamp = datetime.now(timezone.utc).isoformat()
    for record in records:
        if int(record["task_id"]) in requested:
            record["exclusion"] = {
                "status": "EXCLUDED",
                "reason": reason,
                "recorded_utc": timestamp,
            }

    # Validate the complete-axis invariant before changing the manifest.
    _, excluded, excluded_axes = partition_manifest_exclusions(records)
    backup = manifest.with_name(
        f"{manifest.stem}.before_exclusions{manifest.suffix}"
    )
    if not backup.exists():
        backup.write_text(manifest.read_text())
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record) + "\n" for record in records))
    os.replace(temporary, manifest)

    audit = {
        "status": "RECORDED",
        "manifest": str(manifest),
        "backup_manifest": str(backup),
        "recorded_utc": timestamp,
        "reason": reason,
        "excluded_task_ids": sorted(int(record["task_id"]) for record in excluded),
        "excluded_axis_blocks": [
            {"R_angstrom": r, "axis": axis, "reason": block_reason}
            for (r, axis), block_reason in sorted(excluded_axes.items())
        ],
    }
    audit_path = manifest.parent / "exclusion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


def analyze(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest).resolve()
    workdir = Path(args.workdir).resolve()
    all_records = [
        json.loads(x) for x in manifest.read_text().splitlines() if x.strip()
    ]
    try:
        records, excluded_records, excluded_axes = (
            partition_manifest_exclusions(all_records)
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not records:
        raise SystemExit("No active field tasks remain after exclusions")
    grouped: dict[tuple[float, str, float], dict[int, dict[str, Any]]] = defaultdict(dict)
    print(
        f"ANALYZE inventory: 0/{len(records)} active "
        f"({len(excluded_records)} excluded of {len(all_records)})",
        flush=True,
    )
    for record_number, record in enumerate(records, 1):
        if record.get("residual_group") != "C2v":
            raise SystemExit(
                "This manifest predates the direction-adapted C2v protocol. "
                "Prepare a new field work directory; do not mix C1 and C2v tasks."
            )
        path = Path(record["output_h5"])
        if not path.exists():
            raise SystemExit(f"Missing task {record['task_id']}: {path}")
        data = load_h5(path)
        if not bool(data["attrs"].get("converged", False)):
            raise SystemExit(f"Incomplete task {record['task_id']}: {path}")
        if str(data["attrs"].get("residual_group", "")) != "C2v":
            raise SystemExit(
                f"Task {record['task_id']} is not a C2v residual-symmetry result: {path}"
            )
        grouped[(float(record["R_angstrom"]), record["axis"], float(record["field_magnitude_au"]))][int(record["field_sign"])] = record
        if record_number % 100 == 0 or record_number == len(records):
            print(
                f"ANALYZE inventory: {record_number}/{len(records)}",
                flush=True,
            )

    zero_root = Path(records[0]["zero_h5"]).parents[2]
    identities = identity_index(zero_root)
    pair_rows: list[dict[str, Any]] = []
    alpha_matrices: dict[tuple[float, str, float], np.ndarray] = {}
    even_hamiltonians: dict[tuple[float, str, float], np.ndarray] = {}
    even_hamiltonians_rank6: dict[tuple[float, str, float], np.ndarray] = {}
    zero_hamiltonians: dict[float, np.ndarray] = {}
    zero_hamiltonians_rank6: dict[float, np.ndarray] = {}
    zero_term_blocks: dict[float, dict[str, list[int]]] = {}
    matrix_archive: dict[str, np.ndarray] = {}
    zero_gaps: dict[float, float] = {}
    grouped_items = sorted(grouped.items())
    print(
        f"ANALYZE signed pairs: 0/{len(grouped_items)}",
        flush=True,
    )
    for pair_number, ((r, axis, field), signs) in enumerate(grouped_items, 1):
        if set(signs) != {-1, 1}:
            raise SystemExit(f"Incomplete signed pair at R={r}, axis={axis}, F={field}")
        zero = load_h5(Path(signs[1]["zero_h5"]))
        minus = load_h5(Path(signs[-1]["output_h5"]))
        plus = load_h5(Path(signs[1]["output_h5"]))
        # Normalize zero-field key names to the field-frame names.
        zero_frame = {
            "mo_coeff": zero["mo_coeff"], "ci_vectors": zero["ci_vectors"],
            "e_states_Eh": zero["e_states_Eh"],
        }
        mol = molecule(r, verbose=0, memory_mb=args.memory_mb)
        s = np.asarray(mol.intor_symmetric("int1e_ovlp"))
        target = [identities[r][name] for name in TARGET_NAMES]
        ozm = state_overlap(zero_frame, minus, s)
        ozp = state_overlap(zero_frame, plus, s)
        sel_m, pm, svm = select_target(ozm, target)
        sel_p, pp, svp = select_target(ozp, target)
        hm = pm @ np.diag(minus["e_states_Eh"][sel_m]) @ pm.T
        hp = pp @ np.diag(plus["e_states_Eh"][sel_p]) @ pp.T
        h0 = np.diag(zero["e_states_Eh"][target])
        p6m, sv6m = polar(ozm)
        p6p, sv6p = polar(ozp)
        hm6 = p6m @ np.diag(minus["e_states_Eh"]) @ p6m.T
        hp6 = p6p @ np.diag(plus["e_states_Eh"]) @ p6p.T
        h06 = np.diag(zero["e_states_Eh"])
        zero_hamiltonians[r] = h0
        zero_hamiltonians_rank6[r] = h06
        zero_term_blocks[r] = {
            "X_1Sigma_g+": [identities[r]["X_1Sigma_g+"]],
            "Bp_1Sigma_g+": [identities[r]["Bp_1Sigma_g+"]],
            "Delta_g": [
                identities[r]["Delta_g_x2-y2"], identities[r]["Delta_g_xy"]
            ],
            "A_1Pi_u": [identities[r]["A_1Pi_u_y"], identities[r]["A_1Pi_u_x"]],
        }
        zero_gaps[r] = float(h0[0, 0] - 0.5 * (h0[1, 1] + h0[2, 2]))
        even_hamiltonians[(r, axis, field)] = 0.5 * (hp + hm)
        even_hamiltonians_rank6[(r, axis, field)] = 0.5 * (hp6 + hm6)
        alpha = -(hp + hm - 2.0 * h0) / field**2
        alpha = 0.5 * (alpha + alpha.T)
        ae, av = np.linalg.eigh(alpha)
        alpha_matrices[(r, axis, field)] = alpha
        archive_key = f"{r_label(r)}_{axis}_{f_label(field)}"
        matrix_archive[f"{archive_key}_alpha"] = alpha
        matrix_archive[f"{archive_key}_alpha_eigenvectors"] = av
        matrix_archive[f"{archive_key}_H_minus_in_zero_frame"] = hm
        matrix_archive[f"{archive_key}_H_plus_in_zero_frame"] = hp
        matrix_archive[f"{archive_key}_H6_minus_in_zero_frame"] = hm6
        matrix_archive[f"{archive_key}_H6_plus_in_zero_frame"] = hp6
        matrix_archive[f"{archive_key}_rank6_zero_minus_sv"] = sv6m
        matrix_archive[f"{archive_key}_rank6_zero_plus_sv"] = sv6p
        matrix_archive[f"{archive_key}_overlap_zero_minus"] = ozm
        matrix_archive[f"{archive_key}_overlap_zero_plus"] = ozp

        plus_selected = {
            "mo_coeff": plus["mo_coeff"], "ci_vectors": plus["ci_vectors"][sel_p],
        }
        minus_selected = {
            "mo_coeff": minus["mo_coeff"], "ci_vectors": minus["ci_vectors"][sel_m],
        }
        opm = state_overlap(minus_selected, plus_selected, s)
        _, svpm = polar(opm)
        g_trace = float((3.0 - np.sum(svpm**2)) / (4.0 * field**2))

        full_pm = state_overlap(minus, plus, s)
        full_sv = np.linalg.svd(full_pm, compute_uv=False)
        g6_trace = float((NSTATE - np.sum(full_sv**2)) / (4.0 * field**2))
        active_min = subspace_sv(minus["mo_coeff"], plus["mo_coeff"], s, NCORE, NCORE + NCAS)
        retained_min = subspace_sv(minus["mo_coeff"], plus["mo_coeff"], s, 0, NRETAIN)

        gm = np.mean(minus["state_rdm1"][sel_m], axis=0)
        gp = np.mean(plus["state_rdm1"][sel_p], axis=0)
        dm = orthogonal_ao_density(minus["mo_coeff"], gm, s)
        dp = orthogonal_ao_density(plus["mo_coeff"], gp, s)
        rdm_diff = float(np.linalg.norm(dm - dp))
        nm = np.linalg.eigvalsh(0.5 * (gm + gm.T))[::-1]
        np_ = np.linalg.eigvalsh(0.5 * (gp + gp.T))[::-1]
        noon_diff = float(np.max(np.abs(nm - np_)))
        parity = float(np.max(np.abs(np.sort(minus["e_states_Eh"][sel_m]) - np.sort(plus["e_states_Eh"][sel_p]))))
        spin_max = float(max(np.max(np.abs(minus["state_spin_square"][sel_m])), np.max(np.abs(plus["state_spin_square"][sel_p]))))
        # The raw +F/-F AO-density difference is intentionally diagnostic, not
        # a gate: inversion relates the two densities spatially, so they are
        # not expected to be equal in a fixed AO frame.  NOONs and energies are
        # inversion invariants and remain strict parity gates.
        pair_checks = {
            "zero_target_subspace_sv_min": (
                min(float(svm.min()), float(svp.min()))
                >= GATES["zero_target_subspace_sv_min"]
            ),
            "plus_minus_target_subspace_sv_min": (
                float(svpm.min())
                >= GATES["plus_minus_target_subspace_sv_min"]
            ),
            "active_subspace_sv_min": active_min >= GATES["active_subspace_sv_min"],
            "retained_subspace_sv_min": retained_min >= GATES["retained_subspace_sv_min"],
            "energy_parity_abs_Eh_max": parity <= GATES["energy_parity_abs_Eh_max"],
            "ensemble_noon_abs_max": noon_diff <= GATES["ensemble_noon_abs_max"],
            "spin_square_abs_max": spin_max <= GATES["spin_square_abs_max"],
        }
        failed_pair_gates = [name for name, ok in pair_checks.items() if not ok]
        passed = not failed_pair_gates
        pair_rows.append({
            "R_angstrom": r, "axis": axis, "field_au": field,
            "selected_minus_roots": " ".join(map(str, sel_m)),
            "selected_plus_roots": " ".join(map(str, sel_p)),
            "alpha_eig_0": float(ae[0]), "alpha_eig_1": float(ae[1]), "alpha_eig_2": float(ae[2]),
            "alpha_X": float(alpha[0, 0]),
            "alpha_Delta_x2-y2": float(alpha[1, 1]),
            "alpha_Delta_xy": float(alpha[2, 2]),
            "alpha_X_Delta_x2-y2": float(alpha[0, 1]),
            "alpha_X_Delta_xy": float(alpha[0, 2]),
            "alpha_Delta_component_coupling": float(alpha[1, 2]),
            "target_projector_metric_trace": g_trace,
            "rank6_projector_metric_trace": g6_trace,
            "zero_target_sv_min": min(float(svm.min()), float(svp.min())),
            "plus_minus_target_sv_min": float(svpm.min()),
            "active_sv_min": active_min, "retained_sv_min": retained_min,
            "energy_parity_abs_Eh": parity,
            "ensemble_rdm_fro_diagnostic": rdm_diff,
            "ensemble_noon_abs": noon_diff,
            "spin_square_abs_max": spin_max,
            "bright_channel_X_weight": float(abs(av[0, -1]) ** 2),
            "failed_gates": " ".join(failed_pair_gates),
            "status": "PASS" if passed else "FAIL",
        })
        if not passed:
            print(
                f"PAIR FAIL R={r:.10g} axis={axis} F={field:.1e}: "
                + ", ".join(failed_pair_gates),
                flush=True,
            )
        if pair_number % 10 == 0 or pair_number == len(grouped_items):
            print(
                f"ANALYZE signed pairs: {pair_number}/{len(grouped_items)}",
                flush=True,
            )

    by_limit: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_limit[(float(row["R_angstrom"]), row["axis"])].append(row)
    limit_rows: list[dict[str, Any]] = []
    limit_matrix_archive: dict[str, np.ndarray] = {}
    overall = all(x["status"] == "PASS" for x in pair_rows)
    print(f"ANALYZE zero-field limits: 0/{len(by_limit)}", flush=True)
    for limit_number, ((r, axis), rows) in enumerate(sorted(by_limit.items()), 1):
        rows.sort(key=lambda x: x["field_au"])
        fields = [x["field_au"] for x in rows]
        matrices = [even_hamiltonians[(r, axis, f)] for f in fields]
        h_c1_zero, alpha_limit, alpha_spread, alpha_stability = (
            fit_even_hamiltonian_limit(fields, matrices)
        )
        matrices6 = [even_hamiltonians_rank6[(r, axis, f)] for f in fields]
        h6_c1_zero, alpha6_limit, alpha6_spread, alpha6_stability = (
            fit_even_hamiltonian_limit(fields, matrices6)
        )
        alpha_eigenvalues, alpha_eigenvectors = np.linalg.eigh(alpha_limit)
        zero_offset = 0.5 * (
            h_c1_zero - zero_hamiltonians[r]
            + (h_c1_zero - zero_hamiltonians[r]).T
        )
        zero_offset_abs_max = float(np.max(np.abs(np.linalg.eigvalsh(zero_offset))))
        zero_offset6 = 0.5 * (
            h6_c1_zero - zero_hamiltonians_rank6[r]
            + (h6_c1_zero - zero_hamiltonians_rank6[r]).T
        )
        zero_offset6_abs_max = float(
            np.max(np.abs(np.linalg.eigvalsh(zero_offset6)))
        )
        metric_limit, metric_spread, metric_stability = fit_limit(
            fields, [x["target_projector_metric_trace"] for x in rows]
        )
        rank6_metric_limit, rank6_metric_spread, rank6_metric_stability = fit_limit(
            fields, [x["rank6_projector_metric_trace"] for x in rows]
        )
        spreads = [
            alpha_spread, alpha6_spread, metric_spread, rank6_metric_spread
        ]
        stabilities = [
            alpha_stability, alpha6_stability,
            metric_stability, rank6_metric_stability,
        ]
        pair_passed = all(row["status"] == "PASS" for row in rows)
        limit_checks = {
            "all_signed_pairs_pass": pair_passed,
            "fit_relative_spread_max": (
                max(spreads) <= GATES["fit_relative_spread_max"]
            ),
            "fit_relative_stability_max": (
                max(stabilities) <= GATES["fit_relative_spread_max"]
            ),
            "common_frame_zero_offset_abs_Eh_max": (
                zero_offset_abs_max
                <= GATES["common_frame_zero_offset_abs_Eh_max"]
            ),
            "rank6_common_frame_zero_offset_abs_Eh_max": (
                zero_offset6_abs_max
                <= GATES["common_frame_zero_offset_abs_Eh_max"]
            ),
        }
        failed_limit_gates = [
            name for name, ok in limit_checks.items() if not ok
        ]
        passed = not failed_limit_gates
        overall = overall and passed
        limit_key = f"{r_label(r)}_{axis}"
        limit_matrix_archive[f"{limit_key}_alpha_F0"] = alpha_limit
        limit_matrix_archive[f"{limit_key}_alpha_eigenvectors_F0"] = alpha_eigenvectors
        limit_matrix_archive[f"{limit_key}_H_subgroup_F0"] = h_c1_zero
        limit_matrix_archive[f"{limit_key}_H_subgroup_minus_D2h_H0"] = zero_offset
        limit_matrix_archive[f"{limit_key}_alpha6_F0"] = alpha6_limit
        limit_matrix_archive[f"{limit_key}_H6_subgroup_F0"] = h6_c1_zero
        limit_matrix_archive[f"{limit_key}_H6_subgroup_minus_D2h_H0"] = zero_offset6
        term_energies = {
            name: float(np.mean(np.diag(zero_hamiltonians_rank6[r])[indices]))
            for name, indices in zero_term_blocks[r].items()
        }
        ground_identity = min(term_energies, key=term_energies.get)
        ground_block = zero_term_blocks[r][ground_identity]
        response_block = alpha6_limit[np.ix_(ground_block, ground_block)]
        ground_alpha = (
            float(response_block[0, 0]) if len(ground_block) == 1
            else float(np.linalg.eigvalsh(response_block)[-1])
        )
        target_alpha_trace = float(np.trace(alpha_limit))
        delta_geom = (
            target_alpha_trace / (2.0 * metric_limit)
            if metric_limit > 1.0e-14 else float("nan")
        )
        limit_rows.append({
            "R_angstrom": r, "axis": axis,
            "X_minus_Delta_mean_Eh": zero_gaps[r],
            "zero_field_ground_identity": ground_identity,
            "alpha_eig_0_F0": float(alpha_eigenvalues[0]),
            "alpha_eig_1_F0": float(alpha_eigenvalues[1]),
            "alpha_eig_2_F0": float(alpha_eigenvalues[2]),
            "alpha_X_F0": float(alpha_limit[0, 0]),
            "alpha_Delta_x2-y2_F0": float(alpha_limit[1, 1]),
            "alpha_Delta_xy_F0": float(alpha_limit[2, 2]),
            "alpha_X_Delta_x2-y2_F0": float(alpha_limit[0, 1]),
            "alpha_X_Delta_xy_F0": float(alpha_limit[0, 2]),
            "alpha_Delta_component_coupling_F0": float(alpha_limit[1, 2]),
            "physical_ground_quadratic_response_F0": ground_alpha,
            "trace_target_alpha_F0": target_alpha_trace,
            "target_projector_metric_trace_F0": metric_limit,
            "rank6_projector_metric_trace_F0": rank6_metric_limit,
            "Delta_geom_target_Eh": delta_geom,
            "Delta_geom_target_eV": delta_geom * EH_TO_EV,
            "bright_channel_X_weight_F0": float(abs(alpha_eigenvectors[0, -1]) ** 2),
            "common_frame_zero_offset_abs_max_Eh": zero_offset_abs_max,
            "rank6_common_frame_zero_offset_abs_max_Eh": zero_offset6_abs_max,
            "max_fit_spread": max(spreads), "max_fit_stability": max(stabilities),
            "signed_pair_failure_count": sum(
                row["status"] != "PASS" for row in rows
            ),
            "failed_gates": " ".join(failed_limit_gates),
            "status": "PASS" if passed else "FAIL",
        })
        if not passed:
            print(
                f"LIMIT FAIL R={r:.10g} axis={axis}: "
                + ", ".join(failed_limit_gates),
                flush=True,
            )
        if limit_number % 10 == 0 or limit_number == len(by_limit):
            print(
                f"ANALYZE zero-field limits: {limit_number}/{len(by_limit)}",
                flush=True,
            )

    lookup = {(x["R_angstrom"], x["axis"]): x for x in limit_rows}
    symmetry_rows = []
    xy_pass_by_r: dict[float, bool] = defaultdict(lambda: True)
    xy_audit_values = sorted(
        {float(row["R_angstrom"]) for row in limit_rows if row["axis"] == "y"}
    )
    for r in xy_audit_values:
        if (r, "x") not in lookup or (r, "y") not in lookup:
            continue
        for key in ("alpha_eig_0_F0", "alpha_eig_1_F0", "alpha_eig_2_F0", "target_projector_metric_trace_F0"):
            x, y = float(lookup[(r, "x")][key]), float(lookup[(r, "y")][key])
            rel = abs(x - y) / max(abs(x), abs(y), 1.0)
            passed = rel <= GATES["xy_relative_error_max"]
            overall = overall and passed
            xy_pass_by_r[r] = xy_pass_by_r[r] and passed
            symmetry_rows.append({
                "R_angstrom": r, "quantity": key, "x_value": x, "y_value": y,
                "relative_error": rel,
                "failed_gates": "" if passed else "xy_relative_error_max",
                "status": "PASS" if passed else "FAIL",
            })

    global_rows = []
    global_geometry_exclusions = []
    incomplete_global_geometries = []
    r_values = sorted({float(record["R_angstrom"]) for record in all_records})
    for r in r_values:
        missing_required = [
            axis for axis in ("x", "z") if (r, axis) not in lookup
        ]
        if missing_required:
            if all((r, axis) in excluded_axes for axis in missing_required):
                global_geometry_exclusions.append({
                    "R_angstrom": r,
                    "excluded_axes": " ".join(missing_required),
                    "reason": " | ".join(
                        excluded_axes[(r, axis)] for axis in missing_required
                    ),
                    "status": "EXCLUDED",
                })
            else:
                overall = False
                incomplete_global_geometries.append({
                    "R_angstrom": r,
                    "missing_required_axes": " ".join(missing_required),
                    "status": "REVIEW",
                })
            continue
        perpendicular = lookup[(r, "x")]
        parallel = lookup[(r, "z")]
        alpha_perp = float(perpendicular["physical_ground_quadratic_response_F0"])
        alpha_parallel = float(parallel["physical_ground_quadratic_response_F0"])
        trace_a_perp = float(perpendicular["trace_target_alpha_F0"])
        trace_a_parallel = float(parallel["trace_target_alpha_F0"])
        trace_g_perp = float(perpendicular["target_projector_metric_trace_F0"])
        trace_g_parallel = float(parallel["target_projector_metric_trace_F0"])
        trace_a_iso = (2.0 * trace_a_perp + trace_a_parallel) / 3.0
        trace_g_iso = (2.0 * trace_g_perp + trace_g_parallel) / 3.0
        delta_iso = (
            trace_a_iso / (2.0 * trace_g_iso)
            if trace_g_iso > 1.0e-14 else float("nan")
        )
        identity_consistent = (
            perpendicular["zero_field_ground_identity"]
            == parallel["zero_field_ground_identity"]
        )
        point_pass = (
            perpendicular["status"] == "PASS"
            and parallel["status"] == "PASS"
            and identity_consistent
            and xy_pass_by_r[r]
        )
        overall = overall and point_pass
        global_rows.append({
            "R_angstrom": r,
            "zero_field_ground_identity": perpendicular["zero_field_ground_identity"],
            "X_minus_Delta_mean_Eh": perpendicular["X_minus_Delta_mean_Eh"],
            "alpha_perpendicular_au": alpha_perp,
            "alpha_parallel_au": alpha_parallel,
            "alpha_isotropic_au": (2.0 * alpha_perp + alpha_parallel) / 3.0,
            "Tr_A_target_perpendicular_au": trace_a_perp,
            "Tr_A_target_parallel_au": trace_a_parallel,
            "Tr_A_target_isotropic_au": trace_a_iso,
            "Tr_gFF_target_perpendicular_au": trace_g_perp,
            "Tr_gFF_target_parallel_au": trace_g_parallel,
            "Tr_gFF_target_isotropic_au": trace_g_iso,
            "Delta_geom_target_perpendicular_Eh": perpendicular["Delta_geom_target_Eh"],
            "Delta_geom_target_parallel_Eh": parallel["Delta_geom_target_Eh"],
            "Delta_geom_target_isotropic_Eh": delta_iso,
            "Delta_geom_target_perpendicular_eV": perpendicular["Delta_geom_target_eV"],
            "Delta_geom_target_parallel_eV": parallel["Delta_geom_target_eV"],
            "Delta_geom_target_isotropic_eV": delta_iso * EH_TO_EV,
            "xy_audit_status": "PASS" if xy_pass_by_r[r] else "FAIL",
            "status": "PASS" if point_pass else "FAIL",
        })

    out = workdir / "analysis"
    write_csv(out / "signed_field_pairs.csv", pair_rows)
    write_csv(out / "zero_field_limits.csv", limit_rows)
    write_csv(out / "xy_symmetry_audit.csv", symmetry_rows)
    write_csv(out / "global_response_observables.csv", global_rows)
    write_csv(out / "excluded_tasks.csv", [
        {
            "task_id": int(record["task_id"]),
            "R_angstrom": float(record["R_angstrom"]),
            "axis": str(record["axis"]),
            "field_magnitude_au": float(record["field_magnitude_au"]),
            "field_sign": int(record["field_sign"]),
            "reason": str(record["exclusion"].get("reason", "")),
            "recorded_utc": str(record["exclusion"].get("recorded_utc", "")),
            "status": "EXCLUDED",
        }
        for record in excluded_records
    ])
    write_csv(
        out / "global_geometry_exclusions.csv", global_geometry_exclusions
    )
    write_csv(
        out / "incomplete_global_geometries.csv", incomplete_global_geometries
    )
    np.savez_compressed(out / "signed_field_matrices.npz", **matrix_archive)
    np.savez_compressed(out / "zero_field_limit_matrices.npz", **limit_matrix_archive)
    pair_gate_failures = Counter(
        gate
        for row in pair_rows if row["status"] != "PASS"
        for gate in str(row["failed_gates"]).split()
    )
    limit_gate_failures = Counter(
        gate
        for row in limit_rows if row["status"] != "PASS"
        for gate in str(row["failed_gates"]).split()
    )
    xy_gate_failures = Counter(
        gate
        for row in symmetry_rows if row["status"] != "PASS"
        for gate in str(row["failed_gates"]).split()
    )
    summary = {
        "status": "PASS" if overall else "REVIEW",
        "audit_status": (
            "PASS_WITH_DOCUMENTED_EXCLUSIONS"
            if overall and excluded_records else
            ("PASS" if overall else "REVIEW")
        ),
        "method": (
            "rank-three polar-transport signed-field response with "
            "direction-adapted C2v root blocks"
        ),
        "curvature_fit": (
            "H_even(F)=H_subgroup(0)-A*F^2/2+B*F^4 with a free common-frame intercept"
        ),
        "manifest_task_count": len(all_records),
        "analyzed_task_count": len(records),
        "excluded_task_count": len(excluded_records),
        "excluded_axis_block_count": len(excluded_axes),
        "pair_count": len(pair_rows), "limit_count": len(limit_rows),
        "global_geometry_count": len(global_rows),
        "excluded_global_geometry_count": len(global_geometry_exclusions),
        "incomplete_global_geometry_count": len(incomplete_global_geometries),
        "failure_counts": {
            "signed_field_pairs": sum(row["status"] != "PASS" for row in pair_rows),
            "zero_field_limits": sum(row["status"] != "PASS" for row in limit_rows),
            "xy_symmetry_audits": sum(row["status"] != "PASS" for row in symmetry_rows),
            "global_geometries": sum(row["status"] != "PASS" for row in global_rows),
        },
        "failed_gate_counts": {
            "signed_field_pairs": dict(pair_gate_failures),
            "zero_field_limits": dict(limit_gate_failures),
            "xy_symmetry_audits": dict(xy_gate_failures),
        },
        "gates": GATES,
        "crossing_interpretation": (
            "Pdeg mu Pdeg is zero: test finite rank-three quadratic response and projector geometry; "
            "do not infer a linear Stark cusp from degeneracy alone."
        ),
        "exclusion_policy": (
            "Only complete signed-field (R,axis) ladders may be excluded. "
            "A geometry lacking an explicitly excluded x or z axis is omitted "
            "from global isotropic observables without interpolation; all "
            "remaining active gates must pass."
        ),
        "excluded_axis_blocks": [
            {"R_angstrom": r, "axis": axis, "reason": reason}
            for (r, axis), reason in sorted(excluded_axes.items())
        ],
        "output_directory": str(out),
        "primary_table": str(out / "global_response_observables.csv"),
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def self_test(_: argparse.Namespace) -> None:
    assert r_label(1.652734375) == "R_1p652734375"
    assert f_label(-5e-5) == "F_m5.0e-05"
    assert parse_task_id_spec("8-10,12,15-15") == [8, 9, 10, 12, 15]
    mock_manifest = [
        {
            "task_id": task_id,
            "R_angstrom": 1.0,
            "axis": "z",
            "exclusion": {
                "status": "EXCLUDED",
                "reason": "test complete-axis exclusion",
            },
        }
        for task_id in range(8)
    ]
    active, excluded, excluded_axes = partition_manifest_exclusions(mock_manifest)
    assert not active and len(excluded) == 8
    assert excluded_axes[(1.0, "z")] == "test complete-axis exclusion"
    axes = sum(
        3 if any(abs(r - x) < 1e-10 for x in DEFAULT_XY_AUDIT_R) else 2
        for r in DEFAULT_R
    )
    assert axes >= 2 * len(DEFAULT_R)
    global_r = [1.0 + 0.05 * i for i in range(45)]
    global_axes = 2 * len(global_r) + sum(
        any(abs(r - audit) < 1.0e-10 for audit in DEFAULT_XY_AUDIT_R)
        for r in global_r
    )
    assert global_axes * len(DEFAULT_FIELDS) * 2 == 760
    h0 = np.diag([-1.0, -1.0, -1.0])
    alpha = np.diag([2.0, 5.0, 9.0])
    f = 1.0e-3
    hp = h0 - 0.5 * alpha * f * f
    hm = h0 - 0.5 * alpha * f * f
    recovered = -(hp + hm - 2 * h0) / f**2
    assert np.max(np.abs(recovered - alpha)) < 1.0e-9
    fields = np.asarray(DEFAULT_FIELDS)
    matrices = [alpha + np.diag([1.0, 2.0, 3.0]) * value**2 for value in fields]
    matrix_limit, spread, stability = fit_matrix_limit(fields, matrices)
    assert np.max(np.abs(matrix_limit - alpha)) < 1.0e-10
    assert spread < 1.0e-10 and stability < 1.0e-10
    shifted_h0 = h0 + np.diag([2.0e-8, -1.0e-8, 0.5e-8])
    even_h = [
        shifted_h0 - 0.5 * alpha * value**2
        + np.diag([1.0e5, 2.0e5, 3.0e5]) * value**4
        for value in fields
    ]
    fitted_h0, fitted_alpha, even_spread, even_stability = (
        fit_even_hamiltonian_limit(fields, even_h)
    )
    assert np.max(np.abs(fitted_h0 - shifted_h0)) < 1.0e-12
    assert np.max(np.abs(fitted_alpha - alpha)) < 1.0e-7
    assert even_spread < 1.0e-6 and even_stability < 0.05
    symmetric = np.asarray([[0.8, 0.2], [0.2, 0.5]])
    antisymmetric = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    projected, projection_report = project_singlet_ci(symmetric + 1.0e-3 * antisymmetric)
    assert np.max(np.abs(projected - projected.T)) == 0.0
    assert abs(np.linalg.norm(projected) - 1.0) < 1.0e-14
    assert projection_report["antisymmetric_removed_relative_norm"] > 0.0
    for axis in "xyz":
        axes = subgroup_axes(axis)
        assert np.allclose(axes @ axes.T, np.eye(3), atol=1.0e-14)
        assert abs(np.linalg.det(axes) - 1.0) < 1.0e-14
        assert sorted(subgroup_root_order(axis)) == list(range(NSTATE))
        assert sum(
            len(indices) for _, indices in FIELD_STATE_BLOCKS[axis]
        ) == NSTATE

    class MockFCI:
        def __init__(self):
            self.nroots = 1
            self.converged = True

        def kernel(self, h1, h2, norb, nelec, ci0=None, nroots=1, **kwargs):
            candidates = [
                np.asarray((0.0, 1.0, 0.0, 0.0)),
                np.asarray((1.0, 0.0, 0.0, 0.0)),
                np.asarray((0.0, 0.0, 1.0, 0.0)),
                np.asarray((0.0, 0.0, 0.0, 1.0)),
            ]
            self.converged = [True] * len(candidates)
            return np.arange(len(candidates), dtype=float), candidates

    class MockHomingFCI(SubspaceHomingFCISolver, MockFCI):
        pass

    mock = MockHomingFCI()
    mock.nroots = 2
    mock.homing_desired_roots = 2
    mock.homing_irrep = "mock"
    mock.homing_min_sv_seen = 1.0
    mock.homing_last_min_sv = 1.0
    mock.homing_min_local_sv_seen = 1.0
    mock.homing_last_local_sv = 1.0
    mock.homing_last_candidate_count = 2
    mock.homing_anchor_references = np.stack((np.eye(4)[0], np.eye(4)[2]))
    mock_e, mock_ci = mock.kernel(
        None, None, 2, (1, 1),
        # These local guesses point to the wrong two candidate roots.  The
        # fixed zero-field anchor above must still recover roots 1 and 2.
        ci0=[np.eye(4)[1], np.eye(4)[3]],
    )
    assert np.allclose(mock_e, (1.0, 2.0))
    assert np.linalg.svd(
        np.stack((np.eye(4)[0], np.eye(4)[2]))
        @ np.stack(mock_ci).T,
        compute_uv=False,
    ).min() > 1.0 - 1.0e-14

    # When two candidate subspaces both pass the immutable-anchor identity
    # gate, local continuity must win.  Anchor-only scoring would choose
    # energies (0,1); dual homing must retain the incoming (2,3) subspace.
    smooth = MockHomingFCI()
    smooth.nroots = 2
    smooth.homing_desired_roots = 2
    smooth.homing_irrep = "smooth"
    smooth.homing_min_sv_seen = 1.0
    smooth.homing_last_min_sv = 1.0
    smooth.homing_min_local_sv_seen = 1.0
    smooth.homing_last_local_sv = 1.0
    smooth.homing_last_candidate_count = 2
    smooth.homing_anchor_references = np.stack((
        np.sqrt(0.6) * np.eye(4)[0] + np.sqrt(0.4) * np.eye(4)[2],
        np.sqrt(0.6) * np.eye(4)[1] + np.sqrt(0.4) * np.eye(4)[3],
    ))
    smooth_e, smooth_ci = smooth.kernel(
        None, None, 2, (1, 1),
        ci0=[np.eye(4)[2], np.eye(4)[3]],
    )
    assert np.allclose(smooth_e, (2.0, 3.0))
    assert smooth.homing_last_min_sv > HOMING_ANCHOR_SV_MIN
    assert smooth.homing_last_local_sv > 1.0 - 1.0e-14
    singleton_flat = np.arange(36.0)
    singleton_refs = SubspaceHomingFCISolver._references(
        singleton_flat, 1, 4, (2, 2)
    )
    assert len(singleton_refs) == 1
    assert singleton_refs[0].shape == (6, 6)
    assert np.array_equal(singleton_refs[0].ravel(), singleton_flat)
    real_cas_singleton = np.arange(4900.0)
    real_cas_refs = SubspaceHomingFCISolver._references(
        real_cas_singleton, 1, 8, (4, 4)
    )
    assert len(real_cas_refs) == 1
    assert real_cas_refs[0].shape == (70, 70)
    assert np.array_equal(real_cas_refs[0].ravel(), real_cas_singleton)
    print(
        "SELF-TEST PASS: labels, C2v root blocks and dual anchored/local flat/split "
        "root homing, "
        "free-intercept curvature, and strict-singlet CI projection"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("self-test")
    p.set_defaults(func=self_test)
    p = sub.add_parser("prepare")
    p.add_argument("--zero-root", required=True)
    p.add_argument("--workdir", default="C2_SA6_global_field_work_v2")
    p.add_argument("--grid-csv")
    p.add_argument(
        "--r-values",
        help="Comma-separated override; default reads field_use=1 from geometry_grid.csv",
    )
    p.add_argument(
        "--xy-audit-r",
        default=",".join(str(x) for x in DEFAULT_XY_AUDIT_R),
    )
    p.add_argument("--fields", default=",".join(f"{x:g}" for x in DEFAULT_FIELDS))
    p.set_defaults(func=prepare)
    p = sub.add_parser("run-task")
    p.add_argument("--manifest", required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--memory-mb", type=int, default=120000)
    p.add_argument("--force", action="store_true")
    p.add_argument("--restart-h5")
    p.add_argument("--rescue", action="store_true")
    p.set_defaults(func=run_task)
    p = sub.add_parser("status")
    p.add_argument("--manifest", required=True)
    p.add_argument("--retry-file")
    p.set_defaults(func=task_status)
    p = sub.add_parser("exclude-tasks")
    p.add_argument("--manifest", required=True)
    p.add_argument(
        "--task-ids", required=True,
        help="Comma-separated ids or inclusive ranges, for example 8-15",
    )
    p.add_argument("--reason", required=True)
    p.set_defaults(func=exclude_tasks)
    p = sub.add_parser("analyze")
    p.add_argument("--manifest", required=True)
    p.add_argument("--workdir", default="C2_SA6_global_field_work_v2")
    p.add_argument("--memory-mb", type=int, default=120000)
    p.set_defaults(func=analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
