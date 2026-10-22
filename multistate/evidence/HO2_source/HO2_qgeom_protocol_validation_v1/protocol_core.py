#!/usr/bin/env python3
"""Protocol-generic machinery for the HO2 response validation package.

The original pilot deliberately used equal-weight SA2-CASSCF.  This module
keeps its directly transported CAS(13,9), symmetry, overlap, and finite-field
conventions, while allowing either a state-specific ground-state optimization
(``ss``) or an equal-weight four-state optimization (``sa4``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ho2_core import (
    BASIS,
    CAS_FOLLOW_SV_MIN,
    FULL_NELECAS,
    NCAS,
    NCORE,
    NELECAS,
    NRETAIN,
    SPIN,
    SPIN_PENALTY_SHIFT,
    SS_TARGET,
    STATE_IRREP,
    build_reference_mf,
    embed_active_ci,
    load_h5,
    pass_h5,
    qc_modules,
    save_h5,
    spin_and_rdm,
    tag_symmetry_orbitals,
)


VERSION = "1.2.5"
DEFAULT_FIELDS = (2.5e-4, 5.0e-4, 7.5e-4, 1.0e-3)
DEFAULT_MANIFOLD_FIELDS = DEFAULT_FIELDS
DEFAULT_FIELD_INDICES = "4:6,9:11,15:18"
DEFAULT_MANIFOLD_INDICES = "2:11,15:18"
DEFAULT_MANIFOLD_RANK = 2
PROTOCOLS = {"ss": 1, "sa4": 4}
GROUND_OVERLAP_WARN = 0.75
SUBSPACE_OVERLAP_WARN = 0.75


def protocol_nstate(protocol: str) -> int:
    try:
        return PROTOCOLS[str(protocol).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown protocol {protocol!r}; choose ss or sa4") from exc


def protocol_label(protocol: str) -> str:
    return "SS-CASSCF(13,9)" if protocol == "ss" else "SA4-CASSCF(13,9)"


def validation_root(workdir: Path) -> Path:
    return Path(workdir) / "protocol_validation"


def zero_path(workdir: Path, protocol: str, source_index: int) -> Path:
    return (
        validation_root(workdir)
        / protocol
        / "zero"
        / f"irc_{int(source_index):03d}.h5"
    )


def field_label(value: float) -> str:
    sign = "p" if value > 0.0 else "m"
    return f"F_{sign}{abs(float(value)):.1e}"


def field_path(workdir: Path, source_index: int, signed_field: float) -> Path:
    return protocol_field_path(workdir, "ss", source_index, signed_field)


def protocol_field_path(
    workdir: Path,
    protocol: str,
    source_index: int,
    signed_field: float,
    *,
    manifold: bool = False,
) -> Path:
    directory = "manifold_fields" if manifold else "fields"
    return (
        validation_root(workdir)
        / str(protocol)
        / directory
        / f"irc_{int(source_index):03d}"
        / f"{field_label(signed_field)}.h5"
    )


def metadata_path(workdir: Path) -> Path:
    return validation_root(workdir) / "protocol_metadata.json"


def load_metadata(workdir: Path) -> dict[str, Any]:
    path = metadata_path(workdir)
    if not path.exists():
        raise SystemExit(f"Missing {path}; run prepare_protocol.py first")
    return json.loads(path.read_text())


def pilot_workdir(workdir: Path) -> Path:
    return Path(load_metadata(workdir)["pilot_workdir"]).resolve()


def pilot_zero_path(workdir: Path, source_index: int) -> Path:
    return pilot_workdir(workdir) / "zero" / f"irc_{int(source_index):03d}.h5"


def ci_roots_n(ci: Any, expected: int | None = None) -> list[np.ndarray]:
    """Return CI vectors without assuming the original pilot's two roots."""
    if isinstance(ci, (list, tuple)):
        roots = [np.asarray(item) for item in ci]
    else:
        array = np.asarray(ci)
        if expected == 1 and array.ndim == 2:
            roots = [array]
        elif array.ndim >= 3:
            roots = [array[i] for i in range(array.shape[0])]
        elif array.ndim == 2:
            roots = [array]
        else:
            raise ValueError(f"Cannot decode CI roots from shape {array.shape}")
    if expected is not None and len(roots) != int(expected):
        raise ValueError(f"Expected {expected} CI roots, got {len(roots)}")
    answer = []
    for root in roots:
        value = np.array(root, copy=True)
        norm = float(np.linalg.norm(value))
        if norm <= 0.0:
            raise ValueError("Encountered a zero-norm CI vector")
        answer.append(value / norm)
    return answer


def kernel_ci0(ci: Sequence[np.ndarray] | None, nstate: int):
    if ci is None:
        return None
    roots = ci_roots_n(ci, nstate)
    return roots[0] if nstate == 1 else roots


def transport_active_ci_n(
    old_mol,
    new_mol,
    old_mo: np.ndarray,
    new_mo: np.ndarray,
    old_ci: Any,
    nstate: int,
):
    """Transport one or four CI vectors in the directly followed active CAS."""
    _, _, gto, _, _, _, _, fci_addons, _, _ = qc_modules()
    cross = np.asarray(gto.intor_cross("int1e_ovlp", old_mol, new_mol))
    old_active = np.asarray(old_mo)[:, NCORE:NRETAIN]
    new_active = np.asarray(new_mo)[:, NCORE:NRETAIN]
    active_overlap = old_active.T @ cross @ new_active
    left, singular_values, right_h = np.linalg.svd(active_overlap)
    rotation = left @ right_h
    transported = []
    for root in ci_roots_n(old_ci, nstate):
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


def state_overlap_n(
    left: dict[str, Any], right: dict[str, Any], s_cross: np.ndarray
) -> np.ndarray:
    """Exact nonorthogonal many-electron overlap for rectangular state frames."""
    *_, fci_addons, _, _ = qc_modules()[0:10]
    retained = (
        np.asarray(left["mo_coeff"])[:, :NRETAIN].T
        @ np.asarray(s_cross)
        @ np.asarray(right["mo_coeff"])[:, :NRETAIN]
    )
    left_ci = [embed_active_ci(x) for x in ci_roots_n(left["ci_vectors"])]
    right_ci = [embed_active_ci(x) for x in ci_roots_n(right["ci_vectors"])]
    out = np.empty((len(left_ci), len(right_ci)))
    for i, bra in enumerate(left_ci):
        for j, ket in enumerate(right_ci):
            out[i, j] = float(
                fci_addons.overlap(
                    bra, ket, NRETAIN, FULL_NELECAS, s=retained
                ).real
            )
    return out


def same_geometry_overlap_n(left: dict[str, Any], right: dict[str, Any], mol):
    return state_overlap_n(
        left, right, np.asarray(mol.intor_symmetric("int1e_ovlp"))
    )


def cross_geometry_overlap_n(left, right, left_mol, right_mol):
    _, _, gto, _, _, _, _, _, _, _ = qc_modules()
    cross = np.asarray(gto.intor_cross("int1e_ovlp", left_mol, right_mol))
    return state_overlap_n(left, right, cross)


def build_protocol_casscf(
    mf,
    protocol: str,
    *,
    newton: bool,
    grad_tol: float,
    max_macro: int,
    max_step: float,
):
    """Build state-specific or equal-weight SA4 CASSCF with identical settings."""
    _, fci, _, _, mcscf, _, _, fci_addons, _, _ = qc_modules()
    nstate = protocol_nstate(protocol)
    mc = mcscf.CASSCF(mf, NCAS, NELECAS, ncore=NCORE)
    solver = fci.direct_spin1_symm.FCI(mc.mol)
    solver.spin = SPIN
    solver.wfnsym = STATE_IRREP
    solver.conv_tol = 1.0e-11
    solver.max_cycle = 300
    solver.nroots = nstate
    solver = fci_addons.fix_spin_(
        solver, shift=SPIN_PENALTY_SHIFT, ss=SS_TARGET
    )
    solver.nroots = nstate
    mc.fcisolver = solver
    mc.conv_tol = 1.0e-10
    mc.conv_tol_grad = grad_tol
    mc.max_cycle_macro = max_macro
    mc.max_cycle_micro = 12 if not newton else 20
    mc.max_stepsize = max_step
    mc.ah_level_shift = 0.5 if not newton else 1.0
    mc.ah_grad_trust_region = 1.0
    mc.canonicalization = False
    if nstate > 1:
        mc = mc.state_average_([1.0 / nstate] * nstate)
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


def optimize_protocol(
    mf,
    mo0: np.ndarray,
    ci0: Sequence[np.ndarray] | None,
    protocol: str,
):
    """Converge a protocol through settle, tight Newton, and floor stages."""
    nstate = protocol_nstate(protocol)
    stages = (
        ("settle", False, 2.0e-5, 120, 0.015),
        ("tight_newton", True, 1.0e-6, 180, 0.006),
        ("floor_newton", True, 1.0e-6, 260, 0.002),
    )
    mo = np.asarray(mo0)
    ci = None if ci0 is None else ci_roots_n(ci0, nstate)
    reports = []
    converged = False
    mc = None
    for name, newton, grad_tol, max_macro, max_step in stages:
        mc = build_protocol_casscf(
            mf,
            protocol,
            newton=newton,
            grad_tol=grad_tol,
            max_macro=max_macro,
            max_step=max_step,
        )
        started = time.time()
        try:
            mc.kernel(
                tag_symmetry_orbitals(mf.mol, mo),
                ci0=kernel_ci0(ci, nstate),
            )
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            mc.converged = False
        reports.append({
            "name": name,
            "converged": bool(mc.converged),
            "wall_seconds": time.time() - started,
            "error": error,
        })
        if getattr(mc, "mo_coeff", None) is not None:
            mo = np.asarray(mc.mo_coeff)
        if getattr(mc, "ci", None) is not None:
            ci = ci_roots_n(mc.ci, nstate)
        if newton and bool(mc.converged):
            converged = True
            break
    if mc is None or ci is None or not converged:
        raise RuntimeError(
            f"{protocol_label(protocol)} did not converge: {json.dumps(reports)}"
        )
    return mc, mo, ci, reports


def physical_energies_n(mc, ci: Sequence[np.ndarray], nstate: int):
    roots = ci_roots_n(ci, nstate)
    if nstate == 1:
        penalized = np.asarray([float(mc.e_tot)])
    else:
        penalized = np.asarray(mc.e_states, dtype=float)
    spins, _, _ = spin_and_rdm(roots)
    physical = penalized - SPIN_PENALTY_SHIFT * (spins - SS_TARGET)
    return physical, penalized


def pack_result(mc, mo: np.ndarray, ci: Sequence[np.ndarray], protocol: str):
    nstate = protocol_nstate(protocol)
    roots = ci_roots_n(ci, nstate)
    physical, penalized = physical_energies_n(mc, roots, nstate)
    order = np.argsort(physical)
    ordered_roots = [roots[int(i)] for i in order]
    physical = np.asarray(physical)[order]
    penalized = np.asarray(penalized)[order]
    spins, rdm1, noons = spin_and_rdm(ordered_roots)
    return {
        "mo_coeff": np.asarray(mo),
        "ci_vectors": np.stack(ordered_roots),
        "e_states_Eh": physical,
        "e_states_penalized_Eh": penalized,
        "state_spin_square": spins,
        "state_rdm1_active": rdm1,
        "state_noons_active": noons,
    }


def complete_h5(path: Path, protocol: str | None = None) -> bool:
    if not pass_h5(path):
        return False
    if protocol is None:
        return True
    try:
        return str(load_h5(path)["attrs"].get("protocol")) == protocol
    except Exception:
        return False


__all__ = [
    "BASIS",
    "CAS_FOLLOW_SV_MIN",
    "DEFAULT_FIELDS",
    "DEFAULT_FIELD_INDICES",
    "DEFAULT_MANIFOLD_FIELDS",
    "DEFAULT_MANIFOLD_INDICES",
    "DEFAULT_MANIFOLD_RANK",
    "GROUND_OVERLAP_WARN",
    "SUBSPACE_OVERLAP_WARN",
    "VERSION",
    "build_reference_mf",
    "ci_roots_n",
    "complete_h5",
    "cross_geometry_overlap_n",
    "field_path",
    "load_h5",
    "load_metadata",
    "metadata_path",
    "optimize_protocol",
    "pack_result",
    "pilot_workdir",
    "pilot_zero_path",
    "protocol_field_path",
    "protocol_label",
    "protocol_nstate",
    "same_geometry_overlap_n",
    "save_h5",
    "transport_active_ci_n",
    "validation_root",
    "zero_path",
]
