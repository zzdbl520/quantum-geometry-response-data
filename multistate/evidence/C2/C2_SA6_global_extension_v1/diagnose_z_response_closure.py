#!/usr/bin/env python3
"""Fixed-orbital CASCI diagnosis of the longitudinal C2 response manifold.

The test freezes the certified zero-field SA6-CASSCF orbitals and repeats the
signed z-field calculation at the CASCI level.  It separates a discontinuous
orbital-optimization branch from a failure of the rank-three CI manifold
itself.  Zero-field B1u and Au CASCI roots are also sampled to expose the
symmetry-required longitudinal response channels.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from c2_sa6_global_field import (
    BASIS,
    EH_TO_EV,
    FIELD_STATE_BLOCKS,
    GATES,
    NCAS,
    NCORE,
    NELECAS,
    NSTATE,
    SPIN_PENALTY_SHIFT,
    TARGET_NAMES,
    build_field_mf,
    closest_point,
    ci_roots,
    field_molecule,
    fit_even_hamiltonian_limit,
    fit_limit,
    identity_index,
    install_subspace_homing,
    load_h5,
    molecule,
    point_index,
    polar,
    project_singlet_roots,
    qc_modules,
    r_label,
    save_h5,
    select_target,
    state_overlap,
    subgroup_root_order,
    tag_symmetry_orbitals,
    write_csv,
)


AXIS = "z"
AUXILIARY_IRREPS = ("B1u", "Au")


def make_d2h_molecule(r: float, memory_mb: int):
    _, _, gto, _, _, _, _, _ = qc_modules()
    return gto.M(
        atom=f"C 0 0 {-r/2:.14f}; C 0 0 {r/2:.14f}",
        unit="Angstrom",
        basis=BASIS,
        charge=0,
        spin=0,
        symmetry="D2h",
        verbose=4,
        max_memory=memory_mb,
    )


def all_solver_roots_converged(mc) -> bool:
    solvers = getattr(mc.fcisolver, "fcisolvers", ())
    if not solvers:
        return bool(getattr(mc, "converged", True))
    return all(
        bool(np.all(np.atleast_1d(getattr(solver, "converged", True))))
        for solver in solvers
    )


def build_fixed_casci(mf, anchor_ci: Sequence[np.ndarray]):
    _, fci, _, mcscf, _, fci_addons, _, _ = qc_modules()
    mc = mcscf.CASCI(mf, NCAS, NELECAS, ncore=NCORE)
    solvers = []
    anchor_offset = 0
    for irrep, zero_indices in FIELD_STATE_BLOCKS[AXIS]:
        block_size = len(zero_indices)
        block_anchor = anchor_ci[anchor_offset:anchor_offset + block_size]
        if len(block_anchor) != block_size:
            raise RuntimeError(
                f"Missing fixed-CASCI anchors for {irrep}: "
                f"expected {block_size}, received {len(block_anchor)}"
            )
        anchor_offset += block_size
        solver = fci.direct_spin1_symm.FCI(mc.mol)
        solver.spin = 0
        solver.wfnsym = irrep
        solver.nroots = block_size
        solver.conv_tol = 1.0e-12
        solver.max_cycle = 300
        solver = fci_addons.fix_spin_(
            solver, shift=SPIN_PENALTY_SHIFT, ss=0.0
        )
        solver = install_subspace_homing(
            solver, block_size, irrep, block_anchor
        )
        solvers.append(solver)
    if anchor_offset != NSTATE:
        raise RuntimeError(
            f"Fixed-CASCI anchor partition covered {anchor_offset} roots"
        )
    mc = mcscf.state_average_mix_(
        mc, solvers, [1.0 / NSTATE] * NSTATE
    )
    mc.canonicalization = False
    return mc


def fixed_casci_frame(
    zero: dict[str, Any],
    zero_h5: Path,
    r: float,
    signed_field: float,
    memory_mb: int,
    output_h5: Path,
) -> dict[str, Any]:
    _, fci, _, _, _, _, _, spin_op = qc_modules()
    zero_ci, zero_projection = project_singlet_roots(zero["ci_vectors"])
    order = subgroup_root_order(AXIS)
    ordered_ci = [zero_ci[index] for index in order]
    mo0 = np.asarray(zero["mo_coeff"])
    mol = field_molecule(r, AXIS, memory_mb=memory_mb)
    field = np.asarray((0.0, 0.0, signed_field))
    mf = build_field_mf(mol, field, mo0)
    mc = build_fixed_casci(mf, ordered_ci)
    started = time.time()
    mc.kernel(tag_symmetry_orbitals(mol, mo0), ci0=ordered_ci)
    wall_seconds = time.time() - started
    if not all_solver_roots_converged(mc):
        raise RuntimeError(
            f"Fixed-orbital CASCI did not converge at Fz={signed_field:+.1e}"
        )

    penalized = np.asarray(mc.e_states, dtype=float)
    raw_ci = ci_roots(mc.ci)
    spins = np.asarray([
        spin_op.spin_square0(vector, NCAS, NELECAS)[0]
        for vector in raw_ci
    ])
    if float(np.max(np.abs(spins))) > GATES["spin_square_abs_max"]:
        raise RuntimeError(
            f"Fixed CASCI returned spin contamination {np.max(abs(spins)):.3e}"
        )
    energies = penalized - SPIN_PENALTY_SHIFT * spins
    projected_ci, final_projection = project_singlet_roots(raw_ci)
    rdm1 = np.stack([
        fci.direct_spin1.make_rdm1(vector, NCAS, NELECAS)
        for vector in projected_ci
    ])
    noons = np.stack([
        np.linalg.eigvalsh(0.5 * (density + density.T))[::-1]
        for density in rdm1
    ])
    homing = [
        {
            "irrep": str(getattr(solver, "homing_irrep", "unknown")),
            "last_anchor_sv": float(
                getattr(solver, "homing_last_min_sv", float("nan"))
            ),
            "last_local_sv": float(
                getattr(solver, "homing_last_local_sv", float("nan"))
            ),
            "candidate_count": int(
                getattr(solver, "homing_last_candidate_count", solver.nroots)
            ),
        }
        for solver in mc.fcisolver.fcisolvers
    ]
    mo = np.asarray(mc.mo_coeff)
    s = np.asarray(mol.intor_symmetric("int1e_ovlp"))
    orbital_change = float(
        np.max(np.abs(mo.T @ s @ mo0 - np.eye(mo.shape[1])))
    )
    attrs = {
        "kind": "C2_fixed_orbital_CASCI_z_field_diagnostic",
        "R_angstrom": r,
        "axis": AXIS,
        "signed_field_au": signed_field,
        "zero_h5": str(zero_h5),
        "method": "fixed zero-field SA6 orbitals; CASCI(8,8)/aug-cc-pVTZ",
        "residual_group": "C2v",
        "converged": True,
        "wall_seconds": wall_seconds,
        "maximum_abs_spin_square": float(np.max(np.abs(spins))),
        "maximum_fixed_orbital_change": orbital_change,
        "root_homing_reports_json": homing,
        "zero_ci_projection_json": zero_projection,
        "final_ci_projection_json": final_projection,
    }
    save_h5(output_h5, {
        "e_states_Eh": energies,
        "e_states_spin_penalized_Eh": penalized,
        "mo_coeff": mo,
        "ci_vectors": np.asarray(projected_ci),
        "state_rdm1": rdm1,
        "state_noons": noons,
        "state_spin_square": spins,
    }, attrs)
    return {
        "mo_coeff": mo,
        "ci_vectors": np.asarray(projected_ci),
        "e_states_Eh": energies,
        "state_noons": noons,
        "state_spin_square": spins,
        "wall_seconds": wall_seconds,
        "output_h5": str(output_h5),
    }


def solve_auxiliary_irrep(
    mol,
    mf,
    mo: np.ndarray,
    irrep: str,
    nroots: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    _, fci, _, mcscf, _, _, _, _ = qc_modules()
    solver = fci.direct_spin0_symm.FCI(mol)
    solver.wfnsym = irrep
    solver.nroots = nroots
    solver.conv_tol = 1.0e-12
    solver.max_cycle = 300
    mc = mcscf.CASCI(mf, NCAS, NELECAS, ncore=NCORE)
    mc.fcisolver = solver
    mc.canonicalization = False
    mc.kernel(tag_symmetry_orbitals(mol, mo))
    energies = np.atleast_1d(np.asarray(mc.e_tot, dtype=float))
    if nroots == 1:
        roots = [np.asarray(mc.ci)]
    else:
        roots = [np.asarray(root) for root in mc.ci]
    if len(roots) != len(energies):
        raise RuntimeError(
            f"{irrep} CASCI returned {len(energies)} energies and "
            f"{len(roots)} roots"
        )
    return energies, roots


def auxiliary_channel_spectrum(
    zero: dict[str, Any],
    r: float,
    target: Sequence[int],
    nroots: int,
    memory_mb: int,
    output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, fci, _, _, _, _, _, _ = qc_modules()
    mol = make_d2h_molecule(r, memory_mb)
    mo = np.asarray(zero["mo_coeff"])
    mf = build_field_mf(mol, np.zeros(3), mo)
    zero_ci, _ = project_singlet_roots(zero["ci_vectors"])
    zero_energies = np.asarray(zero["e_states_Eh"], dtype=float)
    mol.set_common_orig_((0.0, 0.0, 0.0))
    rz_ao = np.asarray(mol.intor("int1e_r", comp=3)[2])
    c_active = mo[:, NCORE:NCORE + NCAS]
    rz_active = c_active.T @ rz_ao @ c_active
    identity_by_index = {index: TARGET_NAMES[position] for position, index in enumerate(target)}

    rows: list[dict[str, Any]] = []
    archive: dict[str, np.ndarray] = {}
    channel_mu2: dict[str, float] = defaultdict(float)
    for irrep in AUXILIARY_IRREPS:
        energies, roots = solve_auxiliary_irrep(
            mol, mf, mo, irrep, nroots
        )
        archive[f"{irrep}_energies_Eh"] = energies
        archive[f"{irrep}_ci_vectors"] = np.asarray(roots)
        for auxiliary_root, (energy, ket) in enumerate(zip(energies, roots)):
            for target_index in target:
                bra = zero_ci[target_index]
                transition_dm = fci.direct_spin0.trans_rdm1(
                    bra, ket, NCAS, NELECAS
                )
                mu_z = -float(np.einsum(
                    "pq,qp->", rz_active, np.asarray(transition_dm)
                ).real)
                gap = float(energy - zero_energies[target_index])
                mu2 = mu_z * mu_z
                channel_mu2[irrep] += mu2
                rows.append({
                    "channel_irrep": irrep,
                    "auxiliary_root": auxiliary_root,
                    "auxiliary_energy_Eh": float(energy),
                    "target_identity": identity_by_index[target_index],
                    "target_solver_order": target_index,
                    "target_energy_Eh": float(zero_energies[target_index]),
                    "energy_gap_Eh": gap,
                    "energy_gap_eV": gap * EH_TO_EV,
                    "mu_z_au": mu_z,
                    "abs_mu_z_au": abs(mu_z),
                    "alpha_diagonal_contribution_au": (
                        2.0 * mu2 / gap if abs(gap) > 1.0e-12 else float("nan")
                    ),
                    "metric_diagonal_contribution_au": (
                        mu2 / gap**2 if abs(gap) > 1.0e-12 else float("nan")
                    ),
                })
    rows.sort(key=lambda row: row["abs_mu_z_au"], reverse=True)
    np.savez_compressed(output / "auxiliary_channel_roots.npz", **archive)
    summary = {
        "auxiliary_irreps": list(AUXILIARY_IRREPS),
        "requested_roots_per_irrep": nroots,
        "summed_mu_z_squared_by_irrep": dict(sorted(channel_mu2.items())),
        "largest_abs_mu_z_au": (
            float(rows[0]["abs_mu_z_au"]) if rows else 0.0
        ),
        "largest_transition": rows[0] if rows else None,
    }
    return rows, summary


def compare_pair(
    zero_frame: dict[str, Any],
    minus: dict[str, Any],
    plus: dict[str, Any],
    target: Sequence[int],
    s_ao: np.ndarray,
    field: float,
) -> tuple[dict[str, Any], np.ndarray, float]:
    ozm = state_overlap(zero_frame, minus, s_ao)
    ozp = state_overlap(zero_frame, plus, s_ao)
    sel_m, pm, svm = select_target(ozm, target)
    sel_p, pp, svp = select_target(ozp, target)
    minus_selected = {
        "mo_coeff": minus["mo_coeff"],
        "ci_vectors": minus["ci_vectors"][sel_m],
    }
    plus_selected = {
        "mo_coeff": plus["mo_coeff"],
        "ci_vectors": plus["ci_vectors"][sel_p],
    }
    omp = state_overlap(minus_selected, plus_selected, s_ao)
    _, svpm = polar(omp)
    hm = pm @ np.diag(minus["e_states_Eh"][sel_m]) @ pm.T
    hp = pp @ np.diag(plus["e_states_Eh"][sel_p]) @ pp.T
    h0 = np.diag(zero_frame["e_states_Eh"][target])
    even_h = 0.5 * (hm + hp)
    alpha = -(hm + hp - 2.0 * h0) / field**2
    alpha = 0.5 * (alpha + alpha.T)
    metric_trace = float(
        (len(target) - np.sum(svpm**2)) / (4.0 * field**2)
    )
    parity = float(np.max(np.abs(
        np.sort(minus["e_states_Eh"][sel_m])
        - np.sort(plus["e_states_Eh"][sel_p])
    )))
    zero_sv = min(float(svm.min()), float(svp.min()))
    pm_sv = float(svpm.min())
    checks = {
        "zero_target_subspace_sv_min": (
            zero_sv >= GATES["zero_target_subspace_sv_min"]
        ),
        "plus_minus_target_subspace_sv_min": (
            pm_sv >= GATES["plus_minus_target_subspace_sv_min"]
        ),
        "energy_parity_abs_Eh_max": (
            parity <= GATES["energy_parity_abs_Eh_max"]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    row = {
        "field_au": field,
        "selected_minus_roots": " ".join(map(str, sel_m)),
        "selected_plus_roots": " ".join(map(str, sel_p)),
        "zero_target_sv_min": zero_sv,
        "plus_minus_target_sv_min": pm_sv,
        "energy_parity_abs_Eh": parity,
        "target_projector_metric_trace": metric_trace,
        "alpha_trace_au": float(np.trace(alpha)),
        "failed_gates": " ".join(failed),
        "status": "PASS" if not failed else "FAIL",
    }
    return row, even_h, metric_trace


def run(args: argparse.Namespace) -> None:
    zero_root = args.zero_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    r, point_dir = closest_point(point_index(zero_root), args.R)
    zero_h5 = point_dir / "point.h5"
    zero = load_h5(zero_h5)
    identities = identity_index(zero_root)[r]
    target = [identities[name] for name in TARGET_NAMES]
    zero_ci, _ = project_singlet_roots(zero["ci_vectors"])
    zero_frame = {
        "mo_coeff": np.asarray(zero["mo_coeff"]),
        "ci_vectors": np.asarray(zero_ci),
        "e_states_Eh": np.asarray(zero["e_states_Eh"], dtype=float),
    }
    (output / "zero_source.txt").write_text(str(zero_h5) + "\n")
    fields = sorted(set(args.fields))
    if len(fields) < 4 or any(field <= 0.0 for field in fields):
        raise SystemExit("Supply at least four positive field magnitudes")

    print(f"FIXED-CASCI zero reproduction at R={r:.10f} A", flush=True)
    zero_fixed = fixed_casci_frame(
        zero, zero_h5, r, 0.0, args.memory_mb,
        output / "fields" / "F_0.h5"
    )
    s_ao = np.asarray(molecule(r, verbose=0).intor_symmetric("int1e_ovlp"))
    overlap_zero = state_overlap(zero_frame, zero_fixed, s_ao)
    _, _, zero_target_sv = select_target(overlap_zero, target)
    zero_full_sv = np.linalg.svd(overlap_zero, compute_uv=False)
    zero_reproduction = {
        "target_subspace_sv_min": float(zero_target_sv.min()),
        "rank6_subspace_sv_min": float(zero_full_sv.min()),
        "rank6_singular_values": zero_full_sv.tolist(),
        "status": (
            "PASS"
            if float(zero_target_sv.min())
            >= GATES["zero_target_subspace_sv_min"]
            else "FAIL"
        ),
    }

    frames: dict[float, dict[int, dict[str, Any]]] = defaultdict(dict)
    total = 2 * len(fields)
    completed = 0
    for field in fields:
        for sign in (-1, 1):
            signed = sign * field
            label = f"F_{'p' if sign > 0 else 'm'}{field:.1e}"
            print(
                f"FIXED-CASCI {completed + 1}/{total}: Fz={signed:+.1e}",
                flush=True,
            )
            frames[field][sign] = fixed_casci_frame(
                zero, zero_h5, r, signed, args.memory_mb,
                output / "fields" / f"{label}.h5",
            )
            completed += 1

    pair_rows = []
    even_hamiltonians = []
    metric_values = []
    for field in fields:
        row, even_h, metric_trace = compare_pair(
            zero_frame, frames[field][-1], frames[field][1],
            target, s_ao, field,
        )
        pair_rows.append(row)
        even_hamiltonians.append(even_h)
        metric_values.append(metric_trace)
        print(
            f"PAIR {row['status']} F={field:.1e}: "
            f"zero_sv={row['zero_target_sv_min']:.8f}, "
            f"pm_sv={row['plus_minus_target_sv_min']:.8f}",
            flush=True,
        )

    h_zero_fit, alpha_limit, fit_spread, fit_stability = (
        fit_even_hamiltonian_limit(fields, even_hamiltonians)
    )
    metric_limit, metric_spread, metric_stability = fit_limit(
        fields, metric_values
    )
    h0 = np.diag(zero_frame["e_states_Eh"][target])
    zero_offset = 0.5 * (
        h_zero_fit - h0 + (h_zero_fit - h0).T
    )
    zero_offset_max = float(
        np.max(np.abs(np.linalg.eigvalsh(zero_offset)))
    )
    delta_geom = (
        float(np.trace(alpha_limit)) / (2.0 * metric_limit)
        if metric_limit > 1.0e-14 else float("nan")
    )
    limit_checks = {
        "fit_relative_spread_max": (
            max(fit_spread, metric_spread)
            <= GATES["fit_relative_spread_max"]
        ),
        "fit_relative_stability_max": (
            max(fit_stability, metric_stability)
            <= GATES["fit_relative_spread_max"]
        ),
        "common_frame_zero_offset_abs_Eh_max": (
            zero_offset_max
            <= GATES["common_frame_zero_offset_abs_Eh_max"]
        ),
    }
    failed_limit_gates = [
        name for name, passed in limit_checks.items() if not passed
    ]
    limit = {
        "alpha_matrix_F0_au": alpha_limit.tolist(),
        "alpha_eigenvalues_F0_au": np.linalg.eigvalsh(alpha_limit).tolist(),
        "alpha_trace_F0_au": float(np.trace(alpha_limit)),
        "target_projector_metric_trace_F0": metric_limit,
        "Delta_geom_target_Eh": delta_geom,
        "Delta_geom_target_eV": delta_geom * EH_TO_EV,
        "fit_relative_spread": fit_spread,
        "fit_relative_stability": fit_stability,
        "metric_fit_relative_spread": metric_spread,
        "metric_fit_relative_stability": metric_stability,
        "common_frame_zero_offset_abs_max_Eh": zero_offset_max,
        "failed_gates": failed_limit_gates,
        "status": "PASS" if not failed_limit_gates else "FAIL",
    }

    print("AUXILIARY B1u/Au channel spectrum", flush=True)
    auxiliary_rows, auxiliary_summary = auxiliary_channel_spectrum(
        zero, r, target, args.aux_roots, args.memory_mb, output
    )
    fixed_pairs_pass = all(row["status"] == "PASS" for row in pair_rows)
    if zero_reproduction["status"] != "PASS":
        diagnosis = "ZERO_TO_C2V_MAPPING_OR_ROOT_SELECTION_FAILURE"
        recommendation = (
            "Repair the fixed-orbital zero-field reproduction before "
            "interpreting any response."
        )
    elif fixed_pairs_pass:
        diagnosis = "ORBITAL_OPTIMIZATION_BRANCH_BIFURCATION"
        recommendation = (
            "The rank-three CI manifold is continuous on frozen orbitals. "
            "The optimized-CASSCF failure comes from orbital-stationary-branch "
            "switching; use a coupled-perturbed/constrained-orbital treatment "
            "or report the certified transverse response."
        )
    else:
        diagnosis = "RANK_THREE_FIXED_ORBITAL_MANIFOLD_NOT_CONTINUOUS"
        recommendation = (
            "Enlarge the longitudinal parent/target space using the sampled "
            "B1u and Au response channels before any z-response production."
        )

    summary = {
        "status": (
            "PASS" if zero_reproduction["status"] == "PASS"
            and fixed_pairs_pass
            and not failed_limit_gates else "REVIEW"
        ),
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "method": (
            "fixed zero-field SA6-CASSCF orbitals with signed-field "
            "CASCI(8,8), direction-adapted C2v root homing"
        ),
        "R_angstrom": r,
        "zero_h5": str(zero_h5),
        "target_names": list(TARGET_NAMES),
        "target_solver_indices": target,
        "field_magnitudes_au": fields,
        "zero_reproduction": zero_reproduction,
        "pair_count": len(pair_rows),
        "pair_failure_count": sum(
            row["status"] != "PASS" for row in pair_rows
        ),
        "minimum_zero_target_sv": min(
            row["zero_target_sv_min"] for row in pair_rows
        ),
        "minimum_plus_minus_target_sv": min(
            row["plus_minus_target_sv_min"] for row in pair_rows
        ),
        "fixed_orbital_limit": limit,
        "auxiliary_channels": auxiliary_summary,
        "output_directory": str(output),
    }
    write_csv(output / "fixed_casci_signed_pairs.csv", pair_rows)
    write_csv(output / "auxiliary_channels.csv", auxiliary_rows)
    np.savez_compressed(
        output / "fixed_casci_limit_matrices.npz",
        H_subgroup_F0=h_zero_fit,
        H_D2h_F0=h0,
        alpha_F0=alpha_limit,
    )
    (output / "fixed_casci_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def self_test() -> None:
    fields = [5.0e-5, 1.0e-4, 2.5e-4, 5.0e-4]
    h0 = np.diag((-1.0, -0.9, -0.8))
    alpha = np.diag((2.0, 4.0, 6.0))
    matrices = [
        h0 - 0.5 * alpha * field**2
        + np.eye(3) * 1.0e5 * field**4
        for field in fields
    ]
    recovered_h0, recovered_alpha, spread, stability = (
        fit_even_hamiltonian_limit(fields, matrices)
    )
    assert np.max(np.abs(recovered_h0 - h0)) < 1.0e-12
    assert np.max(np.abs(recovered_alpha - alpha)) < 1.0e-7
    assert spread < 1.0e-6 and stability < 0.05
    print("SELF-TEST PASS: fixed-CASCI response fit and decision scaffolding")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--zero-root", type=Path)
    parser.add_argument("--R", type=float, default=3.0390625)
    parser.add_argument(
        "--output", type=Path,
        default=Path("C2_SA6_z_closure_diagnostic_v1"),
    )
    parser.add_argument(
        "--fields", type=float, nargs="+",
        default=(5.0e-5, 1.0e-4, 2.5e-4, 5.0e-4),
    )
    parser.add_argument("--aux-roots", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=120000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.zero_root is None:
        raise SystemExit("--zero-root is required unless --self-test is used")
    if args.aux_roots < 1:
        raise SystemExit("--aux-roots must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
