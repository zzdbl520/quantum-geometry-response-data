#!/usr/bin/env python3
"""Continue the accepted C2 rank-six SA-CASSCF frame to one geometry."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
from pyscf import fci, gto, mcscf, scf
from pyscf.fci import addons as fci_addons
from pyscf.fci import spin_op

from c2_sa6_core import (
    NCAS,
    NCORE,
    NELECAS,
    NSTATE,
    active_noons,
    assign_linear_labels,
    build_solvers,
    molecule,
    orbital_labels,
    projected_seed_guess,
    restricted_dipole,
    run_rhf,
    subspace_singular_values,
)


def load_frame(path: Path):
    with h5py.File(path, "r") as h5:
        r = float(h5.attrs["R_angstrom"])
        mo = np.asarray(h5["mo_coeff"])
        ci = [np.asarray(x) for x in np.asarray(h5["ci_vectors"])]
    if len(ci) != NSTATE:
        raise ValueError(f"Expected {NSTATE} restart CI roots, found {len(ci)}")
    return r, mo, ci


def transformed_restart(restart_h5: Path, mol, s_ao):
    prev_r, prev_mo, prev_ci = load_frame(restart_h5)
    prev_mol = molecule(prev_r, symmetry=True)
    projected = scf.addons.project_mo_nr2nr(prev_mol, prev_mo, mol)
    mo0, projection_info = projected_seed_guess(mol, projected, s_ao)
    s_cross = gto.intor_cross("int1e_ovlp", prev_mol, mol)
    old_active = prev_mo[:, NCORE : NCORE + NCAS]
    new_active = np.asarray(mo0[:, NCORE : NCORE + NCAS])
    active_rotation = old_active.T @ s_cross @ new_active
    active_rotation_sv = np.linalg.svd(active_rotation, compute_uv=False)

    ci0 = []
    for ci in prev_ci:
        transformed = fci_addons.transform_ci_for_orbital_rotation(
            ci, NCAS, NELECAS, active_rotation
        )
        norm = np.linalg.norm(transformed)
        if norm < 1.0e-12:
            raise RuntimeError("Transformed restart CI vector has zero norm")
        ci0.append(np.asarray(transformed) / norm)
    return prev_r, projected, mo0, ci0, {
        "restart_h5": str(restart_h5),
        "restart_R_angstrom": prev_r,
        "raw_projected_full_MO_metric_error": float(
            np.max(np.abs(projected.T @ s_ao @ projected - np.eye(projected.shape[1])))
        ),
        "active_rotation_singular_values": active_rotation_sv.tolist(),
        "symmetry_projection": projection_info,
    }


def write_states(path: Path, rows):
    fields = (
        "solver_order",
        "irrep",
        "irrep_root",
        "energy_Eh",
        "relative_energy_mEh",
        "global_energy_rank",
        "spin_square",
        "multiplicity",
        "local_assignment",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def copy_ci_frame(ci):
    if isinstance(ci, (list, tuple)):
        return [np.array(x, copy=True) for x in ci]
    array = np.asarray(ci)
    if array.ndim >= 3 and array.shape[0] == NSTATE:
        return [np.array(x, copy=True) for x in array]
    raise ValueError("Cannot interpret the state-averaged CI frame")


def build_optimizer(
    mf,
    mol,
    out,
    max_memory_mb,
    name,
    optimizer,
    grad_tolerance,
    max_macro,
    max_stepsize,
    level_shift,
    spin_penalty_shift,
):
    solvers, state_meta = build_solvers(
        mol, spin_penalty_shift=spin_penalty_shift
    )
    mc = mcscf.CASSCF(mf, NCAS, NELECAS)
    mc.max_memory = max_memory_mb
    mc.conv_tol = 1.0e-10
    mc.conv_tol_grad = grad_tolerance
    mc.max_cycle_macro = max_macro
    mc.max_cycle_micro = 8 if optimizer == "first_order" else 20
    mc.max_stepsize = max_stepsize
    mc.ah_level_shift = level_shift
    mc.ah_grad_trust_region = 1.5
    mc.kf_interval = 2
    mc.canonicalization = False
    mc.chkfile = str(out / f"sa6_{name}.chk")
    mc = mcscf.state_average_mix_(mc, solvers, [1.0 / NSTATE] * NSTATE)
    if optimizer == "newton":
        mc = mc.newton()
        mc.max_cycle_macro = max_macro
        mc.max_cycle_micro = 20
        mc.max_stepsize = max_stepsize
        mc.ah_level_shift = level_shift
        mc.ah_grad_trust_region = 1.5
        mc.kf_interval = 2
        mc.canonicalization = False
        mc.chkfile = str(out / f"sa6_{name}.chk")
    return mc, state_meta


def optimize_stage(mc, mo0, ci0, name):
    best = {
        "gradient": float("inf"),
        "energy_Eh": None,
        "macro_iteration": None,
        "mo": np.array(mo0, copy=True),
        "ci": copy_ci_frame(ci0),
    }

    def callback(envs):
        gradient = envs.get("norm_gall", envs.get("norm_gorb"))
        mo = envs.get("mo")
        ci = envs.get("fcivec")
        if gradient is None or mo is None or ci is None:
            return
        gradient = float(gradient)
        if np.isfinite(gradient) and gradient < best["gradient"]:
            best["gradient"] = gradient
            best["energy_Eh"] = float(envs.get("e_tot", np.nan))
            best["macro_iteration"] = int(envs.get("imacro", -1))
            best["mo"] = np.array(mo, copy=True)
            best["ci"] = copy_ci_frame(ci)

    mc.callback = callback
    started = time.time()
    mc.kernel(mo0, ci0=ci0)
    wall_seconds = time.time() - started
    if mc.converged:
        next_mo = np.asarray(mc.mo_coeff)
        next_ci = copy_ci_frame(mc.ci)
        source = "converged_final_frame"
    else:
        next_mo, next_ci = best["mo"], best["ci"]
        source = "historical_minimum_gradient_frame"
    report = {
        "name": name,
        "converged": bool(mc.converged),
        "wall_seconds": wall_seconds,
        "best_gradient": best["gradient"],
        "best_energy_Eh": best["energy_Eh"],
        "best_macro_iteration": best["macro_iteration"],
        "continuation_frame_source": source,
    }
    print("===== OPTIMIZER STAGE =====", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return next_mo, next_ci, report


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    restart_h5 = args.restart_h5.resolve()

    mol = molecule(args.R, symmetry=True)
    s_ao = mol.intor_symmetric("int1e_ovlp")
    prev_r, _projected_prev, mo0, ci0, restart_info = transformed_restart(
        restart_h5, mol, s_ao
    )
    mf = run_rhf(mol, out)

    stages = (
        ("settle", "first_order", 1.0e-4, 80, 0.020, 0.50),
        ("damped_newton", "newton", 1.0e-6, 120, 0.010, 0.50),
        ("tight_settle", "first_order", 5.0e-6, 160, 0.005, 1.00),
        ("tight_newton", "newton", 1.0e-6, 120, 0.002, 1.00),
    )
    stage_reports = []
    stage_mo, stage_ci = mo0, ci0
    mc = None
    state_meta = None
    wall_seconds = 0.0
    for name, optimizer, grad_tol, max_macro, max_step, level_shift in stages:
        mc, state_meta = build_optimizer(
            mf,
            mol,
            out,
            args.max_memory_mb,
            name,
            optimizer,
            grad_tol,
            max_macro,
            max_step,
            level_shift,
            args.spin_penalty_shift,
        )
        stage_mo, stage_ci, stage_report = optimize_stage(
            mc, stage_mo, stage_ci, name
        )
        stage_reports.append(stage_report)
        wall_seconds += stage_report["wall_seconds"]
        if optimizer == "newton" and mc.converged:
            break
    penalized_energies = np.asarray(mc.e_states, dtype=float)
    ci_roots = [np.asarray(x) for x in list(mc.ci)]
    if penalized_energies.size != NSTATE or len(ci_roots) != NSTATE:
        raise RuntimeError(
            f"Expected {NSTATE} roots, got {penalized_energies.size} energies and "
            f"{len(ci_roots)} CI vectors"
        )

    spin_results = [spin_op.spin_square0(ci, NCAS, NELECAS) for ci in ci_roots]
    spin_squares = np.asarray([result[0] for result in spin_results], dtype=float)
    multiplicities = np.asarray([result[1] for result in spin_results], dtype=float)
    energies = penalized_energies - args.spin_penalty_shift * spin_squares

    energy_floor = float(energies.min())
    energy_order = np.argsort(energies)
    rank = {int(index): int(place + 1) for place, index in enumerate(energy_order)}
    rows = []
    for index, (meta, energy, ss, multiplicity) in enumerate(
        zip(state_meta, energies, spin_squares, multiplicities)
    ):
        rows.append(
            {
                "solver_order": index,
                "irrep": meta["irrep"],
                "irrep_root": meta["irrep_root"],
                "energy_Eh": float(energy),
                "relative_energy_mEh": float(1000.0 * (energy - energy_floor)),
                "global_energy_rank": rank[index],
                "spin_square": float(ss),
                "multiplicity": float(multiplicity),
                "assignment": "unassigned",
            }
        )
    degeneracy = assign_linear_labels(rows)
    for row in rows:
        row["local_assignment"] = row.pop("assignment")

    final_mo = np.asarray(mc.mo_coeff)
    final_orth_error = float(
        np.max(np.abs(final_mo.T @ s_ao @ final_mo - np.eye(final_mo.shape[1])))
    )
    _, final_labels = orbital_labels(mol, final_mo, s_ao)
    active_sv_vs_restart = subspace_singular_values(
        mo0[:, NCORE : NCORE + NCAS],
        final_mo[:, NCORE : NCORE + NCAS],
        s_ao,
    )
    retained_sv_vs_restart = subspace_singular_values(
        mo0[:, : NCORE + NCAS],
        final_mo[:, : NCORE + NCAS],
        s_ao,
    )
    mu = restricted_dipole(mol, final_mo, ci_roots)
    noons = active_noons(ci_roots)
    write_states(out / "states.csv", rows)
    for axis_index, axis in enumerate("xyz"):
        np.savetxt(out / f"mu_P_{axis}.csv", mu[axis_index], delimiter=",")
    np.savetxt(out / "active_noons.csv", noons, delimiter=",")

    h5_path = out / "point.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["R_angstrom"] = args.R
        h5.attrs["restart_R_angstrom"] = prev_r
        h5.attrs["method"] = "SA6-CASSCF(8,8)/aug-cc-pVTZ; 3Ag+B1g+B2u+B3u"
        h5.attrs["optimizer"] = "staged trust-region first-order/Newton"
        h5.attrs["spin_penalty_shift_Eh"] = args.spin_penalty_shift
        h5.attrs["energies_corrected_for_spin_penalty"] = bool(
            args.spin_penalty_shift > 0.0
        )
        h5.attrs["converged"] = bool(mc.converged)
        h5.attrs["state_irreps_json"] = json.dumps(
            [row["irrep"] for row in rows]
        )
        h5.create_dataset("e_states_Eh", data=energies)
        if args.spin_penalty_shift > 0.0:
            h5.create_dataset(
                "e_states_spin_penalized_Eh", data=penalized_energies
            )
        h5.create_dataset("mo_coeff", data=final_mo)
        h5.create_dataset("ci_vectors", data=np.asarray(ci_roots))
        h5.create_dataset("state_spin_square", data=spin_squares)
        h5.create_dataset("active_noons", data=noons)
        h5.create_dataset("mu_P_xyz", data=mu)

    gate = {
        "casscf_converged": bool(mc.converged),
        "final_mo_orth_error_le_1e-8": final_orth_error <= 1.0e-8,
        "all_states_singlet_abs_S2_le_1e-6": bool(
            np.max(np.abs(spin_squares)) <= 1.0e-6
        ),
        "Pi_u_split_le_1e-7_Eh": degeneracy["Pi_u_component_split_Eh"]
        <= 1.0e-7,
        "Delta_g_split_le_1e-7_Eh": degeneracy["Delta_g_component_split_Eh"]
        <= 1.0e-7,
        "active_min_sv_vs_projected_restart_ge_0p995": float(
            active_sv_vs_restart.min()
        )
        >= 0.995,
        "retained_min_sv_vs_projected_restart_ge_0p995": float(
            retained_sv_vs_restart.min()
        )
        >= 0.995,
    }
    report = {
        "status": "PASS" if all(gate.values()) else "REVIEW",
        "R_angstrom": args.R,
        "restart": restart_info,
        "method": "equal-weight symmetry-complete SA6-CASSCF(8,8)/aug-cc-pVTZ",
        "optimizer": "staged trust-region first-order/Newton",
        "spin_penalty_shift_Eh": args.spin_penalty_shift,
        "energies_corrected_for_spin_penalty": bool(
            args.spin_penalty_shift > 0.0
        ),
        "optimizer_stages": stage_reports,
        "casscf_converged": bool(mc.converged),
        "average_energy_Eh": float(np.mean(energies)),
        "states": rows,
        "degeneracy_diagnostics": degeneracy,
        "final_mo_orthogonality_error": final_orth_error,
        "final_active_orbital_irreps": final_labels[NCORE : NCORE + NCAS],
        "active_subspace_singular_values_vs_projected_restart": (
            active_sv_vs_restart.tolist()
        ),
        "retained_subspace_singular_values_vs_projected_restart": (
            retained_sv_vs_restart.tolist()
        ),
        "mu_P_eigenvalues_au": {
            axis: np.linalg.eigvalsh(mu[k]).tolist()
            for k, axis in enumerate("xyz")
        },
        "wall_seconds": wall_seconds,
        "gate": gate,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=float, required=True)
    parser.add_argument("--restart-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-memory-mb", type=int, default=120000)
    parser.add_argument("--spin-penalty-shift", type=float, default=0.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
