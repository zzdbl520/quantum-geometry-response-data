#!/usr/bin/env python3
"""Symmetry-complete SA6-CASSCF center test for singlet C2.

The archived C1 SA4 scan does not contain complete linear-molecule multiplets.
At R=1.70 Angstrom this script instead optimizes one common CAS(8,8) orbital
frame for the six D2h components

    3 Ag + 1 B1g + 1 B2u + 1 B3u,

which represent two Sigma_g+ candidates, a complete Delta_g pair, and a
complete Pi_u pair.  Two independently constructed orbital guesses are
supported so that agreement is a numerical result rather than an assumption.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.linalg
from pyscf import fci, gto, lib, mcscf, scf, symm
from pyscf.fci import addons as fci_addons


IRREP_ORDER = ("Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u")
CORE_COUNTS = {"Ag": 1, "B1u": 1}
ACTIVE_COUNTS = {
    "Ag": 2,
    "B1u": 2,
    "B2u": 1,
    "B3u": 1,
    "B2g": 1,
    "B3g": 1,
}
STATE_BLOCKS = (("Ag", 3), ("B1g", 1), ("B2u", 1), ("B3u", 1))
NCORE = 2
NCAS = 8
NELECAS = (4, 4)
NSTATE = 6


def molecule(r_angstrom: float, symmetry: bool = True):
    return gto.M(
        atom=(
            f"C 0 0 {-r_angstrom / 2:.12f}; "
            f"C 0 0 {+r_angstrom / 2:.12f}"
        ),
        unit="Angstrom",
        basis="aug-cc-pVTZ",
        charge=0,
        spin=0,
        symmetry="D2h" if symmetry else False,
        verbose=4,
    )


def nearest_seed_file(pilot_dir: Path, r: float):
    candidates = []
    for path in pilot_dir.glob("R_*.h5"):
        try:
            with h5py.File(path, "r") as h5:
                rp = float(h5.attrs["R_angstrom"])
                converged = bool(h5.attrs.get("converged", False))
            if converged:
                candidates.append((abs(rp - r), rp, path))
        except (OSError, KeyError, ValueError):
            continue
    if not candidates:
        raise FileNotFoundError(f"No converged R_*.h5 point in {pilot_dir}")
    _, rp, path = min(candidates, key=lambda row: (row[0], row[1]))
    return path, rp


def metric_orthonormalize(coeff, s_ao):
    metric = coeff.T @ s_ao @ coeff
    eig, vec = np.linalg.eigh(0.5 * (metric + metric.T))
    if eig.min() < 1.0e-10:
        raise RuntimeError(
            f"Orbital metric is singular: minimum eigenvalue={eig.min():.3e}"
        )
    result = coeff @ (vec @ np.diag(eig ** -0.5) @ vec.T)
    error = float(
        np.max(np.abs(result.T @ s_ao @ result - np.eye(result.shape[1])))
    )
    return result, error


def load_archived_seed(pilot_dir: Path, r: float, mol, s_ao):
    seed_path, seed_r = nearest_seed_file(pilot_dir, r)
    with h5py.File(seed_path, "r") as h5:
        seed_mo = np.asarray(h5["mo_coeff"])
        seed_energies = np.asarray(h5["e_states_Eh"], dtype=float)
        seed_method = str(h5.attrs.get("method", "unknown"))
    seed_mol = molecule(seed_r, symmetry=False)
    seed_mo = scf.addons.project_mo_nr2nr(seed_mol, seed_mo, mol)
    seed_mo, orth_error = metric_orthonormalize(seed_mo, s_ao)
    return seed_mo, {
        "file": str(seed_path),
        "R_angstrom": float(seed_r),
        "method": seed_method,
        "energies_Eh": seed_energies.tolist(),
        "average_energy_Eh": float(np.mean(seed_energies)),
        "projection_orth_error": orth_error,
    }


def run_rhf(mol, out: Path):
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.max_cycle = 200
    mf.max_memory = 120000
    mf.chkfile = str(out / "rhf.chk")
    mf.kernel()
    if not mf.converged:
        newton = mf.newton()
        newton.conv_tol = 1.0e-12
        newton.max_cycle = 100
        newton.kernel()
        mf = newton
    if not mf.converged:
        raise RuntimeError("D2h RHF did not converge")
    return mf


def orbital_labels(mol, mo, s_ao):
    raw_labels = symm.label_orb_symm(
        mol, mol.irrep_name, mol.symm_orb, mo, s=s_ao, check=True
    )
    # PySCF releases differ here: older versions commonly return integer irrep
    # IDs, whereas PySCF 2.13 may return irrep-name strings directly.  Normalize
    # both representations without changing the tagged orbital array.
    ids = []
    names = []
    for label in raw_labels:
        if isinstance(label, (str, np.str_)):
            name = str(label)
            irrep_id = symm.irrep_name2id(mol.groupname, name)
        else:
            irrep_id = int(label)
            name = symm.irrep_id2name(mol.groupname, irrep_id)
        ids.append(int(irrep_id))
        names.append(name)
    return np.asarray(ids, dtype=int), names


def subspace_singular_values(left, right, s_ao):
    return np.linalg.svd(left.T @ s_ao @ right, compute_uv=False)


def diagnose_guess(mol, mo, archived_mo, s_ao, construction, extra=None):
    _, labels = orbital_labels(mol, mo, s_ao)
    core_sv = subspace_singular_values(
        archived_mo[:, :NCORE], mo[:, :NCORE], s_ao
    )
    active_sv = subspace_singular_values(
        archived_mo[:, NCORE : NCORE + NCAS],
        mo[:, NCORE : NCORE + NCAS],
        s_ao,
    )
    retained_sv = subspace_singular_values(
        archived_mo[:, : NCORE + NCAS], mo[:, : NCORE + NCAS], s_ao
    )
    orth_error = float(
        np.max(np.abs(mo.T @ s_ao @ mo - np.eye(mo.shape[1])))
    )
    if orth_error > 1.0e-8:
        raise RuntimeError(
            f"Initial MO frame is not S-orthonormal: max error={orth_error:.3e}"
        )
    result = {
        "construction": construction,
        "orth_error": orth_error,
        "core_min_subspace_sv_vs_archived": float(core_sv.min()),
        "active_min_subspace_sv_vs_archived": float(active_sv.min()),
        "core_plus_active_min_subspace_sv_vs_archived": float(retained_sv.min()),
        "active_orbital_irreps": labels[NCORE : NCORE + NCAS],
    }
    if extra:
        result.update(extra)
    return result


def balanced_rhf_guess(mf, archived_mo, s_ao):
    probe = mcscf.CASSCF(mf, NCAS, NELECAS)
    cas_list = mcscf.caslst_by_irrep(
        probe,
        mf.mo_coeff,
        ACTIVE_COUNTS,
        CORE_COUNTS,
        s=s_ao,
        base=0,
    )
    mo = mcscf.sort_mo_by_irrep(
        probe,
        mf.mo_coeff,
        ACTIVE_COUNTS,
        CORE_COUNTS,
        s=s_ao,
    )
    diag = diagnose_guess(
        mf.mol,
        mo,
        archived_mo,
        s_ao,
        "D2h RHF orbitals selected by explicit valence irrep counts",
        {"active_original_indices_zero_based": [int(x) for x in cas_list]},
    )
    return mo, diag


def symmetry_block_map(mol, s_ao):
    """Return S-orthonormal SALCs for every irrep.

    ``mol.symm_orb`` spans symmetry-adapted AO blocks, but those columns are
    not guaranteed to satisfy C^T S C = I.  Projector SVDs require an
    orthonormal coordinate basis, so each block is Lowdin-orthogonalized in
    its own overlap metric before it is used.
    """
    result = {}
    for name, irrep_id, coeff in zip(
        mol.irrep_name, mol.irrep_id, mol.symm_orb
    ):
        coeff = np.asarray(coeff)
        metric = coeff.T @ s_ao @ coeff
        eig, vec = np.linalg.eigh(0.5 * (metric + metric.T))
        if eig.min() < 1.0e-12:
            raise RuntimeError(
                f"Symmetry AO block {name} is singular: "
                f"minimum eigenvalue={eig.min():.3e}"
            )
        lowdin = vec @ np.diag(eig ** -0.5) @ vec.T
        orth_coeff = coeff @ lowdin
        orth_error = float(
            np.max(
                np.abs(
                    orth_coeff.T @ s_ao @ orth_coeff
                    - np.eye(orth_coeff.shape[1])
                )
            )
        )
        if orth_error > 1.0e-9:
            raise RuntimeError(
                f"Failed to orthonormalize symmetry AO block {name}: "
                f"max error={orth_error:.3e}"
            )
        result[name] = (orth_coeff, int(irrep_id))
    return result


def projected_seed_guess(mol, archived_mo, s_ao):
    """Best fixed-count D2h approximation to the archived core/active spaces."""
    blocks = symmetry_block_map(mol, s_ao)
    seed_core = archived_mo[:, :NCORE]
    seed_active = archived_mo[:, NCORE : NCORE + NCAS]
    core_columns = []
    core_ids = []
    active_columns = []
    active_ids = []
    coords_by_irrep = {}
    block_diagnostics = {}

    for name in IRREP_ORDER:
        if name not in blocks:
            continue
        sym_orb, irrep_id = blocks[name]
        n_sym = sym_orb.shape[1]
        n_core = CORE_COUNTS.get(name, 0)
        n_active = ACTIVE_COUNTS.get(name, 0)

        if n_core:
            overlap = sym_orb.T @ s_ao @ seed_core
            left, singular, _ = np.linalg.svd(overlap, full_matrices=False)
            if left.shape[1] < n_core:
                raise RuntimeError(f"Insufficient {name} core directions")
            core_coord = left[:, :n_core]
            core_columns.extend(
                [sym_orb @ core_coord[:, [k]] for k in range(n_core)]
            )
            core_ids.extend([irrep_id] * n_core)
            core_singular = singular[:n_core].tolist()
        else:
            core_coord = np.zeros((n_sym, 0))
            core_singular = []

        complement = scipy.linalg.null_space(core_coord.T)
        if n_active:
            available = sym_orb @ complement
            overlap = available.T @ s_ao @ seed_active
            left, singular, _ = np.linalg.svd(overlap, full_matrices=False)
            if left.shape[1] < n_active:
                raise RuntimeError(f"Insufficient {name} active directions")
            active_coord = complement @ left[:, :n_active]
            active_columns.extend(
                [sym_orb @ active_coord[:, [k]] for k in range(n_active)]
            )
            active_ids.extend([irrep_id] * n_active)
            active_singular = singular[:n_active].tolist()
        else:
            active_coord = np.zeros((n_sym, 0))
            active_singular = []

        selected_coord = np.hstack((core_coord, active_coord))
        coords_by_irrep[name] = selected_coord
        block_diagnostics[name] = {
            "core_count": n_core,
            "active_count": n_active,
            "core_projection_singular_values": core_singular,
            "active_projection_singular_values": active_singular,
        }

    if len(core_columns) != NCORE or len(active_columns) != NCAS:
        raise RuntimeError(
            f"Projected guess has {len(core_columns)} core and "
            f"{len(active_columns)} active orbitals"
        )

    external_columns = []
    external_ids = []
    for name in IRREP_ORDER:
        if name not in blocks:
            continue
        sym_orb, irrep_id = blocks[name]
        selected = coords_by_irrep.get(name, np.zeros((sym_orb.shape[1], 0)))
        complement = scipy.linalg.null_space(selected.T)
        external_columns.extend(
            [sym_orb @ complement[:, [k]] for k in range(complement.shape[1])]
        )
        external_ids.extend([irrep_id] * complement.shape[1])

    columns = core_columns + active_columns + external_columns
    ids = core_ids + active_ids + external_ids
    mo = np.hstack(columns)
    mo = lib.tag_array(mo, orbsym=np.asarray(ids, dtype=int))
    diag = diagnose_guess(
        mol,
        mo,
        archived_mo,
        s_ao,
        "Maximum-overlap D2h projection with fixed valence irrep counts",
        {"projection_by_irrep": block_diagnostics},
    )
    return mo, diag


def build_solvers(mol, spin_penalty_shift=0.0):
    solvers = []
    state_meta = []
    for irrep, nroots in STATE_BLOCKS:
        solver = fci.direct_spin0_symm.FCI(mol)
        solver.wfnsym = irrep
        solver.nroots = nroots
        solver.conv_tol = 1.0e-11
        solver.max_cycle = 200
        if spin_penalty_shift > 0.0:
            solver = fci_addons.fix_spin_(
                solver, shift=spin_penalty_shift, ss=0.0
            )
        solvers.append(solver)
        for root in range(nroots):
            state_meta.append({"irrep": irrep, "irrep_root": root})
    return solvers, state_meta


def assign_linear_labels(rows):
    ag = [row for row in rows if row["irrep"] == "Ag"]
    b1g = next(row for row in rows if row["irrep"] == "B1g")
    delta_ag = min(ag, key=lambda row: abs(row["energy_Eh"] - b1g["energy_Eh"]))
    remaining_ag = sorted(
        [row for row in ag if row is not delta_ag], key=lambda row: row["energy_Eh"]
    )
    if len(remaining_ag) != 2:
        raise RuntimeError("Expected two Sigma_g+ candidates after Delta assignment")
    remaining_ag[0]["assignment"] = "X 1Sigma_g+ candidate"
    remaining_ag[1]["assignment"] = "B' 1Sigma_g+ candidate"
    delta_ag["assignment"] = "B 1Delta_g (x2-y2 component) candidate"
    b1g["assignment"] = "B 1Delta_g (xy component) candidate"
    for row in rows:
        if row["irrep"] == "B2u":
            row["assignment"] = "A 1Pi_u (y component) candidate"
        elif row["irrep"] == "B3u":
            row["assignment"] = "A 1Pi_u (x component) candidate"

    b2u = next(row for row in rows if row["irrep"] == "B2u")
    b3u = next(row for row in rows if row["irrep"] == "B3u")
    return {
        "Pi_u_component_split_Eh": float(abs(b2u["energy_Eh"] - b3u["energy_Eh"])),
        "Delta_g_component_split_Eh": float(
            abs(delta_ag["energy_Eh"] - b1g["energy_Eh"])
        ),
        "Delta_g_Ag_root": int(delta_ag["irrep_root"]),
    }


def restricted_dipole(mol, mo, ci_roots):
    mol.set_common_orig_((0.0, 0.0, 0.0))
    r_ao = mol.intor("int1e_r", comp=3)
    c_core = np.asarray(mo[:, :NCORE])
    c_active = np.asarray(mo[:, NCORE : NCORE + NCAS])
    result = np.zeros((3, NSTATE, NSTATE))
    for i, bra in enumerate(ci_roots):
        for j, ket in enumerate(ci_roots):
            transition_dm = fci.direct_spin0.trans_rdm1(
                bra, ket, NCAS, NELECAS
            )
            dm_ao = c_active @ np.asarray(transition_dm) @ c_active.T
            if i == j:
                dm_ao = dm_ao + 2.0 * (c_core @ c_core.T)
            result[:, i, j] = -np.einsum("aij,ji->a", r_ao, dm_ao)
    return 0.5 * (result + np.swapaxes(result, 1, 2))


def active_noons(ci_roots):
    rows = []
    for ci in ci_roots:
        dm = fci.direct_spin0.make_rdm1(ci, NCAS, NELECAS)
        rows.append(np.linalg.eigvalsh(dm)[::-1])
    return np.asarray(rows)


def write_state_csv(path, rows):
    fields = (
        "solver_order",
        "irrep",
        "irrep_root",
        "energy_Eh",
        "relative_energy_mEh",
        "global_energy_rank",
        "assignment",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mol = molecule(args.R, symmetry=True)
    s_ao = mol.intor_symmetric("int1e_ovlp")
    archived_mo, archived_info = load_archived_seed(
        args.pilot_dir.resolve(), args.R, mol, s_ao
    )
    mf = run_rhf(mol, out)

    ci0 = None
    if args.restart_h5 is not None:
        restart_path = args.restart_h5.resolve()
        with h5py.File(restart_path, "r") as h5:
            mo0 = np.asarray(h5["mo_coeff"])
            ci0 = [np.asarray(ci) for ci in np.asarray(h5["ci_vectors"])]
        guess_info = diagnose_guess(
            mol,
            mo0,
            archived_mo,
            s_ao,
            "Restart from completed first-order SA6 MO/CI frame",
            {"restart_h5": str(restart_path)},
        )
    elif args.init == "balanced_rhf":
        mo0, guess_info = balanced_rhf_guess(mf, archived_mo, s_ao)
    else:
        mo0, guess_info = projected_seed_guess(mol, archived_mo, s_ao)

    solvers, state_meta = build_solvers(mol)
    mc = mcscf.CASSCF(mf, NCAS, NELECAS)
    mc.max_memory = 120000
    mc.conv_tol = 1.0e-10
    mc.conv_tol_grad = 1.0e-6
    mc.max_cycle_macro = 250
    mc.max_cycle_micro = 10
    mc.chkfile = str(out / "sa6_casscf.chk")
    mc = mcscf.state_average_mix_(mc, solvers, [1.0 / NSTATE] * NSTATE)
    if args.optimizer == "newton":
        mc = mc.newton()
        mc.max_cycle_macro = 100
        mc.max_cycle_micro = 20
        mc.chkfile = str(out / "sa6_newton.chk")

    start = time.time()
    mc.kernel(mo0, ci0=ci0)
    wall_seconds = time.time() - start
    energies = np.asarray(mc.e_states, dtype=float)
    ci_roots = list(mc.ci)
    if energies.size != NSTATE or len(ci_roots) != NSTATE:
        raise RuntimeError(
            f"Expected {NSTATE} states, got {energies.size} energies and "
            f"{len(ci_roots)} CI vectors"
        )

    rows = []
    for index, (meta, energy) in enumerate(zip(state_meta, energies)):
        rows.append(
            {
                "solver_order": index,
                "irrep": meta["irrep"],
                "irrep_root": meta["irrep_root"],
                "energy_Eh": float(energy),
            }
        )
    energy_floor = float(energies.min())
    energy_order = np.argsort(energies)
    rank = {int(index): int(place + 1) for place, index in enumerate(energy_order)}
    for row in rows:
        row["relative_energy_mEh"] = float(
            1000.0 * (row["energy_Eh"] - energy_floor)
        )
        row["global_energy_rank"] = rank[row["solver_order"]]
        row["assignment"] = "unassigned"
    degeneracy = assign_linear_labels(rows)

    final_mo = np.asarray(mc.mo_coeff)
    final_orth_error = float(
        np.max(
            np.abs(final_mo.T @ s_ao @ final_mo - np.eye(final_mo.shape[1]))
        )
    )
    _, final_labels = orbital_labels(mol, final_mo, s_ao)
    final_active_sv = subspace_singular_values(
        archived_mo[:, NCORE : NCORE + NCAS],
        final_mo[:, NCORE : NCORE + NCAS],
        s_ao,
    )
    mu = restricted_dipole(mol, final_mo, ci_roots)
    mu_spectra = {
        axis: np.linalg.eigvalsh(mu[k]).tolist()
        for k, axis in enumerate("xyz")
    }
    noons = active_noons(ci_roots)

    write_state_csv(out / "states.csv", rows)
    for axis_index, axis in enumerate("xyz"):
        np.savetxt(
            out / f"mu_P_{axis}.csv",
            mu[axis_index],
            delimiter=",",
            fmt="%.16e",
        )
    np.savetxt(out / "active_noons.csv", noons, delimiter=",", fmt="%.16e")

    with h5py.File(out / "sa6_center.h5", "w") as h5:
        h5.attrs["R_angstrom"] = args.R
        h5.attrs["method"] = "SA6-CASSCF(8,8)/aug-cc-pVTZ; 3Ag+B1g+B2u+B3u"
        h5.attrs["initialization"] = args.init
        h5.attrs["optimizer"] = args.optimizer
        h5.attrs["converged"] = bool(mc.converged)
        h5.attrs["state_irreps_json"] = json.dumps(
            [row["irrep"] for row in rows]
        )
        h5.create_dataset("e_states_Eh", data=energies)
        h5.create_dataset("mo_coeff", data=final_mo)
        h5.create_dataset("ci_vectors", data=np.asarray(ci_roots))
        h5.create_dataset("active_noons", data=noons)

    gate = {
        "casscf_converged": bool(mc.converged),
        "final_mo_orth_error_le_1e-8": final_orth_error <= 1.0e-8,
        "Pi_u_split_le_1e-7_Eh": degeneracy["Pi_u_component_split_Eh"] <= 1.0e-7,
        "Delta_g_split_le_1e-7_Eh": degeneracy["Delta_g_component_split_Eh"] <= 1.0e-7,
    }
    report = {
        "status": "PASS" if all(gate.values()) else "REVIEW",
        "R_angstrom": args.R,
        "method": "equal-weight symmetry-complete SA6-CASSCF(8,8)/aug-cc-pVTZ",
        "state_model": "3Ag + B1g + B2u + B3u",
        "initialization": args.init,
        "optimizer": args.optimizer,
        "pyscf_version": getattr(__import__("pyscf"), "__version__", "unknown"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "wall_seconds": wall_seconds,
        "rhf_energy_Eh": float(mf.e_tot),
        "rhf_note": (
            "RHF is used only as a symmetry-orbital generator; its open pi-shell "
            "occupation is not a physical C2 reference."
        ),
        "archived_C1_SA4_seed": archived_info,
        "initial_guess": guess_info,
        "casscf_converged": bool(mc.converged),
        "average_energy_Eh": float(mc.e_average),
        "states": rows,
        "degeneracy_diagnostics": degeneracy,
        "final_mo_orth_error": final_orth_error,
        "final_active_orbital_irreps": final_labels[NCORE : NCORE + NCAS],
        "final_active_subspace_singular_values_vs_archived": final_active_sv.tolist(),
        "mu_P_eigenvalues_au": mu_spectra,
        "gate": gate,
        "interpretation": (
            "A rank-six model is accepted only after both independent starts "
            "converge to the same stationary solution and restore the Pi_u and "
            "Delta_g component degeneracies."
        ),
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        required=True,
        help="Archived C1 SA4 directory containing R_*.h5 files",
    )
    parser.add_argument("--R", type=float, default=1.70)
    parser.add_argument(
        "--init",
        choices=("balanced_rhf", "projected_seed"),
        required=True,
    )
    parser.add_argument(
        "--optimizer",
        choices=("first_order", "newton"),
        default="first_order",
    )
    parser.add_argument(
        "--restart-h5",
        type=Path,
        help="Optional completed SA6 HDF5 file supplying MO and CI restart data",
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
