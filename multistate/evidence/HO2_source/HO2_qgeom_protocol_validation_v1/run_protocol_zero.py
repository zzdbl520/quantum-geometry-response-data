#!/usr/bin/env python3
"""Run SS and/or SA4 zero-field transport on the validation grid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ho2_core import (
    BASIS,
    CAS_FOLLOW_SV_MIN,
    build_molecule,
    build_reference_mf,
    build_rohf_seed,
    load_grid,
    project_mos,
    qc_modules,
    write_json,
)
from protocol_core import (
    GROUND_OVERLAP_WARN,
    SUBSPACE_OVERLAP_WARN,
    VERSION,
    ci_roots_n,
    complete_h5,
    cross_geometry_overlap_n,
    load_h5,
    optimize_protocol,
    pack_result,
    pilot_zero_path,
    protocol_label,
    protocol_nstate,
    same_geometry_overlap_n,
    save_h5,
    transport_active_ci_n,
    validation_root,
    zero_path,
)


def run_geometry(
    row,
    workdir: Path,
    protocol: str,
    basis: str,
    memory_mb: int,
    parent,
    force: bool,
):
    source_index = int(row["source_index"])
    output = zero_path(workdir, protocol, source_index)
    if complete_h5(output, protocol) and not force:
        print(
            f"SKIP {protocol} zero source_index={source_index} path={output}",
            flush=True,
        )
        return load_h5(output)
    if output.exists() and not force:
        raise RuntimeError(f"Refusing incomplete zero file {output}; use --force")

    nstate = protocol_nstate(protocol)
    mol = build_molecule(row, basis=basis, memory_mb=memory_mb, symmetry=True)
    started = time.time()
    projection = None
    parent_overlap = None
    parent_status = None
    parent_source_index = None

    if parent is None:
        pilot_seed_path = pilot_zero_path(workdir, source_index)
        if not complete_h5(pilot_seed_path):
            raise RuntimeError(f"Missing SA2 seed checkpoint {pilot_seed_path}")
        pilot_seed = load_h5(pilot_seed_path)
        mo0 = np.asarray(pilot_seed["mo_coeff"])
        pilot_ci = ci_roots_n(pilot_seed["ci_vectors"], 2)
        ci0 = [pilot_ci[0]] if protocol == "ss" else None
        mf = build_reference_mf(mol, mo0, pilot_ci, field_x=0.0)
    else:
        parent_row, parent_data = parent
        parent_source_index = int(parent_row["source_index"])
        parent_mol = build_molecule(
            parent_row, basis=basis, memory_mb=memory_mb, symmetry=True
        )
        orbital_basis_mf = build_rohf_seed(mol)
        mo0, projection = project_mos(
            parent_mol,
            mol,
            np.asarray(parent_data["mo_coeff"]),
            orbital_basis_mf,
        )
        if projection["retained_projection_sv_min"] < CAS_FOLLOW_SV_MIN:
            raise RuntimeError(
                "Direct core+CAS continuation failed at source index "
                f"{source_index}: retained sv_min="
                f"{projection['retained_projection_sv_min']:.6f}"
            )
        ci0, ci_transport = transport_active_ci_n(
            parent_mol,
            mol,
            np.asarray(parent_data["mo_coeff"]),
            mo0,
            parent_data["ci_vectors"],
            nstate,
        )
        projection["active_ci_transport"] = ci_transport
        mf = build_reference_mf(mol, mo0, ci0, field_x=0.0)

    mc, mo, ci, stages = optimize_protocol(mf, mo0, ci0, protocol)
    arrays = pack_result(mc, mo, ci, protocol)
    current = dict(arrays)

    if parent is not None:
        parent_row, parent_data = parent
        parent_mol = build_molecule(
            parent_row, basis=basis, memory_mb=memory_mb, symmetry=True
        )
        parent_overlap = cross_geometry_overlap_n(
            parent_data, current, parent_mol, mol
        )
        parent_sv = np.linalg.svd(parent_overlap, compute_uv=False)
        parent_ground_best = float(np.max(np.abs(parent_overlap[0, :])))
        if protocol == "ss":
            parent_status = (
                "PASS"
                if parent_ground_best >= GROUND_OVERLAP_WARN
                else "LOW_GROUND_STATE_OVERLAP"
            )
        else:
            parent_status = (
                "PASS"
                if float(parent_sv.min()) >= SUBSPACE_OVERLAP_WARN
                else "FOUR_STATE_SUBSPACE_REPLACEMENT"
            )
        if parent_status != "PASS":
            print(
                f"WARN {protocol} zero source_index={source_index} "
                f"status={parent_status} subspace_sv_min={parent_sv.min():.6f} "
                f"ground_best_overlap={parent_ground_best:.6f}",
                flush=True,
            )
    else:
        parent_sv = None
        parent_ground_best = None

    pilot_path = pilot_zero_path(workdir, source_index)
    pilot_overlap = None
    pilot_ground_overlap = None
    pilot_subspace_sv = None
    if complete_h5(pilot_path):
        pilot = load_h5(pilot_path)
        pilot_overlap = same_geometry_overlap_n(pilot, current, mol)
        pilot_ground_overlap = float(abs(pilot_overlap[0, 0]))
        if protocol == "sa4":
            pilot_subspace_sv = np.linalg.svd(
                pilot_overlap, compute_uv=False
            ).tolist()

    energies = np.asarray(arrays["e_states_Eh"], dtype=float)
    attrs = {
        "package_version": VERSION,
        "converged": True,
        "calculation": f"zero_field_{protocol}_casscf",
        "protocol": protocol,
        "source_index": source_index,
        "grid_position": int(row["grid_position"]),
        "s_cart_A": float(row["s_cart_A"]),
        "basis": basis,
        "active_space": protocol_label(protocol),
        "nstate": nstate,
        "state_symmetry": "lowest doublet B1 state(s) in exact linear C2v",
        "parent_source_index": parent_source_index,
        "projection": projection,
        "parent_continuity_status": parent_status,
        "parent_ground_best_overlap": parent_ground_best,
        "parent_subspace_sv": None if parent_sv is None else parent_sv.tolist(),
        "parent_state_overlap_abs": (
            None if parent_overlap is None else np.abs(parent_overlap).tolist()
        ),
        "sa2_ground_to_protocol_ground_abs": pilot_ground_overlap,
        "sa2_two_state_in_protocol_sv": pilot_subspace_sv,
        "sa2_to_protocol_overlap_abs": (
            None if pilot_overlap is None else np.abs(pilot_overlap).tolist()
        ),
        "stage_reports": stages,
        "state_average_energy_Eh": float(np.mean(energies)),
        "energy_gap_Eh": (
            float(energies[1] - energies[0]) if len(energies) > 1 else None
        ),
        "wall_seconds": time.time() - started,
        "coords_json": row["coords_json"],
    }
    save_h5(output, arrays, attrs)
    gap_text = (
        f" gap={energies[1]-energies[0]:.6e}" if len(energies) > 1 else ""
    )
    print(
        f"PASS {protocol} zero source_index={source_index} "
        f"E0={energies[0]:.12f}{gap_text} path={output}",
        flush=True,
    )
    return load_h5(output)


def run_protocol(args, rows, protocol: str):
    seed_positions = [i for i, row in enumerate(rows) if int(row["is_seed"])]
    if len(seed_positions) != 1:
        raise RuntimeError(f"Expected one seed geometry, found {len(seed_positions)}")
    seed_pos = seed_positions[0]
    seed = run_geometry(
        rows[seed_pos],
        args.workdir,
        protocol,
        args.basis,
        args.memory_mb,
        None,
        args.force,
    )
    completed = {int(rows[seed_pos]["source_index"])}

    if args.direction in ("all", "lower"):
        parent = (rows[seed_pos], seed)
        for pos in range(seed_pos - 1, -1, -1):
            current = run_geometry(
                rows[pos], args.workdir, protocol, args.basis,
                args.memory_mb, parent, args.force
            )
            completed.add(int(rows[pos]["source_index"]))
            parent = (rows[pos], current)
            if int(rows[pos]["source_index"]) == args.stop_source_index:
                break

    if args.direction in ("all", "upper"):
        parent = (rows[seed_pos], seed)
        for pos in range(seed_pos + 1, len(rows)):
            current = run_geometry(
                rows[pos], args.workdir, protocol, args.basis,
                args.memory_mb, parent, args.force
            )
            completed.add(int(rows[pos]["source_index"]))
            parent = (rows[pos], current)
            if int(rows[pos]["source_index"]) == args.stop_source_index:
                break

    inventory = {
        "package_version": VERSION,
        "protocol": protocol,
        "requested_direction": args.direction,
        "completed_source_indices_this_invocation": sorted(completed),
        "pass_count_total": sum(
            complete_h5(zero_path(args.workdir, protocol, int(row["source_index"])), protocol)
            for row in rows
        ),
        "expected_count": len(rows),
    }
    path = validation_root(args.workdir) / protocol / "zero_inventory.json"
    write_json(path, inventory)
    print(json.dumps(inventory, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="HO2_qgeom_protocol_validation_work"
    )
    parser.add_argument("--protocol", choices=("ss", "sa4", "both"), default="both")
    parser.add_argument("--memory-mb", type=int, default=30000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument(
        "--direction", choices=("all", "seed", "lower", "upper"), default="all"
    )
    parser.add_argument("--stop-source-index", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.workdir = Path(args.workdir).resolve()
    rows = load_grid(args.workdir)
    if not rows:
        raise SystemExit("Empty geometry grid; run prepare_protocol.py first")
    *_, lib, _, _, _, _, _, _ = qc_modules()
    lib.num_threads(int(args.threads))
    protocols = ("ss", "sa4") if args.protocol == "both" else (args.protocol,)
    for protocol in protocols:
        run_protocol(args, rows, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
