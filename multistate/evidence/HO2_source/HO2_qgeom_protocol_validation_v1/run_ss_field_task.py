#!/usr/bin/env python3
"""Run one state-specific signed-field validation task."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ho2_core import build_molecule, build_reference_mf, grid_by_source_index, qc_modules
from protocol_core import (
    GROUND_OVERLAP_WARN,
    VERSION,
    ci_roots_n,
    complete_h5,
    load_h5,
    optimize_protocol,
    pack_result,
    same_geometry_overlap_n,
    save_h5,
)


def read_manifest(path: Path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def run_task_record(
    record, workdir: Path, memory_mb: int, threads: int, force: bool
):
    task_id = int(record["task_id"])
    source_index = int(record["source_index"])
    signed_field = float(record["signed_field_au"])
    output = Path(record["output_path"])
    if complete_h5(output, "ss") and not force:
        return {"task_id": task_id, "status": "SKIP", "output_path": str(output)}
    if output.exists() and not force:
        raise RuntimeError(f"Refusing incomplete field file {output}; use --force")

    _, _, _, lib, _, _, _, _, _, _ = qc_modules()
    lib.num_threads(int(threads))
    row = grid_by_source_index(workdir)[source_index]
    zero = load_h5(Path(record["zero_path"]))
    if str(zero["attrs"].get("protocol")) != "ss":
        raise RuntimeError(f"Not an SS zero anchor: {record['zero_path']}")
    mol = build_molecule(
        row, basis=str(record["basis"]), memory_mb=memory_mb, symmetry=True
    )
    zero_ci = ci_roots_n(zero["ci_vectors"], 1)
    zero_mo = np.asarray(zero["mo_coeff"])
    mf = build_reference_mf(mol, zero_mo, zero_ci, field_x=signed_field)
    started = time.time()
    mc, mo, ci, stages = optimize_protocol(mf, zero_mo, zero_ci, "ss")
    arrays = pack_result(mc, mo, ci, "ss")
    current = {
        "mo_coeff": arrays["mo_coeff"],
        "ci_vectors": arrays["ci_vectors"],
    }
    overlap = same_geometry_overlap_n(zero, current, mol)
    ground_overlap = float(abs(overlap[0, 0]))
    if ground_overlap < GROUND_OVERLAP_WARN:
        raise RuntimeError(
            f"Field task {task_id} lost the SS ground state: "
            f"|overlap|={ground_overlap:.6f}"
        )

    arrays.update({
        "state_energy_rank": np.asarray([0], dtype=int),
        "zero_state_overlap": overlap,
    })
    attrs = {
        "package_version": VERSION,
        "converged": True,
        "calculation": "signed_field_ss_casscf",
        "protocol": "ss",
        "task_id": task_id,
        "source_index": source_index,
        "grid_position": int(record["grid_position"]),
        "signed_field_au": signed_field,
        "field_magnitude_au": abs(signed_field),
        "basis": str(record["basis"]),
        "zero_path": str(record["zero_path"]),
        "field_convention": "H(F)=H0+F*x_electron; nuclear linear term omitted",
        "zero_ground_overlap_abs": ground_overlap,
        "stage_reports": stages,
        "wall_seconds": time.time() - started,
        "threads": int(threads),
        "memory_mb": int(memory_mb),
    }
    save_h5(output, arrays, attrs)
    return {
        "task_id": task_id,
        "status": "PASS",
        "protocol": "ss",
        "source_index": source_index,
        "signed_field_au": signed_field,
        "zero_ground_overlap_abs": ground_overlap,
        "output_path": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--workdir")
    parser.add_argument("--memory-mb", type=int, default=30000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    records = read_manifest(manifest)
    if not 0 <= args.task_id < len(records):
        raise SystemExit(f"--task-id must lie in [0,{len(records)-1}]")
    workdir = (
        Path(args.workdir).resolve()
        if args.workdir
        else manifest.parents[2]
    )
    result = run_task_record(
        records[args.task_id], workdir, args.memory_mb, args.threads, args.force
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
