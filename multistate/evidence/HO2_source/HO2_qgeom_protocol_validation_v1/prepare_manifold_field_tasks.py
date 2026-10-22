#!/usr/bin/env python3
"""Prepare signed-field SA4 tasks for a locally followed low-energy manifold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ho2_core import BASIS, load_grid, parse_floats, parse_indices, write_json
from protocol_core import (
    DEFAULT_MANIFOLD_FIELDS,
    DEFAULT_MANIFOLD_INDICES,
    DEFAULT_MANIFOLD_RANK,
    VERSION,
    complete_h5,
    protocol_field_path,
    validation_root,
    zero_path,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="HO2_qgeom_protocol_validation_work"
    )
    parser.add_argument("--indices", default=DEFAULT_MANIFOLD_INDICES)
    parser.add_argument(
        "--fields",
        default=",".join(f"{value:g}" for value in DEFAULT_MANIFOLD_FIELDS),
    )
    parser.add_argument("--rank", type=int, default=DEFAULT_MANIFOLD_RANK)
    parser.add_argument("--basis", default=BASIS)
    args = parser.parse_args()

    if not 1 <= args.rank <= 4:
        raise SystemExit("--rank must lie in [1,4] for the SA4 candidate frame")
    workdir = Path(args.workdir).resolve()
    rows = load_grid(workdir)
    if not rows:
        raise SystemExit("Empty geometry grid; run prepare_protocol.py")
    available = {int(row["source_index"]): row for row in rows}
    selected = parse_indices(args.indices, max(available) + 1)
    missing_grid = sorted(set(selected) - set(available))
    if missing_grid:
        raise SystemExit(f"Indices not present in validation grid: {missing_grid}")
    fields = parse_floats(args.fields)
    missing_zero = [
        index for index in selected
        if not complete_h5(zero_path(workdir, "sa4", index), "sa4")
    ]
    if missing_zero:
        raise SystemExit(
            "SA4 zero transport is incomplete at: "
            + ",".join(map(str, missing_zero))
        )

    records = []
    for source_index in selected:
        row = available[source_index]
        for magnitude in fields:
            for sign in (-1, 1):
                signed = sign * magnitude
                records.append({
                    "task_id": len(records),
                    "protocol": "sa4",
                    "manifold_rank": int(args.rank),
                    "source_index": source_index,
                    "grid_position": int(row["grid_position"]),
                    "signed_field_au": signed,
                    "field_magnitude_au": magnitude,
                    "basis": args.basis,
                    "zero_path": str(zero_path(workdir, "sa4", source_index)),
                    "output_path": str(protocol_field_path(
                        workdir, "sa4", source_index, signed, manifold=True
                    )),
                })

    root = validation_root(workdir) / "sa4"
    manifest = root / "manifold_field_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    report = {
        "package_version": VERSION,
        "protocol": "sa4",
        "manifold_rank": int(args.rank),
        "manifest": str(manifest),
        "selected_source_indices": selected,
        "geometry_count": len(selected),
        "field_magnitudes_au": fields,
        "task_count": len(records),
        "signed_fields_are_independent_from_zero": True,
        "candidate_state_count": 4,
    }
    write_json(root / "manifold_field_manifest_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
