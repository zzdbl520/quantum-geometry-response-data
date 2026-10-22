#!/usr/bin/env python3
"""Create local-window signed-field tasks anchored to SS zero checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ho2_core import BASIS, load_grid, parse_floats, parse_indices, write_json
from protocol_core import (
    DEFAULT_FIELDS,
    DEFAULT_FIELD_INDICES,
    VERSION,
    complete_h5,
    field_path,
    validation_root,
    zero_path,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="HO2_qgeom_protocol_validation_work"
    )
    parser.add_argument("--indices", default=DEFAULT_FIELD_INDICES)
    parser.add_argument(
        "--fields", default=",".join(f"{value:g}" for value in DEFAULT_FIELDS)
    )
    parser.add_argument("--basis", default=BASIS)
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    rows = load_grid(workdir)
    if not rows:
        raise SystemExit("Empty geometry grid; run prepare_protocol.py")
    available = {int(row["source_index"]): row for row in rows}
    size = max(available) + 1
    selected_indices = parse_indices(args.indices, size)
    missing_grid = sorted(set(selected_indices) - set(available))
    if missing_grid:
        raise SystemExit(f"Indices not present in validation grid: {missing_grid}")
    fields = parse_floats(args.fields)
    missing_zero = [
        index
        for index in selected_indices
        if not complete_h5(zero_path(workdir, "ss", index), "ss")
    ]
    if missing_zero:
        raise SystemExit(
            "SS zero transport is incomplete at: "
            + ",".join(map(str, missing_zero))
        )

    records = []
    for source_index in selected_indices:
        row = available[source_index]
        for magnitude in fields:
            for sign in (-1, 1):
                signed = sign * magnitude
                records.append({
                    "task_id": len(records),
                    "protocol": "ss",
                    "source_index": source_index,
                    "grid_position": int(row["grid_position"]),
                    "signed_field_au": signed,
                    "field_magnitude_au": magnitude,
                    "basis": args.basis,
                    "zero_path": str(zero_path(workdir, "ss", source_index)),
                    "output_path": str(field_path(workdir, source_index, signed)),
                })

    root = validation_root(workdir) / "ss"
    manifest = root / "field_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    report = {
        "package_version": VERSION,
        "protocol": "ss",
        "manifest": str(manifest),
        "selected_source_indices": selected_indices,
        "geometry_count": len(selected_indices),
        "field_magnitudes_au": fields,
        "task_count": len(records),
        "signed_fields_are_independent_from_zero": True,
    }
    write_json(root / "field_manifest_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
