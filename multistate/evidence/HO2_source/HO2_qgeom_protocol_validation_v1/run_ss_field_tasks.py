#!/usr/bin/env python3
"""Run selected SS signed-field tasks with process-level parallelism."""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ho2_core import parse_indices, write_json
from protocol_core import complete_h5, validation_root
from run_ss_field_task import read_manifest, run_task_record


def safe_run(payload):
    record, workdir, memory_mb, threads, force = payload
    try:
        return run_task_record(
            record, Path(workdir), memory_mb, threads, force
        )
    except Exception as exc:
        return {
            "task_id": int(record["task_id"]),
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--task-ids",
        help="Comma-separated task IDs or inclusive slices, e.g. 8 or 0:15",
    )
    parser.add_argument("--workdir")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=30000)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    workdir = (
        Path(args.workdir).resolve()
        if args.workdir
        else manifest.parents[2]
    )
    records = read_manifest(manifest)
    selected_ids = (
        parse_indices(args.task_ids, len(records))
        if args.task_ids
        else list(range(len(records)))
    )
    selected = [records[i] for i in selected_ids]
    payloads = [
        (record, str(workdir), args.memory_mb, args.threads_per_worker, args.force)
        for record in selected
    ]
    results = []
    if args.workers <= 1:
        for payload in payloads:
            result = safe_run(payload)
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(safe_run, payload) for payload in payloads]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)

    failures = [item for item in results if item["status"] == "FAIL"]
    inventory = {
        "requested_task_count": len(selected),
        "pass_count": sum(item["status"] == "PASS" for item in results),
        "skip_count": sum(item["status"] == "SKIP" for item in results),
        "failure_count": len(failures),
        "failures": failures,
        "total_manifest_tasks": len(records),
        "total_completed_tasks": sum(
            complete_h5(Path(record["output_path"]), "ss")
            for record in records
        ),
    }
    path = validation_root(workdir) / "ss" / "field_inventory.json"
    write_json(path, inventory)
    print(json.dumps(inventory, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
