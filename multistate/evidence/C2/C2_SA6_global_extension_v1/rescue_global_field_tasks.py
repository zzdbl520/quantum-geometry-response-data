#!/usr/bin/env python3
"""Rescue missing signed-field tasks by continuation from nearby PASS fields.

Independent geometry/axis groups run in parallel.  Inside one group, tasks run
serially and the closest completed field of the same sign is preferred, so a
newly rescued point becomes the next continuation anchor.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def excluded(record: dict) -> bool:
    value = record.get("exclusion")
    return isinstance(value, dict) and value.get("status") == "EXCLUDED"


def completed(record: dict) -> bool:
    path = Path(record["output_h5"])
    if not path.exists():
        return False
    try:
        import h5py
        with h5py.File(path, "r") as h5:
            return (
                bool(h5.attrs.get("converged", False))
                and str(h5.attrs.get("residual_group", "")) == "C2v"
            )
    except Exception:
        return False


def signed_field(record: dict) -> float:
    return float(record["field_sign"]) * float(record["field_magnitude_au"])


def best_anchor(target: dict, candidates: list[dict]) -> dict:
    target_field = signed_field(target)
    return min(
        candidates,
        key=lambda record: (
            int(record["field_sign"]) != int(target["field_sign"]),
            abs(signed_field(record) - target_field),
        ),
    )


def rescue_group(
    group: list[dict], manifest: Path, script: Path, log_root: Path,
    memory_mb: int,
) -> list[dict]:
    pending = [record for record in group if not completed(record)]
    results = []
    while pending:
        passed = [record for record in group if completed(record)]
        if not passed:
            results.extend({"task_id": int(record["task_id"]), "status": "NO_ANCHOR"} for record in pending)
            break
        choices = []
        for target in pending:
            anchor = best_anchor(target, passed)
            score = (
                int(anchor["field_sign"]) != int(target["field_sign"]),
                abs(signed_field(anchor) - signed_field(target)),
            )
            choices.append((score, int(target["task_id"]), target, anchor))
        _, task, target, anchor = min(choices, key=lambda item: (item[0], item[1]))
        log = log_root / f"task_{task:05d}.stdout"
        command = [
            sys.executable, "-u", str(script), "run-task",
            "--manifest", str(manifest), "--index", str(task),
            "--memory-mb", str(memory_mb), "--rescue",
            "--restart-h5", str(anchor["output_h5"]),
        ]
        if Path(target["output_h5"]).exists():
            command.append("--force")
        environment = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment.setdefault(key, "1")
        with log.open("w") as handle:
            result = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, env=environment
            )
        status = "PASS" if result.returncode == 0 and completed(target) else "FAIL"
        results.append({
            "task_id": task, "status": status,
            "R_angstrom": target["R_angstrom"], "axis": target["axis"],
            "field_au": signed_field(target),
            "restart_task_id": int(anchor["task_id"]),
            "restart_field_au": signed_field(anchor), "log": str(log),
        })
        pending.remove(target)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=30000)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    all_records = load_records(manifest)
    excluded_records = [record for record in all_records if excluded(record)]
    records = [record for record in all_records if not excluded(record)]
    missing = [record for record in records if not completed(record)]
    groups = {}
    for record in records:
        key = (float(record["R_angstrom"]), record["axis"])
        groups.setdefault(key, []).append(record)
    affected = [groups[key] for key in groups if any(not completed(record) for record in groups[key])]
    log_root = manifest.parent / "rescue_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "c2_sa6_global_field.py"
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                rescue_group, group, manifest, script, log_root, args.memory_mb
            )
            for group in affected
        ]
        for future in concurrent.futures.as_completed(futures):
            for row in future.result():
                results.append(row)
                print(json.dumps(row), flush=True)
    results.sort(key=lambda row: row["task_id"])
    summary = {
        "manifest_task_count": len(all_records),
        "active_task_count": len(records),
        "excluded_task_count": len(excluded_records),
        "initial_missing_count": len(missing),
        "pass_count": sum(row["status"] == "PASS" for row in results),
        "failure_count": sum(row["status"] != "PASS" for row in results),
        "results": results,
    }
    (manifest.parent / "rescue_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 2 if summary["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
