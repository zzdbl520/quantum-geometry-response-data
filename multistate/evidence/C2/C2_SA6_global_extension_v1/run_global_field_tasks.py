#!/usr/bin/env python3
"""Run or retry independent global signed-field manifest tasks locally."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


def records(path: Path) -> list[dict]:
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


def run_one(
    script: Path, manifest: Path, record: dict, log_root: Path,
    memory_mb: int, force_retries: bool,
) -> tuple[int, int, str]:
    task = int(record["task_id"])
    if completed(record):
        return task, 0, "SKIP"
    log = log_root / f"task_{task:05d}.stdout"
    command = [
        sys.executable, "-u", str(script), "run-task",
        "--manifest", str(manifest), "--index", str(task),
        "--memory-mb", str(memory_mb),
    ]
    if force_retries and Path(record["output_h5"]).exists():
        command.append("--force")
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment.setdefault(key, "1")
    with log.open("w") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, env=environment
        )
    return task, result.returncode, str(log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=30000)
    parser.add_argument("--force-retries", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    workdir = manifest.parent
    log_root = workdir / "task_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "c2_sa6_global_field.py"
    all_records = records(manifest)
    excluded_records = [record for record in all_records if excluded(record)]
    todo = [record for record in all_records if not excluded(record)]
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_one, script, manifest, record, log_root,
                args.memory_mb, args.force_retries,
            )
            for record in todo
        ]
        for future in concurrent.futures.as_completed(futures):
            task, code, detail = future.result()
            if code:
                failures.append(task)
                print(f"FAIL task {task}: {detail}", flush=True)
            elif detail != "SKIP":
                print(f"PASS task {task}", flush=True)
    summary = {
        "manifest_task_count": len(all_records),
        "active_task_count": len(todo),
        "excluded_task_count": len(excluded_records),
        "failure_count": len(failures),
        "failure_indices": sorted(failures), "logs": str(log_root),
    }
    print(json.dumps(summary, indent=2))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
