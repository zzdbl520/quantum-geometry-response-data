#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 PILOT_WORKDIR VALIDATION_WORKDIR" >&2
  exit 2
fi

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pilot_workdir="$1"
validation_workdir="$2"
workers="${WORKERS:-4}"
threads="${THREADS_PER_WORKER:-4}"
memory_mb="${MEMORY_MB:-30000}"

python "$package_dir/prepare_protocol.py" \
  --pilot-workdir "$pilot_workdir" \
  --workdir "$validation_workdir" \
  --indices 0:20

python -u "$package_dir/run_protocol_zero.py" \
  --workdir "$validation_workdir" \
  --protocol both \
  --threads "$threads" \
  --memory-mb "$memory_mb"

python "$package_dir/prepare_ss_field_tasks.py" \
  --workdir "$validation_workdir" \
  --indices "4:6,9:11,15:18" \
  --fields "2.5e-4,5.0e-4,7.5e-4,1.0e-3"

python -u "$package_dir/run_ss_field_tasks.py" \
  --manifest "$validation_workdir/protocol_validation/ss/field_manifest.jsonl" \
  --workdir "$validation_workdir" \
  --workers "$workers" \
  --threads-per-worker "$threads" \
  --memory-mb "$memory_mb"

python "$package_dir/analyze_protocol_validation.py" \
  --workdir "$validation_workdir" \
  --memory-mb "$memory_mb"

python "$package_dir/plot_protocol_validation.py" \
  --workdir "$validation_workdir"
