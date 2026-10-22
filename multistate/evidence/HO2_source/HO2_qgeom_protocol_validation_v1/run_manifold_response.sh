#!/usr/bin/env bash
set -euo pipefail

package=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workdir=${1:-HO2_qgeom_protocol_validation_work}
workers=${WORKERS:-4}
threads=${THREADS_PER_WORKER:-4}
memory_mb=${MEMORY_MB:-30000}
indices=${MANIFOLD_INDICES:-2:11,15:18}
fields=${MANIFOLD_FIELDS:-2.5e-4,5.0e-4,7.5e-4,1.0e-3}
rank=${MANIFOLD_RANK:-2}

python "$package/prepare_manifold_field_tasks.py" \
  --workdir "$workdir" \
  --indices "$indices" \
  --fields "$fields" \
  --rank "$rank"

manifest="$workdir/protocol_validation/sa4/manifold_field_manifest.jsonl"
python -u "$package/run_manifold_field_tasks.py" \
  --manifest "$manifest" \
  --workdir "$workdir" \
  --workers "$workers" \
  --threads-per-worker "$threads" \
  --memory-mb "$memory_mb"

python "$package/analyze_manifold_response.py" \
  --manifest "$manifest" \
  --workdir "$workdir" \
  --memory-mb "$memory_mb"

python "$package/plot_manifold_response.py" --workdir "$workdir"
python "$package/plot_ho2_publication.py" --workdir "$workdir"
