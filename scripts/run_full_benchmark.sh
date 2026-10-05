#!/usr/bin/env bash
# Full MOT17-FRCNN benchmark: SORT baseline then the occlusion-aware tracker.
# Run sequentially so both jobs do not contend for the GPU.
set -u

cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=./venv/bin/python

mkdir -p logs results

run() {
    local tracker=$1
    echo "=== ${tracker} start $(date -Is) ==="
    ${PY} -m mot.pipeline \
        --sequences all \
        --tracker "${tracker}" \
        --out-dir "results/full_${tracker}" \
        2>&1 | tee "logs/full_${tracker}.log"
    echo "=== ${tracker} exit=${PIPESTATUS[0]} $(date -Is) ==="
}

run sort
run occlusion

echo "ALL BENCHMARKS COMPLETE $(date -Is)"