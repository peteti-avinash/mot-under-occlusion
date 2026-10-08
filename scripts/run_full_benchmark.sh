#!/usr/bin/env bash
# Full MOT17-FRCNN benchmark: SORT baseline then the occlusion-aware tracker.
# Run sequentially so both jobs do not contend for the GPU.
set -uo pipefail

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
    local status=${PIPESTATUS[0]}
    echo "=== ${tracker} exit=${status} $(date -Is) ==="
    return "${status}"
}

run sort || exit $?
run occlusion || exit $?

echo "ALL BENCHMARKS COMPLETE $(date -Is)"
