#!/usr/bin/env bash
set -euo pipefail

# Fair, paper-style benchmark settings:
# - Training: unconstrained (fast)
# - Testing/benchmark: CPU-only + single-thread + batch=1 (end-to-end)
#
# Usage:
#   bash scripts/run_fair_cpu_single_thread.sh [dataset]
#
# Example:
#   bash scripts/run_fair_cpu_single_thread.sh dataset1

DATASET="${1:-dataset1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export MLFE_THREADS="1"
export MLFE_DEVICE="cpu"

export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export VECLIB_MAXIMUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"

echo "[FairMode] dataset=${DATASET} MLFE_THREADS=${MLFE_THREADS} MLFE_DEVICE=${MLFE_DEVICE}"

python3 scripts/run_full_experiment_cycle.py \
  --dataset "${DATASET}" \
  --threads 1 \
  --device cpu \
  --apply-to test
