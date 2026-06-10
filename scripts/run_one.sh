#!/usr/bin/env bash
# Generic single-run launcher with a structured, download-friendly save layout.
#
# Usage:
#   bash scripts/run_one.sh <TAG> <ENV_NAME> <SEED> [extra flags...]
#
# Results land in:
#   exp/<TAG>/<ENV_NAME>/seed<SEED>/run_<timestamp>/{flags.json, train.csv, eval.csv}
#   exp/<TAG>/<ENV_NAME>/seed<SEED>/stdout_<timestamp>.log
#
# wandb runs offline (local files only, under ./wandb); CSVs are the primary logs.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=$1
ENV=$2
SEED=$3
shift 3

STAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="exp/${TAG}/${ENV}/seed${SEED}"
mkdir -p "${SAVE_DIR}"

echo "[run_one] tag=${TAG} env=${ENV} seed=${SEED} extra: $*" | tee "${SAVE_DIR}/stdout_${STAMP}.log"

WANDB_MODE=offline python main.py \
  --env_name="${ENV}" \
  --seed="${SEED}" \
  --save_dir="${SAVE_DIR}" \
  --flat_save_dir=True \
  --run_name="run_${STAMP}" \
  --run_group="${TAG}" \
  --wandb_mode=offline \
  "$@" 2>&1 | tee -a "${SAVE_DIR}/stdout_${STAMP}.log"
