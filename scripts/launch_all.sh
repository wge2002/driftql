#!/usr/bin/env bash
# Dispatch every command in a commands file across GPUs.
#
# Usage:
#   GPUS="0,1,2,3" JOBS_PER_GPU=1 bash scripts/launch_all.sh scripts/phase1_commands.txt
#
# Lines starting with '#' and blank lines are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

CMD_FILE=${1:-scripts/phase1_commands.txt}
GPUS=${GPUS:-"0"}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
SLOTS=$((NGPU * JOBS_PER_GPU))

mapfile -t CMDS < <(grep -vE '^\s*(#|$)' "$CMD_FILE")
echo "[launch_all] dispatching ${#CMDS[@]} runs on GPUs [$GPUS], ${JOBS_PER_GPU} job(s)/GPU"

i=0
for cmd in "${CMDS[@]}"; do
  gpu=${GPU_ARR[$((i % NGPU))]}
  echo "[launch_all] GPU ${gpu}: ${cmd}"
  CUDA_VISIBLE_DEVICES=${gpu} bash -c "${cmd}" &
  i=$((i + 1))
  sleep 5
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 30; done
done
wait
echo "[launch_all] all runs finished."
