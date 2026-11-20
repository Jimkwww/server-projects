#!/usr/bin/env bash
# G_Q_corr / G_Q_corr_frozen training campaign (2 variants x 3 seeds = 6 runs).
# One detached tmux session; 3 GPUs each run one seed's learnable + frozen pair.
set -euo pipefail

cd /home/user_wzk/wzk/projects/SSM/SSM
CONFIG=configs/resolved_global_qcorr.json
MANIFEST=configs/data_raw.json
OUT=runs/global_qcorr
mkdir -p "$OUT" logs

run_one() {
  local variant="$1" seed="$2" gpu="$3"
  local name="${variant}_seed${seed}"
  echo "[$(date +%T)] start $name on cuda:$gpu" | tee -a "logs/global_qcorr.launch.log"
  python -m ssm_exp.engine \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --out "$OUT/$name" --variant "$variant" --seed "$seed" \
    --device "cuda:$gpu" --workers 4 --augment-max 0.002 \
    > "$OUT/$name.log" 2>&1
  echo "[$(date +%T)] done $name" | tee -a "logs/global_qcorr.launch.log"
}

# 3 parallel GPU workers, each running its seed pair sequentially.
(
  run_one G_Q_corr 17 0
  run_one G_Q_corr_frozen 17 0
) &
(
  run_one G_Q_corr 42 1
  run_one G_Q_corr_frozen 42 1
) &
(
  run_one G_Q_corr 2026 2
  run_one G_Q_corr_frozen 2026 2
) &
wait

echo "ALL RUNS DONE" | tee -a "logs/global_qcorr.launch.log"
touch logs/global_qcorr.DONE
