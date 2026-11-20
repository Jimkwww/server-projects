#!/usr/bin/env bash
# Q_corr physical-gate round training campaign (6 variants x 3 seeds = 18 runs).
# Each GPU runs one seed's 6 variants sequentially (seed -> GPU mapping fixed).
# One detached tmux session; see logs/qcorr_gates.launch.log for progress.
set -euo pipefail

cd /home/user_wzk/wzk/projects/SSM/SSM
CONFIG=configs/resolved_qcorr_gates.json
MANIFEST=configs/data_raw.json
OUT=runs/qcorr_gates
mkdir -p "$OUT" logs

VARIANTS=(Q_corr_gate_c Q_corr_gate_d Q_corr_gate_cd Q_corr_token_c Q_corr_token_d Q_corr_token_cd)

run_one() {
  local variant="$1" seed="$2" gpu="$3"
  local name="${variant}_seed${seed}"
  echo "[$(date +%T)] start $name on cuda:$gpu" | tee -a "logs/qcorr_gates.launch.log"
  python -m ssm_exp.engine \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --out "$OUT/$name" --variant "$variant" --seed "$seed" \
    --device "cuda:$gpu" --workers 4 --augment-max 0.002 \
    > "$OUT/$name.log" 2>&1
  echo "[$(date +%T)] done $name" | tee -a "logs/qcorr_gates.launch.log"
}

# 3 parallel GPU workers, each running its seed's 6 variants sequentially.
for v in "${VARIANTS[@]}"; do run_one "$v" 17 0; done &
for v in "${VARIANTS[@]}"; do run_one "$v" 42 1; done &
for v in "${VARIANTS[@]}"; do run_one "$v" 2026 2; done &
wait

echo "ALL RUNS DONE" | tee -a "logs/qcorr_gates.launch.log"
touch logs/qcorr_gates.DONE
