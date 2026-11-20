#!/usr/bin/env bash
# Round-3 review: unified order-B extended CFO eval on all 24 existing checkpoints.
# Read-only extension; outputs to predictions_round3_review/ and reports/round3_review/.
set -euo pipefail
cd /home/user_wzk/wzk/projects/SSM/SSM
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM

RUNS=""
for d_v in "runs/clean G" "runs/clean S1" "runs/round2 S1_corr" "runs/round3 G_corr" \
          "runs/round3_aug G" "runs/round3_aug G_corr" "runs/round3_aug S1" "runs/round3_aug S1_corr"; do
  set -- $d_v
  d=$1; v=$2
  for s in 17 42 2026; do
    if [ -f "$d/${v}_seed${s}/best.pt" ]; then
      RUNS="${RUNS}${RUNS:+,}$d/${v}_seed${s}"
    fi
  done
done

FREQS="0.0001,0.00025,0.0005,0.001,0.002,0.003,0.004,0.005,0.0075,0.01,0.02"

python scripts/eval_extended_cfo.py \
  --runs "$RUNS" \
  --manifest configs/data_raw.json \
  --frequencies "$FREQS" \
  --pred-dir predictions_round3_review \
  --out reports/round3_review/extended_cases.jsonl
