#!/usr/bin/env bash
# Final frozen test evaluation: 12 augmented checkpoints (G/G_corr/S1/S1_corr x 3 seeds)
# on the independent test split, order B. No re-training, no test-based selection.
set -euo pipefail
cd /home/user_wzk/wzk/projects/SSM/SSM
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM

RUNS=""
for v in G G_corr S1 S1_corr; do
  for s in 17 42 2026; do
    RUNS="${RUNS}${RUNS:+,}runs/round3_aug/${v}_seed${s}"
  done
done

python scripts/eval_extended_cfo.py \
  --runs "$RUNS" \
  --manifest configs/data_raw.json \
  --split test \
  --frequencies "0.001,0.002,0.005,0.01,0.02" \
  --pred-dir predictions_test \
  --out reports/final_test/test_cases.jsonl
