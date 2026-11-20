#!/usr/bin/env bash
# Round-5 Review Section 四: Q_corr_frozen training worker (3 seeds, sequential on one GPU).
# Usage: run_round5_review_train.sh <gpu_index>
cd /home/user_wzk/wzk/projects/SSM/SSM
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM
GPU=$1
OUT=runs/round5_review
mkdir -p "$OUT"
for s in 17 42 2026; do
  name=Q_corr_frozen_seed${s}
  {
    echo "# cmd: python -m ssm_exp.engine --config configs/resolved_round5_review.json --manifest configs/data_raw.json --out $OUT/$name --variant Q_corr_frozen --seed $s --device cuda:$GPU --workers 4"
    echo "# started: $(date '+%Y-%m-%d %H:%M:%S')"
    python -m ssm_exp.engine --config configs/resolved_round5_review.json --manifest configs/data_raw.json \
      --out "$OUT/$name" --variant Q_corr_frozen --seed "$s" --device "cuda:$GPU" --workers 4 2>&1
    echo "# finished: $(date '+%Y-%m-%d %H:%M:%S')"
  } | tee "$OUT/$name.log"
done
