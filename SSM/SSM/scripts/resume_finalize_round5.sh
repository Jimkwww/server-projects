#!/usr/bin/env bash
# Round-5 resume: the 21 new runs were already evaluated (eval_cases.jsonl). This appends
# the 3 S1_corr baselines, runs the benchmark, and re-aggregates. Run detached in tmux.
set -euo pipefail
cd /home/user_wzk/wzk/projects/SSM/SSM
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM
RPT=reports/round5_iq

# 1) append 3 S1_corr baseline evaluations (pre-round-5 checkpoints lack aux_mean/aux_std)
python scripts/eval_round5.py \
  --runs runs/round3_aug/S1_corr_seed17,runs/round3_aug/S1_corr_seed42,runs/round3_aug/S1_corr_seed2026 \
  --manifest configs/data_raw.json --device cuda:0 \
  --pred-dir predictions_round5_iq --out "$RPT/eval_cases.jsonl"

# 2) benchmark (one seed per variant: architecture-identical across seeds)
python scripts/bench_round5.py \
  --runs runs/round5_iq/R0_seed17,runs/round5_iq/C0_seed17,runs/round5_iq/Rb_seed17,runs/round5_iq/Cb_seed17,runs/round5_iq/Q_linear_seed17,runs/round5_iq/Q_corr_seed17,runs/round5_iq/R_corr_seed17,runs/round3_aug/S1_corr_seed17 \
  --device cuda:0 --out "$RPT/benchmark.json"

# 3) aggregate (reads eval_cases.jsonl + reused round4 eval/bench for S1/G/G_corr)
python scripts/aggregate_round5.py

echo "RESUME DONE ($(date '+%H:%M:%S'))"
