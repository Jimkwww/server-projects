#!/usr/bin/env bash
# Round-5 finalize: wait for all 21 round5 training runs, then unified order-B eval
# (21 new + 3 S1_corr baselines), benchmark, and aggregation. Run detached in tmux.
set -euo pipefail
cd /home/user_wzk/wzk/projects/SSM/SSM
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM
OUT=runs/round5_iq
RPT=reports/round5_iq
PRED=predictions_round5_iq
mkdir -p "$RPT" "$PRED"

VARIANTS=(R0 C0 Rb Cb Q_linear Q_corr R_corr)
for v in "${VARIANTS[@]}"; do
  for s in 17 42 2026; do
    while [ ! -f "$OUT/${v}_seed$s/DONE.json" ]; do sleep 30; done
  done
done
echo "all 21 round5 runs done ($(date '+%H:%M:%S'))"

# build comma-separated run list: 21 round5 + 3 S1_corr baselines
RUNS=""
for v in "${VARIANTS[@]}"; do
  for s in 17 42 2026; do
    RUNS="${RUNS:+$RUNS,}$OUT/${v}_seed$s"
  done
done
for s in 17 42 2026; do
  RUNS="${RUNS:+$RUNS,}runs/round3_aug/S1_corr_seed$s"
done

rm -f "$RPT/eval_cases.jsonl"
python scripts/eval_round5.py \
  --runs "$RUNS" --manifest configs/data_raw.json --device cuda:0 \
  --pred-dir "$PRED" --out "$RPT/eval_cases.jsonl"

BENCH=""
for v in "${VARIANTS[@]}"; do
  BENCH="${BENCH:+$BENCH,}$OUT/${v}_seed17"
done
BENCH="${BENCH},runs/round3_aug/S1_corr_seed17"
python scripts/bench_round5.py --runs "$BENCH" --device cuda:0 --out "$RPT/benchmark.json"

python scripts/aggregate_round5.py
echo "FINALIZE DONE ($(date '+%H:%M:%S'))"
