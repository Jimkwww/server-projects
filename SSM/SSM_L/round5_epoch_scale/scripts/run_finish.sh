#!/usr/bin/env bash
# Round5 finish: re-dispatch the 72 no-CFO finetune jobs (fixed python -m invocation),
# then regenerate analyze/figures/hashes. Run in tmux.
ROOT=/home/user_wzk/wzk/projects/SSM/SSM_L
R5=$ROOT/round5_epoch_scale
cd "$ROOT" || exit 1
rm -f "$R5/runs/downstream/FINISHED_V2.flag"

echo "[$(date '+%F %T')] dispatch no-CFO finetune (72 jobs)"
python "$R5/scripts/dispatch.py" --jobs "$R5/scripts/finetune_jobs.txt" --gpus 3 \
  --logdir "$R5/logs/finetune" --retries 1 || echo "finetune had failures (see logs)"

echo "[$(date '+%F %T')] analyze + report"
python "$R5/scripts/analyze_r5.py" || exit 1

echo "[$(date '+%F %T')] figures"
python "$R5/scripts/make_figures.py" || echo "figures failed (non-fatal)"

echo "[$(date '+%F %T')] hashes + checkpoint sha256"
python "$R5/scripts/make_hashes.py" || echo "hashes failed (non-fatal)"

echo "[$(date '+%F %T')] ALL DONE"
touch "$R5/runs/downstream/FINISHED_V2.flag"
