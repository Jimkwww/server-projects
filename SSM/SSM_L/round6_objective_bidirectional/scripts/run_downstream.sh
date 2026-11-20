#!/usr/bin/env bash
# Round6 master: wait for 21 pretrain runs -> frozen readout -> no-CFO finetune -> analyze.
# Launch once, in tmux (after pretrain finished, or instead of it if you run pretrain
# separately). Idempotent (dispatch skips jobs whose marker exists). 3 GPUs.
ROOT=/home/user_wzk/wzk/projects/SSM/SSM_L
R6=$ROOT/round6_objective_bidirectional
cd "$ROOT"

echo "[$(date '+%F %T')] waiting for 21 pretrain DONE.json ..."
while true; do
  N=$(ls "$R6"/runs/pretrain/f1/{G_denoise,G_joint,Q_denoise,Q_joint,Q_bi_mae,Q_bi_denoise,Q_bi_joint}_seed{17,42,2026}/DONE.json 2>/dev/null | wc -l)
  if [ "$N" -ge 21 ]; then
    echo "[$(date '+%F %T')] all 21 pretrain runs done"
    break
  fi
  sleep 60
done

echo "[$(date '+%F %T')] regenerating job lists"
python "$R6/scripts/make_jobs.py" || exit 1

echo "[$(date '+%F %T')] dispatch frozen readout (42 jobs) on 3 GPUs"
python "$R6/scripts/dispatch.py" --jobs "$R6/scripts/readout_jobs.txt" --gpus 3 \
  --logdir "$R6/logs/readout" --retries 1 || echo "readout had failures (see logs)"

echo "[$(date '+%F %T')] dispatch no-CFO finetune (42 jobs) on 3 GPUs"
python "$R6/scripts/dispatch.py" --jobs "$R6/scripts/finetune_jobs.txt" --gpus 3 \
  --logdir "$R6/logs/finetune" --retries 1 || echo "finetune had failures (see logs)"

echo "[$(date '+%F %T')] analyze + report"
python "$R6/scripts/analyze_r6.py" || exit 1

echo "[$(date '+%F %T')] figures"
python "$R6/scripts/make_figures.py" || echo "figures failed (non-fatal)"

echo "[$(date '+%F %T')] encoder latency"
python "$R6/scripts/latency_r6.py" --device cuda:1 || echo "latency failed (non-fatal)"

echo "[$(date '+%F %T')] hashes + checkpoint sha256"
python "$R6/scripts/make_hashes.py" || echo "hashes failed (non-fatal)"

echo "[$(date '+%F %T')] ALL DOWNSTREAM DONE"
