#!/usr/bin/env bash
# Round5 master: wait for 18 pretrain runs -> frozen readout -> no-CFO finetune -> analyze.
# Launch once, in tmux. Idempotent (dispatch skips jobs whose marker exists).
ROOT=/home/user_wzk/wzk/projects/SSM/SSM_L
R5=$ROOT/round5_epoch_scale
cd "$ROOT"

echo "[$(date '+%F %T')] waiting for 18 pretrain DONE.json ..."
while true; do
  N=$(ls "$R5"/runs/pretrain/f{1,2}/{G,G_Qlearn,Q}_seed{17,42,2026}/DONE.json 2>/dev/null | wc -l)
  if [ "$N" -ge 18 ]; then
    echo "[$(date '+%F %T')] all 18 pretrain runs done"
    break
  fi
  sleep 60
done

echo "[$(date '+%F %T')] regenerating job lists"
python "$R5/scripts/make_jobs.py" || exit 1

echo "[$(date '+%F %T')] dispatch frozen readout (180 jobs)"
python "$R5/scripts/dispatch.py" --jobs "$R5/scripts/readout_jobs.txt" --gpus 3 \
  --logdir "$R5/logs/readout" --retries 1 || echo "readout had failures (see logs)"

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
