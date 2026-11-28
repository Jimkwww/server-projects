#!/bin/bash
# Launch 8 runs (4 models x 2 tasks), seed17, round-robin on 3 GPUs, run inside tmux.
cd /home/user_wzk/wzk/projects/SSM/SSM_EM/em_memory_suite/public_baselines_rml_radar_v1 || exit 1
EPOCHS=${1:-30}
mkdir -p runs
echo "[$(TZ=Asia/Shanghai date '+%m-%d %H:%M CST')] launching 8 runs, epochs=$EPOCHS"
JOBS=("rml q" "rml structured" "rml emind" "rml wavesfm" "radar q" "radar structured" "radar emind" "radar wavesfm")
i=0
for job in "${JOBS[@]}"; do
  set -- $job; task=$1; model=$2
  gpu=$((i % 3))
  ( conda run -n SFM python train.py --task "$task" --model "$model" --gpu "cuda:$gpu" --epochs "$EPOCHS" \
      > "runs/${task}_${model}.log" 2>&1 ) &
  echo "  [$gpu] $task/$model"
  i=$((i+1))
done
wait
echo "[$(TZ=Asia/Shanghai date '+%m-%d %H:%M CST')] all runs done, generating summary"
python summary.py
echo "[$(TZ=Asia/Shanghai date '+%m-%d %H:%M CST')] done"
