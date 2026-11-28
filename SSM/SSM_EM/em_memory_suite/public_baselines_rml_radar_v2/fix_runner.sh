#!/bin/bash
# Fix pass: (1) re-measure efficiency per-model in independent processes, (2) rerun Q seed42/2026.
set -u
cd /home/user_wzk/wzk/projects/SSM/SSM_EM/em_memory_suite/public_baselines_rml_radar_v2 || exit 1
PY=/home/user_wzk/miniconda3/envs/SFM/bin/python
echo "[$(TZ=Asia/Shanghai date '+%H:%M:%S CST')] efficiency re-measure (serial, cuda:0)"
mkdir -p efficiency
for task in rml radar; do
  for model in q structured emind wavesfm; do
    $PY -u efficiency.py --task $task --model $model --gpu cuda:0 --out efficiency/${task}_${model}.json 2>&1 | grep -vE 'W1127|Warning|warn|deprecated|use_return'
  done
done
$PY merge_efficiency.py
echo "[$(TZ=Asia/Shanghai date '+%H:%M:%S CST')] Q seed42/2026 rerun (4 jobs, 3 GPUs)"
declare -a QJOBS=( "rml q 42" "radar q 42" "rml q 2026" "radar q 2026" )
i=0
for job in "${QJOBS[@]}"; do
  set -- $job; task=$1; model=$2; seed=$3
  gpu=$((i % 3))
  ( $PY -u train.py --task "$task" --model "$model" --gpu "cuda:$gpu" --seed "$seed" --lr 3e-4 --epochs 30 --out runs \
      > logs/fix_q_s${seed}_${task}.log 2>&1 ) &
  echo "  [cuda:$gpu] Q s$seed $task"
  i=$((i+1))
done
wait
echo "[$(TZ=Asia/Shanghai date '+%H:%M:%S CST')] fix pass done"
