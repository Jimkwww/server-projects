#!/bin/bash
# Section 6 pretrain orchestration: one seed per GPU, 2 corpora x 3 models sequential.
# Usage: run_pretrain_scale.sh <seed> <gpu_index>
set -e
seed=$1
gpu=$2
ROOT=/home/user_wzk/wzk/projects/SSM/SSM_L
R3=$ROOT/round3_scale_transfer
cd $ROOT
for corpus in f1 f2; do
  CFG=$R3/configs/manifest_${corpus}.json
  for model in G G_Qlearn Q; do
    OUT=$R3/runs/pretrain_scale/${corpus}/${model}_seed${seed}
    echo "=== $(date +%H:%M:%S) ${corpus}/${model}_seed${seed} on cuda:${gpu} ==="
    python $R3/scripts/pretrain_scale.py \
      --config $ROOT/configs/pretrain.json \
      --manifest $CFG \
      --backbone $model --task mae \
      --out $OUT --runs-root $R3/runs/pretrain_scale/${corpus} \
      --corpus ${corpus^^} --max-steps 7800 --seed $seed --device cuda:${gpu} --workers 4
  done
done
echo "=== $(date +%H:%M:%S) seed $seed all done ==="
