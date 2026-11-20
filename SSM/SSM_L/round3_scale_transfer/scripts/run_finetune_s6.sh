#!/bin/bash
# Section 6 downstream part 2: RML finetune (50 ep, CFO aug) for the 18 F1/F2 encoders.
# One seed per GPU, 2 corpora x 3 models sequential. Run AFTER pretrain_scale.
# Usage: run_finetune_s6.sh <seed> <gpu_index>
set -e
seed=$1
gpu=$2
ROOT=/home/user_wzk/wzk/projects/SSM/SSM_L
R3=$ROOT/round3_scale_transfer
cd $ROOT
for corpus in f1 f2; do
  for model in G G_Qlearn Q; do
    PRET=$R3/runs/pretrain_scale/${corpus}/${model}_seed${seed}/encoder_final.pt
    OUT=$R3/runs/finetune_s6/${corpus}_${model}_seed${seed}
    if [ -f "$OUT/best.pt" ]; then echo "=== skip ${corpus}_${model}_seed${seed} ==="; continue; fi
    echo "=== $(date +%H:%M:%S) finetune ${corpus}_${model}_seed${seed} on cuda:${gpu} ==="
    python -m round2_overnight.downstream --mode finetune \
      --config $ROOT/configs/finetune.json --manifest $ROOT/configs/data.json \
      --backbone $model --pretrained $PRET \
      --out $OUT --seed $seed --device cuda:${gpu} --workers 4
  done
done
echo "=== $(date +%H:%M:%S) seed $seed finetunes all done ==="
