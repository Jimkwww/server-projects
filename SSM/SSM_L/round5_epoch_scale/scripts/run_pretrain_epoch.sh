#!/usr/bin/env bash
# Round5 fixed-epoch pretraining: one seed's 6 runs (f1/f2 x G/G_Qlearn/Q) on one GPU.
# Usage: bash run_pretrain_epoch.sh <seed> <gpu_index>
SEED=$1
GPU=$2
ROOT=/home/user_wzk/wzk/projects/SSM/SSM_L
CFG=$ROOT/configs/pretrain.json
cd "$ROOT"

FAILED=0
for CORPUS in f1 f2; do
  for BACKBONE in G G_Qlearn Q; do
    OUT=$ROOT/round5_epoch_scale/runs/pretrain/$CORPUS/${BACKBONE}_seed$SEED
    SHARED=$ROOT/round5_epoch_scale/runs/pretrain/$CORPUS/shared_init
    MANIFEST=$ROOT/round3_scale_transfer/configs/manifest_$CORPUS.json
    LOG=$ROOT/round5_epoch_scale/logs/pretrain_${CORPUS}_${BACKBONE}_seed${SEED}.log
    echo "[$(date '+%F %T')] start $CORPUS/$BACKBONE seed$SEED on cuda:$GPU"
    if python round5_epoch_scale/scripts/pretrain_epoch.py \
        --config "$CFG" --manifest "$MANIFEST" --backbone "$BACKBONE" \
        --out "$OUT" --shared-init-dir "$SHARED" --corpus "$CORPUS" \
        --epochs 50 --seed "$SEED" --device "cuda:$GPU" --workers 4 >> "$LOG" 2>&1; then
      echo "[$(date '+%F %T')] done  $CORPUS/$BACKBONE seed$SEED"
    else
      echo "[$(date '+%F %T')] FAIL  $CORPUS/$BACKBONE seed$SEED (exit=$?)"
      FAILED=$((FAILED+1))
    fi
  done
done
echo "[$(date '+%F %T')] ALL DONE seed$SEED (failures=$FAILED)"
