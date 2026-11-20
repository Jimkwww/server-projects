#!/usr/bin/env bash
# Round6 pretraining: 21 runs (7 groups x 3 seeds) on GPU1 ONLY (cuda:1), serially.
# Launch in tmux:  bash round6_objective_bidirectional/scripts/run_pretrain.sh
# Idempotent: a run whose encoder_final.pt already exists is skipped.
ROOT=/home/user_wzk/wzk/projects/SSM/SSM_L
R6=$ROOT/round6_objective_bidirectional
CFG=$R6/configs/pretrain.json
MANIFEST=$ROOT/round3_scale_transfer/configs/manifest_f1.json
R6_SHARED=$R6/runs/pretrain/f1/shared_init
R5_SHARED=$ROOT/round5_epoch_scale/runs/pretrain/f1/shared_init
LOG_DIR=$R6/logs
mkdir -p "$LOG_DIR"
cd "$ROOT"

GPU=1  # cuda:1

FAILED=0
TOTAL=0
for SPEC in "G denoise" "G joint" "Q denoise" "Q joint" "Q_bi mae" "Q_bi denoise" "Q_bi joint"; do
  set -- $SPEC
  BACKBONE=$1
  TASK=$2
  for SEED in 17 42 2026; do
    NAME=${BACKBONE}_${TASK}_seed${SEED}
    OUT=$R6/runs/pretrain/f1/$NAME
    LOG=$LOG_DIR/pretrain_${NAME}.log
    if [ -f "$OUT/encoder_final.pt" ]; then
      echo "[$(date '+%F %T')] skip  $NAME (final exists)"
      continue
    fi
    TOTAL=$((TOTAL+1))
    echo "[$(date '+%F %T')] start $NAME on cuda:$GPU"
    if python round6_objective_bidirectional/scripts/pretrain_r6.py \
        --config "$CFG" --manifest "$MANIFEST" --backbone "$BACKBONE" --task "$TASK" \
        --out "$OUT" --shared-init-dir "$R6_SHARED" --round5-shared-init-dir "$R5_SHARED" \
        --corpus f1 --epochs 50 --seed "$SEED" --device "cuda:$GPU" --workers 4 \
        >> "$LOG" 2>&1; then
      echo "[$(date '+%F %T')] done  $NAME"
    else
      echo "[$(date '+%F %T')] FAIL  $NAME (exit=$?)"
      FAILED=$((FAILED+1))
    fi
  done
done
echo "[$(date '+%F %T')] ALL PRETRAIN DONE (ran=$TOTAL failures=$FAILED)"
