#!/usr/bin/env bash
# Round6 extension: Q_bi denoise + joint (2 tasks x 3 seeds = 6 runs), 3 GPUs in parallel.
# Launch inside tmux:  bash round6_objective_bidirectional/scripts/run_pretrain_qbi_extra.sh
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

# (backbone task seed gpu) -> 6 runs, 2 per GPU
RUNS=(
  "Q_bi denoise 17 0"
  "Q_bi denoise 42 0"
  "Q_bi denoise 2026 1"
  "Q_bi joint 17 1"
  "Q_bi joint 42 2"
  "Q_bi joint 2026 2"
)

PIDS=()
for ENTRY in "${RUNS[@]}"; do
  set -- $ENTRY
  BACKBONE=$1; TASK=$2; SEED=$3; GPU=$4
  NAME=${BACKBONE}_${TASK}_seed${SEED}
  OUT=$R6/runs/pretrain/f1/$NAME
  LOG=$LOG_DIR/pretrain_${NAME}.log
  if [ -f "$OUT/encoder_final.pt" ]; then
    echo "[$(date '+%F %T')] skip  $NAME (final exists)"
    continue
  fi
  echo "[$(date '+%F %T')] start $NAME on cuda:$GPU"
  python round6_objective_bidirectional/scripts/pretrain_r6.py \
      --config "$CFG" --manifest "$MANIFEST" --backbone "$BACKBONE" --task "$TASK" \
      --out "$OUT" --shared-init-dir "$R6_SHARED" --round5-shared-init-dir "$R5_SHARED" \
      --corpus f1 --epochs 50 --seed "$SEED" --device "cuda:$GPU" --workers 4 \
      >> "$LOG" 2>&1 &
  PIDS+=($!)
done

FAILED=0
for P in "${PIDS[@]}"; do
  if ! wait "$P"; then
    FAILED=$((FAILED+1))
  fi
done
echo "[$(date '+%F %T')] Q_bi EXTRA PRETRAIN DONE (launched=${#PIDS[@]} failures=$FAILED)"
