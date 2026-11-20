#!/usr/bin/env bash
# MixT unequal-window phase-1: G / U-F / U-C / E32-S, seed=2024, 50ep (or smoke).
# Usage: bash run_phase1.sh [smoke]   (default = full 50ep)
# Constraint: physical GPU 2 only (CUDA_VISIBLE_DEVICES=2 -> program sees cuda:0).
set -euo pipefail

ROOT=/home/user_wzk/wzk/projects/MixT
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=2
export EMFM_H5=/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5
export EMFM_MANIFEST="$ROOT/manifest/rml2018a_manifest.csv"
PY=~/miniconda3/envs/SFM/bin/python

MODE="${1:-full}"            # full | smoke
EPOCHS=()
[ "$MODE" = "smoke" ] && EPOCHS=(--epochs 1)

RUNS="$ROOT/experiments/unequal_window_cyclic/runs"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/experiments/unequal_window_cyclic/phase1_${MODE}_${TS}.log"

echo "=== MixT phase-1 $MODE start $(date) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

run() {
  local name="$1"; shift
  local out="$RUNS/${name}_50_s2024"
  [ "$MODE" = "smoke" ] && out="$RUNS/${name}_smoke_s2024"
  echo "=== [$name] start $(date) ===" | tee -a "$LOG"
  "$PY" -m "$@" --h5 "$EMFM_H5" --manifest "$EMFM_MANIFEST" \
    --device cuda:0 --seed 2024 --workers 2 "${EPOCHS[@]}" --output "$out" 2>&1 | tee -a "$LOG"
  echo "=== [$name] done $(date) ===" | tee -a "$LOG"
}

run G       emfm.train --mode train        --config emfm/configs/global_matched_50ep.json
run U-F     emfm.train_unequal_window      --config emfm/configs/unequal_window_fixed_50ep.json
run U-C     emfm.train_unequal_window      --config emfm/configs/unequal_window_cyclic_50ep.json
run E32-S   emfm.train_singlescale_window  --config emfm/configs/equal_window32_shift_50ep.json

echo "=== ALL_DONE $(date) ===" | tee -a "$LOG"
