#!/usr/bin/env bash
# 多尺度 token 化 Phase 2 消融（按 analysis.md §6 顺序），GPU 1 串行。
#   1) T0 / T2 多种子（seed 42 / 2026，配合已有 seed 17）——锚定筛选对照、确认 T2 F1 稳健性
#   2) T2 冻结随机滤波器控制（freeze_conv）——区分"可学习" vs "多尺度感受野"
#   3) T3(db4) 元数据位置编码（time_center + band embedding）——检验"展平正弦位置错位"假设
#   4) T4 同样元数据位置编码——看 DWT 是否同样回补
# 用法：
#   bash run_phase2.sh          # 正式：全量 50ep
#   SMOKE=1 bash run_phase2.sh  # 冒烟：每 run 1 epoch
set -uo pipefail

ROOT=/home/user_wzk/wzk/projects/MultToken
cd "$ROOT"

H5=/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5
MANIFEST=/home/user_wzk/wzk/projects/SwinE/manifest/rml2018a_manifest.csv
PY=~/miniconda3/envs/SFM/bin/python
SMOKE=${SMOKE:-0}

mkdir -p runs
TS=$(date +%Y%m%d_%H%M%S)
LOG="runs/phase2_${TS}.log"
: > "$LOG"

echo "=== MultToken phase2 start $(date) SMOKE=$SMOKE ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

launch() {
  local name="$1" cfg="$2" seed="$3" gpu="$4"
  local out="runs/$name"
  local mode="train"
  if [ "$SMOKE" = "1" ]; then
    out="runs/smoke_$name"
    mode="smoke"
  fi
  if [ -d "$out" ]; then
    echo "[skip] $out 已存在，不覆盖" | tee -a "$LOG"
    return 0
  fi
  echo "[start] $name cfg=$cfg seed=$seed gpu=$gpu" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m mtk.train \
    --config "mtk/configs/$cfg.json" \
    --h5 "$H5" --manifest "$MANIFEST" \
    --mode "$mode" --device cuda:0 --seed "$seed" --workers 2 \
    --output "$out" >> "$LOG" 2>&1
  echo "[done ] $name rc=$?" | tee -a "$LOG"
}

for spec in \
  "p2_t0_s42        t0_baseline         42" \
  "p2_t0_s2026      t0_baseline         2026" \
  "p2_t2_s42        t2_multiscale       42" \
  "p2_t2_s2026      t2_multiscale       2026" \
  "p2_t2_frozen_s17 t2_frozen           17" \
  "p2_t3_db4_meta_s17 t3_wpt_db4_j2_meta 17" \
  "p2_t4_meta_s17   t4_dwt_meta         17" ; do
  set -- $spec
  launch "$1" "$2" "$3" 1
done

wait
echo "=== ALL_DONE $(date) ===" | tee -a "$LOG"
