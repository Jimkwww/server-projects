#!/usr/bin/env bash
# 控制 B（单尺度大核卷积 + 16 维瓶颈）正式训练：3 种子（17/42/2026），50 epoch。
# 先做 1-epoch 冒烟，再跑正式；固定输出目录，已存在则跳过（不覆盖）。
# 用法：
#   bash run_control_b.sh          # 冒烟 + 3 正式
#   SMOKE_ONLY=1 bash run_control_b.sh  # 只冒烟
set -uo pipefail

ROOT=/home/user_wzk/wzk/projects/MultToken
cd "$ROOT"

H5=/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5
MANIFEST=/home/user_wzk/wzk/projects/SwinE/manifest/rml2018a_manifest.csv
PY=~/miniconda3/envs/SFM/bin/python
SMOKE_ONLY=${SMOKE_ONLY:-0}

mkdir -p runs
TS=$(date +%Y%m%d_%H%M%S)
LOG="runs/control_b_${TS}.log"
: > "$LOG"

echo "=== control B start $(date) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

launch() {
  local name="$1" seed="$2" mode="$3"
  local out="runs/$name"
  if [ -d "$out" ]; then
    echo "[skip] $out 已存在，不覆盖" | tee -a "$LOG"
    return 0
  fi
  echo "[start] $name seed=$seed mode=$mode" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=1 "$PY" -m mtk.train \
    --config mtk/configs/b_single_conv.json \
    --h5 "$H5" --manifest "$MANIFEST" \
    --mode "$mode" --device cuda:0 --seed "$seed" --workers 2 \
    --output "$out" >> "$LOG" 2>&1
  echo "[done ] $name rc=$?" | tee -a "$LOG"
}

# 1-epoch 冒烟（不训练正式权重，仅验证流水线）
launch "smoke_b_s17" 17 smoke

if [ "$SMOKE_ONLY" = "1" ]; then
  echo "=== SMOKE_ONLY done $(date) ===" | tee -a "$LOG"
  exit 0
fi

for spec in \
  "b_s17    17" \
  "b_s42    42" \
  "b_s2026  2026" ; do
  set -- $spec
  launch "$1" "$2" train
done

wait
echo "=== ALL_DONE $(date) ===" | tee -a "$LOG"
