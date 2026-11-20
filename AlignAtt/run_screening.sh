#!/usr/bin/env bash
# 阶段2 固定预算监督筛选：A0/A1/A2/A3/A4，seed=17，50 epoch（文档 §8 阶段2）。
# 每个实验一个 detached tmux；目录已存在则跳过。GPU1。
set -euo pipefail

ROOT=/home/user_wzk/wzk/projects/AlignAtt
cd "$ROOT"

SEED=17
# 换卡：ALIGN_GPU=2 ./run_screening.sh
for name in a0_global a1_zero_delay a2_magnitude a3_full a4_real_mlp; do
  ./run.sh "$name" "$SEED" 50 128
  sleep 3
done
echo "=== 筛选实验已全部入队（seed=$SEED，GPU=${ALIGN_GPU:-1}）==="
