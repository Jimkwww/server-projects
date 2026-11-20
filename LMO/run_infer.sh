#!/bin/bash
# LEMMO Q16 interference 复现推理脚本 (GPU2, A100 sm_80)
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate LMO
cd /home/user_wzk/wzk/projects/LMO

# A100 = sm_80, 覆盖 compat.py 默认的 sm_90
export TILELANG_DEFAULT_TARGET='{"kind":"cuda","arch":"sm_80"}'
export FLA_TILELANG=1

SIGNAL="${1:?usage: run_infer.sh <signal.npy> [question] [max_new_tokens]}"
QUESTION="${2:-这段信号中存在哪种干扰？}"
MAXNEW="${3:-64}"

python infer.py \
  --model LEMMO_Recognition \
  --config configs/LEMMO_Recognition.local.json \
  --signal "$SIGNAL" \
  --sample-rate 20000000 \
  --task interference \
  --question "$QUESTION" \
  --device cuda:2 \
  --max-new-tokens "$MAXNEW"
