#!/usr/bin/env bash
# 在 GPU 1 上排队：等物理 GPU 1 空闲（显存 < 500 MiB，空闲约 1 MiB）后再串行跑 Phase 2。
# 用法：bash queue_phase2_gpu1.sh    （建议放 tmux/detach 里跑）
set -uo pipefail

ROOT=/home/user_wzk/wzk/projects/MultToken
cd "$ROOT"

TS=$(date +%Y%m%d_%H%M%S)
QUEUE_LOG="runs/queue_phase2_${TS}.log"
echo "=== queue start $(date) ===" | tee "$QUEUE_LOG"

while true; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
  if [ -z "$mem" ]; then
    echo "[warn] 读不到 GPU1 显存，10s 后重试" | tee -a "$QUEUE_LOG"
    sleep 10
    continue
  fi
  if [ "$mem" -lt 500 ]; then
    echo "[go] GPU1 空闲（${mem} MiB），开始跑 Phase 2" | tee -a "$QUEUE_LOG"
    break
  fi
  echo "[wait] GPU1 忙（${mem} MiB），60s 后再查" | tee -a "$QUEUE_LOG"
  sleep 60
done

bash "$ROOT/run_phase2.sh" 2>&1 | tee -a "$QUEUE_LOG"
echo "=== queue done $(date) ===" | tee -a "$QUEUE_LOG"
