#!/usr/bin/env bash
# 控制 B 排队：等物理 GPU 1 空闲后跑 run_control_b.sh（冒烟 + 3 正式种子）。
set -uo pipefail

ROOT=/home/user_wzk/wzk/projects/MultToken
cd "$ROOT"

TS=$(date +%Y%m%d_%H%M%S)
QLOG="runs/queue_control_b_${TS}.log"
echo "=== queue start $(date) ===" | tee "$QLOG"

while true; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
  if [ -z "$mem" ]; then
    echo "[warn] 读不到 GPU1 显存，10s 后重试" | tee -a "$QLOG"; sleep 10; continue
  fi
  if [ "$mem" -lt 500 ]; then
    echo "[go] GPU1 空闲（${mem} MiB），开始跑控制 B" | tee -a "$QLOG"; break
  fi
  echo "[wait] GPU1 忙（${mem} MiB），60s 后再查" | tee -a "$QLOG"
  sleep 60
done

bash "$ROOT/run_control_b.sh" 2>&1 | tee -a "$QLOG"
echo "=== queue done $(date) ===" | tee -a "$QLOG"
