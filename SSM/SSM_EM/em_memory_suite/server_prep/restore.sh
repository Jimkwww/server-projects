#!/usr/bin/env bash
# 死机/重启后的一键恢复：重跑所有 runner（--retry-failed 跳过已完成 job，接着跑）。
# 用法： bash server_prep/restore.sh
set -e
cd "$(dirname "$0")/.."

# 1. 主实验三个分片（GPU1/GPU2 全天 + GPU0 夜班），各自恢复自己的 plan 断点续跑
for gpu in main3_cuda1 main3_cuda2; do
  plan="results/$gpu/plan/plan.json"
  if [ -f "$plan" ] && [ ! -f "results/$gpu/plan/RUNNING.lock" ]; then
    tmux new-session -d -s "em_$gpu" \
      "cd $(pwd) && conda run -n SFM python -m emmem.runner --plan $plan --retry-failed 2>&1 | tee results/$gpu/runner.log"
    echo "[restore] 已恢复 $gpu runner（tmux em_$gpu）"
  elif [ -f "results/$gpu/plan/RUNNING.lock" ]; then
    echo "[restore] $gpu 锁文件还在（可能有活跃进程），跳过；确认无进程后 rm 锁文件再跑"
  fi
done
# GPU0 夜班调度器（nohup 后台 + PID 文件，含断点续跑）
if ! pgrep -f 'bash server_prep/night_scheduler.sh' >/dev/null 2>&1; then
  nohup bash server_prep/night_scheduler.sh >> results/night_scheduler.log 2>&1 &
  echo "[restore] 已恢复 GPU0 夜班调度器（nohup 后台）"
fi

# 2. 历史基线（如果没跑完）
if [ "$(ls results/historical/*.json 2>/dev/null | wc -l)" -lt 24 ]; then
  tmux new-session -d -s em_hist \
    "cd $(pwd) && bash server_prep/run_historical.sh 2>&1 | tee results/historical/run.log"
  echo "[restore] 已恢复历史基线（tmux em_hist）"
fi

echo "[restore] 完成。查看：tmux ls"
