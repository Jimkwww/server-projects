#!/usr/bin/env bash
# GPU0 夜班调度器（主实验分担版）：CST 21:00-09:00 跑 results/main3_cuda0 的 plan，
# 白天 kill 释放 GPU0。用 PID 文件 + 进程组管理（不依赖 tmux 命令，避免嵌套 server 问题）。
# 挂 nohup 后台（不经 tmux），由 restore.sh 一并恢复。
ROOT=/home/user_wzk/wzk/projects/SSM/SSM_EM/em_memory_suite
cd "$ROOT" || exit 1
PLAN="results/main3_cuda0/plan/plan.json"
COMP="results/main3_cuda0/plan/completion.json"
PIDFILE="results/main3_cuda0/runner.pid"
LOGFILE="results/main3_cuda0/runner.log"
LOCK="results/main3_cuda0/plan/RUNNING.lock"

log() { echo "[$(TZ=Asia/Shanghai date '+%m-%d %H:%M CST')] $*"; }

is_night() {
  local h; h=$(TZ=Asia/Shanghai date +%H)
  [ "$h" -ge 21 ] || [ "$h" -lt 9 ]
}

gpu0_idle() {
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)
  [ -n "$used" ] && [ "$used" -lt 500 ]
}

all_done() {
  [ -f "$COMP" ] && grep -q '"all_done": true' "$COMP"
}

running() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

stop_runner() {
  if [ -f "$PIDFILE" ]; then
    local PID; PID=$(cat "$PIDFILE")
    # 杀整个进程组（runner + 正在跑的 job 子进程）
    kill -TERM -"$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
    sleep 2
    kill -KILL -"$PID" 2>/dev/null  # SIGKILL 兜底（runner 的 finally 不会执行）
    rm -f "$PIDFILE"
    rm -f "$LOCK"  # 硬杀时 runner 的 finally: lock.unlink() 不执行，这里手动清理
    log "已停止夜班任务 (PID $PID)，已完成 job 保留，下次续跑"
  fi
}

start_runner() {
  # 启动前清理残留 RUNNING.lock（若上次被硬杀，finally 没删掉）
  if [ -f "$LOCK" ]; then
    local LPID; LPID=$(cat "$LOCK" 2>/dev/null)
    if [ -z "$LPID" ] || ! kill -0 "$LPID" 2>/dev/null; then
      rm -f "$LOCK"
      log "清理残留 RUNNING.lock（旧 PID $LPID 已死）"
    fi
  fi
  # setsid 使 runner 自成进程组，便于整体 kill
  setsid conda run -n SFM python -m emmem.runner --plan "$PLAN" --retry-failed >> "$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  log "夜班启动 GPU0 主实验分片 (PID $!)"
}

while true; do
  if is_night; then
    if ! all_done; then
      if ! running && gpu0_idle; then
        start_runner
      fi
    fi
  else
    # 白天窗口：释放 GPU0
    if running; then
      stop_runner
    fi
  fi
  sleep 60
done
