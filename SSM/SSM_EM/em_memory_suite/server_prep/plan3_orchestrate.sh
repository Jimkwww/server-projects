#!/usr/bin/env bash
# Phase-3 orchestration: wait for 27 scale pretrains, then launch 3-GPU AMC downstream.
# Runs INSIDE tmux so it survives Claude-session teardown.
cd /home/user_wzk/wzk/projects/SSM/SSM_EM/em_memory_suite || exit 1
echo "[$(TZ=Asia/Shanghai date '+%m-%d %H:%M CST')] 等待 27 个规模预训练 DONE..."
while true; do
  N=$(ls results/plan3_scale/*/DONE.json 2>/dev/null | wc -l)
  if [ "$N" -ge 27 ]; then
    echo "[$(TZ=Asia/Shanghai date '+%m-%d %H:%M CST')] 27 预训练全部完成，启动下游"
    break
  fi
  sleep 300
done

for s in 0 1 2; do
  tmux new-session -d -s "p3d$s" "cd /home/user_wzk/wzk/projects/SSM/SSM_EM/em_memory_suite && conda run -n SFM python server_prep/plan3_scale_downstream.py --gpu cuda:$s --out results/plan3_scale_downstream --shard $s --nshards 3 2>&1 | tee results/plan3_downstream_shard$s.log"
done
echo "[$(TZ=Asia/Shanghai date '+%m-%d %H:%M CST')] 下游已启动（p3d0/1/2）"
