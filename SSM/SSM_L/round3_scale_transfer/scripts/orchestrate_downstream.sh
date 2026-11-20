#!/bin/bash
# Section 6 downstream orchestrator (runs after the 18 pretrain_scale runs finish).
# 1. wait for all 18 pretrain DONE.json
# 2. launch 3 finetune tmux sessions (seed 17/42/2026 -> GPU 0/1/2)
# 3. wait for all 18 finetune best.pt
# 4. section6_probe.py (RML + radar unified LR) on cuda:0
# 5. eval_finetune_s6.py (clean/CFO) on cuda:0
# 6. section7_acceptance.py
set -u
ROOT=/home/user_wzk/wzk/projects/SSM/SSM_L
R3=$ROOT/round3_scale_transfer
LOG=$R3/logs/orchestrate_downstream.log
mkdir -p $R3/logs
exec > >(tee -a $LOG) 2>&1
echo "=== orchestrator start $(date +%F_%T) ==="

MODELS=(G G_Qlearn Q)
CORPORA=(f1 f2)
SEEDS=(17 42 2026)
GPUS=(0 1 2)

count_done() {
  local n=0
  for c in f1 f2; do for m in G G_Qlearn Q; do for s in 17 42 2026; do
    [ -f "$R3/runs/pretrain_scale/$c/${m}_seed$s/DONE.json" ] && n=$((n+1))
  done; done; done
  echo $n
}

count_ft() {
  local n=0
  for c in f1 f2; do for m in G G_Qlearn Q; do for s in 17 42 2026; do
    [ -f "$R3/runs/finetune_s6/${c}_${m}_seed$s/best.pt" ] && n=$((n+1))
  done; done; done
  echo $n
}

echo "=== [1/6] waiting for 18 pretrain DONE.json ==="
for i in $(seq 1 240); do
  n=$(count_done)
  echo "$(date +%T) pretrain done: $n/18"
  [ "$n" -ge 18 ] && break
  sleep 30
done
if [ "$(count_done)" -lt 18 ]; then
  echo "TIMEOUT waiting for pretrains (only $(count_done)/18 done)"
  exit 1
fi
echo "=== all 18 pretrains done at $(date +%F_%T) ==="

echo "=== [2/6] launching 3 finetune tmux sessions ==="
for i in 0 1 2; do
  seed=${SEEDS[$i]}; gpu=${GPUS[$i]}
  sname="s6f_$seed"
  tmux kill-session -t $sname 2>/dev/null
  setsid tmux new-session -d -s $sname -c $ROOT \
    "source /home/user_wzk/miniconda3/etc/profile.d/conda.sh && conda activate SFM && bash $R3/scripts/run_finetune_s6.sh $seed $gpu" </dev/null
  echo "launched tmux $sname (seed $seed, cuda:$gpu)"
done

echo "=== [3/6] waiting for 18 finetune best.pt ==="
for i in $(seq 1 360); do
  n=$(count_ft)
  echo "$(date +%T) finetune done: $n/18"
  [ "$n" -ge 18 ] && break
  sleep 30
done
if [ "$(count_ft)" -lt 18 ]; then
  echo "TIMEOUT waiting for finetunes (only $(count_ft)/18 done). Missing:"
  for c in f1 f2; do for m in G G_Qlearn Q; do for s in 17 42 2026; do
    [ -f "$R3/runs/finetune_s6/${c}_${m}_seed$s/best.pt" ] || echo "  ${c}_${m}_seed$s"
  done; done; done
  exit 1
fi
echo "=== all 18 finetunes done at $(date +%F_%T) ==="

echo "=== [4/6] section6_probe.py (unified LR: RML + radar) ==="
python $R3/scripts/section6_probe.py --device cuda:0 || echo "PROBE FAILED"

echo "=== [5/6] eval_finetune_s6.py (clean/CFO) ==="
python $R3/scripts/eval_finetune_s6.py --device cuda:0 || echo "EVAL FAILED"

echo "=== [6/6] section7_acceptance.py ==="
python $R3/scripts/section7_acceptance.py || echo "ACCEPTANCE FAILED"

echo "=== orchestrator DONE at $(date +%F_%T) ==="
