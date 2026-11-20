#!/usr/bin/env bash
# 多尺度 token 化筛选（Phase 1）：8 个 run，3 卡并行，每卡顺序执行。
#   - T0 seed=2024：与 Global 基线 0.3869 交叉验证
#   - T0/T1/T2/T3(haar)/T3(db4)/T4/T5 均 seed=17：同种子单因素筛选
# 用法：
#   bash run_screening.sh          # 正式：全量 50ep
#   SMOKE=1 bash run_screening.sh  # 冒烟：每 run 1 epoch，输出到 runs/smoke_*
# 约束：目录已存在则跳过（不覆盖）；用 detached tmux 跑，关终端不断。
set -uo pipefail

ROOT=/home/user_wzk/wzk/projects/MultToken
cd "$ROOT"

H5=/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5
MANIFEST=/home/user_wzk/wzk/projects/SwinE/manifest/rml2018a_manifest.csv
PY=~/miniconda3/envs/SFM/bin/python
SMOKE=${SMOKE:-0}

mkdir -p runs
TS=$(date +%Y%m%d_%H%M%S)
LOG="runs/screening_${TS}.log"
: > "$LOG"

echo "=== MultToken screening start $(date) SMOKE=$SMOKE ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

# launch <name> <config> <seed> <gpu>
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

# 只跑物理 GPU 1（顺序）。GPU 0 被 user_cx 占用；GPU 2 被 MixT（tmux mixt_G）占用。
# 若 GPU 2 空闲，可把下面第二组注释打开并改成 2，恢复两卡并行。
for spec in \
  "sup_t0_s2024      t0_baseline      2024" \
  "sup_t0_s17        t0_baseline      17" \
  "sup_t1_s17        t1_dct           17" \
  "sup_t2_s17        t2_multiscale    17" \
  "sup_t3_haar_j2_s17 t3_wpt_haar_j2  17" \
  "sup_t3_db4_j2_s17 t3_wpt_db4_j2    17" \
  "sup_t4_s17        t4_dwt           17" \
  "sup_t5_s17        t5_stft          17" ; do
  set -- $spec
  launch "$1" "$2" "$3" 1
done

wait
echo "=== ALL_DONE $(date) ===" | tee -a "$LOG"
