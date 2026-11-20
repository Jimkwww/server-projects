#!/usr/bin/env bash
# 运行单个 AlignAtt 实验（detached tmux，断网安全，跑完可 tmux attach 查看）。
# 用法: ./run.sh <config名> <seed> [epochs=50] [batch=128] [输出目录]
#   例: ./run.sh a3_full 17           # 主候选 A3，seed=17，50 epoch
#   例: ./run.sh a3_full 17 1         # 1 epoch 冒烟（epochs<50 时输出目录自动加 _smoke）
set -euo pipefail

NAME="${1:?config name (a0_global/a1_zero_delay/a2_magnitude/a3_full/a4_real_mlp/a5_shuffle/a6_relation_only)}"
SEED="${2:?seed}"
EPOCHS="${3:-50}"
BATCH="${4:-128}"

ROOT=/home/user_wzk/wzk/projects/AlignAtt
SWIN=/home/user_wzk/wzk/projects/SwinE
H5=/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5
MANIFEST=$SWIN/manifest/rml2018a_manifest.csv
PY=~/miniconda3/envs/SFM/bin/python

if [ "$EPOCHS" -lt 50 ]; then
  OUT="${5:-$ROOT/runs/${NAME}_smoke}"
else
  OUT="${5:-$ROOT/runs/${NAME}_s${SEED}}"
fi
LOG="$OUT.log"
TMUX_SESS="align_${NAME}_s${SEED}_e${EPOCHS}"

if [ -d "$OUT" ]; then
  echo "[skip] $OUT 已存在，不重复启动" >&2
  exit 0
fi

cd "$ROOT"
# 环境变量直接内联进命令：tmux 会话不继承脚本的 export（tmux server 环境是独立的）。
GPU="${ALIGN_GPU:-1}"     # 用 ALIGN_GPU=2 ./run.sh ... 换卡
CMD="env PYTHONPATH=$SWIN/pretrain_v1:$ROOT CUDA_VISIBLE_DEVICES=$GPU $PY -m align.train_align --config configs/${NAME}.json --h5 $H5 --manifest $MANIFEST --output $OUT --device cuda:0 --seed $SEED --workers 2 --epochs $EPOCHS --batch-size $BATCH 2>&1 | tee $LOG"
tmux new-session -d -s "$TMUX_SESS" "$CMD"
echo "已启动 tmux 会话: $TMUX_SESS"
echo "日志: $LOG"
echo "查看: tmux attach -t $TMUX_SESS   (Ctrl-B D 脱离)"
