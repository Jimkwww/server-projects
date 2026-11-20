#!/usr/bin/env bash
# Round-4 structure training worker: run assigned variant:seed specs sequentially on one GPU.
# Usage: run_round4_train.sh <gpu_index> <variant:seed> [<variant:seed> ...]
cd /home/user_wzk/wzk/projects/SSM/SSM
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM
GPU=$1; shift
OUT=runs/round4_structure
mkdir -p "$OUT"
for spec in "$@"; do
  v=${spec%%:*}; s=${spec##*:}
  name=${v}_seed${s}
  {
    echo "# cmd: python -m ssm_exp.engine --config configs/resolved_round4.json --manifest configs/data_raw.json --out $OUT/$name --variant $v --seed $s --device cuda:$GPU --workers 4"
    echo "# started: $(date '+%Y-%m-%d %H:%M:%S')"
    python -m ssm_exp.engine --config configs/resolved_round4.json --manifest configs/data_raw.json \
      --out "$OUT/$name" --variant "$v" --seed "$s" --device "cuda:$GPU" --workers 4 2>&1
    echo "# finished: $(date '+%Y-%m-%d %H:%M:%S')"
  } | tee "$OUT/$name.log"
done
