#!/bin/bash
set -u
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM
cd /home/user_wzk/wzk/projects/SSM/SSM
mkdir -p reports/round2 predictions_round2 logs
rm -f logs/recheck.DONE
python scripts/recheck_order.py \
  --runs runs/clean --manifest configs/data_raw.json \
  --frequencies=0.0001,0.00025,0.0005,0.001,0.002,0.005,-0.0001,-0.00025,-0.0005,-0.001,-0.002,-0.005 \
  --phases=0.7853981634,1.5707963268,3.1415926536,4.7123889804 \
  --device cuda:0 --workers 4 --batch-size 128 \
  --pred-dir predictions_round2 \
  --out reports/round2/recheck_cases.jsonl \
  > logs/recheck.log 2>&1
echo "recheck exit=$?" >> logs/recheck.log
touch logs/recheck.DONE
