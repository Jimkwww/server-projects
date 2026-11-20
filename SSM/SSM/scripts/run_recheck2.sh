#!/bin/bash
# Round-2 part 2: frozen-protocol perturbation eval for the 3 new models (9 runs).
# Same grid and code path as part-1 recheck (order A/B), separate output files.
set -e
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM
cd /home/user_wzk/wzk/projects/SSM/SSM
mkdir -p predictions_round2_new reports/round2
python scripts/recheck_order.py \
  --runs runs/round2 \
  --manifest configs/data_raw.json \
  --frequencies=0.0001,0.00025,0.0005,0.001,0.002,0.005,-0.0001,-0.00025,-0.0005,-0.001,-0.002,-0.005 \
  --phases=0.7853981634,1.5707963268,3.1415926536,4.7123889804 \
  --device cuda:0 \
  --workers 4 \
  --pred-dir predictions_round2_new \
  --out reports/round2/newmodel_cases.jsonl
touch logs/recheck2.DONE
