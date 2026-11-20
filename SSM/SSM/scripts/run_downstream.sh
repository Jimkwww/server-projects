#!/bin/bash
# Downstream evaluation: perturbation robustness (cuda:0) + benchmark (cuda:1),
# run in parallel inside a tmux session so it survives VSCode/SSH disconnects.
set -u
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM
cd /home/user_wzk/wzk/projects/SSM/SSM
mkdir -p logs
rm -f logs/downstream.DONE

# Perturbation evaluation (CFO + phase grid), saves predictions.
python scripts/evaluate_campaign.py --runs runs/clean --manifest configs/data.json \
  --frequencies=-0.005,-0.002,-0.001,0.001,0.002,0.005 --split val \
  --device cuda:0 --workers 4 --batch-size 128 --save-predictions \
  > logs/perturbation.log 2>&1 &
PERT=$!

# Benchmark: latency/throughput/memory/length scaling for the 5 seed17 checkpoints.
for v in G S0 S1 S2 S2_concat; do
  echo "=== benchmark $v ==="
  python -m ssm_exp.benchmark --run runs/clean/${v}_seed17 --device cuda:1 \
    >> logs/benchmark.log 2>&1
done
echo "BENCHMARK_DONE" >> logs/benchmark.log &
BENCH=$!

wait $PERT
PERT_RC=$?
wait $BENCH
BENCH_RC=$?
echo "perturbation exit=$PERT_RC benchmark exit=$BENCH_RC" >> logs/run_downstream.log
touch logs/downstream.DONE
echo "DOWNSTREAM_ALL_DONE (pert=$PERT_RC bench=$BENCH_RC)"
