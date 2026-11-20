#!/usr/bin/env bash
# Launch 3 detached tmux sessions, one per GPU, for round-4 structure training (15 runs).
# Assignment (balanced by estimated per-run cost): GPU0=A64, GPU1=A128,
# GPU2=B_front+D_complex. Each session runs its variants sequentially.
cd /home/user_wzk/wzk/projects/SSM/SSM
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM
tmux new-session -d -s r4_gpu0 'bash scripts/run_round4_train.sh 0 A64:17 A64:42 A64:2026'
tmux new-session -d -s r4_gpu1 'bash scripts/run_round4_train.sh 1 A128:17 A128:42 A128:2026'
tmux new-session -d -s r4_gpu2 'bash scripts/run_round4_train.sh 2 B_front:17 B_front:42 B_front:2026 D_complex:17 D_complex:42 D_complex:2026'
echo "launched: r4_gpu0, r4_gpu1, r4_gpu2"
tmux ls
