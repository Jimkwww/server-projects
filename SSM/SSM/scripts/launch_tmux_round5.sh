#!/usr/bin/env bash
# Launch 3 detached tmux sessions, one per GPU, for round-5 IQ training (21 runs).
# Balanced by per-run cost (R_corr has the O(L) sequential branch, slightly heavier).
cd /home/user_wzk/wzk/projects/SSM/SSM
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM
tmux new-session -d -s r5_gpu0 'bash scripts/run_round5_train.sh 0 R0:17 R0:42 R0:2026 C0:17 C0:42 C0:2026 Rb:17'
tmux new-session -d -s r5_gpu1 'bash scripts/run_round5_train.sh 1 Rb:42 Rb:2026 Cb:17 Cb:42 Cb:2026 Q_linear:17 Q_linear:42'
tmux new-session -d -s r5_gpu2 'bash scripts/run_round5_train.sh 2 Q_linear:2026 Q_corr:17 Q_corr:42 Q_corr:2026 R_corr:17 R_corr:42 R_corr:2026'
echo "launched: r5_gpu0, r5_gpu1, r5_gpu2"
tmux ls
