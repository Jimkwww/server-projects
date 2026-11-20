#!/bin/bash
# Launch the round-2 training supervisor inside a tmux session.
# Usage: launch_tmux_round2.sh <seed> [seed ...]
# The tmux server is daemonized and survives VSCode/SSH disconnects.
set -e
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM
cd /home/user_wzk/wzk/projects/SSM/SSM
mkdir -p logs runs/round2
exec python scripts/supervise_round2.py "$@" 2>&1 | tee -a logs/supervisor_round2.log
