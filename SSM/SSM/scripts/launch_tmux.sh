#!/bin/bash
# Launch the detached campaign supervisor inside a tmux session.
# The tmux server is daemonized and survives VSCode/SSH disconnects.
set -e
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM
cd /home/user_wzk/wzk/projects/SSM/SSM
mkdir -p logs
exec python scripts/supervise_clean.py 2>&1 | tee -a logs/supervisor.log
