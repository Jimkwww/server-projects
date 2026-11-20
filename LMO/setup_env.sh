#!/bin/bash
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n LMO python=3.13 -y
conda activate LMO
pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
echo "=== installing requirements ==="
pip install -r /home/user_wzk/wzk/projects/LMO/requirements.txt
echo "=== ENV DONE ==="
python -c "import torch, transformers, fla, triton; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available()); print(\"transformers\", transformers.__version__); print(\"fla\", fla.__version__)"
