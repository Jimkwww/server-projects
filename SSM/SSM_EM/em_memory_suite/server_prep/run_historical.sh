#!/usr/bin/env bash
# 历史基线：G/Q × AMC/radar × probe/finetune，GPU0（空闲卡，不占主实验 GPU1/2）。
cd /home/user_wzk/wzk/projects/SSM/SSM_EM/em_memory_suite
PY="conda run -n SFM python"
for BB in G Q; do
  for TASK in "modulation data/rml2018_clsdev/manifest.json" "radar_waveform data/radchar_tiny/manifest.json"; do
    set -- $TASK
    TN=$1; MAN=$2
    $PY server_prep/historical_baseline.py --manifest "$MAN" --task "$TN" \
      --out results/historical --backbone "$BB" --seeds 17,42,2026 \
      --modes probe,finetune --epochs 50 --device cuda:0
  done
done
echo "[historical] ALL DONE"
