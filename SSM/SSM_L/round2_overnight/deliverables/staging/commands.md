# Round-2 launch / resume commands

```bash
cd /home/user_wzk/wzk/projects/SSM/SSM_L
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh && conda activate SFM
# D (structure ablation, 3-GPU orchestrator, resume-safe)
python round2_overnight/scripts/orchestrate_d.py --phase all
# A (random frozen probes)
python round2_overnight/scripts/run_a.py
# B (corrected denoise eval)
python round2_overnight/scripts/run_b.py --device cuda:2
# C (low-label frozen probe)
python round2_overnight/scripts/run_c.py --device cpu
# E (radar transfer; cache on GPU, head/LR on CPU)
python round2_overnight/scripts/run_e.py --device cuda:0
# CFO eval after D finetunes
python round2_overnight/scripts/eval_finetune_cfo_r2.py --device cuda:0
# aggregation + curves
python round2_overnight/scripts/aggregate.py
```

All long jobs were launched under `tmux` (sessions ssml_r2_d / _a / _b / _c / _e).
Checkpoint SHA256 + split hashes are in reports/split_hashes.json and reports/checkpoint_sha256.json.
