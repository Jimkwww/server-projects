# Round6 运行命令
## 预训练（21 run）
```bash
# 串行（canonical，GPU1）
tmux new-session -d -s r6_pretrain \
  "bash round6_objective_bidirectional/scripts/run_pretrain.sh"
# Q_bi denoise/joint 补充 6 run，3 卡并行
tmux new-session -d -s r6_pretrain_qbi_extra \
  "bash round6_objective_bidirectional/scripts/run_pretrain_qbi_extra.sh"
```
## 下游（readout + finetune，3 卡）
```bash
tmux new-session -d -s r6_downstream \
  "bash round6_objective_bidirectional/scripts/run_downstream.sh"
```
## 单条示例
```bash
# 预训练 Q_bi-denoise seed17
python round6_objective_bidirectional/scripts/pretrain_r6.py \
  --config round6_objective_bidirectional/configs/pretrain.json \
  --manifest round3_scale_transfer/configs/manifest_f1.json \
  --backbone Q_bi --task denoise --out round6_objective_bidirectional/runs/pretrain/f1/Q_bi_denoise_seed17 \
  --shared-init-dir round6_objective_bidirectional/runs/pretrain/f1/shared_init \
  --round5-shared-init-dir round5_epoch_scale/runs/pretrain/f1/shared_init \
  --corpus f1 --epochs 50 --seed 17 --device cuda:0
```
