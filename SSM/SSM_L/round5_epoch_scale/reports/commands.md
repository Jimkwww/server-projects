# Round5 运行与恢复命令

## 1. 预训练（18 run，3 GPU）

单 run（脚本会跳过已完成的 `encoder_final.pt`）：

```bash
cd /home/user_wzk/wzk/projects/SSM/SSM_L
python round5_epoch_scale/scripts/pretrain_epoch.py \
  --config configs/pretrain.json \
  --manifest round3_scale_transfer/configs/manifest_{f1,f2}.json \
  --backbone {G,G_Qlearn,Q} \
  --out round5_epoch_scale/runs/pretrain/{f1,f2}/{backbone}_seed{seed} \
  --shared-init-dir round5_epoch_scale/runs/pretrain/{f1,f2}/shared_init \
  --corpus {f1,f2} --epochs 50 --seed {17,42,2026} --device cuda:{0,1,2} --workers 4
```

批量（1 种子 = 6 run 串行，绑定 1 张卡）：

```bash
bash round5_epoch_scale/scripts/run_pretrain_epoch.sh <seed> <gpu_index>   # 已分别在 tmux 三窗口跑 17/0、42/1、2026/2
```

崩溃恢复：加 `--resume`（从 `last.pt` 恢复 RNG/优化器/调度器/aux）。

## 2. 下游（等待预训练 18×DONE.json 后自动）

```bash
bash round5_epoch_scale/scripts/run_downstream.sh   # 已在 tmux 窗口 downstream 跑
```

内部四步：
1. `make_jobs.py` → `readout_jobs.txt`（180）/ `finetune_jobs.txt`（72）
2. `dispatch.py --jobs readout_jobs.txt --gpus 3` → 冻结读出
3. `dispatch.py --jobs finetune_jobs.txt --gpus 3` → 无 CFO 微调
4. `analyze_r5.py` + `make_figures.py` + `make_hashes.py` → 报告

所有 job 幂等：`done: <marker>` 存在即跳过。单个 job 失败可单独重跑对应命令行
（命令在 `readout_jobs.txt` / `finetune_jobs.txt`，`@@GPU@@` 替换为 0/1/2）。

## 3. 报告重生成（不改动任何权重/结果，纯读）

```bash
python round5_epoch_scale/scripts/analyze_r5.py     # 汇总表 + conclusion + PPT + raw json
python round5_epoch_scale/scripts/make_figures.py   # 训练曲线 + 读出趋势图
python round5_epoch_scale/scripts/make_hashes.py    # config/data + checkpoint sha256
```

## 4. 打包（无权重）

```bash
python round5_epoch_scale/scripts/package_r5.py     # 产出 round5_epoch_scale_package.tar.gz
```

## 5. 监控

```bash
tmux attach -t r5_pretrain        # 4 窗口：seed17 / seed42 / seed2026 / downstream
ls round5_epoch_scale/runs/pretrain/*/*/DONE.json | wc -l   # 预训练完成数（目标 18）
tail -f round5_epoch_scale/logs/pretrain_f1_G_seed17.log    # 单 run 进度
```
