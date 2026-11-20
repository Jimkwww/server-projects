# SSM_L 第一轮 · 启动与恢复命令

## 0. 环境

```bash
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM            # torch 2.14.0+cu126, 复数 fp32/complex64
cd /home/user_wzk/wzk/projects/SSM/SSM_L
```

3×A100 已就绪；每张卡一个进程。长任务必须放 tmux，防 vscode 关闭导致中断。

## 1. 一键全量编排（推荐）

在 tmux 里跑编排器，按依赖分阶段、3 卡并行、自动断点续跑/跳过已完成、seed-17 阶段先验收：

```bash
# 创建持久 tmux 会话（setsid 保证脱离当前 shell）
setsid tmux new-session -d -s ssml_round1 -c /home/user_wzk/wzk/projects/SSM/SSM_L \
  'source /home/user_wzk/miniconda3/etc/profile.d/conda.sh && conda activate SFM && \
   python scripts/orchestrate.py --all 2>&1 | tee logs/orchestrate_all.log' < /dev/null

# 查看进度
tmux attach -t ssml_round1        # 退出会话 Ctrl-b d
tail -f logs/orchestrate_all.log
```

恢复：编排器对已完成 run（`DONE.json` + 预训练还要求 `encoder_final.pt`）自动跳过，未完成/中断的 run 加 `--resume` 续跑；seed-17 阶段校验失败会停止，不会继续烧剩余预算。

分阶段 / 演练：

```bash
python scripts/orchestrate.py --dry-run              # 只打印将执行的命令
python scripts/orchestrate.py --phase pretrain_s17   # 只跑 seed-17 预训练
python scripts/orchestrate.py --phase pretrain_rest
python scripts/orchestrate.py --phase probe_rest
python scripts/orchestrate.py --phase finetune_rest
```

## 2. 单 run 手动命令（等价于编排器内部命令）

### 预训练（50 epoch）

```bash
python -m pretrain.engine \
  --config configs/pretrain.json --manifest configs/data.json \
  --backbone G --task mae \
  --out runs/pretrain/G_mae_seed17 --runs-root runs/pretrain \
  --seed 17 --device cuda:0 --workers 4
# backbone ∈ {G,Q}; task ∈ {mae,denoise}; seed ∈ {17,42,2026}; device cuda:{0,1,2}
# 中断后续跑：末尾加 --resume
```

### 冻结线性探测（30 epoch，主终点）

```bash
python -m pretrain.downstream --mode probe \
  --config configs/probe.json --manifest configs/data.json \
  --backbone G --pretrained runs/pretrain/G_mae_seed17/encoder_final.pt \
  --out runs/probe/G_mae_seed17 --seed 17 --device cuda:0 --workers 4
# 续跑：末尾加 --resume
```

### 全量微调（50 epoch）

```bash
python -m pretrain.downstream --mode finetune \
  --config configs/finetune.json --manifest configs/data.json \
  --backbone G --pretrained runs/pretrain/G_mae_seed17/encoder_final.pt \
  --out runs/finetune/G_mae_seed17 --seed 17 --device cuda:0 --workers 4
# 续跑：末尾加 --resume
```

恢复语义：`--resume` 比较 checkpoint 内配置（`resume_key`，排除 epochs）与 provenance，一致则从 `last.pt` + 训练/验证 RNG 状态续跑，否则报错；仅 `--smoke` 允许 `--epochs` 覆盖（见 engine.py 内 `resume_key`）。

## 3. 验收 / 聚合 / 曲线 / CFO / 诊断 / 打包

```bash
python scripts/accept.py                  # 4 项正式验收 → reports/acceptance.{json,md}
python -m pytest tests -q                 # 20 项单元/核心测试
python scripts/aggregate.py               # 探测/微调/基线汇总+配对表 → results.json + paired_tables.md
python scripts/curves.py                  # 三张训练/探测/微调曲线 PNG
python scripts/eval_finetune_cfo.py       # 微调 best.pt 在 clean + CFO±0.005 的 OA/F1
python scripts/denoise_diagnostic.py      # 去噪器在合成波形上的独立诊断
python scripts/package.py                 # 汇总交付物 → deliverables/SSM_L_round1_results.zip
```

## 4. 本轮实际执行记录

- 编排器一次 `--all` 完成全部 36 个新 run（12 预训练 + 12 探测 + 12 微调），无 NaN、无失败。
- 6 个监督基线（G-scratch 来自旧工程 round3_aug、Q-scratch 来自 round5_iq）经配置哈希、划分文件 md5、增强顺序、选模规则逐项核对后复用。
- 日志：`logs/{pretrain,probe,finetune}/{group}_seed{s}.log`、`logs/orchestrate_all.log`。
