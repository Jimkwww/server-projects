# Revision-1 commands（复现命令）

工作目录：`/home/user_wzk/wzk/projects/SSM/SSM_L`。所有脚本在
`round4_no_cfo_qstudy/revision1/scripts/`，输出到 `round4_no_cfo_qstudy/revision1/reports/`
与 `round4_no_cfo_qstudy/revision1/runs/`。**不覆盖** `round4_no_cfo_qstudy/reports/` 下的任何
原文件。

## 0. 共享环境

- 依赖与 round-3/round-4 相同：`conda test1` + `torch(cu126)` + `sklearn` + `numpy` + `matplotlib`。
- 导入路径：`sys.path` 指向 `SSM_L` 与 `SSM_L/round4_no_cfo_qstudy/revision1/scripts`。
- 无 CFO、无新预训练/全量微调：全部复用既有 checkpoint。

## 1. 统一冻结读出（Section 3）

```bash
cd /home/user_wzk/wzk/projects/SSM/SSM_L
# 缓存特征（复用已存在，skip 不重算）
python round4_no_cfo_qstudy/revision1/scripts/unified_readout.py --device cuda:0
# 或分步：--cache-only 只缓存；--readout-only 只读特征做读出（不重算特征）
```

产物：`reports/unified_probe.csv`、`reports/unified_probe.md`、`reports/unified_probe_raw.json`、
`runs/predictions/*_pred.npz`、`runs/features/*.pt`。

## 2. SSM 记忆诊断修正（Section 4）

```bash
cd /home/user_wzk/wzk/projects/SSM/SSM_L
# A 组（无监督预训练编码器，固定+重拟合读出）
python round4_no_cfo_qstudy/revision1/scripts/memory_diagnosis.py --group A --device cuda:1
# B 组（下游微调模型，原始 head，per-K OA vs full）
python round4_no_cfo_qstudy/revision1/scripts/memory_diagnosis.py --group B --device cuda:2
```

产物：`reports/memory_A_raw.json`、`reports/memory_B_raw.json`、
`reports/memory_pretrain.csv`（A 组：fixed/refit × split × K）、
`reports/memory_finetuned_original_head.csv`（B 组：K × OA × delta_vs_full）、
`runs/memory_features/*.pt`。

一致性校验（每编码器，随脚本自动跑）：`K_none_matches_original_maxdiff`、
`segment_vs_serial_K*_maxdiff`、`K_eq_N_equals_full_maxdiff`、`K1_equals_b_maxdiff`。

## 3. 状态统计补充（Section 5）

```bash
cd /home/user_wzk/wzk/projects/SSM/SSM_L
python round4_no_cfo_qstudy/revision1/scripts/state_stats.py --device cuda:0
```

产物：`reports/state_statistics.json`、`reports/state_statistics.md`、
`reports/figs/state_stats_rho_quantiles.png`、`state_stats_retention.png`、
`state_stats_tau_hist.png`。

## 4. Q 2D 公平性审计（Section 6，无重跑）

```bash
cd /home/user_wzk/wzk/projects/SSM/SSM_L
python round4_no_cfo_qstudy/revision1/scripts/fairness_audit.py
```

产物：`reports/fairness_audit.json`、`reports/fairness_audit.md`。

## 5. 雷达 eval 主表 + P1 配对表（Section 1.2 / 7）

```bash
cd /home/user_wzk/wzk/projects/SSM/SSM_L
python round4_no_cfo_qstudy/revision1/scripts/radar_eval_agg.py
```

产物：`reports/radar_eval_main.json`、`reports/tables_radar_eval.md`。

## 6. 打包（不含权重）

```bash
cd /home/user_wzk/wzk/projects/SSM/SSM_L/round4_no_cfo_qstudy
tar czf revision1_package.tar.gz revision1 \
  --exclude='revision1/runs/features/*.pt' \
  --exclude='revision1/runs/memory_features/*.pt' \
  --exclude='revision1/scripts/__pycache__'
```

（特征缓存 `.pt` 体积大，可复算，不打包；预测 `.npz` 保留。）

## 7. tmux 启动记录（本次实际执行）

```bash
tmux new-session -d -s r1_unified "python round4_no_cfo_qstudy/revision1/scripts/unified_readout.py --device cuda:0 > round4_no_cfo_qstudy/revision1/logs_unified.log 2>&1"
tmux new-session -d -s r1_memA "python round4_no_cfo_qstudy/revision1/scripts/memory_diagnosis.py --group A --device cuda:1 > round4_no_cfo_qstudy/revision1/logs_memA.log 2>&1"
tmux new-session -d -s r1_memB "python round4_no_cfo_qstudy/revision1/scripts/memory_diagnosis.py --group B --device cuda:2 > round4_no_cfo_qstudy/revision1/logs_memB.log 2>&1"
```
