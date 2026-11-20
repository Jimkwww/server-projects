# Revision-1 状态（2026-11-11 完成）

**状态：全部完成。** 本目录是对 `round4_no_cfo_qstudy` 的**补正与补充验证**，不覆盖原报告 /
结果 / 权重 / 预测；无 CFO、无新预训练 / 全量微调，全部复用既有 checkpoint 重提取特征、
重训线性读出、重跑状态与记忆诊断。

## 结果速览（修正后）

- **问题 1（预训练收益）**：RML 收益 1.8–3.1pp、雷达 0.4–1.0pp；「Global 吃去噪、SSM 吃 MAE」
  仅 RML 成立（雷达 Global 反转为 MAE>denoise）。
- **问题 2（Q 学习阶段）**：该 2×2 设置下未证明学习有必要、固定 Q 有竞争力；雷达 eval 排序
  C(86.76)>B(86.48)>D(86.43)≈A(86.42)，「A 最优/学习有害」撤销。Q+MAE=41.91% 用可学习 Q。
- **问题 3（Q vs G_corr）**：Q 滤波相关整体优 ~1.6pp，但维度/初始化/统计量未隔离。
- **问题 4（SSM 记忆）**：撤销「16–64 token 有效记忆」。冻结预训练编码器重拟合读出在 K=1 即
  恢复 ~full（mae RML 0.315 vs 0.328）；下游微调模型仅温和渐进短窗依赖；状态统计无双峰。
- **问题 5（统一冻结读出）**：RML 上 G(0.3172) 最好、Q 描述子有损；雷达上学习Q(0.8042) 最好、
  Q 描述子有益（任务相关）。

## 交付物（均在 `revision1/`）

| 文件 | 对应节 |
|---|---|
| `reports/correction.md` | 一（6 条逐条修正） |
| `reports/completion_audit.csv` | 二 |
| `reports/unified_probe.csv / .md / _raw.json` | 三（统一冻结读出） |
| `runs/predictions/*_pred.npz` | 三（逐样本预测 + sha256） |
| `reports/memory_A_raw.json / memory_B_raw.json` | 四（记忆诊断原始） |
| `reports/memory_pretrain.csv`（A 组 fixed/refit） | 四 |
| `reports/memory_finetuned_original_head.csv`（B 组） | 四 |
| `reports/memory_summary.md` | 四（汇总） |
| `reports/state_statistics.json / .md` + `figs/state_stats_*.png` | 五 |
| `reports/fairness_audit.json / .md` | 六 |
| `reports/tables_radar_eval.md / radar_eval_main.json` | 雷达 eval 主表 + P1 配对 |
| `reports/revised_conclusion.md` | 七（5 问题最终答案） |
| `reports/acceptance_revision1.md` | 七（验收） |
| `reports/commands.md` | 七（复现命令） |
| `reports/config.md` | 七（配置） |
| `reports/source_diff.md` | 七（源码改动清单） |
| `reports/latest_status.md` | 七（本文件） |
| `scripts/*.py` | 全部脚本（common_r1 / unified_readout / memory_diagnosis / memory_summary / state_stats / fairness_audit / radar_eval_agg） |
| `revision1_package.tar.gz` | 打包（不含权重与特征缓存） |

## 一致性 / 验收要点

- 记忆诊断一致性校验全过：`K_none==原始前向(0.0)`、`segment_vs_serial≤1.2e-6`、`K>=N==full(0.0)`、
  `K1==直接写入(0.0)`；旧 SSM S1 权重映射前向 maxdiff=0.0，B scratch RML full 与历史一致。
- 统一读出 36 次 LR 拟合全部收敛（`converged=1`）。
- fairness_audit：固定/学习 Q 预训练初始权重逐字节一致、A/B 冻结 hash 前后一致、C/D 确有更新。

## 复现

`reports/commands.md`（tmux 启动记录见第 7 节）。
