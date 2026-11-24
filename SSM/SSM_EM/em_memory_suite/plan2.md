# 第二阶段计划：验证优越性 + 扩大预训练数据与微调任务

目标：在已完成的主矩阵（34 变体，多分支结构+递推有效、辅助目标无贡献）基础上，进一步验证 structured 的优越性，并扩大预训练语料与下游任务。

## 实验 1：扩大微调任务（跨域/跨任务迁移）

- 新任务：
  - TECHRec 技术识别（LTE/WiFi/DVB-T 3 类，已接入 `data/techrec`）
  - HisarMod 跨数据集调制（26 类，接入中）
- 用主实验 F1 预训练 checkpoint（structured / generic_ssm / generic_transformer / generic_ssm_recon，seed 17/42/2026）
- 下游：冻结 probe + 全微调（50ep，本包协议）
- 输出：新任务 probe/finetune 对比表，验证 structured 是否在新任务/新数据集仍最优

## 实验 2：扩大预训练数据

- 预训练语料：F1 100k → F1 100k + RML2018A 额外 100k 分层采样 = 200k
- 重新预训练 4 个关键模型（structured / generic_ssm / generic_transformer / generic_ssm_recon），3 seed
- 下游全任务：AMC + radar + TECHRec + HisarMod
- 输出：200k 预训练 vs 100k 预训练的下游对比，看"更多预训练数据是否进一步放大 structured 优势"

## 实验 3：参数匹配（H9，可选）

- scale_medium（dim=128 depth=6）的 structured vs generic_ssm vs generic_transformer
- 验证 structured 优势是否随模型规模保持（不是小模型的偶然）

## 输出（跑完打包）

- report/plan2_report.md（中文，结果先行）
- report/plan2_summary.csv（新任务 + 扩大预训练的完整对比表）
- report/plan2_figures/
- deliverables/em_memory_suite_plan2_results.zip
