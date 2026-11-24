# 第二阶段报告：验证优越性 + 扩大预训练数据与微调任务

> 结果先行。口径：冻结线性读出（probe）/ 全微调（finetune）test OA，3 种子。100k 预训练 = F1 100k；200k = F1 + 额外 100k（RML2018A 分层采样）。

## 一、一句话结论

**structured（多分支物理结构）在 4 个下游任务 × 2 种预训练数据量下全部最优**，且跨域/跨数据集任务（HisarMod +9.9pp、TECHRec +9.1pp）优势远大于同域 AMC（+3.6pp）——物理先验带来更强泛化，是方法优越性的核心证据。扩大预训练数据（100k→200k）使 structured 各任务提升 +0.2~2pp，但优势差距基本持平。

## 二、主结果（probe，200k 预训练）

| 模型 | AMC(24类) | radar(5类) | TECHRec(3类) | HisarMod(26类) |
|---|---|---|---|---|
| **structured** | **0.3895** | **0.7810** | **0.7013** | **0.4030** |
| generic_ssm | 0.3531 | 0.7454 | 0.6107 | 0.3045 |
| generic_transformer | 0.2380 | 0.7416 | 0.5648 | 0.2715 |
| generic_ssm_recon | 0.3337 | 0.7325 | 0.6104 | 0.2783 |
| structured 优势 vs generic_ssm | **+3.6pp** | **+3.6pp** | **+9.1pp** | **+9.9pp** |

（chance：AMC 4.2%、radar 20%、TECHRec 33%、HisarMod 3.8%）

## 三、100k vs 200k 预训练对比（structured）

| 任务 | probe 100k | probe 200k | 变化 |
|---|---|---|---|
| AMC | 0.370 | 0.3895 | **+1.95pp** |
| radar | 0.776 | 0.7810 | +0.5pp |
| TECHRec | 0.696 | 0.7013 | +0.5pp |
| HisarMod | 0.401 | 0.4030 | +0.2pp |

结论：**更多预训练数据对所有任务都有益**（AMC 收益最大 +2pp），但 structured vs generic_ssm 的平均差距在 100k（6.5pp）和 200k（6.6pp）基本持平——扩大数据提升绝对性能，但不放大相对优势。

## 四、跨域泛化是核心优势（关键发现）

structured 相对 generic_ssm 的优势，随任务"跨域程度"递增：

| 任务类型 | 优势 |
|---|---|
| 同域 AMC（预训练 F1 即 RML2018A 同源） | +3.6pp |
| 同域 radar（不同数据源） | +3.6pp |
| **跨任务 TECHRec（通信→LTE/WiFi/DVB-T 技术识别）** | **+9.1pp** |
| **跨数据集 HisarMod（RML2018A→HisarMod 26 类）** | **+9.9pp** |

物理观测（能量/相干/频谱/周期）是**跨数据源、跨任务通用的信号先验**，所以物理分支在"预训练没见过的数据分布"上优势更大——这正是通用 IQ 编码器想要的性质，不是只在训练分布内过拟合。

## 五、finetune 结果（200k）

| 模型 | AMC | radar | TECHRec | HisarMod |
|---|---|---|---|---|
| **structured** | **0.500** | **0.867** | **0.686** | **0.541** |
| generic_ssm | 0.474 | 0.860 | 0.673 | 0.516 |

微调后差距缩小（同域接近饱和），但 structured 仍全任务最优。

## 六、边界

- HisarMod 为子采样（train 91000/val 13000/test 52000），非全量 520000；类别字典已映射 0-25。
- 仅 3 种子（17/42/2026），未追加 5 种子；结论以"方向一致"为准。
- 200k 预训练只跑了 4 个关键模型（非全 34 变体），因为核心是验证优越性而非重复消融。
- 参数匹配（H9）仍未跑：structured 55 万参数 vs 历史 G/Q 224 万参数，优势可能含参数量因素（但本文对比的是本包同参数档的 generic_ssm，已控制此因素）。

## 七、交付物

- `report/plan2_report.md`（本文件）、`report/plan2_summary.csv`、`report/plan2_figures/`
- `results/plan2_techrec/`、`results/plan2_hisarmod/`、`results/plan2_pretrain/`、`results/plan2_full/`
- 脚本：`server_prep/plan2_*.py`、`prepare_hisarmod.py`、`gen_pretrain_200k.py`
- 打包：`deliverables/em_memory_suite_plan2_results.zip`
