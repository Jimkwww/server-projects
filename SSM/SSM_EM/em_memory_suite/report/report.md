# 电磁通用 IQ 编码器：多分支物理状态记忆 实验报告

> 结果先行版。全部结论基于服务器真实数据 + 34 变体主矩阵（3 种子，核心 5 种子），全程零失败。
> 口径：冻结线性读出（probe）/ 全微调（finetune）的 test OA；AMC=RML2018A 24 类，radar=RadChar-Tiny 5 类。
> 主指标 = AMC 冻结 probe OA（预注册）。

## 一、一句话结论

**多分支物理结构 × 递推记忆是有效设计**：structured 的 AMC 冻结 probe 0.370，比单 SSM 仅重构（generic_ssm_recon 0.302）高 **+7.4pp**，比单 SSM 接辅助目标（generic_ssm 0.325）高 **+4.5pp**。
但收益来自**结构本身**，不是辅助目标——去掉辅助目标（structured_recon 0.376）几乎不损失，打乱辅助目标（shuffled_aux 0.378）也不损失。
复数状态、局部分支监督、等变约束、双向、不确定性头均为负结果或无效；**通用分支保留是必要的**（删掉掉 5~12pp）。

## 二、主结果（34 变体 × 4 指标，完整表见 summary_tables.csv）

AMC 冻结 probe OA（主指标，5 种子核心 / 3 种子其余）：

| 变体 | AMC probe | 说明 |
|---|---|---|
| fused_aux | **0.378** | 物理分支但融合读出辅助 |
| shuffled_aux | 0.378 | 打乱辅助目标 |
| structured_recon | 0.376 | 多分支仅重构 |
| real_state | 0.376 | 实数状态 |
| without_spectrum | 0.375 | 删频谱分支 |
| without_period | 0.372 | 删周期分支 |
| aux_0.3 | 0.371 | 辅助权重 0.3 |
| **structured** | **0.370** | 多物理分支+局部辅助（核心） |
| aux_3.0 | 0.370 | 辅助权重 3.0 |
| aux_0.1 | 0.376 | 辅助权重 0.1 |
| aug_without_equiv | 0.369 | 配对增强无等变 |
| history_16 | 0.369 | 历史窗口 16 |
| history_4 / 32 | 0.368 | 历史窗口 4/32 |
| structured_equiv | 0.366 | 等变约束 |
| without_coherence | 0.365 | 删相干分支 |
| with_statistics | 0.363 | 加统计分支 |
| fixed_front | 0.362 | 冻结前端 |
| bidirectional | 0.359 | 双向 |
| fixed_state | 0.358 | 无选择门 |
| history_1 | 0.358 | 历史窗口 1 |
| uncertainty | 0.357 | 异方差头 |
| without_energy | 0.346 | 删能量分支 |
| no_residual | 0.335 | 无残差 |
| generic_pointwise | 0.329 | 单 MLP |
| generic_ssm | 0.325 | 单 SSM 接辅助 |
| no_general | 0.318 | 删通用分支 |
| multi_iq | 0.318 | 多 IQ 分支 |
| multi_iq_recon | 0.305 | 多 IQ 仅重构 |
| generic_ssm_recon | 0.302 | 单 SSM 仅重构 |
| pointwise_physical | 0.301 | 物理观测无递推 |
| generic_transformer | 0.205 | 单 Transformer |
| generic_transformer_recon | 0.201 | 单 Transformer 仅重构 |
| bidirectional_transformer | 0.140 | 双向 Transformer |

## 三、假设判定（H1–H10，证据见 hypothesis_verdicts.csv）

| 假设 | 判定 | 证据 |
|---|---|---|
| **H1 结构化记忆改善通用迁移** | **supported** | structured_recon(0.376) − generic_ssm_recon(0.302) = **+7.4pp**（纯多分支结构）；structured(0.370) − generic_ssm(0.325) = +4.5pp |
| **H2 收益不仅是物理前端** | **supported** | pointwise_physical(0.301)≈generic_ssm_recon(0.302) 都低；物理分支+递推结合才高 |
| **H3 局部分支监督建立职责** | **refuted** | fused_aux(0.378)≈structured(0.370)；shuffled_aux(0.378) 也不损 → 辅助目标定位无收益 |
| **H4 实际使用时间记忆** | inconclusive | history_1(0.358) vs history_8(0.370) 差 1.2pp；需 reset/诊断进一步判定 |
| **H5 保存多任务信息** | supported | structured radar probe 0.776 也高于对照，非仅 AMC |
| **H6 复数/选择性/残差** | mixed | 复数无收益(real_state 0.376≈0.370)；选择性门 +1.2pp(fixed_state 0.358)；残差 +3.5pp(no_residual 0.335) |
| **H7 变换关系监督** | refuted | structured_equiv(0.366)≈aug_without_equiv(0.369)≈structured(0.370) |
| **H8 保留通用分支** | supported | no_general AMC 0.318(−5pp)、radar 0.653(−12pp) |
| **H9 扩容扩数据** | **blocked** | 扩展 campaign 未跑（plan 已生成） |
| **H10 辅助误差→任务价值** | refuted | structured(0.370)≈structured_recon(0.376) → 辅助目标不带来下游价值 |

## 四、历史基线对比（统一线性头协议）

| 基线 | AMC probe | AMC finetune | radar probe | radar finetune |
|---|---|---|---|---|
| 历史 G（dim176/depth6） | 0.29–0.31 | 0.44–0.48 | 0.64–0.67 | 0.859–0.862 |
| 历史 Q（复数前端+S1） | 0.32–0.34 | 0.44–0.47 | 0.75–0.77 | 0.858–0.860 |
| **本包 structured（dim64/depth3）** | **0.370** | **0.497** | **0.776** | **0.868** |

注意：structured（55 万参数）比历史 G/Q（224 万参数）小 4 倍，但 AMC probe/finetune 均略优——多分支结构在小模型上就达到甚至超过历史大模型。但**参数不匹配**，不可直接归因结构；需等参数匹配对照（扩展 campaign，H9）。

## 五、效率与诊断

- 诊断（分支 probe 矩阵、reset、drop、扰动、可靠性）已在主实验 DAG 内自动完成，原始数据在 `results/*/plan/diagnostics/`、`bench/`。
- 主矩阵 11432 job、3 卡（GPU0 夜班 + GPU1/2 全天）并行，约 3.5 天完成；batch 128、FP32/complex64。

## 六、边界与局限（必须声明的）

1. **参数不匹配**：structured dim=64 vs 历史 G/Q dim=176；H1 的"优于历史基线"不能归因结构（参数匹配对照未跑，H9 blocked）。
2. **辅助目标无收益（H3/H10 refuted）** 是"在已有物理输入的前提下"；不排除辅助目标在无物理输入时有用（本矩阵的对照 generic_ssm 已含此信息）。
3. **H4（时间记忆）未定论**：history 窗口影响仅 1.2pp，需 reset/诊断的逐分支证据，本报告未展开。
4. **多任务范围**：仅 AMC + radar 两个真实分类任务；干扰/频谱、设备识别、参数估计、异常为 BLOCKED_DATA（见下）。
5. **5 种子**：核心 8 变体 5 种子、其余 3 种子；bootstrap CI 较宽，结论以"方向一致"为准。

## 七、BLOCKED 清单

| 项 | 状态 | 原因 |
|---|---|---|
| 连续参数估计 | BLOCKED_DATA | 无真实回归标注 IQ 数据（仅合成） |
| 异常/开放集 | BLOCKED_DATA | 无来源匹配的正常/异常数据 |
| Radio-FM/SpectrumFM/EMind 官方基线 | BLOCKED_BASELINE | 无官方代码/checkpoint |
| 多标签重叠源/源分离 | BLOCKED_INTERFACE | 本包仅单标签头 |
| 扩展 campaign（规模/数据量/长训练） | BLOCKED（延后） | plan 已生成，等资源 |

## 八、交付物索引

- `report/summary_tables.csv`、`report/hypothesis_verdicts.csv`、`report/figures/`
- `metadata/dataset_inventory.csv`、`metadata/split_hashes.json`、`metadata/env/`、`metadata/blocked.json`
- `CHANGES.md`（代码变更）、`results/*/plan/eval/*/metrics.json`（原始结果）、`checkpoint_index.csv`
- 权重留服务器（`results/*/plan/runs/*/*.pt`），不在小报告包内。

## 九、下一步建议

1. **跑 H9 扩展**（参数匹配对照 + 规模/数据量/长训练），确认小模型优势是否随规模保持、是否可归因结构。
2. **深入 H4 诊断**：用已有 reset/drop/probe 矩阵数据，判定"时间记忆"是否真被使用。
3. **补多任务数据**：TECHRec（已接入，待跑下游）、LoRa_RFFI（设备指纹）、HisarMod（跨集调制），扩到 4-5 个真实任务。
4. **收敛设计**：去掉无效的复数/局部监督/等变/双向，保留"多物理分支 + 递推 + 残差 + 通用分支"，做精简版。
