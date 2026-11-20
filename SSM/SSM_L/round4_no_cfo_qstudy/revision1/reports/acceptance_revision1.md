# Revision-1 acceptance record

## 约束自查

- [x] 不覆盖原报告/结果/权重/预测：所有产物在 `revision1/` 下；原 `round4_no_cfo_qstudy/reports/**` 未改动。
- [x] 无 CFO：本轮无任何新训练；读出的下游 config `augment_cfo_max=0.0`。
- [x] 无新预训练/全量微调：全部复用既有 `encoder_final.pt` / `best.pt`。
- [x] 实现错误定位而非全量重跑：仅重做 P2 记忆诊断与雷达 eval 聚合；P0/P1 训练未重跑。

## 逐项验收

| 项 | 检查 | 结果 |
|---|---|---|
| 一 correction.md | 6 条修正均含原表述/修正/证据 | 完成 |
| 二 completion_audit.csv | 五列（要求/实现/已有文件/状态/补充动作） | 完成 |
| 三 unified_readout | 4 编码器×2任务×3seed；encoder_final；scaler(train)；收敛检查；无 eval 选超参 | 完成（见 unified_probe.csv/md） |
| 四 memory_diagnosis | A/B 组；K∈{1,4,8,16,32,64,full}+偏移；一致性校验全过；K=none==原始前向 | 完成（见 memory_*.csv） |
| 五 state_statistics | 18 模型固定子集；rho/tau/gate/RMS/保留积；截断规则 | 完成（见 state_statistics.json/md + figs） |
| 六 fairness_audit | 初始权重逐字节一致；A/B 冻结 hash 前后一致；C/D 更新；resume；未核验项标注 | 完成（见 fairness_audit.md） |
| 七 交付物 | 全部落盘 revision1/；打包无权重 | 完成（见清单） |

## 一致性校验（记忆诊断，全部通过）

- `K_none_matches_original_maxdiff = 0.0`（每编码器）
- `segment_vs_serial_K*_maxdiff ≤ ~1.2e-6`（每编码器）
- `K_eq_N_equals_full_maxdiff = 0.0`（K>=N 零偏移 == full）
- `K1_equals_b_maxdiff = 0.0`（K=1 切断全部历史，等价于直接写入项）
- 旧 SSM S1 scratch 权重映射前向 maxdiff = 0.0；B_scratch RML full OA 与历史 0.390064 一致

## 未核验 / 限制

- fairness_audit 中 round2 vs round4 两套预训练循环的 bit 级等价性未重跑（见 fairness_audit.md 第 6 节）。
- RML test 封存，统一读出与记忆诊断的 RML 主结果均为 val。
