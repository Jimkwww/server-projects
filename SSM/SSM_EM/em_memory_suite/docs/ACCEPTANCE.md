# 验收清单

## 代码准入

- [ ] tests全部通过；保存版本、硬件、用时、日志。
- [ ] 串行/并行scan数值与梯度一致，非2幂长度与initial state正确。
- [ ] 无mask干净信息旁路；因果配置不看未来；不同batch样本互不污染。
- [ ] 辅助头局部分支梯度隔离；冻结probe不更新encoder。
- [ ] 分类/回归/分割形状、目标单位、missing labels明确。
- [ ] epoch边界恢复与连续训练一致；GPU环境独立复验。
- [ ] 发现错误修复后重跑相关测试并使受影响run失效，不保留旧结果混用。

## 研究准入

- [ ] 训练/验证/test及来源groups审计，参数与训练token预算清单。
- [ ] 真实任务范围及缺失任务明确；主任务的评价指标预注册。
- [ ] 原S1/Q与新参考实现分开；官方模型身份和版本可核实。
- [ ] 长度、SNR、频偏单位、归一化、双向性不混淆。

## 结果完整性

- [ ] 主矩阵34变体全状态记录；3seed基础，核心比较5seed。
- [ ] 每任务冻结/微调/从头及标签比例；best/last核心补充。
- [ ] 多规模/数据量/长训练/跨域；等epoch与等step比较区分。
- [ ] 读出probe/状态reset/drop/物理干预/可靠性结果。
- [ ] 鲁棒性label policy正确；无意义扰动标N/A而不是伪结果。
- [ ] latency/throughput/memory/params包含前端，当前kernel局限明确。
- [ ] 失败、OOM、BLOCKED_DATA/BASELINE不从分母消失。
- [ ] 不同loss值不横向当表征评分，选择只用val。

## 判断用语

“指定信息可读出”不等于“每个隐状态唯一对应物理量”。
“分支删除损害任务”不独立证明物理因果机制。
“多任务胜出”须列任务数、negative transfer及seed差异。
“基础模型”需预训来源/规模/迁移范围证据，不由网络名字决定。
“效率更好”须包含完整前端与实际部署内核条件。

## 交付目录

report/report.md, report/summary_tables.csv, report/hypothesis_verdicts.csv,
report/figures/, metadata/dataset_inventory.csv, metadata/split_hashes.json,
metadata/env/, configs/, logs/, failures.json, code_diff.patch,
checkpoint_index.csv（path,SHA256,size,source_run），tests.log。

对每个假设填：status=supported/refuted/inconclusive/blocked；直接证据路径；效应大小；限制。
