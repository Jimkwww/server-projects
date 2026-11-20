# 三轮实验勘误与修正记录（归档）

按时间顺序汇总本项目三轮实验中的统计/口径/bug 修正。所有修正均已在对应轮次的报告中体现；此处为归档索引。

## Round 1 → Round 2 复核（`reports/round2/`）

1. **标准差口径**：统一改为样本标准差（÷(n−1)），与第一轮口径一致（此前部分汇总误用总体标准差 ÷n）。
2. **预处理顺序**：区分 order A（先归一化再扰动）与 order B（先扰动再归一化）。经 15 个第一轮 checkpoint 逐 run 复现确认 order A clean OA 精确复现、无实现错误；A/B 差异区分「3 seed 均值 <0.8pp」与「逐 run 最大 1.28pp」（S2_concat seed42 CFO=−0.005）。
3. **低 SNR 措辞**：撤回「低 SNR 全随机」——S1 在 −2dB ≈ 14.7% > 24 类随机猜测 4.17%。
4. **撤回「clean 与鲁棒性双升是 SSM 特有」**：对照未配齐（此前比较的是「SSM+仅复相关」与「Transformer+完整描述」，同时改变了骨干与描述组合），待补 G_corr。

## Round 2 → Round 3（`reports/round3/`）

5. **补 G_corr 对照**（G + 仅复相关，3 seed）后：复相关 clean 收益在 SSM 上为正（S1_corr−S1 +1.86pp 全正）、在 G 上为负（G_corr−G −1.37pp 全负）；CFO 鲁棒性收益两骨干通用。
6. **机制措辞**：能量+相关的互补机制尚不确定，不作「载波补偿/状态控制」解释。

## Round 3 复核（`reports/round3_review/`）

7. **预测 npz 文件名冲突 bug**：原 `eval_extended_cfo.py` 用 `run.name`（仅 basename）命名逐样本预测，导致 `runs/clean/G_seed17` 与 `runs/round3_aug/G_seed17` 同名互相覆盖（276=12×23，无增强组 npz 被增强组覆盖）。已改父目录前缀，冒烟确认 jsonl 行数 == npz 数（552==552）。
8. **结论口径三分层**：不再使用「clean 收益 SSM 特有」「G 上复相关已被增强替代」「SSM 获得独特频偏补偿能力」，改为观察 / 可能解释 / 未证明机制；「增强范围外」仅指超出人工注入 ±0.002，不代表真实数据总频偏范围。
9. **扩展频偏评估**：新增 ±0.003 / ±0.004 / ±0.0075 / ±0.01 / ±0.02，确认增强后 G_corr 在 clean 与全部频偏绝对领先。

## 最终 test 评估（`reports/final_test/`）

10. **test 独立评估**：test 从未用于 checkpoint 选择/调参（best.pt 按 clean val OA 最早 tie 冻结）。test 排序与 val **完全一致**（G_corr > G > S1_corr > S1）。
11. **一处 val→test 量级差异（如实报告）**：clean 上 S1_corr−S1 在 val 为 +0.90pp、test 为 +0.11pp（≈无差异），即「复相关对 SSM 的 clean 增益」在独立测试集上基本消失。不改变排序，也不改变「该 clean 增益本身很小」的判断。
