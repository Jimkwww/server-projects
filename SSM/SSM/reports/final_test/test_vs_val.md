# 最终 test 结论：test vs val 排序核对

> 区分验证集开发结果与独立测试集结果。以下对比同一 order-B 协议下 val 与 test 的 3-seed 均值。

## 排序对比（OA 均值，pp；从高到低）

| 条件 | val 排序 | test 排序 | 是否一致 |
|---|---|---|---|
| clean | G_corr > G > S1_corr > S1 | G_corr > G > S1_corr > S1 | 一致 |
| CFO=+0.005 | G_corr > G > S1_corr > S1 | G_corr > G > S1_corr > S1 | 一致 |
| CFO=−0.005 | G_corr > G > S1_corr > S1 | G_corr > G > S1_corr > S1 | 一致 |
| CFO=+0.01 | G_corr > G > S1_corr > S1 | G_corr > G > S1_corr > S1 | 一致 |
| CFO=−0.01 | G_corr > G > S1_corr > S1 | G_corr > G > S1_corr > S1 | 一致 |
| CFO=+0.02 | G_corr > G > S1_corr > S1 | G_corr > G > S1_corr > S1 | 一致 |
| CFO=−0.02 | G_corr > G > S1_corr > S1 | G_corr > G > S1_corr > S1 | 一致 |

## 结论

（由 aggregate_test.py 汇总；最终文字结论见 test_conclusion.md）
