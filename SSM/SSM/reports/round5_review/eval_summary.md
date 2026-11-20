# Round 5 Review 评价：Q_corr_frozen 对照结论（Section 五）

数据源：`reports/round5_review/eval_comparison.json` / `.md`（order-B val，未用 test，
3 seed，mean ± 样本标准差 ddof=1，差值单位 pp=OA 小数差×100）。全文不构成统计显著表述。

## clean val 主表

| variant | seed17 | seed42 | seed2026 | OA mean ± std | Macro-F1 mean ± std |
|---|---|---|---|---|---|
| S1 | 0.4298 | 0.4321 | 0.4250 | 0.4290 ± 0.0036 | 0.4084 ± 0.0028 |
| S1_corr | 0.4327 | 0.4446 | 0.4365 | 0.4379 ± 0.0061 | 0.4200 ± 0.0082 |
| Q_corr | 0.4500 | 0.4612 | 0.4519 | 0.4544 ± 0.0060 | 0.4343 ± 0.0045 |
| Q_corr_frozen | 0.4622 | 0.4679 | 0.4423 | 0.4575 ± 0.0135 | 0.4396 ± 0.0161 |

## 三个核心比较

### (1) Q_corr − Q_corr_frozen（学习滤波是否加值？）

clean per-seed 配对差：−1.22 / −0.67 / +0.96pp，mean −0.31 ± 1.13pp。

**结论：未显示学习滤波带来额外价值。** 三个种子里两个为负、一个为正，方向不一致，
且 |mean|（0.31pp）远小于样本标准差（1.13pp）。frozen 与 learnable 的 clean 表现相当
（甚至 frozen 均值略高，但差异在种子波动内）。CFO 各点同样正负混合、无一致方向。

### (2) Q_corr_frozen − S1（固定滤波 + 相关 readout 是否已有帮助？）

clean per-seed 配对差：+3.24 / +3.59 / +1.73pp，mean +2.85 ± 0.99pp。

**结论：是——固定（不学习的）滤波 + corr readout 本身已带来一致的正向收益。**
三种子一致为正；CFO 各点亦一致为正（+0.005 下 +5.58pp、−0.005 下 +3.64pp）。
这说明增益主要来自「复数递归滤波器结构 + 相邻复相关 readout」本身，而非滤波器
参数的可学习性。

### (3) Q_corr − S1_corr（复核原始 Round-5 对照）

clean mean +1.65 ± 0.10pp（三种子一致正）；但 CFO ±0.005 下 −3.50 / −4.84pp（三种子一致负）。

**结论：与原始 Round-5 一致——clean 上 Q_corr 优于 S1_corr，大 CFO 下明显更差。**

## 结论边界（按任务书约束）

- **frozen ≈ learnable 仅表示「未显示学习滤波的明确额外价值」**，不等于「学习滤波
  必然无用」；本配置、本滤波器集、本数据规模下不排除更大数据/更长时间下学习有收益。
- **frozen 有效仅支持「这一组滤波器」**：冻结的是 Q_corr 的初始随机滤波器
  （seed+70002），不构成对任意滤波器/任意初始化的支持。
- **不宣称单种子/单点结论**：所有结论基于 3 seed 的 mean±std 与 per-seed 配对差；
  seed2026 的 frozen clean OA（0.4423）明显低于另两种子，样本标准差（1.35pp）较大，
  已在均值±std 中体现。
- **不做 auto-CFO 补偿 / 物理状态声明**：Q_corr_frozen 与 Q_corr 一样在大 CFO 下
  退化（±0.005 绝对 OA 0.34 vs clean 0.46），无任何自动频偏补偿能力。
