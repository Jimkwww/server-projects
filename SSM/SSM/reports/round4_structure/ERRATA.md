# Round-4 勘误（过程记录，不影响结论）

1. **C_bi_front 初次漏发**：训练 launcher 首次把 GPU2 分给 B_front+D_complex，漏掉了 C_bi_front 的 3 个 run。发现后单独补发（约 58 分钟后），三个 run 正常完成（best_epoch 48/39/39）。结果无影响，仅训练完成时间延后。

2. **聚合脚本两处 bug（已修复并重跑）**：
   - train_stats 段对 attention 基线（G/G_corr 无 SSM 模块，诊断里 rho/gate 为 None）调用 `statistics.mean([None,...])` 崩溃；改为过滤 None。
   - paired_diffs 段循环变量 `for a,b in comps` 覆盖了 argparse 命名空间 `a`，导致 benchmark 段 `a.bench` 报错；循环变量改名为 `va,vb`。
   - 修复后完整重跑 `scripts/aggregate_round4.py`，所有 10 个 .md 报告重新生成。

3. 中间文件（benchmark_partial.json / benchmark_cbi.json / eval_cases_baseline|r4|cbi.jsonl）已从交付物中移除，合并结果保留为 benchmark.json 与 eval_cases.jsonl。

---

## Round-5 前置修正（2025 记录；不新增 Round-4 训练，原始数据 eval_cases.jsonl 未改）

4. **掉点表单位统一为百分点(pp)**：原 `cfo_curves.md`「Relative drop」表与 `phase_curves.md`「Drop」表、`paired_diffs.md` 的表头虽写「pp」，数值实为 OA 小数差。已按「OA 小数差 × 100 = pp」换算（均值与标准差同步 ×100），重跑 `aggregate_round4.py` 重新生成三个报告。**原始 OA 数据仍保留在 `eval_cases.jsonl` 及各 `*_curves.md` 的 Absolute OA 表中，未覆盖。** 关键数值换算示例：S1 在 CFO +0.005 的相对掉点由 `+0.1431` 改为 `+14.31pp`；相位掉点绝对值最大约 `0.98pp`（D_complex@π/4）。

5. **相位表述更正**：`conclusion.md` Q8 原写「对全局相位几乎不变（下降 <0.01pp）」，实为 OA 小数 0.01 量级 = **<1pp**。已改为「相对 clean 的波动 <1pp」。

6. **「标准差 > 均值」不再单独作为噪声判定**：`conclusion.md` Q1/Q2 原以「样本标准差大于均值」判定 A128 增益为噪声。已改为以「三种子方向不一致」为更直接证据，标准差大于均值仅作次要旁证。

7. **复数前端结论限定在当前配置**：`conclusion.md` Q3 与「解释」段补充说明 B_front 的损失混杂了 (a) 无偏置 vs 有偏置、(b) 复数前端初始尺度失配（等效实数矩阵方差 1/8，为 `Linear(16,176)` 方差 1/48 的 6 倍）两个因素，不能直接归因于复数绑定本身。Round-5 将分离这两个混杂。

8. **初始化方差表述澄清（含二次勘误）**：`design_impl.md` 与 `correctness.md` 补充区分「实部/虚部各自方差」「复数总方差 `E|W|²`」「等效实数矩阵元素方差」。**二次勘误（Round-5 实测）**：`torch.randn(cfloat)` 的实/虚部各 N(0,1/2)（复数标准正态总方差=1），故 `ComplexLinear` 实/虚部各自方差 = **1/(2·in) = 1/16**、总复方差 `E|W|²` = **1/in = 1/8**、等效实数矩阵元素方差 = **1/16**，是 `nn.Linear(16,176)` 方差 1/48 的 **3 倍**（不是此前结论里写的 6 倍）。`conclusion.md` Q3 限定与 `correctness.md` 已同步修正；`design_impl.md` 保留原文并加勘误注记（保留历史原始表述）。
