# Round-4 结构实验：设计 + 实现说明

## 0. 研究问题

基于 S1（输入驱动的衰减+写入，`h_n = a_n h_{n-1} + g_n (B u_n)`），回答四个问题：

1. 双向扫描是否优于单向 SSM？
2. 复数结构化 IQ 前端是否优于普通实数 Patch 映射？
3. 双向 + 复数前端能否叠加？
4. 复数特征贯穿所有层是否优于仅前端复数？

本轮**授权**：代码实现、正确性验证、训练、val 评估、效率测量、打包。不覆盖已有结果、不使用 test、不逐个训练请求确认。

## 1. 冻结协议

- RML2018A 24 类，原 cls_dev train/val 划分；**不读 test、不推理 test、不按 test 选点**。
- length=1024，patch=8 → 128 token；实数宽 176，复数宽 88（=176 个实数分量）。
- depth=6，FFN 比 4，dropout=0；SSM 状态 128 复数；SSM 无显式位置编码。
- 50 epoch，batch=128，lr=3e-4；AdamW/weight decay/warmup/cosine/grad_clip 冻结。
- fp32/complex64；种子 17/42/2026。
- CFO 增强 U(-0.002,0.002) 逐样本；顺序：raw IQ → CFO → joint max-abs norm → model（order B）。
- checkpoint：clean val OA 最高、最早打破平局；**不按 CFO 结果换 checkpoint**。

## 2. 基线（复用既有增强 checkpoint）

| variant | 说明 | 来源 |
|---|---|---|
| G | 注意力 + 正弦位置 | runs/round3_aug |
| G_corr | 注意力 + 复相关物理特征（3 维） | runs/round3_aug |
| S1 | 单向选择性复数 SSM（状态 128） | runs/round3_aug |

G_corr 含复相关特征而新模型不含——G_corr 是**实用强基线**，不是单因素对照。

## 3. 实验 A：双向 S1（A64 / A128）

- 仅替换 SSM 混合器，实数前端/残差/LayerNorm/FFN/头全部不变。
- 正向与反向各用独立状态参数、输入映射 B、选择网络、读取矩阵 C。
  - `h_fwd[n] = a_f[n] h_fwd[n-1] + g_f[n] (B_f u[n])`
  - `h_bwd[n] = a_b[n] h_bwd[n+1] + g_b[n] (B_b u[n])`
  - `y[n] = Re(C_f h_fwd[n] + C_b h_bwd[n]) + d u[n]`
- 直通项只加一次（单一 `d`）；无未记录的双向平均或额外增益。
- 反向通过翻转 token 维实现（翻转 → 前向扫描 → 翻转回），**不**做 patch 内反转、**不**做 IQ 共轭；两个方向均为零边界初态。
- 并行扫描 + 串行参考（两者容差内一致，见 correctness）。
- A64 = 正向 64 + 反向 64；A128 = 正向 128 + 反向 128。
- 双向**不**称作"全局注意力"（机制完全不同）。

## 4. 实验 B：复数前端（B_front）

- 每个 patch 的 8 个 IQ 点 = 8 维复数向量；无偏 `ComplexLinear(8,88)`；打包 88 复数 → 176 实数。
- 后续完全沿用原始单向 S1（普通 LayerNorm、实数 FFN、实数头），状态 128。
- 复数线性：`z_R = W_R I − W_I Q`，`z_I = W_I I + W_R Q`（已验证等价于实数权重绑定）。
- 只测试前端复数权重约束；**不**声称端到端复数或等变性。
- 参数量与 `Linear(16,176)` 的比较 + 初始化输出尺度见 correctness/benchmark。

## 5. 实验 C：双向 + 复数前端（C_bi_front）

- 前端 = B_front 完全一致；混合器 = A128 双向。
- 对照：C_bi_front vs A128（复数前端在双向下的增量）、C_bi_front vs B_front（双向在复数前端下的增量）。
- 固定在 A128（不做 A64/A128 的 val 挑选）。

## 6. 实验 D：复数贯穿（D_complex）

- 固定单向（主要与 B_front 对比）。
- 复数 IQ patch → `ComplexLinear(8,88)` → 6 个复数块 → 复数归一化 → 均值池化 → 打包 Re/Im 为 176 实数 → `Linear(176,24)`。
- 每块：复数归一化 → 单向选择性复数 SSM → 复数残差 → 复数归一化 → 复数 FFN → 复数残差。
  - B：88→128 复数；C：128→88 复数（保留完整复数，**不取实部**）；直通项 88 维实数缩放，初值 1。
  - alpha/delta/theta/g 沿用 S1；选择器读取当前归一化复数特征的 Re+Im（打包 176 实数）+ 原 3 零槽 = 179 → 64 → 256。
  - 无实数频率注入。
- 复数归一化：`rms=sqrt(mean(|z|²)+eps)`，`out = gamma * z / rms`，gamma 88 维实数初值 1，无 shift（eps=1e-6）。
- 复数 FFN：无偏 `ComplexLinear(88,352)` → 径向非线性 `z*sigmoid(|z|)` → 无偏 `ComplexLinear(352,88)`；最终也过复数归一化。
- 该项同时改变 norm+FFN+读取，因此收益**不可归因到单一部件**；选择器读 Re/Im，故**不**声称严格等变性。

## 7. 关键实现决策（记录）

- 复数前端打包顺序：`[Re0,Im0,Re1,Im1,...]` 交错（对齐已有 `[I0,Q0,I1,Q1,...]` 约定）。
- `ComplexLinear` 复数项初始化 real/imag iid `N(0,1/in)`（沿用 ComplexSSM B/C 约定），**非** `nn.Linear` 的 Xavier-uniform。正确区分：**实部/虚部各自方差 = 1/in**；**复数总方差 `E|W|²` = 2/in**；**等效实数矩阵元素方差 = 1/in**，是 `nn.Linear(16,176)` 默认方差 1/48 的 6 倍。两者输出尺度差异在 correctness 中实测并记录。前端之后紧跟 LayerNorm/残差，尺度差异大多被吸收。

> 勘误（Round-5 发现）：上段「1/in / 2/in / 1/in」为对 `torch.randn(cfloat)` 的误解。实际 `torch.randn(cfloat)` 实/虚部各 N(0,1/2)，故 `ComplexLinear` 实/虚部各自方差 = 1/(2·in)，总复方差 `E|W|²` = 1/in，等效实数矩阵元素方差 = 1/(2·in)，是 `nn.Linear` 方差 1/48 的 3 倍（非 6 倍）。以 correctness.md 实测为准。
- 复数归一化 eps=1e-6。
- D_complex 的 SSM 读取**无**单独的 readout gain（等价于 gain=1；S1 的 `ssm_gain` 配置值恒为 1.0，故无实际差异）。
- 双向由 variant 名（A64/A128/C_bi_front）编码，`direction` 配置字段保持 `forward`；反向为内部翻转实现，未新增方向标志位。
- 状态统计（state RMS、rho 均值/近 1 比例、gate 均值/饱和率、grad clip）在训练时按 `diagnostic_every` 记录，覆盖 ComplexSSM/BidirectionalSSM/ComplexSelectiveSSM 三类。

## 8. 参数量（实数标量，复数 ×2）

| variant | 实数参数量 |
|---|---|
| G | 2,251,592 |
| G_corr | 2,252,120 |
| S1 | 2,216,936 |
| A64 | 2,286,056 |
| A128 | 2,928,872 |
| B_front | 2,215,352 |
| C_bi_front | 2,927,288 |
| D_complex | 1,192,352 |

- B_front − S1 = −1584（恰为复数前端 vs 实数前端的参数差），证明 B_front 其余路径与 S1 完全一致。
- D_complex 参数明显最少（复数 FFN 无偏 + 权重共享 + 更紧凑的复数 SSM），必须报告（见 benchmark）。
- A64 略大于 S1（两套选择网络）；A128/C_bi_front ≈ 2.93M。

## 9. 训练执行

- 15 run = 5 配置 × 3 种子（A64/A128/B_front/C_bi_front/D_complex），分 3 块 GPU 顺序跑。
- 执行顺序：审计复现 → 新模块+正确性 → smoke/过拟合 → 正式训练 → 冻结评估 → 效率 → 聚合打包。
- 正确性失败的配置不占满训练预算：修复并记录受影响的 run、重跑。
- 源码 SHA256（round4，含全部新模块）：`3c3761479b15e78dc163daaefdc8ba59a323533b3a9ff9bbcd767fd66968c2c8`
- 数据 manifest（raw，normalization=none）：configs/data_raw.json；audit：reports/server_audit.md。
- 配置：configs/resolved_round4.json（augment_cfo_max=0.002，augment_on_raw=true）。

## 10. 评价协议

- clean val OA + Macro-F1；SNR≤0 / >0 分组 + 逐 SNR。
- CFO 网格 0 / ±0.001 / ±0.002 / ±0.005；扩展 ±0.01 / ±0.02（单独展示，不用于选择）。
- 全局相位 0 / π/4 / π/2 / π / 3π/2（raw IQ 常数旋转，再 max_abs norm）。
- 同一样本/扰动/顺序；报告绝对 OA 与相对下降；3 种子均值 + 样本标准差 + 逐种子配对差。
