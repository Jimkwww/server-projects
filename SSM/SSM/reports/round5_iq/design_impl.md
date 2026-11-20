# Round-5 复数前端公平对照与 IQ 辅助支路实验：设计 + 实现说明

## 0. 研究问题

在 Round-4 的复数前端结论（`B_front` 相对 `S1` 掉 ~4pp，但存在 bias/初始化混杂）基础上，本轮做两条独立实验线，强调**用源码、数值测试、训练记录证明"实现 = 任务定义"**：

- **Line A（复数前端公平对照）**：把复数前端与实数前端放在**完全相同**的初始化、bias 配置、共享主干、优化器条件下对照，隔离"复数是精度损失"与"只是初始化/参数化差异"。
- **Line B（IQ 辅助支路）**：在**自由实主线（原 S1）不动**的前提下，给原始 IQ 加一条可学习的复递归支路，看显式时域结构能否带来干净增益或鲁棒性增益。

本轮**授权**：代码实现、正确性验证（20 项契约）、训练、val 评估、效率测量、打包。**不读 test、不按 test 选点、不逐个训练请求确认、不覆盖既有结果**。

## 1. 冻结协议（与历史 Global/S1 一致）

- RML2018A 24 类，原 `cls_dev` train/val 划分；**不读 test、不推理 test**。
- L=1024，patch=8 → 128 token；实数宽 176，复数宽 88（=176 个实数分量）。
- depth=6，FFN 比 4，dropout=0；SSM 状态 128 复数；SSM 无显式位置编码。
- 50 epoch，batch=128，lr=3e-4；AdamW / weight decay / warmup / cosine / grad_clip 冻结。
- fp32 / complex64；种子 17/42/2026。
- **Order-B 增广**：raw IQ → 每样本 CFO（U(−0.002, 0.002)）→ 联合 max-abs 归一化 → 模型。
- **checkpoint 选择**：clean val OA，最早 tie。

## 2. Line A：复数前端公平对照（R0 / C0 / Rb / Cb）

四个前端：`R0` 实 `Linear(16→176)` 无 bias；`C0` 复 `ComplexLinear(8→88)` 无 bias；`Rb`/`Cb` 同构但带 bias（bias 初始化为 0，可学习）。

### 2.1 匹配初始化

- `round5_front_weights`（[model.py:313](ssm_exp/model.py#L313)）生成 `W_R, W_I ~ U(−1/√(2·in_c), 1/√(2·in_c))`，`in_c=8`，每项方差 = `1/(6·in_c) = 1/48`。
- `complex_to_real_interleaved`（[model.py:325](ssm_exp/model.py#L325)）把复权重展开成等价实矩阵 `M = [[W_R, −W_I],[W_I, W_R]]`，按交错 I/Q 排列（`M[0::2,0::2]=W_R; M[0::2,1::2]=−W_I; M[1::2,0::2]=W_I; M[1::2,1::2]=W_R`）。
- `FairRealFront`（[model.py:338](ssm_exp/model.py#L338)）把 `M` 拷贝为实 `Linear` 权重（自由参数）；`FairComplexFront`（[model.py:357](ssm_exp/model.py#L357)）把 `W_R+jW_I` 绑定为复权重。
- **结果**：init 时 R 与 C 输出逐位一致（数值测试验证差异 ~1e-8）。

### 2.2 A2 初始化尺度

- `nn.Linear(16,176)` 默认 `kaiming_uniform(a=√5)` = `U(−1/√16, 1/√16)`，方差 `1/(3·16)=1/48`。
- Round-5 复前端的等价实矩阵逐项方差 = `1/48`，**与 `nn.Linear` 严格对齐**。
- **注意**：这**不是** Round-4 `ComplexLinear` 的 `N(0,1/in)` 约定。Round-4 复前端实/虚各 `N(0,1/2)`，逐项方差 `1/(2·in)=1/16`，等价实矩阵方差 = `3×` `nn.Linear`。这正是 Round-4 复数前端存在**初始化混杂**的量化来源，本轮通过 A2 消除。

### 2.3 A3 优化器与梯度绑定

- 四个前端用**同一优化器、同一 lr**。
- C 用单个复 `W` 参数（实/虚绑定）；R 用 `M` 的两个自由副本。复权重对实/虚的梯度会合并到同一 `W` 上，等价于 R 的两个副本梯度聚合——保证更新路径结构一致，差异只来自"绑定 vs 自由"这一结构性因素。

### 2.4 共享主干

`build_round5_model`（[model.py:647](ssm_exp/model.py#L647)）从参考 `S1` 拷贝 blocks/norm/head 权重，保证四个前端之后的主干（S1 单向复数 SSM）**同种子下逐位一致**。

## 3. Line B：自由实主线 + IQ 辅助支路（Q_linear / Q_corr / R_corr）

### 3.1 主线

主线 = 原 `PatchEmbedding` + 单向 S1，**完全不动**（复用参考 S1 的前端/blocks/norm/head）。

### 3.2 辅助支路

支路读取**增广+归一化后**的原始 IQ（与主线输入同一个 `x`，在 patch embedding 之前）：

- `s[m] = I[m] + jQ[m]`（[model.py:426](ssm_exp/model.py#L426)）。
- 递归 `q[m] = a·q[m−1] + b·s[m]`，`q` 为 8 维复状态；`q[−1]=0` 每样本重置；连续跑满 1024 点（**不做每 patch 重置**），每个输入只算一次。
- `a = exp(−softplus(raw_alpha))·exp(j·theta)`（[model.py:427](ssm_exp/model.py#L427)），8 维复数。
- 初始化：`tau=1/alpha` 对数均匀 2..128（`alpha=1/tau`）；`theta = linspace(−π,π,9)[:-1]`；`b` 实/虚各 `N(0,0.1)`。

### 3.3 三种 readout（均 → 16 实数）

- **Q_linear**：`state_mean_pack`（[model.py:380](ssm_exp/model.py#L380)）每 patch 内 8 维复状态取均值 → 实/虚交错 16 维。
- **Q_corr**：`state_corr_pack`（[model.py:388](ssm_exp/model.py#L388)）每 patch 内相邻 7 对的复相关 `r = Σ q[t]·q[t−1]^* / (‖q[t]‖·‖q[t−1]‖)` → 实/虚 16 维。
- **R_corr**：`RealRecurBranch`（[model.py:433](ssm_exp/model.py#L433)）8 组 2 维实状态 `x_k[m] = A_k x_k[m−1] + B_k [I[m],Q[m]]^T`；`A_k` 为一般 2×2 实矩阵，forward 用 `A_k = ρ_k·A_raw_k / max(1,‖A_raw_k‖₂)` 强制谱范数 `<1`（`ρ_k=sigmoid(logit_k)∈(0,1)`），`B_k` 自由 2×2；readout 走与 Q_corr 相同的相关打包。初始 `A_raw=R(θ)`、`B=[[b_R,−b_I],[b_I,b_R]]`，使初始状态序列与 Q_corr 一致。

### 3.4 融合与标准化

- 融合：`u_new = u + W_f · f_std`，`W_f` 为 `Linear(16,176)` 无 bias、**零初始化**（[model.py:567](ssm_exp/model.py#L567)）。
- 标准化：`f_std = (f − aux_mean)/aux_std`，`aux_mean/aux_std` 是注册 buffer，由 `fit_aux`（[engine.py:55](ssm_exp/engine.py#L55)）在 8192 个固定样本（`np.random.default_rng(0)` 抽取、max-abs 归一、无 CFO）上算得，训练时冻结。
- **零初始化梯度时序**（数学性质，非 detach）：step 1 时 `dL/dθ_branch = W_f^T · grad_u = 0`，只有 `W_f` 得非零梯度；step 2 起支路参数才获得梯度。已写入实现契约并数值验证。

## 4. 随机流分离（可复现）

- 数据 loader shuffle：`Generator().manual_seed(seed)`。
- CFO 增广流：`Generator(device).manual_seed(seed+100000)`。
- Line A 前端生成器：`seed+70001`。
- Line B 支路生成器：`seed+70002`。
- 同 seed 下 7 个新变体的 perm 与 CFO 逐位一致（契约第 17 项 + `audit_round5_input.py` 验证）。

## 5. 参数计数（与设计一致）

| variant | 实数参数量 | 相对 R0 | 说明 |
|---|---|---|---|
| R0 | 2216760 | 0 | 实前端无 bias |
| C0 | 2215352 | −1408 | 复绑定省 8×176=1408 参数（88 复→176 实，绑定 176 个实参） |
| Rb | 2216936 | +176 | R0 + bias(176) |
| Cb | 2215528 | −1232 | C0 + bias(176) |
| Q_linear / Q_corr | 2219784 | +3024 | R0 + 支路 + W_f(16×176=2816) + 支路自身 |
| R_corr | 2219824 | +3064 | Q_corr + 40（R_corr 比 Q_corr 多 40 参） |

## 6. 关键实现决定与"不声称"边界

- **不声称**复数隐藏状态 = 物理状态；**不声称**严格等变 / CFO 补偿；主线统一为原单向 S1。
- 辅助支路显式命名为 `ComplexRecurBranch`/`RealRecurBranch`，**不叫 S1**、不并入 S1 主线（避免污染 S1 定义）。
- 允许"无明确收益"结论；观察/解释/未证实机制分离。
- 全部结论仅限当前冻结配置；复数前端公平对照的结论受"绑定 vs 自由"这一结构性差异约束（见结论文档）。
