# Round-5 实现一致性契约（implementation contract）

本文档在**正式训练前**建立，逐项证明「实际实现 == 任务定义」。列：任务要求 → 源码位置 → 验证方法 → 实际结果。

- 冻结源码哈希（`engine.code_hash()`，训练前）：`c58df87f3dc72a5d6bd7696ec29e9e67e1fab7bea0e486538393791e330781a0`
- 冻结配置：`configs/resolved_round5.json`（`protocol.name=RML2018A_cls_dev_round5_iq`）
- 测试文件：`tests/test_round5.py`（36 项断言，全绿）+ `tests/test_round5_resume.py`（2 项，全绿）；全量回归 `75 passed`
- 支路/前端随机种子偏移：前端 `seed+70001`，IQ 支路 `seed+70002`（`model.py:560/566/570/578`）
- **补充审计（Round 5 Review）**：修复 resume 重算辅助统计量 bug 并新增 `Q_corr_frozen` 变体后，
  源码哈希更新为 `6e7486f7373a90c6e61e47bf7603c68ad4cab5674366853e1e74caab222f506a`；
  历史 21 个运行均未使用 `--resume`，全部"未受影响"（详见
  `reports/round5_review/resume_fix/resume_fix.md`）。

## 前端（Line A）

| # | 任务要求 | 源码位置 | 验证方法 | 实际结果 |
|---|---|---|---|---|
| 1 | 复数乘法 == 实数展开 | `FairComplexFront.forward` (model.py:357)、`complex_to_real_interleaved` (model.py:325) | `test_fair_front_complex_equals_real_expansion` | ✅ `z=W@s` 与 `M@[I0,Q0,...]` 一致（rtol/atol 1e-5） |
| 2 | 交错 IQ 排列正确 | `complex_to_real_interleaved` (model.py:325) | `test_complex_to_real_interleaved_permutation` | ✅ `M[2k,2j]=W_R`、`M[2k,2j+1]=−W_I`、`M[2k+1,2j]=W_I`、`M[2k+1,2j+1]=W_R` |
| 3 | R0/C0/Rb/Cb 初始化前端输出+logits 完全一致 | `build_round5_model` (model.py:647) | `test_a_line_matched_init_output_identity`（5 对） | ✅ 前端输出与 logits 一致（rtol/atol 1e-5，差异来自复数 vs 实数运算 fp 舍入） |
| 4 | 主干 state_dict 一致 / 仅前端不同 | `build_round5_model` (model.py:654) | `test_a_line_shared_backbone_front_differs` | ✅ blocks/norm/head 逐键 `atol=0` 相同；前端 `front.W`(C) 与 `front.proj.weight`(R=M) 满足 `M=expand(W)` |
| 5 | 一步更新后 C 仍绑定 / R 自由 | `FairComplexFront.W`(单复数参数) vs `FairRealFront.proj.weight`(自由实矩阵) | `test_complex_stays_bound_real_free_after_update` | ✅ C 等效矩阵仍满足复结构（`M[even,even]=M[odd,odd]` 且 `M[even,odd]=−M[odd,even]`）；R 一步后不再满足 |
| 6 | 偏置按配置存在/初始化零/可学习 | `FairRealFront.__init__`、`FairComplexFront.__init__` (model.py:338/357) | `test_bias_present_per_config_init_zero_learnable` | ✅ R0/C0 无 bias；Rb/Cb 有 bias 且初值 0；一步后 grad 非零 |

## IQ 支路（Line B）

| # | 任务要求 | 源码位置 | 验证方法 | 实际结果 |
|---|---|---|---|---|
| 7 | 支路读取增强+归一化后的原始 IQ（非 token/标签/SNR/频偏） | `Classifier.forward` 中 `self.aux_descriptor(x)` (model.py:616) | `test_branch_reads_raw_iq_via_hook`（forward hook 捕获输入） | ✅ hook 捕获的支路输入 `atol=0` 等于模型输入 `x` |
| 8 | Q_corr/R_corr 初始状态/描述/完整输出一致 | `RealRecurBranch.__init__` 从 Q_corr 复数递推展开 (model.py:433) | `test_q_corr_r_corr_init_identity` | ✅ 状态 `<1e-3`（fp 舍入）、描述 rtol 1e-4、完整输出 rtol 1e-5 |
| 9 | Q_linear/Q_corr/R_corr 串行==并行 | `ComplexRecurBranch`(parallel_scan) vs `serial_scan`；`RealRecurBranch`(顺序循环) vs 矩阵幂闭式 | `test_complex_recur_branch_serial_equals_parallel`、`test_real_recur_branch_serial_equals_closed_form` | ✅ 复数 1e-4；实数矩阵闭式 1e-4 |
| 10 | 状态跨 patch 连续 / 样本边界重置 | `ComplexRecurBranch.forward` 全 1024 点一次扫描 (model.py:427) | `test_state_continuous_across_patch_reset_at_sample_boundary` | ✅ `q[0]=b·s[0]`；`q[8]=a·q[7]+b·s[8]`（非重置）；批量==逐样本拼接 |
| 11 | 手工 7 对相关、无跨边界 | `state_corr_pack` (model.py:388) | `test_state_corr_manual_7_pairs_no_cross_patch` | ✅ 每 patch 内恰好 7 对相邻，`num/den` 逐项一致，无跨 patch/循环配对 |
| 12 | 零融合 == 同权重 S1 | `aux_fusion` 零初始化 (model.py:584) + `build_round5_model` 复制 S1 主路 | `test_zero_fusion_equals_s1`（4 变体，含 Q_corr_frozen） | ✅ `atol=0` 完全一致 |
| 13 | 第一步更新到达支路参数（不 detach） | `aux_descriptor` 无 detach (model.py:636) | `test_gradient_reaches_branch_params` | ⚠️ 见下方「W_f 零初始化的梯度时序」 |
| 14 | 无跨样本泄漏 | `parallel_scan` / `RealRecurBranch` 顺序循环均为逐样本 | `test_no_cross_sample_leakage` | ✅ 批量==逐样本拼接（rtol/atol 1e-5） |

### W_f 零初始化的梯度时序（item 13 的精确说明）

`u_new = u + W_f·f`，`W_f` 初值为 0。第 1 步：

- `d loss/d W_f = Σ grad_u ⊗ f` **非零**（第 1 步即更新 `W_f`，实测 grad 39.65）；
- `d loss/d f = W_fᵀ·grad_u = 0`，故 `d loss/d 支路参数 = 0`（第 1 步支路参数梯度为 0）。

这是零初始化融合层的**数学必然**，不是 detach：支路参数在优化器中、`requires_grad=True`、前向无 `.detach()`。第 2 步起 `W_f≠0`，梯度正常到达支路参数（`test_gradient_reaches_branch_params` 已验证第 2 步 raw_alpha/theta/b 梯度非零）。契约据此如实记录，不声称「第 1 步即更新支路参数」。

## 维度 / 参数 / dtype

| # | 任务要求 | 验证方法 | 实际结果 |
|---|---|---|---|
| 15 | 维度/参数/dtype 正确 | `test_dims_params_dtype` | ✅ 支路状态 [B,1024,8]；R_corr 状态 [B,1024,8,2]；描述 [B,128,16]；`W_f=Linear(16,176)` 无偏置。参数差：C0 比 R0 少 1408（前端绑定 2816→1408）；Rb/Rc 各 +176（偏置）；R_corr 比 Q_corr +40（支路 72−32）；Q_linear==Q_corr 参数。b 为复数、R_corr 的 A/B 为实数 |

## 训练 / 随机性 / 冻结

| # | 任务要求 | 源码位置 | 验证方法 | 实际结果 |
|---|---|---|---|---|
| 16 | 旧 checkpoint 前向 == 冻结实现 | `ComplexLinear` 未改 (model.py:132) | `test_old_variants_unchanged`（G/S0/S1/S2/B_front/D_complex） | ✅ 旧变体前向/维度不变；`ComplexLinear` 仍是 `randn(cfloat)/sqrt(in)`（实/虚部各 N(0,1/2)→方差 1/16），未改为 Round-5 的 1/48 |
| 17 | 保存预训练 batch 样本 ID/CFO 值/哈希，证明配对模型输入一致 | `engine.py:106-107`（gen/aug_gen 先于模型构建创建） | `test_data_and_aug_stream_independent_of_model_build` | ✅ 数据洗牌序（`gen`）与 CFO 流（`aug_gen`）由各自 `manual_seed(seed)` 决定，与模型构建无关；7 个新变体同 seed 下 perm 与 CFO 逐位 `atol=0` 一致 |
| 18 | 所有支路参数在优化器中、无 detach | `engine.py` 用 `requires_grad=True` 参数构造优化器 | `test_branch_params_in_optimizer_no_detach` | ✅ Q_linear/Q_corr/R_corr 的 aux_branch/aux_fusion 全部在 param_groups 中，`requires_grad=True`，前向有梯度链；Q_corr_frozen 的 aux_branch 冻结（`requires_grad=False`、不在优化器），aux_fusion 仍训练（`test_q_corr_frozen_branch_excluded_from_optimizer`） |
| 19 | 过拟合 + NaN 检查 | `--smoke` 训练（128 样本） | 对 8 个变体（含 Q_corr_frozen）冒烟 5 epoch | ✅ 全部无 NaN、train loss 下降、val OA→1.0（见 `reports/round5_review/smoke_overfit.json`） |
| 20 | 训练前冻结源码哈希/resolved 配置/实现文档 | 本文档顶部 | 本契约 | ✅ 源码哈希、配置路径、测试全绿已记录 |

## 关键实现决定（与任务书逐条对应）

- **A1 匹配初始化**：`build_round5_model` 先构建参考 S1（同 seed 下全局 RNG 一致），再构建变体并用 `load_state_dict(strict=False)` 复制主干（A 线只复制 blocks/norm/head；B 线复制整条 S1 主路 front+blocks/norm/head）。前端共用 `seed+70001` 生成器产生的同一 `W_R,W_I`：C 用复数权重、R 用展开 `M`。
- **A2 初始化尺度**：`round5_front_weights` 生成 `W_R,W_I ~ U(−1/√(2·in), 1/√(2·in))`，等效实数矩阵元素方差 = 1/(6·in) = 1/48，**实测 0.0216 == nn.Linear(16,176) 的 0.0213**（匹配，而非 Round-4 的 1/16=3× 失配）。
- **A3 优化边界**：C 前端是单一复数参数（绑定是结构性的）；优化器/学习率与 R 完全相同，无隐式 C 组 lr 改动。参数化差异（复数 F.linear vs 实数 matmul）在契约 #5 与结论中作为已知局限记录。
- **B 支路定义**：`s[m]=I[m]+jQ[m]`；`q[m]=a·q[m−1]+b·s[m]`；`a=exp(−softplus(raw_alpha))·exp(jθ)`；`q` 8 维复数、`q[−1]=0` 每样本、1024 点连续（不按 patch 重置）、每输入只算一次（非每层）。tau 对数均匀 2–128（α=1/τ）、θ 确定性均匀覆盖 [−π,π)、b 实/虚部各 N(0,0.1)。**不叫 S1**。
- **Q_linear/Q_corr/R_corr**：Q_linear 读 patch 内 8 状态均值→16 实；Q_corr 读 patch 内 7 对相邻复相关（无跨 patch/循环，ε=1e-8）→8 复相关打包→16 实；R_corr 用 8 组 2 维实状态（`A_k` 2×2 非旋转、`B_k` 2×2 自由），谱范数 `<1` 由前向缩放 `A_k=ρ_k·A_raw_k/max(1,‖A_raw_k‖₂)`（ρ=sigmoid<1）保证，初值从 Q_corr 展开（同初始状态序列）。
- **融合**：`u_new=u+W_f·f_std`，`W_f=Linear(16,176)` 无偏置零初始化；描述用固定 8192 训练样本（无增强、max_abs 归一化）按每 seed 初始支路统计冻结（Q_corr/R_corr 同 init 同统计）；缓存 buffer、冻结、梯度保留到支路参数。
- **不声称**：不声称复数隐藏状态=物理状态、不声称严格等变/CFO 补偿；主线统一为原单向 S1。
