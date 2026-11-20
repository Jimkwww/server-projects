# G_Q_corr / G_Q_corr_frozen 验收报告

验收脚本：`scripts/accept_global_qcorr.py`，逐 seed 结果落盘于 `reports/global_qcorr/acceptance/seed{17,42,2026}.json`，resume 测试 `tests/test_global_qcorr_resume.py` 结果见 `check4b_resume.json`。三个 seed 的 4 项检查全部通过，关键数值以 seed17 为代表（其余两 seed 一致）。

## Check 1 — 确实使用 Global 骨干
- 6 个 block 全部为 Attention（`n_attention_blocks=6`），无 SSM 分支。
- 位置编码 `sinusoidal`，前端 `PatchEmbedding`（patch=8），token 形状 `[8,128,176]`。
- 骨干参数与 `G` 完全共享（`backbone_shared_with_G=true`），融合层零初始化（`aux_fusion_zero=0.0`）。
- 零融合时 G_Q_corr / G_Q_corr_frozen 与 G 的 logits 逐元素完全一致（`max_abs_logit_diff=0.0`）。
- 参数量：G=2,251,592；G_Q_corr=2,254,440；G_Q_corr_frozen=2,254,440（冻结版可训练 2,254,408）。

## Check 2 — 确实复用 Q 特征（复相关，非 raw-IQ 相关）
- 辅助分支参数与 `Q_corr` 的 `ComplexRecurBranch` 完全一致（`branch_params_identical=true`）。
- 相同输入下滤波状态 `max_abs_filter_state_diff=0.0`；标准化前/后描述子与 `Q_corr` 一致（`pre/post_std_desc_diff=0.0`）。
- `aux_mean/aux_std` 统计与 Q_corr 相同（`aux_stats_identical=true`）。
- 无 raw-IQ 相关融合路径（`has_raw_iq_corr_fusion=false`）。

## Check 3 — 冻结边界正确 + 可学习分支确实收到梯度并变化
- 可训练性差异仅限 `aux_branch.b / raw_alpha / theta` 三个参数，共 32 个实标量（`n_trainable_diff_real_scalars=32`）。
- 学习版 step2 分支梯度：`raw_alpha=1.18e-4`、`theta=4.73e-4`、`b=4.3e-10`；10 步后 `raw_alpha` 与 `theta` 均已变化。
- 冻结版：`frozen_branch_hash_unchanged=true`，融合层正常收到梯度并更新（`frozen_fusion_grad_abs_sum=8.32`）。
- **重要发现（b 参数不变性）**：8 个复数输入系数 `b` 的梯度 ~4e-10、10 步后数值不变。原因是归一化 lag-1 相关读出对每通道复缩放与全局相位不变——`b_k` 缩放 `q_k` 会在归一化相关中消掉，全局相位同样消掉。因此**实际可学习的只有 16 个参数（8 个衰减 `raw_alpha` + 8 个相位角 `theta`），16 个 `b` 停留在初始化值**。这不是 bug，而是该读出结构的固有对称性，作为 errata 记录。

## Check 4 — 数据 / 增广 / 统计一致，resume 一致
- 同一 seed 首次 batch 的样本 ID 与注入 CFO 完全一致（`first_batch_ids_identical=true`、`injected_cfo_identical=true`）。
- `fit_aux` 统计一致（`aux_ids_sha256=9dc6a50c…`）。
- resume 测试 `tests/test_global_qcorr_resume.py`：保存/恢复后逐位一致，且拒绝缺失 aux 统计的恢复（2 passed）。
