# Round 5 Review 实现验收材料（Section 二）

补充原始契约 [implementation_contract.md](../round5_iq/implementation_contract.md) 的实际数值证据。
详细数值见 `evidence.json`、`smoke_overfit.json`、`test_log.txt`。

## 1. 测试验收

- 测试文件：`tests/test_round5.py`（36 项断言）+ `tests/test_round5_resume.py`（2 项）。
  已随验收材料打包到 `reports/round5_review/tests/`。
- 环境：conda SFM，Python 3.11.16，torch 2.9.1 (cu12.6)，numpy 2.4.6，NVIDIA A100-PCIE-40GB。
- 命令与完整通过日志：`reports/round5_review/test_log.txt`。
- 结果：`tests/test_round5.py` 全绿；`tests/test_round5_resume.py` 全绿；全量回归 **75 passed**。

## 2. Smoke 过拟合（契约 #19，由「待执行」→ 已执行）

8 个变体（R0/C0/Rb/Cb/Q_linear/Q_corr/R_corr/Q_corr_frozen）各 `--smoke --epochs 5`（seed 17，128 样本），
全部无 NaN、train loss 下降、val OA→1.0。汇总见 `smoke_overfit.json`。
其中 Q_corr_frozen 的 `trainable=2219752` / `total=2219784`（冻结支路 32 实数参数），
与 Q_corr（total=2219784，无冻结）一致。

## 3. 初始化输出一致性 / 主干权重一致性

- **B 线 Q_corr_frozen ≡ Q_corr（初始）**：状态序列、描述符、logits 逐位一致（max|Δ|=0，`bit_identical=true`），
  证明冻结变体与原变体共享同一初始状态。
- **A 线主干权重**：R0/C0 的 blocks/norm/head 共 40 个 key 与 S1 逐位一致（`all_r0_equal_s1`、
  `all_c0_equal_s1` 均为 true），证明 matched init 只替换前端、主干共享。

## 4. 第一步更新梯度到支路（契约 #13）

| 步 | W_f 梯度 | 支路 raw_alpha / theta / b 梯度 |
|---|---|---|
| 第 1 步 | 3.712（非零） | 0 / 0 / 0（零初始化融合层的数学必然，非 detach） |
| 第 2 步 | 3.362 | 0.00199 / 0.0291 / 4.48e-7（非零，梯度到达支路） |

## 5. 前端等效权重 / 输出 scale

- `round5_front_weights` 生成 `W_R,W_I ~ U(−1/√(2·in), 1/√(2·in))`，展开实矩阵 `M` 元素方差实测
  **0.0209 ≈ 1/48 = 0.0208**，与 `nn.Linear(16,176)` 默认初始化方差 0.0212 匹配（非 Round-4 的 1/16=3× 失配）。
- 复数前端与等效实数前端在相同输入下输出方差一致（0.3355 vs 0.3355），max|Δ| ≈ 7.2e-7（fp 舍入）。

## 6. 数据 ID / CFO 随机序列记录

训练前已冻结并记录在 `reports/round5_iq/input_audit.json`：3 个 seed × 3 batch，每 batch 128 条
样本 ID 序列与 `ids_cfo_sha256`，证明同 seed 下各变体配对输入与 CFO 增广流一致。
契约 #17 的 `test_data_and_aug_stream_independent_of_model_build` 亦逐位验证了这一点。
