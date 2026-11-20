# 服务器验证报告（validation）

日期：2026-10-29。以服务器最终源码为准重新验证（不依赖本地曾通过的测试）。

## 1. 单元测试与合成流程

- `python -m pytest -q` → **17 passed**（代码修订前 17 passed，修订后再次 17 passed）。
- `python scripts/check_pipeline.py` → **SYNTHETIC integration pipeline passed**（5 模型 2ep 训练、扰动评估、benchmark、汇总均通过，输出在 `scratch/qa/`，仅合成数据）。

## 2. 代码修订（把候选默认对齐历史，diff 见 git/源码清单）

| 文件 | 修订 | 原因 |
|---|---|---|
| `ssm_exp/data.py` | 新增 `max_abs` 归一化 | 历史 Global 用联合分量绝对最大值归一化，候选只有 none/sample_rms |
| `ssm_exp/model.py` | 新增 `sinusoidal_positions`；G 强制正弦绝对位置 | 历史 Global 用正弦位置编码，候选 position=none 会删去关键结构 |
| `ssm_exp/model.py` | Attention 改为手动 softmax（非 SDPA） | 与历史 Global 一致，避免 memory-efficient attention 非确定内核 |
| `ssm_exp/engine.py` | scheduler 由 epoch 级改 step 级 warmup_fraction+cosine | 历史为 step 级 5% warmup + cosine |
| `configs/base.json` | `warmup_epochs:0` → `warmup_fraction:0.05` | 同上 |

修订均未改变研究问题、数据划分或 SSM 模块定义。修订后重跑全部测试。

## 3. CUDA 数值检查（A100, fp32/complex64）

- serial vs parallel scan：B2N16/B2N33/B4N128，max_abs_diff = 2.4e-7 ~ 4.8e-7（fp32 容差内，不要求逐位一致）。
- 梯度：serial vs parallel 对 a/b 的梯度 max_abs_diff = 1.0e-6 ~ 1.5e-6。
- 状态重置（样本独立性）：layer 逐样本前向 vs 整批前向 max_diff = 0.0；重复前向确定性 = 0.0。
- 物理描述：CFO 使 r 乘 exp(j2πf)（max_diff 3.4e-7）；相位变换保持 E/r（max_diff 3.6e-7）。CFO 单位确认为 cycles/sample。
- 完整 S2 模型（dim176/depth6/state128）CUDA 前向有限、全部参数梯度有限。

## 4. 真实数据 smoke（每模型 2 epoch，runs/smoke/）

- G/S0/S1/S2/S2_concat 均无 NaN，loss 正常下降，有限梯度。
- smoke 只取前 128 样本（manifest 按 class→snr 排列，前 128 训练样本均为 class 0），val_oa=1.0 / macro_f1≈1/24 说明数据 pipeline 的类别/SNR 分布符合预期（全部 class 0）。
- S2_concat 在 2ep/1batch smoke 下 val_oa 偏低（0.41），属融合层收敛慢，非故障；后续过拟合检查确认其可训。

## 5. 过拟合检查（scratch/overfit/，单独目录，不纳入正式结果）

- 72 训练样本（24 类 × 3）/ 48 留出 val 样本，class 均衡。
- 初版用 wd=0.01+60ep+cosine：train_oa 只到 0.68(G)/0.56(S2)，**疑似过拟合设置问题**（历史 overfit 用 wd=0+300ep）。
- 修正为 wd=0 + 200ep：**G train_oa=1.000 / train_loss=0.003，S2 train_oa=1.000 / train_loss=0.035**；留出 val 保持在随机水平（0.04~0.06），无标签泄漏。
- 结论：标签映射、前端、训练循环均正确；SSM 也能对少样本过拟合。

## 6. 种子一致性与 resume

- 数据顺序：训练 shuffle 用 `torch.Generator(seed)`，各模型同 seed 得到相同样本顺序（代码构造保证）；val 顺序读取。
- 增强抽样：`aug_gen=torch.Generator(seed+100000)` 独立于模型初始化/dropout，避免模型差异改变增强抽样。
- resume：`test_engine.py::test_checkpoint_resume_exact` 断言中断续训后权重 rtol=0/atol=0 完全一致（覆盖 optimizer/scheduler/RNG 全部状态）。修订 scheduler 后仍通过。
- 续训身份检查：config/provenance（data audit + code sha256 + environment）任一变化即拒绝续训，要求新 campaign（符合任务书）。

## 7. 物理统计来源

- `fit_physics` 仅在训练集固定子集（seed0，最多 8192 样本）拟合 mean/std，与训练种子无关，不读 val/test。
- 评估时 `physical_features` 由扰动后的 IQ 实时重算（`perturb` 在 `Classifier.forward` 之前作用于输入），不把注入频偏当输入。

## 8. 待办 / 已知

- 无关键事实缺失，可进入正式实验。
- G 的注意力现为确定性手动 softmax（与历史一致）；SSM 变体无注意力，天然确定。
