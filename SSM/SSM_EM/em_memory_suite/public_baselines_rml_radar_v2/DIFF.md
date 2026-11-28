# v2 修正差异清单（DIFF）

> 保留全部现有训练结果；仅补跑受 Q seed 固定 bug 影响的运行，不重跑其余模型。
> 以下逐条说明原问题、影响范围、修复方式与修正后的数据。

## 1. 吞吐单位错误（影响：效率表）

- **原问题**：`efficiency.py` 的 `bench_throughput` 返回 `n/dt`，其中 `n` 是 forward 次数（每次
  forward 处理 batch=64 样本），故 `n/dt` 是 **batches/s**，报告却标注为 `samples/s`，少乘了 batch 64。
- **影响**：report v2 效率表 "吞吐(/s)" 列数值实际是 batches/s（如 Q RML "40" 实为 40 batches/s ≈ 2560 samples/s）。
- **修复**：`samples/s = n/dt × batch`，并明确标注 `throughput_batch64_samples_s`。
- **修正后**（见 report 效率表）：各模型吞吐 = 原值 ×64 量级。

## 2. 样本标准差用总体标准差（影响：三 seed 均值表）

- **原问题**：`analyze.py` 用 `np.std`（默认 `ddof=0`，总体标准差），N=3 时应为样本标准差。
- **修复**：改为 `np.std(x, ddof=1)`。三 seed 的 ±std 相应变大（×√(3/2)≈1.22）。

## 3. 数据集子集名称标注不准确（影响：报告表述）

- **原问题**：报告泛写 "RML" / "radar"，未标注准确版本与划分术语。
- **修正**：
  - RML = **RML2018.01A（Deepsig 2018.01a）cls_dev 划分**（24 类，train 19968/val 3120/test 8112）。
  - radar = **RadChar-Tiny**（5 类，train 30000/dev 10000/eval 10000）——本数据集 val 即 dev、test 即 eval。

## 4. WavesFM token 数漏计 cls token（影响：效率表 token 列）

- **原问题**：WavesFM 用 `cls_token=True` + `global_pool='token'`，实际输入序列含 1 个 cls token；
  报告写 token 64/32，应为 **65/33**（RML/radar）。
- **修复**：效率脚本实测 token 数（EMind `(L+4)//4` 已含 cls，WavesFM `L//16+1`，structured/Q 实测）。

## 5. Q 构造函数固定 seed=17（影响：Q 的 seed42/2026 三 seed 运行）

- **原问题**：`train.py` 里 `Encoder(cfg, 'Q', seed=17)` 硬编码 seed；而 `ComplexRecurBranch.__init__`
  用 `torch.Generator().manual_seed(seed)` 独立初始化 `b`（8 复数=16 实数）。结果 seed42/2026 时
  `aux_branch.b` 被错误固定为 seed17 的初始值，只有 front/blocks/norm 随 --seed 变化。
  （已验证：`b(seed42, 硬编码17) == b(seed17)`；`b(seed42, 正确42) != b(seed42, 硬编码17)`。）
- **影响范围**：仅 Q 的 **seed42 与 seed2026** 共 4 组（rml/radar）。seed17 的 Q 运行（v1 八组 +
  lr/预算候选）不受影响（seed17 时硬编码 17 与传 17 等价）。受影响参数仅 16 个实数（占 Q 10.46M 的 ~0.0002%），
  但仍违反"各 seed 只改变初始化随机性"的协议。
- **修复**：`Encoder(cfg,'Q',seed=seed)`（传入 --seed），**补跑 Q seed42/2026 共 4 组**，覆盖原错误产物。
  其余模型（structured/EMind/WavesFM）及 seed17 全部复用，不重跑。
- **修正后**（见 report 三 seed 表）：Q 的 seed42/2026 原始 OA 更新。

## 6. 效率脚本循环内内存未彻底释放（影响：峰值显存测量）

- **原问题**：`efficiency.py` 循环内 `del model` 后，`fwd` 闭包仍持有 model 引用，`empty_cache()`
  时显存未真正释放，峰值 allocated/reserved 可能混入上一模型的残留。
- **修复**：改为**每模型独立进程**测量（进程退出即彻底释放 CUDA），并分别记录 batch1 与 batch64 的
  峰值显存；预热/重复次数加大（warmup 10、latency 50 次、吞吐 8s）。

## 7. 报告结论过强（影响：措辞）

- **原问题**：结论措辞易被读成 "SSM 普遍优于 ViT" 的泛化。
- **修正**：明确限定"在本协议、这两个 SSM 类实现 vs 这两个 ViT 类实现、3 seed 下"的实测排名；
  3 seed 为初步稳定性判断而非统计显著；不把 structured 领先解释为"物理分支有效"。
