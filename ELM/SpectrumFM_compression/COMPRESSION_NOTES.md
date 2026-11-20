# 模型压缩学习存档

> 目标：以后对任意模型做量化/剪枝时，看这个文件就能复习方法论。
> 原则：学到哪记到哪，逐步补充。

## 1. 压缩实验的标准流程

1. **基线测量**（FP32）：效率四件套 + 精度 → 存 benchmark 表
2. **压缩**（量化 / 剪枝 / 蒸馏）
3. **精度评估**：重测任务指标（F1/Acc），和基线比损失
4. **效率复测**：同一测量脚本、同一口径
5. **对比**：每个配置一行追加进 benchmark.csv，最终画「压缩率 vs 精度」曲线

## 2. 评价指标四件套

| 维度 | 指标 | 测量要点 |
|---|---|---|
| 精度 | 任务指标（分类 = F1/Acc） | 和压缩前比「损失多少」是红线 |
| 体积 | 参数量 / 模型大小 / 压缩率 | 参数量 ≠ 大小（量化改大小不改参数量） |
| 计算量 | FLOPs | **实测为主**（FlopCounterMode）；论文公式口径单独记 |
| 速度 | batch=1 延迟（主）+ 大 batch 吞吐（辅） | cuda Event 计时 + warmup + 多次平均 |

已踩过的坑：
- 论文理论 FLOPs 常漏算「看起来小」的部分（1×1 卷积 = 全连接级计算量）→ 实测和论文口径分开记
- batch=1 延迟里 kernel launch 税占大头（小模型上可到 95%）→ 剪模块/算子融合的收益在 batch=1 延迟上最明显
- 显存峰值里参数只占小头（batch=256 时约 10%）→ 压缩参数对省显存帮助有限

## 3. FP16（2026-09-07 实验完成）

**做法（三步，顺序有讲究）**：
1. 构建 FP32 模型 → 加载 FP32 checkpoint（**先加载**）
2. `model = model.half()` 全模型转 FP16（**后转换**，顺序铁律）
3. 输入同步 `.half()`（最易漏，漏了报 `expected Half but found Float`）

**实测结果（SpectrumFM / A100）**：
| 指标 | FP32 | FP16 | 变化 |
|---|---|---|---|
| 模型大小 | 69.2MB | 34.7MB | -50% |
| FLOPs | 4.63G | 4.63G | 不变（与 dtype 无关） |
| batch=1 延迟 | 10.62ms | 10.10ms | -5% |
| 吞吐(b256) | 1790/s | 4701/s | +163% |
| 显存峰值(b256) | 644.8MB | 335.2MB | -48% |
| F1 | 0.6474 | 0.6475 | 零损失 |

**关键结论**：
- FP16 对分类任务是「几乎免费」的：体积减半、吞吐 ~2.6×、显存近减半、精度零损失
- batch=1 延迟只降 5%：被 kernel launch 税主导（~95% 是固定开销），FP16 只加速计算部分；**要降 b1 延迟得靠剪枝（减 kernel 数）**
- FLOPs 与 dtype 无关，量化实验前后不变（剪枝才会动 FLOPs）

**坑**：
- 输入忘了 half → dtype 报错（最高频）
- 顺序铁律：先 load FP32 再 half，避免 load_state_dict 的 dtype 转换歧义
- LayerNorm/softmax 内部自动 FP32、BN running stats 永远是 FP32 buffer——**是保精度设计不是 bug**；排查「意外 FP32」用 named_parameters 逐个查 dtype
- **模块级 parse_args 坑**：被 import 的模块绝不在顶层解析 sys.argv（会吞掉调用方参数）→ 用惰性解析修复

**FP16 训练 vs 推理**：
- 推理：无梯度，`.half()` 直接搞定
- 训练：梯度下溢（<6e-5 变 0）/更新吞没（大数+小数=没变）→ 必须 AMP 三件套（autocast + GradScaler + FP32 master weights）
- AMP 只绑定「FP16 训练」；INT8 量化用不到 AMP（PTQ 不训练；QAT 是浮点训练 + 假量化 + STE）

## 4. INT8 量化（2026-09-07 实验完成，CPU 口径）

**核心概念**：
- 量化 = 浮点范围线性映射到 INT8 的 256 个格子：`x_q = round(x/scale + zp)`；误差 ≤ 半个格子
- scale/zero_point 是每层一把「尺子」；校准 = 定桶宽（统计激活范围）
- zero_point = 浮点 0 在整数轴上的位置；对称量化 zp=0 固定，非对称按 min 对齐
- 动态量化：激活 scale 运行时 min/max 现算（每 batch）；静态量化：校准提前定死
- 量化收益在「大矩阵乘」模型（CNN/LLM）；小而碎的 attention+GRU 模型在 CPU 上收益小甚至为负

**实测结果（SpectrumFM，CPU，merge 后基线 64.1MB，F1 基线 0.6474）**：
| 配置 | 体积 | F1 | batch=1 延迟 |
|---|---|---|---|
| FP32（CPU） | 64.1MB | 0.6474 | 64.9ms |
| INT8 动态（全 Linear+GRU，未 merge） | 31.0MB（2.23×） | 0.6525（持平） | 67.8ms（0.93×） |
| INT8 手写静态（FFN+分类头 65 层） | 40.3MB（1.59×） | 0.6474（零损失） | 107.3ms（0.61×） |
| **merge 全动态**（merge + 全 Linear/GRU 动态） | **26.0MB（2.47×）** | 0.6504（持平） | 0.76×；**batch=256 则 2.43×**（并行负载趋势值） |
| **INT8 完全体**（+卷积手写静态） | **17.1MB（3.76×，逼近物理下限）** | 0.6510（零损失） | 138.5ms（0.36×，手写 int32 代价） |

**完全体的额外发现**：① 体积 17.1MB ≈ 物理下限 17MB——手写 int8 存储无动态量化 packed 格式的 padding 开销，反而最紧凑；② 卷积量化 + 动态量化叠加 F1 依然零损失——最终量化地图（卷积静态 + Linear/GRU 动态 + 敏感层 FP32）在精度上完全成立。

**动态量化**：一行 `quantize_dynamic(model, {nn.Linear, nn.GRU}, dtype=torch.qint8)`。替换时权重量化（from_float 里 quantize_per_tensor）；激活的 quant/dequant 藏在 C++ 算子 `quantized::linear_dynamic` 内部（每批现算 min/max → 量化 → int8 矩阵乘 → 反量化）。每层有往返开销 → 小矩阵模型变慢（实测 0.88~0.93×）。类名坑：DynamicQuantizedLinear 的 `__name__` 也是 "Linear"，靠体积变化确认量化生效。

**静态量化（手写 PTQ，教学版）**：流程 = merge LoRA → 替换目标 Linear → 校准（512 训练样本 + MinMaxObserver）→ finalize（定 scale/zp、量化权重）。公式：权重对称 int8（w_scale = max|w|/127）、激活非对称 quint8、int32 矩阵乘、反量化含 zero_point 修正项 `out = (qx@qw.T − zp·Σqw) × a_scale × w_scale + bias`。结果 **F1 零损失（0.6474 = 0.6474）**——手写量化数学正确。性能慢 1.6~6.4×（int32 matmul 无指令优化 + 每 forward 的 dtype 转换）——教学版验证精度，工程加速需 fbgemm/TensorRT。

**LoRA merge（任何推理/压缩前的标准预处理）**：`W_merged = W + (lora_A.T @ lora_B.T) × scaling`，精确等价，A/B 参数消失（省 5.2MB）。好处：结构标准化（量化器认识）、LoRA 知识连带进 INT8、比「量化 A/B」更彻底（量化只省 3.9MB，merge 全消）。

**踩坑清单**：
- 量化/浮点边界断层：量化 Linear 期望 quint8 输入，上游浮点输出直接流入 → NotImplementedError；每个量化模块要 QuantStub/DeQuantStub 配对（全模型量化自动配，局部量化手动配）
- torch 2.14 eager 量化已弃用，QuantStub 有 dispatch bug → 手写绕开；**新项目用 torchao**
- **对照实验对象混淆**：in-place 修改函数返回同一对象 → FP32 对照被量化污染 → 必须 `copy.deepcopy` 分离
- GRU 只能动态量化（激活循环依赖、范围随时间步漂移，静态校准统计不住）
- 校准数据必须有代表性（分布和推理时一致），否则 scale 定错、精度崩

**动态 vs 静态**：存储大小只由「量化范围 + 是否 merge」决定，动态/静态不影响体积。动态=无需校准/一行 API/RNN 唯一选择；静态=需校准/大矩阵乘模型加速好（2~4×）。混合地图是工程常态：GRU 动态 + FFN 静态 + attention 保持 FP32。

**体积账**（merge 后 64.1MB）：FFN 8.4M 参数→省 25MB；QKV 4.2M→省 12.6MB；GRU 0.79M→省 2.4MB；卷积 pointwise 3.15M→省 9.5MB；全量化工程值 ~18~20MB（17MB 物理下限 − bias 留 FP32 + 元数据的硬成本）。

**merge 全动态的关键发现**：① merge 白赚 5MB 实证（31→26MB，F1 持平）；② **batch=256 首次加速 2.43×**（旧动态 0.88×）——merge 消除 LoRALinear 碎片路径（「动态 Linear + 浮点 LoRA 项」的拼接），让 fused 算子 `linear_dynamic` 的整数算力在大 batch 下充分发挥；③ 「小 batch 亏、大 batch 赚」画像：batch=1 时 quant/dequant 往返开销主导（0.76×），batch 大时矩阵乘规模主导（2.43×）。

**完全体（量化地图最终版）**：卷积 pointwise1/2 → 手写静态（`ManualQuantConv1x1`，1×1 卷积=等价矩阵乘）；Linear/GRU → 动态；depthwise/BN/LN/相对位置 → FP32（太小或敏感）。设计动机：与 merge 全动态唯一差异 = 卷积量化与否，干净对比卷积的贡献。

## 5. 剪枝（2026-09-10 实验 A/B 完成，激进剪枝进行中）

**核心概念**：
- 两个正交决策：判据（magnitude L1 范数 / random 对照 / 敏感度）× 粒度（非结构化=置零不加速 → 必须结构化）
- 结构化剪枝实现 = **成对切片重建**：fc1 删行 + fc2 删列（共享编号空间、同一 keep、排序保序）——是「物理删除」（`w[keep]` 切片产生更小矩阵）不是「置零」
- 多层独立剪：每层是独立编号空间，行/列操作正交；路径 = 各层节点的组合，剪节点自然断路径（一个节点是路径网枢纽，剪 1 个节点断 512 条路径）
- 重要性判据在剪枝前、用完整模型一次性算好（剪完一层再算下一层会失真）
- 流程：裸剪直接评估（结构化剪枝无需重训即可前向）→ LoRA 微调恢复

**实测结果（FFN 中间维度剪枝，GPU，LoRA 微调 5 epochs）**：
| 配置 | 裸剪 F1 | 微调后 F1 | vs 基线 |
|---|---|---|---|
| magnitude 剪25% | 0.6429 | **0.6573** | **+0.99 点** |
| random 剪25% | 0.6439 | 0.6534 | +0.61 点 |
| magnitude 剪50% | 0.6102 | 0.6538 | +0.65 点 |
| random 剪50% | 0.6170 | 0.6526 | +0.52 点 |
| magnitude 剪60% | 0.5845 | 0.6542 | +0.68 点 |
| random 剪60% | 0.5874 | 0.6451 | -0.23 点 |
| magnitude 剪70% | 0.5447 | 0.6430 | -0.44 点 |
| random 剪70% | 0.5506 | 0.6524 | +0.50 点 |

**关键结论**：
- 剪 25% 裸剪几乎无损（-0.4 点）；剪 50%/60% 裸剪掉 3.7/6.3 点但微调后**完全恢复**
- **多 seed 配对验证（5 seeds × 4 配置，2026-09-11 完成）修正了单次结论**：
  - magnitude 与 random **无显著差异**（剪60% t=-0.97、剪70% t=-0.18）——单次的「判据反超/大胜」是随机假象。**教训：含训练的压缩实验必须多 seed 配对验证，单次 ±0.5 点的波动足以翻转结论**
  - **零代价剪枝上限至少 70%**（25%~70% 多 seed 均值全部 ≥ 基线 +0.25~0.55 点）
- 所有配置微调后 ≥ 基线：**剪枝的正则化效应**（多 seed 下方向稳健，幅度 +0.25~0.55 点）
- 裸剪时 magnitude ≈ random：预训练模型神经元重要性均匀、冗余度高 → L1 判据区分度低（对比：随机初始化大模型冗余度高，magnitude 优势明显）
- **实用结论**：本模型 FFN 宽度零代价剪枝上限至少 70%（省参数至 35%）；判据选择在这个模型上几乎不影响结果

**踩坑**：
- `encoder.layers[0]` 是 InputProjection 不是 ConformerEncoderLayer → 遍历剪枝时用 isinstance 跳过
- inject_lora 新建的 lora_A/B 默认在 CPU → 注入后必须再 `model.to(device)`
- in-place 修改函数的对象混淆 → 对照必须 deepcopy（与 INT8 章同一个坑）

## 6. QAT 量化感知训练（2026-09-11 完成）

**核心机制（一行代码的心脏，qat_modules.py）**：
```python
w_q = self.weight + (torch.round(self.weight / w_scale) - self.weight / w_scale).detach() * w_scale
# 前向：w_q = round(w/s)*s，值被"归到格子上"（真实量化误差进 loss）
# 反向：.detach() 掐断量化项梯度 → 梯度直通（STE：前向用 round，反向假装恒等，导数=1）
```
STE 为什么是 1：恒等函数是离 round 最近的可导函数（逐点误差 ≤0.5），且"量化误差零均值"→宏观斜率=1。

**与 PTQ 的区别**：PTQ 校准定死 scale 后推理；QAT 训练中 fake quant（权重 FP32 master + 前向模拟量化 + observer 移动平均更新激活 scale），训练后 finalize 冻结 → 真量化。
**与 LoRA 微调的区别**：QAT 要全量小 lr（2e-4）微调——"适应量化"需要主干参数动起来；无需 AMP（AMP 只绑定 FP16 训练，QAT 是浮点训练 + 假量化）。

**实测结果**：INT4-PTQ 0.6401（-0.73 点）→ **INT4-QAT 0.6494（+0.20 点，比 PTQ 好 0.93 点）**；组合：剪40%+QAT 0.6426（安全）、剪60%+QAT 0.6263（超界）。

**踩坑（全是血泪）**：
- **量化参数必须 `register_buffer`**：普通属性（`self._qweight = ...`）**不进 state_dict** → checkpoint 保存/加载后量化参数丢失 → 加载模型静默走 fake-quant 路径（速度/精度全错）。教训：保存前后验证 dtype/值，别信"能加载就没事"
- GPU 不支持 int32 矩阵乘（`baddbmm_cuda not implemented for 'Int'`）→ 推理态 matmul 用 float32（整数转浮点精确无损，量化误差来自"值离散化"而非算子 dtype）
- buffer 初始化要用最终形状（否则 load_state_dict size mismatch）；finalize 里的标量赋值都要包成 tensor
- 分流依据用 `finalize 状态`（`_a_scale.item() == 0`）而非 `self.training`——训练中评估（eval 模式）也要走 fake quant，否则未 finalize 时报 NoneType
- Conv1d 权重是 3D `[out,in,1]` → finalize 后 squeeze(-1) 保持 2D

## 7. 部署优化三件套（2026-09-12 完成）

**① 位打包（int4_packing.py）**：两个 4bit 值挤一个字节（`packed = (q[:,0::2]+8)<<4 | (q[:,1::2]+8)`），解包数值验证 ✅。**关键认知：部署版 = 丢弃 FP32 master weight + packed qweight**（master 只在训练时需要）——qat_int4 完整 79.8MB → 部署 **12.2MB**；剪40%+QAT → **10.5MB**（含 GRU FP32 3.1MB，再量化 GRU 可再省 2.3MB）。
**② torch.compile 算子融合**：b1 延迟 8.10→**5.55ms（-31%）**、抖动 ±1.95→±0.13ms、b256 +27%。**「launch 税」实证**：融合减 kernel 数量——唯一能显著改善 b1 的手段（量化/剪枝都动不了它）。
**③ torchao 官方 packed 算子（公正速度对比）**：
| 配置 | b1 | b256 吞吐 | 体积 | F1 |
|---|---|---|---|---|
| FP32 | 6.85ms | 2995/s | 64.1MB | 0.6474 |
| **BF16+编译** | 5.98ms | **9806/s** | 32.2MB | 0.6466 |
| INT8 权重级 | 6.70ms | 3016/s | 28.4MB | 0.6475 |
| BF16+INT4 packed | 11.85ms | 1151/s | 29.5MB | 0.6502 |
结论：**速度之王 = BF16+compile**（量化帮不上忙）；官方权重级量化无损但无速度增益（反量化路径）；INT4 packed 小矩阵下更慢（compile 也不支持，退 eager）但精度最高（激活保 BF16）。**手写 INT4 QAT+打包（10.5MB）在同范围下比官方（29.5MB）小**。
踩坑：torchao 0.18 API 是 Config 类风格（`Int8WeightOnlyConfig` 等）；INT4 需 mslk 内核（不可用）→ 用 `tile_packed_to_4d` 格式 + bfloat16；打包算子要在 CUDA 上执行（先 `.to(device)` 再 `quantize_`）。

**本模型压缩的最终图景**：体积最优 = 剪40%+INT4 QAT+打包（10.5MB / F1 0.6426）；速度最优 = BF16+compile（9806/s）；b1 最优 = 融合（-31%）。**没有任何单一方案全赢——「压缩手段 × 收益」各占一格**。

## 8. 压缩实验脚本清单（SpectrumFM_compression/code/）

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `eval.py` | 评估，输出 OA/P/R/F1 | `--half`：FP16 推理 |
| `profile_model.py` | GPU 效率测量 → 追加 benchmark.csv | `--name`（配置名）、`--ckpt`（权重路径）、`--f1`（该配置 F1）、`--half` |
| `convert_to_fp16.py` | FP32 → FP16 转换保存 | 无参数 |
| `int8_dynamic.py` | INT8 动态量化热身（31MB 版） | 无参数 |
| `int8_static.py` | 手写静态 PTQ（65 层版）；**提供 `merge_lora`/`measure_cpu_latency` 供复用** | 无参数 |
| `quick_compare.py` | 快速对照体积/延迟（随机数据校准） | 无参数 |
| `int8_dynamic_merged.py` | merge + 全动态（26MB 版） | 无参数 |
| `int8_full.py` | 完全体（卷积手写静态 + 动态，17.1MB 版） | 无参数 |
| `int4_ptq.py` | INT4 PTQ 基线（全手写 4bit + GRU 动态 INT8） | 无参数 |
| `int4_qat.py` | **INT4 QAT（fake quant + STE）** | `--epochs`（默认 5）、`--lr`（默认 2e-4）、**`--prune-ratio`：剪枝比例——0 = 纯 QAT；0.40 = 剪 40%；0.60 = 剪 60%（注意是「剪掉的比例」，对应 keep_ratio = 1 − 该值）** |
| `prune_ffn.py` | 剪枝实验 A（裸剪评估）；**提供 `prune_model(m, keep_ratio, mode, seed)`/`evaluate` 供复用** | 无参数（配置写在代码里） |
| `prune_finetune.py` | 剪枝实验 B（剪枝 + LoRA 微调恢复）；**提供 `finetune` 供复用** | `--epochs`、`--configs`（all/25/50） |
| `prune_aggressive.py` | 激进剪枝 60%/70% | 无参数 |
| `prune_multiseed.py` | 多 seed 配对验证（mean±std + 显著性），增量落盘 + 续跑 | `--seeds`（默认 5）、`--epochs` |
| `qat_modules.py` | QAT 核心模块（QuantAwareLinear/Conv1x1，fake quant + STE），供 int4_qat 使用 | — |
| `save_models.py` | 保存 4 个压缩模型落盘（prune40/prune60/qat_int4/qat_int4_prune40） | `--quick`（只存剪枝，秒级）、`--qat-epochs`、`--device` |
| `latency_curve.py` | 延迟-batch 曲线（原始结构，7 档） | `--half`（FP16 曲线）、`--ckpt` |
| `latency_curves_all.py` | **6 模型 × 7 档延迟大对比**（内置各压缩结构的重建加载逻辑） | 无参数 |
| `finetune_baseline.py` | 对照实验（未压缩模型同样微调，分离「训练 vs 压缩」的贡献） | 无参数 |
| `int4_packing.py` | INT4 位打包 + 解包验证 + 部署体积统计 | 无参数 |
| `torch_compile_test.py` | torch.compile 融合对比（b1/b256 前后） | 无参数 |
| `torchao_bench.py` | torchao 官方 packed 算子基准（FP32/BF16/INT8wo/INT4packed） | `--only`（只跑指定配置） |

> 复用关系：`int8_static`（merge_lora、measure_cpu_latency）和 `prune_ffn`（prune_model、evaluate）是基础模块，后续脚本都从它们 import——「工具脚本 = 旧积木 + 一个新 API」。
>
> 运行习惯（踩坑教训）：**任何训练脚本先 `--epochs 1` 冒烟（只看跑不跑通），通过后再跑正式版**；长任务 **tmux 挂后台并 `Ctrl+b d` 分离**（attached 状态下 VSCode 断网会向前台进程发中断；跑完**及时 kill** 会话避免残留干扰判断）；跨设备加载 checkpoint 用 `map_location="cpu"`；**新脚本统一加 `torch.save` 保存产物模型**（方便后续效率测量与复现推理——早期脚本只存了结果数据没存模型，导致补测 b256 时要重建重训）；**保存后的 checkpoint 要验证可加载可复现**（QAT 的量化参数丢失 bug 就是"能保存≠保存对了"的教训）。
