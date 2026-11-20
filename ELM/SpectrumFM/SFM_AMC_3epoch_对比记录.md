# SpectrumFM · AMC 3-epoch 对比实验记录

> 服务器（A100）vs 端侧（NVIDIA Thor）初步算力对比，双方均运行同一份 `amc.py --epochs 3` + `eval.py` + `profile_model.py`。

## 0. 一句话结论

- 精度几乎一致（F1 0.6427 vs 0.6425）→ **端侧复现正确**。
- 训练：服务器快约 **2.1–2.2×**（27.4 min vs ~58 min）。
- 推理：b1 延迟端侧慢 **2.2×**，b256 吞吐端侧慢 **4.5×**。
- FLOPs 与参数量两边**完全一致**（同模型同输入，与硬件无关）。

---

## 1. 模型规格（SFM / SpectrumFM）

- **论文**：SpectrumFM，IEEE JSAC 2025（arXiv 2505.06256），频谱基础模型。
- **思路**：先在大段无线电频谱语料上自监督预训练（掩码重建 + 下一 slot 预测）得到骨干，再在 AMC 等下游任务上做 LoRA 微调。

**微调架构 = ConformerClassifier**（`Model/model.py`）：

| 模块 | 规格 |
|---|---|
| 骨干 ConformerEncoder | 16 层，model_dim=256，4 注意力头，FFN 隐维 512 |
| 输入投影 | InputProjection: 2 → 256 |
| 双向 GRU | hidden=256，bidirectional → 输出 512 |
| Dropout | 0.2 |
| 分类头 | Linear(512 → 11) |

- **骨干权重**：`Checkpoint/pretrain_model_2018.pt`（62 MB，加载后冻结）。
- **LoRA 微调**：r=16，α=32，注入 query / key / value / output / fc1 / fc2；只训练 LoRA + GRU + 分类头。
  - 可训练 **2.11M / 总 18.06M**（约 11.7%）。
- **输入**：2×128（I/Q 经 `iq2ap` 转成幅度+相位，再 `normalize`）。

---

## 2. 测试任务（AMC 自动调制识别）

**数据集 RML2016.10A（Deepsig）**：

| 项 | 值 |
|---|---|
| 调制方式 | 11 种：8PSK、AM-DSB、AM-SSB、BPSK、CPFSK、GFSK、PAM4、QAM16、QAM64、QPSK、WBFM |
| SNR 范围 | -20 ~ +18 dB，步长 2 dB，共 20 档 |
| 样本数 | 每 (调制, SNR) 1000 条 → **220,000 条** |
| 单条形状 | 2×128（IQ） |

- **划分**：每个 (调制, SNR) 内部 `train_test_split(test_size=0.1, random_state=42)` → 训练 198,000 / 测试 22,000。
- **预处理**：`iq2ap`（I/Q → 幅度/相位）+ `normalize`。
- **数据文件**：`Data/RML2016.10a_dict.pkl`（611 MB），md5=`61bf35ac7f0b7d8843613453447ea290`。

---

## 3. 本次 3-epoch 训练配置

| 超参 | 值 |
|---|---|
| epochs | 3 |
| batch_size | 256 |
| lr | 1e-3 |
| 优化器 | Adam（只优化 requires_grad=True 的部分 = LoRA + GRU + 分类头）|
| 损失 | CrossEntropyLoss |
| EarlyStopping | patience=10（3 epoch 内不触发）|
| 权重输出 | `Checkpoint/amc/Our.pt` |
| 设备 | 服务器 A100（cuda:0）/ 端侧 Thor |

---

## 4. 结果对比

### 4.1 微调（训练）

| 口径 | 服务器（A100） | 端侧（Thor） | 倍数 |
|---|---|---|---|
| 3-epoch 纯训练+验证 | ~26.1 min | ~58 min（3×(18.5min+55s)） | 快 **2.2×** |
| 3-epoch 全流程（含分SNR+混淆矩阵） | ~27.4 min | ~59 min（同上 + ~1 min） | 快 **2.1×** |

> 说明：`amc.py` 在训练循环后会自动再跑 `test_per_snr()`（分SNR）和 `confusion()`（混淆矩阵），各约 30 s；两边都会跑，只是端侧报的 58 min 只算了训练+验证。

### 4.2 推理（`profile_model.py`）

| 指标 | 服务器（A100） | 端侧（Thor） | 倍数 |
|---|---|---|---|
| b1 延迟 | 10.49 ms | 23.34 ms | 端侧慢 **2.2×** |
| b256 吞吐 | 1827.5 sps | 409.5 sps | 端侧慢 **4.5×** |
| FLOPs/样本 | 4.63 G | 4.63 G | 一致 |
| 参数量 | 18.06 M | 18.06 M | 一致 |
| 有效算力（FLOPs×吞吐） | ~8.5 TFLOPS | ~1.9 TFLOPS | ~4.5× |

> FLOPs（浮点运算次数）由模型结构和输入尺寸决定，与跑在哪台设备无关，所以两边数值相同；区分的是有效吞吐（FLOPs/s，即 FLOPS），这才体现算力差异。

### 4.3 精度（3-epoch，复现验证）

| 指标 | 服务器（A100） | 端侧（Thor） |
|---|---|---|
| OA | 0.6215 | 0.6240 |
| P（加权） | 0.7310 | 0.7383 |
| R（加权） | 0.6215 | 0.6240 |
| F1（加权） | 0.6427 | 0.6425 |

---

## 5. 测量方式（端侧同款命令）

```bash
cd code
python amc.py --epochs 3            # 微调（自动含分SNR+混淆矩阵）
python eval.py                      # 算 OA/P/R/F1
python profile_model.py             # 参数/FLOPs/延迟/吞吐/显存
```

## 6. 环境

| | 服务器 | 端侧 |
|---|---|---|
| GPU | A100 80G | NVIDIA Thor（Jetson Blackwell sm_11.0，统一内存 122GB）|
| CUDA | cu126 | 13.0（JetPack 7.1）|
| torch | — | 2.14.1+cu130 |
