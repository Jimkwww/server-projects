# T2 可学习多尺度卷积前端 — 实现审计

> 目的：为下一轮实验决策提供**基于真实源码/配置/日志/checkpoint**的 T2 事实核查。
> 未启动任何新训练，未修改任何模型。
>
> 全文按三层区分证据强度，每条结论带标签：
> - **[事实]** = 直接来自源码、配置文件、日志或 checkpoint 的可核实内容。
> - **[判断]** = 由现有结果推导的结论（标注推导口径）。
> - **[未验证]** = 当前无法从仓库/日志确认，或需要额外实验才能确认的内容。
>
> 类名映射说明见 §4 顶部。

---

## 1. T0 与 T2 的真实结构

### 1.1 源码与版本

源码路径均为 `mtk/` 包内文件。本目录**不是 git 仓库**，版本凭据是每个 run 的
`metadata.json` 里记录的 `source_hashes`（每个 `.py` 的 sha256）。

**[事实]** Phase 1（`runs/sup_t2_s17`，原始筛选）与 Phase 2（`runs/p2_t2_s42`）之间的源码差异：

| 文件 | Phase1 hash（sup_t2_s17） | Phase2 hash（p2_t2_s42） | 变化 |
|---|---|---|---|
| `tokenizers.py` | `7c728ce9…049f5c` | `20eb2e10…920623` | 变（新增 `freeze_conv` 参数与 metadata 字段） |
| `model.py` | `b7ca09fe…2b65f4d` | `020fad5f…6ad057e9` | 变（新增 `pos_encoding` 分支） |
| `backbone.py` | `d8f3698a…68ce8cca` | `b260e69f…91d1f6c25` | 变（新增 `MetadataPosEmbed`、`forward(..., pos=None)`） |
| `data.py` | `ecb2f0b0…408f032` | 同 | 未变 |
| `train.py` | `aec0708e…53d61a023` | 同 | 未变 |
| `diagnostic.py` / `__init__.py` | 同 | 同 | 未变 |

**[判断]** 三处变化的性质：`freeze_conv` 默认 `False`、`pos_encoding` 默认 `sinusoidal`、`pos` 默认 `None`。对 `t0`/`t2` 这两个**未设置新字段**的 config，默认路径与改动前逐条等价（源码比对确认：T2 无 `freeze_conv` 时走原分支；`pos is None` 时 `GlobalBackbone` 走 `positions()` 原路径）。**[未验证]** 未对 Phase-2 的 T0 做过与冻结基线的逐位复验（唯一一次逐位复验是 Phase-1 的 `sup_t0_s2024` = 0.3869，发生在改动之前）。

### 1.2 T0 tokenizer 源码（`mtk/tokenizers.py:78-108`）

```python
class T0PatchTokenizer(nn.Module):
    """Flatten P consecutive I/Q samples (2P values) and project to D."""

    def __init__(self, patch_size=8, dim=176):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.proj = nn.Linear(2 * patch_size, dim)

    def forward(self, iq, valid):
        B, L, _ = iq.shape
        P = self.patch_size
        iq = iq.masked_fill(~valid[..., None], 0)
        pad = (-L) % P
        iq = F.pad(iq, (0, 0, 0, pad))
        valid = F.pad(valid, (0, pad), value=False)
        T = iq.shape[1] // P
        x = self.proj(iq.reshape(B, T, 2 * P))
        v = valid.reshape(B, T, P).any(-1)
        dev = x.device
        return TokenBatch(
            embeddings=x, valid_mask=v,
            time_center=_grid_centers(T, P, P, dev),
            time_support=_full_like(x, P),
            scale_id=_ones_long(T, dev, 0), band_id=_ones_long(T, dev, 0),
            metadata={"name": "t0", "patch_size": P, "token_count": T,
                      "projection_in_dim": 2 * P, "dim": self.dim})
```

### 1.3 T2 tokenizer 源码（`mtk/tokenizers.py:162-218`）

```python
class T2MultiscaleConv(nn.Module):
    """Three learnable 1-D conv branches with kernel lengths ~P, 2P, 4P, all at
    stride P on the same time grid. Branch outputs are concatenated (total
    channels == 2P) then mapped to D."""

    def __init__(self, patch_size=8, dim=176, kernels=(8, 16, 32), channels=(6, 5, 5),
                 freeze_conv=False):
        super().__init__()
        if sum(channels) != 2 * patch_size:
            raise ValueError("T2 total branch channels must equal 2P to match T0 projection width")
        if len(kernels) != len(channels):
            raise ValueError("kernels and channels must align")
        self.patch_size = patch_size
        self.dim = dim
        self.kernels = tuple(kernels)
        self.channels = tuple(channels)
        self.freeze_conv = freeze_conv
        self.convs = nn.ModuleList()
        for k, c in zip(kernels, channels):
            pad = (k - patch_size) // 2  # keeps output length == L / stride
            self.convs.append(nn.Conv1d(2, c, k, stride=patch_size, padding=pad, bias=False))
        if freeze_conv:
            for conv in self.convs:
                conv.requires_grad_(False)
        self.proj = nn.Linear(sum(channels), dim)

    def forward(self, iq, valid):
        B, L, _ = iq.shape
        P = self.patch_size
        dev = iq.device
        iq = iq.masked_fill(~valid[..., None], 0)
        pad = (-L) % P
        iq = F.pad(iq, (0, 0, 0, pad))
        valid = F.pad(valid, (0, pad), value=False)
        z = iq.transpose(1, 2)  # [B, 2, L]
        outs = [conv(z) for conv in self.convs]  # each [B, c, T]
        T = outs[0].shape[-1]
        cat = torch.cat(outs, dim=1).transpose(1, 2)  # [B, T, 2P]
        x = self.proj(cat)
        v = valid.reshape(B, -1, P).any(-1)
        support = float(max(self.kernels))
        return TokenBatch(
            embeddings=x, valid_mask=v,
            time_center=_grid_centers(T, P, P, dev),
            time_support=torch.full((T,), support, device=dev),
            scale_id=_ones_long(T, dev, 0), band_id=_ones_long(T, dev, 0),
            metadata={"name": "t2", "kernels": list(self.kernels),
                      "channels": list(self.channels), "stride": P,
                      "per_branch_support": list(self.kernels),
                      "token_count": T, "projection_in_dim": sum(self.channels),
                      "dim": self.dim, "learnable_frontend": True,
                      "freeze_conv": self.freeze_conv})
```

### 1.4 模型如何调用 tokenizer、加位置编码

`mtk/model.py:14-39`（`TokenClassifier`）：

```python
    def __init__(self, config, classes=24):
        ...
        self.tokenizer = build_tokenizer(config["tokenizer"])
        bb = config.get("backbone", {})
        self.backbone = GlobalBackbone(
            dim=bb.get("dim", 176), depth=bb.get("depth", 6),
            heads=bb.get("heads", 4), ratio=bb.get("ratio", 4),
            dropout=bb.get("dropout", 0.0))
        self.pos_encoding = bb.get("pos_encoding", "sinusoidal")
        if self.pos_encoding == "metadata":
            self.pos_embed = MetadataPosEmbed(
                dim=self.backbone.dim, num_bands=bb.get("num_bands", 1))
        ...
    def forward(self, iq, valid_mask):
        tb = self.tokenizer(iq, valid_mask)
        pos = None
        if self.pos_encoding == "metadata":
            pos = self.pos_embed(tb.time_center, tb.band_id)
        x = self.backbone(tb.embeddings, tb.valid_mask, pos=pos)
        pooled = x.sum(1) / tb.valid_mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)
```

位置编码在 `mtk/backbone.py:18-24`（`positions`）与 `:123-136`（`GlobalBackbone.forward`）：

```python
def positions(n, d, device, dtype):
    t = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000) / d))
    p = torch.zeros(n, d, device=device)
    p[:, 0::2] = torch.sin(t * freq)
    p[:, 1::2] = torch.cos(t * freq[:d // 2])
    return p.to(dtype)
```

```python
    def forward(self, embeddings, valid, pos=None):
        ...
        if pos is None:
            pos = positions(embeddings.shape[1], embeddings.shape[2], embeddings.device, embeddings.dtype)
        ...
        x = embeddings.masked_fill(~valid[..., None], 0)
        x = (x + pos[None]).masked_fill(~valid[..., None], 0)
        for block in self.blocks:
            x = block(x, valid)
        return self.norm(x).masked_fill(~valid[..., None], 0)
```

**[事实]** T0 与 T2 的 config（`mtk/configs/`）差异只有 `tokenizer` 块：

- `t0_baseline.json` → `"tokenizer": {"name": "t0", "patch_size": 8, "dim": 176}`
- `t2_multiscale.json` → `"tokenizer": {"name": "t2", "patch_size": 8, "dim": 176, "kernels": [8,16,32], "channels": [6,5,5]}`
- `t2_frozen.json` → 同 t2，另加 `"freeze_conv": true`

两者的 `backbone`（`{"dim":176,"depth":6,"heads":4}`）、`data`、`training` 完全一致。位置编码都是默认 `sinusoidal`。

### 1.5 结构对照表

**[事实]** 以下全部从源码与 `metadata.json`/`best_metrics.json` 核实。

| 项目 | T0 | T2 |
|---|---|---|
| 输入张量 | `[B, 1024, 2]`，最后一维是 (I, Q) 双通道 | 同左 |
| 通道排列 | `reshape(B,T,2P)` 后为**逐样本交错** `[I0,Q0,I1,Q1,…]` | `transpose(1,2)` 后 `[B,2,L]`，I、Q 作为 **2 个独立输入通道** |
| 是否复数运算 | 否，普通实值；无 I/Q 权重共享约束 | 否，普通实值 Conv1d；**I 与 Q 各有一套滤波器（不共享）** |
| 分支数 / 核长 | 1（等价核长 8 的线性投影） | 3（8 / 16 / 32） |
| 每分支 Conv1d 规格 | — | 见下表 |
| stride | —（切块） | 8（=P） |
| 激活 / 归一化 / 池化 / 残差 | 无 | **无**（conv 与 proj 之间、proj 之后均无） |
| 分支融合 | — | **沿通道 dim=1 拼接**（`cat`），非相加/加权 |
| 投影 | `Linear(16→176)` | 拼接后 `Linear(16→176)`（16=6+5+5=2P） |
| 最终 token 数 T | 128 | 128 |
| 嵌入维度 D | 176 | 176 |

T2 三个分支的 `nn.Conv1d` 参数 **[事实]**：

| 分支 | kernel | stride | padding=(k−P)//2 | dilation | groups | in→out | bias |
|---|---|---|---|---|---|---|---|
| 0 | 8 | 8 | 0 | 1 | 1 | 2→6 | False |
| 1 | 16 | 8 | 4 | 1 | 1 | 2→5 | False |
| 2 | 32 | 8 | 12 | 1 | 1 | 2→5 | False |

`padding=(k-P)//2` 是刻意设计：对 k∈{8,16,32}、P=8，k−P 全为偶数，使三支卷积输出长度**都**等于 L/P=128，且三支窗口**中心对齐**（见 §2）。

### 1.6 各步骤张量形状（L=1024，pad=0）

**[事实]** T2 forward 的逐步骤形状：

| 步骤 | 形状 |
|---|---|
| `iq`（mask 填充后） | `[B, 1024, 2]` |
| `z = iq.transpose(1,2)` | `[B, 2, 1024]` |
| `conv0(z)` | `[B, 6, 128]` |
| `conv1(z)` | `[B, 5, 128]` |
| `conv2(z)` | `[B, 5, 128]` |
| `cat(outs, dim=1)` | `[B, 16, 128]` |
| `.transpose(1,2)` | `[B, 128, 16]` |
| `proj(cat)` | `[B, 128, 176]` |
| `valid.reshape(B,-1,8).any(-1)` | `[B, 128]` |

### 1.7 参数量与「+576」的来历

**[事实]** 来自 `metadata.json` 的 `parameters` / `tokenizer_parameters`，并在源码上复算一致：

| 前端 | 组成 | 参数量 |
|---|---|---|
| T0 | `Linear(16→176)` = 16×176 + 176 | 2992 |
| T2 | conv 分支：6×2×8 + 5×2×16 + 5×2×32 = 96+160+320（bias=False） | 576 |
| T2 | `Linear(16→176)`（与 T0 同宽） | 2992 |
| T2 合计 | | 3568 |

- 全模型：T0 = 2,251,592；T2 = 2,252,168（tokenizer 差 576，其余 backbone/head 相同）。
- **为什么恰好 +576**：T2 只比 T0 多了三个卷积核的权重（bias=False），而拼接后的通道数被约束为 6+5+5=16=2P，与 T0 的投影输入宽度完全一致，故投影层参数量不变。576 = 3 个卷积分支权重之和，无 bias、无额外层。

---

## 2. 「多尺度」到底改变了什么

### 2.1 感受野与相邻 token 重叠

**[事实]** 卷积输出 token t 读取输入窗口 `[t·P − pad_b, t·P − pad_b + k_b)`：

| 分支 | 核长 k | padding | 窗口（t 号 token） | 感受野 | 相邻 token 重叠 |
|---|---|---|---|---|---|
| 0 | 8 | 0 | `[8t, 8t+8)` | 8 | 0（stride=核长） |
| 1 | 16 | 4 | `[8t−4, 8t+12)` | 16 | 8 样本（50%） |
| 2 | 32 | 12 | `[8t−12, 8t+20)` | 32 | 24 样本（75%） |

- 三支窗口**中心都落在 `8t+4`**，与 T0 的 `time_center = arange(T)*8 + 4` 一致。**[事实]**（源码 `pad=(k−P)//2` + `_grid_centers(T,P,P)` 双侧一致。）
- T0 是不重叠切块（每 token 8 样本，无重叠）；T2 的大核分支在相邻 token 之间**共享样本**（k=16 重叠 8，k=32 重叠 24）。

### 2.2 T2 相比 T0，除「尺度数量」外还改变了什么

**[事实]**（逐条对应源码）：

1. **最大感受野 8 → 32**（4 倍）。
2. **相邻 token 从无重叠 → 有重叠**（大核分支）。
3. **边界行为**：T0 切块恰好铺满 1024，不越界；T2 的 k=32 分支每个 token 在序列两端各多读 12 个**零填充**样本（`[−12,20)`、`[1012,1036)` 等）。卷积自带零填充，非信号真值。
4. **参数量 +576**（§1.7）。
5. **投影输入语义**：T0 投影输入是原始 I/Q 值；T2 投影输入是卷积滤波后的线性组合。
6. **可学习前端**：T2 比 T0 多一层可学习卷积（`learnable_frontend: true`）。

### 2.3 中间全线性，能否折叠为单个更大核卷积？

**[事实]** T2 前端为 `conv(无 bias) → 通道拼接 → Linear(有 bias)`，中间**没有任何非线性/归一化/池化/残差**（§1.5 已核实）。因此整条链是纯线性映射。

**结论（前向函数等价）：能，且已被数值验证。**

- 折叠条件（全部满足）：
  1. 各分支 stride 相同（=8）；
  2. 各分支 padding 满足 `(k−P)//2`（k−P 为偶数，中心对齐）；
  3. 中间无非线性；
  4. 边界零填充在折叠后的单卷积里行为一致（同 padding=12）。
- 折叠结果：T2 前端 ≡ **单个 `Conv1d(2→176, kernel=32, stride=8, padding=12, bias=True)`**。
  - 折叠核：`W_comb[d, cin, j] = Σ_{分支b,通道c} proj.weight[d, gc] · conv_b.weight[c, cin, kk]`，其中 `j = kk + 12 − pad_b`（把三支短核「对齐中心」后求和）。
  - 折叠偏置 = `proj.bias`。

**[事实]** 数值验证（本审计运行，未训练）：随机初始化 T2 与手写折叠单卷积，喂同一随机输入，输出最大绝对差 **5.96e-7**、相对 L2 **1.97e-7**（float32 舍入量级）。

### 2.4 区分「前向等价」与「参数化/初始化/训练等价」

**[判断]** 三者不能混为一谈：

- **前向函数等价（单向）**：成立（§2.3）。**任意一组 T2 权重都能折叠成一个 k=32 的单卷积**；但**反方向不成立**——任意一个 `Conv1d(2→176,k=32)` 不一定能表示成某个 T2（见下一条的约束）。
- **参数化不等价**：折叠后的单卷积核是**受约束**的。约束有两层，不能只叫「16 维线性子空间」：
  1. **秩 ≤ 16**：每个输出维 d 的核 `W_comb[d]`（展平为 64 维向量）是 16 条「分支通道核」经 `proj` 的线性组合，故其秩 ≤ 16；
  2. **时间支持受限**：这 16 条「通道核」各自只在所属分支的核长窗口内非零（6 条 ≤8 抽头、5 条 ≤16 抽头、5 条 ≤32 抽头），因此 `W_comb` 的不同位置区间由不同数量的通道参与（位置 `[12:20)` 有 3 组、`[8:12)∪[20:24)` 有 2 组、`[0:8)∪[24:32)` 只有 1 组）。
  两者叠加后，T2 可表达的核集是「秩约束 + 结构化时间支持」的**双线性簇**，不是线性子空间；而单个 `Conv1d(2→176,k=32)` 的核是任意 176×2×32 张量。
- **初始化与训练不等价**：T2 初始化的是 3 个短核 + 1 个 Linear；单卷积初始化的是 1 个 176×2×32 核。即使前向函数族相同，AdamW 的梯度流、隐式正则、收敛轨迹都不同。

**[事实]** 参数量上：折叠单卷积 176×2×32 + 176 = 11,440 参数，T2 前端只有 3,568。

### 2.5 如果「不能折叠」会是什么阻碍

**[判断]** 在本实现里折叠**成立**；但若出现下列任一情况，折叠将不成立（此处均不成立，仅为界定边界）：
- 分支间有非线性（激活/BN/池化/残差）；
- 各分支 stride 不同；
- padding 不满足中心对齐（k−P 为奇数，或 pad 取整引入半样本错位）；
- 分支融合用**加权/可学习**而非纯拼接（会引入跨分支非线性耦合，但仍可能可折为单卷积+额外线性结构）。

---

## 3. 三个种子的结果核实

### 3.1 run 路径、最佳 epoch、OA、macro-F1

**[事实]** 全部来自各 `runs/*/best_metrics.json`（`epoch`、`oa`、`macro_f1`）：

| run 路径 | tokenizer | seed | best epoch | val OA | val macro-F1 |
|---|---|---|---|---|---|
| `runs/sup_t0_s17` | t0 | 17 | 16 | 0.3933 | 0.3573 |
| `runs/p2_t0_s42` | t0 | 42 | 29 | 0.3837 | 0.3844 |
| `runs/p2_t0_s2026` | t0 | 2026 | 24 | 0.3837 | 0.3726 |
| `runs/sup_t0_s2024` | t0 | 2024 | 21 | 0.3869 | 0.3766 |
| `runs/sup_t2_s17` | t2 | 17 | 22 | 0.3942 | 0.3929 |
| `runs/p2_t2_s42` | t2 | 42 | 34 | 0.3913 | 0.3944 |
| `runs/p2_t2_s2026` | t2 | 2026 | 21 | 0.3968 | 0.3931 |
| `runs/p2_t2_frozen_s17` | t2(frozen) | 17 | 21 | 0.3760 | 0.3532 |

### 3.2 checkpoint 选择标准

**[事实]** `mtk/train.py:221-242`：每个 epoch 结束在验证集上 `improved = val["oa"] > best`（**严格大于**），只有 OA 破纪录才覆盖 `best.pt` 并写 `best_metrics.json = {"epoch": epoch, **val}`。**没有任何按 F1 选 checkpoint 的逻辑。**

**[规则]** **主评价规则 = 按验证 OA 选 checkpoint，保持不变。** 下文的 max-F1 峰值仅作补充观察，**不事后更换主规则**、不与主表混用。

### 3.3 OA 与 F1 是否来自同一个 checkpoint

**[事实]** 是——`best_metrics.json` 的 OA 与 F1 都来自「**OA 最高**」的那个 epoch（同一次 `evaluate`）。但这里有一个必须指出的口径问题：

**[事实]** 若改为按 macro-F1 选 checkpoint，结果会变（扫描 `metrics.jsonl` 得到）：

| run | OA-best ep | F1@OA-best | max F1 | max F1 的 epoch |
|---|---|---|---|---|
| sup_t0_s17 | 16 | 0.3573 | **0.3841** | 37 |
| p2_t0_s42 | 29 | 0.3844 | 0.3844 | 29 |
| p2_t0_s2026 | 24 | 0.3726 | **0.3845** | 28 |
| sup_t2_s17 | 22 | 0.3929 | 0.3929 | 22 |
| p2_t2_s42 | 34 | 0.3944 | 0.3944 | 34 |
| p2_t2_s2026 | 21 | 0.3931 | 0.3947 | 28 |

**[判断]** T2 的 OA-best 与 F1-best 几乎重合；T0 在 seed 17/2026 上 F1 峰值比 OA 峰值晚 ~12–21 个 epoch，OA-best 处的 F1 明显低于其 max-F1。因此**「T2 的 F1 高 +2.2 点」这个数依赖 OA-selection 口径**：若两边都改用 max-F1 口径，T0 F1 ≈ {0.3841, 0.3844, 0.3845}、T2 ≈ {0.3929, 0.3944, 0.3947}，差距收敛到约 **+0.97 点**。方向不变（T2 仍高），幅度被口径放大。**主规则不变（按 OA 选），本表仅为补充口径，不与主表混用。**

### 3.4 配对差值、均值与标准差（口径声明）

**[事实]** 只用同一种子的配对（seed∈{17,42,2026}），std 为**样本标准差（ddof=1，N=3）**：

| 指标 | 配对差值（T2−T0） | 均值 | std |
|---|---|---|---|
| OA | s17 +0.0010；s42 +0.0077；s2026 +0.0131 | **+0.0073** | 0.0050 |
| macro-F1 | s17 +0.0356；s42 +0.0099；s2026 +0.0205 | **+0.0220** | 0.0106 |

- T0 OA：{0.3933, 0.3837, 0.3837} → 0.3869 ± 0.0055
- T2 OA：{0.3942, 0.3913, 0.3968} → 0.3941 ± 0.0028
- T0 F1：{0.3573, 0.3844, 0.3726} → 0.3714 ± 0.0111
- T2 F1：{0.3929, 0.3944, 0.3931} → 0.3934 ± 0.0006

**[判断]** OA 增益小（均值 +0.7 点，跨种子在 +0.1~+1.3 点之间），F1 增益在三种子下**方向一致为正**（+1.0~+3.6 点），且 T2 的 F1 跨种子方差极小（0.0006）。不放大为「显著优于」，仅陈述为「F1 三种子一致为正、幅度受选择口径影响」。

### 3.5 T2 frozen 究竟冻结了什么

**[事实]**（源码 §1.3 + `p2_t2_frozen_s17/best.pt` 数值核对）：

- 冻结：`tokenizer.convs.0/1/2.weight`（3 个卷积，共 576 参数），`requires_grad=False`。
- 未冻结、仍可训练：`tokenizer.proj.weight/bias`（16→176，2992 参数）、全部 backbone、head。
- **初始权重与可学习 T2 是否相同**：是。数值核对（本审计，未训练）：seed 17 下 `T2MultiscaleConv()` 与 `T2MultiscaleConv(freeze_conv=True)` 的三支卷积初始化逐位相等（构造顺序相同、torch 默认初始化消耗 RNG 顺序一致）。
- **冻结后是否真的没动**：是。`p2_t2_frozen_s17/best.pt` 的三支卷积权重与 seed-17 初始化逐位相等；而 `sup_t2_s17/best.pt` 的三支卷积权重与初始化不相等（说明可学习版确实训练了卷积）。

**[事实]** 注意：`metadata.json` 的 `parameters` 仍计入冻结的 576 参数（torch 的 `parameters()` 不含 requires_grad 过滤），所以 frozen 版参数统计仍为 2,252,168；但 AdamW 对 `grad is None` 的参数直接 `continue`，**不更新、也不施加 weight decay**，故冻结权重严格保持初始值。

**[判断 + 未验证]** `t2_frozen` seed17（OA 0.3760 / F1 0.3532）低于同种子 T0（0.3933/0.3573）与可学习 T2（0.3942/0.3929）。这**与「收益来自可学习性而非固定多尺度滤波」一致**，但这是**单种子、单一冻结方案**的结果，且冻结方案本身改变了优化轨迹（少 576 个可训练参数、不同的解空间）；**据此不足以得出「收益来自学习性」的结论**，需要 §5 的单尺度可学习对照来区分。

---

## 4. 用已有结果定位收益

> 类名：仓库、manifest、h5 **均不存**类别名（manifest 只有 `class` 整数列，h5 无属性）。下文类名采用 Deepsig RML2018.01A 的标准 24 类顺序，**[未验证]**（未从仓库内一手来源确认；但与混淆矩阵的族结构一致：索引 0–2 为 ASK/OOK 易分类、5–7 为 PSK 高混淆族、3↔20 呈现教科书式 BPSK↔AM-DSB-SC 混淆、17/18 为 SSB 对、19/20 为 DSB 对）。逐类结论**以索引为准**，类名仅供阅读。

**标准顺序（未验证）**：`0 OOK, 1 4ASK, 2 8ASK, 3 BPSK, 4 QPSK, 5 8PSK, 6 16PSK, 7 32PSK, 8 16APSK, 9 32APSK, 10 64APSK, 11 128APSK, 12 16QAM, 13 32QAM, 14 64QAM, 15 128QAM, 16 256QAM, 17 AM-SSB-WC, 18 AM-SSB-SC, 19 AM-DSB-WC, 20 AM-DSB-SC, 21 FM, 22 GMSK, 23 OQPSK`。

### 4.1 关键前置事实：类别完全均衡

**[事实]** 验证集 3120 样本 = 24 类 × 130（每 (类,SNR) 5 样本）。**所有类别的样本数相等**，不存在「少数类」。因此 T2 的 macro-F1 提升**不可能来自少数类改善或类别再平衡**——macro-F1 就是 24 类 F1 的简单平均，纯由逐类 F1 变化驱动。

### 4.2 逐类 F1（三种子，T2−T0）

**[事实]** 由各 run `best_metrics.json` 的混淆矩阵计算（precision/recall/F1，OA-selection 口径；`#wins` = 三种子中 T2 F1 > T0 F1 的次数）：

| idx | 类名(未验证) | T0 F1（17/42/26） | T2 F1（17/42/26） | T2−T0 均值 | #wins |
|---|---|---|---|---|---|
| 0 | OOK | .519/.727/.662 | .732/.736/.755 | **+0.105** | 3/3 |
| 1 | 4ASK | .617/.590/.577 | .578/.614/.556 | −0.012 | 1/3 |
| 2 | 8ASK | .687/.612/.602 | .598/.595/.601 | **−0.036** | 0/3 |
| 3 | BPSK | .504/.573/.595 | .714/.697/.711 | **+0.150** | 3/3 |
| 4 | QPSK | .669/.563/.532 | .558/.642/.576 | +0.004 | 2/3 |
| 5 | 8PSK | .370/.177/.191 | .218/.259/.361 | +0.033 | 2/3 |
| 6 | 16PSK | .029/.059/.259 | .294/.229/.195 | **+0.124** | 2/3 |
| 7 | 32PSK | .026/.329/.325 | .190/.243/.272 | +0.009 | 1/3 |
| 8 | 16APSK | .450/.429/.404 | .404/.356/.468 | −0.019 | 1/3 |
| 9 | 32APSK | .291/.261/.302 | .271/.183/.302 | −0.032 | 1/3 |
| 10 | 64APSK | .192/.145/.150 | .176/.153/.196 | +0.013 | 2/3 |
| 11 | 128APSK | .264/.143/.200 | .173/.166/.106 | **−0.054** | 1/3 |
| 12 | 16QAM | .382/.143/.228 | .245/.224/.179 | **−0.035** | 1/3 |
| 13 | 32QAM | .029/.199/.093 | .104/.128/.072 | −0.005 | 1/3 |
| 14 | 64QAM | .105/.154/.165 | .169/.153/.175 | +0.024 | 2/3 |
| 15 | 128QAM | .139/.144/.074 | .115/.152/.193 | +0.034 | 2/3 |
| 16 | 256QAM | .157/.119/.170 | .158/.103/.142 | −0.014 | 1/3 |
| 17 | AM-SSB-WC | .556/.463/.275 | .146/.400/.337 | **−0.137** | 1/3 |
| 18 | AM-SSB-SC | .026/.417/.439 | .508/.439/.455 | **+0.173** | 3/3 |
| 19 | AM-DSB-WC | .500/.543/.453 | .534/.543/.489 | +0.023 | 3/3 |
| 20 | AM-DSB-SC | .353/.437/.358 | .494/.463/.385 | **+0.064** | 3/3 |
| 21 | FM | .676/.744/.636 | .807/.748/.749 | **+0.083** | 3/3 |
| 22 | GMSK | .603/.678/.638 | .643/.680/.558 | −0.013 | 2/3 |
| 23 | OQPSK | .431/.579/.615 | .603/.560/.599 | +0.046 | 1/3 |

**[判断]** 三种子一致改善（3/3）的类：**idx 0(OOK)、3(BPSK)、18(AM-SSB-SC)、19(AM-DSB-WC)、20(AM-DSB-SC)、21(FM)**，另有 6(16PSK) 2/3 大幅改善。三种子一致退化（0~1/3）的类：**idx 2(8ASK)、11(128APSK)、12(16QAM)、17(AM-SSB-WC)**。

**[判断]** F1 提升**不是均匀的**，而是集中在少数类：
- 大头贡献：18(SSB-SC) +0.173、3(BPSK) +0.150、6(16PSK) +0.124、0(OOK) +0.105、21(FM) +0.083、20(DSB-SC) +0.064。
- 明显被拉低：17(SSB-WC) −0.137、2(8ASK) −0.036、11(128APSK) −0.054、12(16QAM) −0.035。注意：idx 17/18 是同类调制对（名称未核实），一升一降**不能单凭 F1 反推发生了 17→18 的预测转移**（F1 一升一降也可能是两者各自与第三类的混淆变化导致）；若要确证转移，需看两者之间的 off-diagonal 混淆单元，本审计未逐单元展开，故不下「再分配」结论。
- 教科书级难对 **3(BPSK)↔20(AM-DSB-SC)** 在 T2 下双向改善（+0.150 / +0.064）。

### 4.3 逐 SNR 表现

**[事实]** 各 run `best_metrics.json` 的 `per_snr`，三种子取均值：

| 频段 | T0 均值 | T2 均值 | T2−T0 |
|---|---|---|---|
| 低 SNR（≤0 dB，11 档） | 0.1356 | 0.1460 | **+0.0104** |
| 高 SNR（>0 dB，15 档） | 0.5711 | 0.5761 | **+0.0050** |

**[判断]** T2 相对 T0 的 OA 增益在低/高 SNR 两段**都很小**（各 +1.0 / +0.5 点），逐档差值在 ±0.05 内波动、无单调规律。**不支持「T2 主要在低 SNR 改善」**，也与 §2 频域前端「低 SNR 好、高 SNR 崩」的模式不同——T2 是全局小幅抬升，不是低 SNR 专项收益。

### 4.4 解释的区分（不凭宏平均猜测）

**[判断]** 基于 §4.1–4.3 的事实，可排除/支持的解读：

- **排除「类别更均衡」**：类别本就完全均衡（§4.1），无此效应。
- **排除「少数类改善」**：无少数类。
- **排除「低 SNR 改善」**：§4.3 显示低 SNR 段几乎无差异。
- **支持「特定类别改善」**：F1 增益集中在 idx {0,3,6,18,19,20,21}，且被 idx {2,11,12,17} 的退化部分抵消。
- **不能凭一升一降证明预测转移**：真实类别均衡不代表预测分布均衡；两个类别的 F1 一升一降不足以单独证明二者之间发生了预测转移（需 off-diagonal 混淆单元佐证，本审计未展开）。
- **[未验证]** 这些类改善是否对应「更大感受野捕捉了符号/成形脉冲时间结构」——这是解释，无对照实验，不能下结论。§5 的单尺度对照正是为区分「感受野 32」与「三支分解/参数化」而设计。

---

## 5. 计算成本与下一轮最小对照

### 5.1 已有成本数据（含口径）

**[事实]** 来自各 `metadata.json` 与 `metrics.jsonl`；均为 **A100-SXM4-40GB、fp32、batch=128、workers=2、cls_dev train/val**，测量的是**全模型**（tokenizer + backbone + head）的训练吞吐与峰值显存，**不含**前端单独的 FLOPs/延迟剖分。

| 前端 | 参数量 | 峰值显存(MB) | 训练吞吐(样本/s) |
|---|---|---|---|
| T0 | 2,251,592 | 1478 | 2620–2633（均值，3 run） |
| T2 | 2,252,168 | 1479 | 2621–2622（均值，Phase2 的 2 run） |
| T2 frozen | 2,252,168 | 1478 | 2634 |

**[事实]** 口径陷阱：
- `sup_t2_s17`（Phase 1）的吞吐均值仅 1551 样本/s（中位 1351、区间 1321–2590，双峰）——该 run 跑在 Phase-1 多卡并行、机器被 `user_cx`/`user_gh` 争抢的时段，**不可作为 T2 的成本读数**。Phase-2 在空闲 GPU 1 上重跑的 T2（`p2_t2_s42`/`s2026`）吞吐 2621–2622 样本/s，与 T0 几乎相同。
- **参数量增幅 ≠ 计算增幅**：T2 只多 576 参数（+0.026%），三支小卷积的 FLOPs 相对 6 层全局注意力可忽略，实测吞吐/显存与 T0 无差异。
- **缺失**：前端单独的前向/反向延迟与 FLOPs 未单独测量；端到端延迟未单独测（`train_samples_s` 含反向与数据加载）。

### 5.2 下一轮最多两个单尺度可学习卷积控制（只提配置，不训练）

> 基于 §2 的折叠结论，T2 本质上等于「一个 k=32 的卷积前端 + 一个受约束的 16→176 参数化」。因此两个控制都用**单一核长 32**（=T2 最大感受野），stride=8、T=128、D=176、同一时间网格、同一 padding=12（中心对齐），仅改变「如何从 32 抽头映射到 D」。

**控制 A —— 匹配最大感受野 + 输出时间网格（最简单单卷积）**

- 结构：`Conv1d(2 → 176, kernel=32, stride=8, padding=12, bias=True)`，输出直接是 D=176，无投影层。
- 前端参数：176×2×32 + 176 = **11,440**（比 T2 的 3,568 大）。
- 控制了什么：把 T2 的「三支分解 + 16→176 投影」替换为「一个 32 抽头卷积直出 D」；感受野、网格、中心、边界与 T2 完全一致。
- 仍存在的差异（不可直接归因的点）：核无约束（全秩）、参数量更大、初始化/训练轨迹不同、**没有 16→176 投影瓶颈**。
- 若 A ≈ T2：收益只来自「32 抽头感受野」，三支分解与投影瓶颈不必要；若 A ≈ T0：说明收益依赖 T2 的受约束参数化或投影，而非单纯大感受野。

**控制 B —— 匹配参数/计算预算（单卷积 + 同一投影瓶颈）**

- 结构：`Conv1d(2 → 16, kernel=32, stride=8, padding=12, bias=False)` + `Linear(16 → 176, bias=True)`。
- 前端参数：16×2×32 + (16×176+176) = 1024 + 2992 = **4,016**（最接近 T2 的 3,568；conv 部分 1024 vs T2 三支合计 576，差 448）。
- 控制了什么：保留与 T0/T2 完全相同的 `Linear(16→176)` 投影，只把「三支 (8,16,32) 卷积」换成「单支 32 抽头卷积」。**B 仍保留秩 ≤ 16 的瓶颈**（16 通道经 176 维投影）；它放开的只是**短/中核的时间支持范围**——16 条通道核从「各自限于 8/16/32 抽头」变为「全部可跨满 32 抽头」。即 B 与 T2 的差异在**时间支持约束**，不在低秩。
- 仍存在的差异：conv 是单核全秩初始化（vs T2 三支短核各自初始化后经 proj 混合）；conv 参数 1024 vs 576；同一函数（T2 可折叠出的核）在 B 里是「16 条通道核都满 32 抽头」的参数化，在 T2 里是「三条分支窗口受限」的参数化。
- 若 B ≈ T2：说明在本协议下「单尺度大核 + 16 维瓶颈」已能达到相近效果，短/中/长三种时间支持并非必要；若 B ≈ T0（而 T2 更好）：说明 T2 的三支短核分解/初始化本身（隐式正则）是关键，值得保留。

**[未验证]** 两个控制均**未实现、未训练**。实现需要新增一个单尺度卷积 tokenizer（或对 `build_tokenizer` 增加一个通用 `single_conv` 后端）；此处只给规格，不含任何训练启动。核长搜索、注意力、小波、MAE 等扩展均不在本控制范围内。

---

## 附：事实 / 判断 / 未验证 一览（摘要）

- **[事实]** T2 前端 = 3 支 `Conv1d(2→{6,5,5}, k={8,16,32}, stride=8, pad=(k−8)//2, bias=False)` 拼接后 `Linear(16→176)`；无激活/归一化/池化/残差；I/Q 为普通双通道实值卷积、不共享；T=128、D=176、中心对齐于 `8t+4`；参数 3,568（=T0 的 2,992 + 576）。
- **[事实]** T2 前端前向等价于单个 `Conv1d(2→176,k=32,stride=8,pad=12,bias=True)`（数值验证 max 差 5.96e-7，单向成立：任意 T2 可折叠、任意单卷积未必能写成 T2），但参数化/初始化/训练不等价（核集是「秩≤16 + 时间支持受限」的双线性簇、11,440 vs 3,568 参数）。
- **[事实]** checkpoint 按 val OA 严格更大选择；OA/F1 同 checkpoint；T0 的 F1 峰值晚于 OA 峰值（口径影响 F1 差距 2.2→约 0.97）。
- **[事实]** T2 frozen 冻结的正是三支卷积（576 参数），初始权重与可学习 T2 seed17 逐位一致、冻结后从未变化；proj/backbone/head 仍训练。
- **[事实]** 类别完全均衡（每类 130），T2 的 F1 增益集中在 idx {0,3,6,18,19,20,21}，被 idx {2,11,12,17} 抵消一部分；低/高 SNR 段 OA 增益都很小。
- **[判断]** T2 的 F1 三种子一致为正、OA 增益小；frozen 结果与「收益依赖可学习性」一致但不足以定论。
- **[未验证]** 类名映射（Deepsig 标准顺序，仓库无一手来源）；「大感受野捕捉时间结构」这一解释；两个单尺度控制的实际表现。
