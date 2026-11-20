# 基线核验（baseline_audit）

版本 v1.0；日期 2026-09-21。对应 `01_multiscale_tokenization_experiment.md` 第 3 节「开始前必须完成的基线核验」。所有数字从实际配置/日志/数据确认，非历史聊天记录。

## 1. 数据集

| 项 | 值 |
|---|---|
| 数据集 | RML2018A（`GOLD_XYZ_OSC.0001_1024.hdf5`） |
| 路径 | `/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5`（21449148312 B） |
| X 形状 | `(2555904, 1024, 2)` float32，`x[...,0]`=I、`x[...,1]`=Q |
| 输入长度 N | 1024（所有样本定长，无变长/padding） |
| 类别 | 24 类（class 0..23，one-hot Y 与 manifest 一致） |
| SNR | 26 档，-20..30 dB（步长 2），manifest 与 h5 的 Z 一致 |
| 划分 | `cls_dev`：train 19968 / val 3120 / test 8112；另有 smoke 256 / overfit 240（本实验不用） |
| manifest | `/home/user_wzk/wzk/projects/SwinE/manifest/rml2018a_manifest.csv`，sha256 `3c76d726ef0b141f213b2267f8b5933210f80776fe458d1aefd63bc9006f246d` |
| 归一化 | `max_abs`：对每样本按 **分量绝对最大值** 缩放（非复包络模） |

元数据说明：h5 仅含 X/Y/Z，**没有采样率、符号率字段**（fs_hz/fs_valid 在 collate 里恒为 0）。因此本阶段**不编造 Hz/秒/符号单位**，跨采样率/符号率结论留待有元数据的数据集（文档第 7 节）。

## 2. 主干与 token 化（Global 基线）

- 模型：`kind=global`，patch_size=8、dims=[176]、depths=[6]、heads=[4]、ratio=4、dropout=0。
- 原始 token：P=8、步长=P（不重叠）、padding=右补零到 8 的倍数。N=1024 → **T=128** 个 patch，每个 patch 展平 2P=16 维。
- 嵌入：`nn.Linear(16, 176)`（即 `embed`）。位置编码：正弦绝对位置（`positions`）。分类头：LayerNorm 后 mean-pool + `Linear(176, 24)`。
- 参数：全模型约 2.25M（基线 Global）。

## 3. 训练协议

- 优化器 AdamW，lr=3e-4，weight_decay=0.01，warmup 5%（余弦退火），grad_clip=1.0，fp32，batch=128，epochs=50。
- 随机种子：`seed_all`（python/numpy/torch/cuda），DataLoader 用 `torch.Generator(seed)` 与 `seed+1`（train/val 独立）。
- checkpoint 选择：按 val OA 最高的 epoch 存 `best.pt`；每 epoch 存 `last.pt`。
- 本阶段**无预训练**（Phase 4 才接回 MAE/自监督）。

## 4. 基线复现结果

复用现有冻结运行（`SwinE/pretrain_v1/runs/`）作为复现依据：

| run | val OA |
|---|---|
| `sup_global_50_s2024` | 0.3869 (ep21) |
| `sup_global_50_s2025` | 0.3801 (ep30) |
| `sup_global_50_s2026` | 0.3837 (ep24) |
| **从头均值** | **0.3835 ± 0.0034** |
| `ft_global_50_s{2024,2025,2026}` | 0.4237 / 0.4045 / 0.3990（微调，共享预训练 checkpoint） |

本项目的 T0（`mtk` 复刻的同一 patch 嵌入 + 同一主干）将在筛选阶段以 seed=2024 再跑一次，与 0.3869 做交叉验证：**若 T0 seed2024 逐位一致，则证明 tokenizer 框架与冻结基线训练协议一致**，后续 T1–T5 的差异可单因素归因于前端。

## 5. 代码来源与冻结说明

- 冻结基线代码在 `SwinE/pretrain_v1/emfm/`（不改动）。本项目在 `MultToken/mtk/` 独立实现。
- `mtk/data.py`：函数体与冻结 `emfm/data.py` 逐句一致（仅模块 docstring 不同）。冻结源 sha256 `530ed67d…d0e5`。
- `mtk/backbone.py` 的 `positions`/`Attention`/`Block`：与冻结 `emfm/models.py` 同段逐句一致；新增 `GlobalBackbone` 仅把基线 `IQEncoder` 中「位置编码 + 6 个全局 Block + LayerNorm」抽出，不含 embed。冻结源 sha256 `5a8f6ee0…9587`。
- `mtk/train.py`：训练协议（seed/优化器/调度/eval/checkpoint/数据划分）与冻结 `emfm/train.py` 一致；仅把模型换成 `TokenClassifier`。冻结源 sha256 `1acb1123…0ae9`。
- 环境：SFM conda（torch 2.14.0+cu126、numpy 2.4.6、scipy 1.17.1、PyWavelets 1.9.0）。

## 6. 硬件与并行约束（记录）

- 3×A100-40G。**GPU 0 被其他用户（user_cx，PID 1804095）占用**，本实验只使用物理 GPU 1、2。
- 单 run 实测约 6.7 s/epoch、峰值显存 1.48GB（冻结基线 `sup_global_50_*` 的 metrics.jsonl）。50ep 单 run 约 7 分钟。
