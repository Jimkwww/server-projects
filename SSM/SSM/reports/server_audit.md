# 服务器审计报告（server_audit）

日期：2026-10-29。执行：服务器 agent。本报告核对历史 Global 工程、cls_dev 数据划分与预处理，并把新工程候选默认修订为与历史对齐的 `configs/resolved.json`。

## 1. 路径勘误（重要）

任务书写「代码在 `/projects/SSM/`」。本机**不存在 `/projects`**，真实路径为：

- 项目根：`/home/user_wzk/wzk/projects/SSM/SSM/`（zip 解压后多套一层 `SSM/`）
- 历史 Global 工程：`/home/user_wzk/wzk/projects/SwinE/supervised_v1/emfm/`（冻结），另有 `pretrain_v1/emfm/` 与 MultToken 复刻 `MultToken/mtk/`
- 冻结 cls_dev manifest：`/home/user_wzk/wzk/projects/SwinE/manifest/rml2018a_manifest.csv`
- RML2018A 原始数据：`/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5`

以下所有命令把任务书里的 `cd /projects/SSM` 替换为本项目根。

## 2. 历史 Global 定位

- **训练入口**：`SwinE/supervised_v1/emfm/train.py`（`--mode train --h5 ... --manifest ... --config ...`）。
- **模型代码**：`SwinE/supervised_v1/emfm/models.py` 的 `kind=global` 分支（`IQEncoder`，dims=[176]、depths=[6]、heads=[4]、patch_size=8）。
- **数据读取**：`SwinE/supervised_v1/emfm/data.py` 的 `RML2018A` 类，从 manifest csv 按 `subset=cls_dev & split` 选行，按行序取 `sample_id` 索引 hdf5。
- **基线数值**（`SwinE/pretrain_v1/GLOBAL_BASELINE.md`，多重来源一致）：
  - 从头 Global 3 种子（**2024/2025/2026**）：0.3869 / 0.3801 / 0.3837 → **0.3835±0.0034**。
  - 微调 Global（共享预训练 ckpt）：0.4237 / 0.4045 / 0.3990 → 0.4091±0.0130。
  - 与 `MultToken/baseline_audit.md` 的冻结复现交叉验证一致。

> 关键差异：历史 Global 的 0.3869 背景数用的是**种子 2024/2025/2026**；本轮任务书明确要求**种子 17/42/2026**（与 AlignAtt/MultToken 后期 Global 基线一致，AlignAtt `a0_global_s{17,42,2026}` 实测 0.393/0.384/0.384）。因此本轮 G 的 17/42/2026 结果**不会**在数值上等于 0.3869，按协议 0.3869 只作背景，不声称复现。

## 3. cls_dev 数据划分与身份

- manifest sha256 = `3c76d726ef0b141f213b2267f8b5933210f80776fe458d1aefd63bc9006f246d`（实测一致）。
- manifest 列：`subset,split,sample_id,class,snr`。`sample_id` = hdf5 原始行号；`class`/`snr` 与 hdf5 的 Y(one-hot argmax)/Z 完全一致（抽验 2000 行 0 不一致）。
- cls_dev 规模：train **19968** / val **3120** / test **8112**（另有 smoke 256、overfit 240，本轮不用）。每类 1300，每 SNR 档 1200（26 档 -20..30 dB 步长 2）。
- manifest 按 (class,snr) 分桶，桶内 test(13)/val(5)/train(32) 交叠排列。历史 dataset 按 **csv 文件行序** 取样本，因此导出 NPY 时**保持 manifest 行序**（已验证 train 顺序与文件行序一致），不排序、不重切。
- 划分身份：**沿用既有 cls_dev 索引**，未重新随机切分（符合任务书禁止项）。

## 4. 历史实际设置 vs 当前设置

| 项目 | 历史实际（SwinE/MultToken Global） | SSM 候选默认（base.json） | 是否对齐 | 处理 |
|---|---|---|---|---|
| 样本长度 | 1024（定长，无 padding） | 1024 | ✅ | 一致 |
| IQ 排列 | `[B,L,2]`，`x[...,0]=I, x[...,1]=Q` | NLC→转置 `[B,2,L]` | ✅ | 一致 |
| patch 内 flatten 顺序 | `[I0,Q0,I1,Q1,...]` | 同 | ✅ | 一致 |
| 类别映射 | Y one-hot argmax→int 0..23 | 同 | ✅ | 一致 |
| 采样率 | **未知**（h5 无 fs 字段） | N/A | ✅ | 扰动只写 cycles/sample，不写 Hz |
| 输入归一化 | `max_abs`（I/Q 联合分量绝对最大值，非复包络模） | `none` | ❌ | **已修 data.py 增加 max_abs**，resolved 用 max_abs |
| patch | 8 | 8 | ✅ | 一致 |
| dim | 176 | 176 | ✅ | 一致 |
| depth | 6 | 6 | ✅ | 一致 |
| heads | 4 | 4 | ✅ | 一致 |
| 位置编码 | G：正弦绝对位置；SSM：无 | G：none（候选） | ❌ | **已修 model.py 增加 sinusoidal，G 强制 sinusoidal**，SSM 变体 position=none |
| pooling | LayerNorm→token 均值 | 同 | ✅ | 一致（定长全有效） |
| norm | pre-norm（mixer/FFN 各一） | 同 | ✅ | 一致 |
| FFN | 4 倍、GELU、dropout=0 | 同 | ✅ | 一致（历史 FFN 第二层后有额外 dropout，dropout=0 时无差异） |
| dropout | 0 | 0 | ✅ | 一致 |
| batch | 128 | 128 | ✅ | 一致 |
| 优化器 | AdamW(lr=3e-4, wd=0.01)，betas/eps 用 PyTorch 默认(.9,.999)/(1e-8) | 同 | ✅ | 一致 |
| weight decay | 0.01 | 0.01 | ✅ | 一致 |
| scheduler | **step 级**：warmup 5% 总步数 + 余弦 | epoch 级 cosine、warmup=0 | ❌ | **已修 engine.py 改为 step 级 warmup_fraction=0.05** |
| warmup | 5% 总步数 | 0（候选） | ❌ | resolved 用 0.05 |
| grad_clip | 1.0 | 1.0 | ✅ | 一致 |
| 增强 | 无 | 无 | ✅ | 一致 |
| 早停 | 无（跑满 50ep，best val OA 选 ckpt，平局最早） | 同 | ✅ | 一致 |
| 精度 | fp32 | fp32 | ✅ | 一致 |
| epoch | 50 | 50 | ✅ | 一致 |
| 种子 | 历史 Global 2024/2025/2026；本轮任务书 17/42/2026 | 17 | — | 按任务书 **17/42/2026**，0.3869 只作背景 |
| 数据 shuffle | train shuffle=True generator(seed)，val 顺序 generator(seed+1) | train generator(seed)，val 顺序 | ✅ | 一致（val 顺序读取） |
| 物理统计 | 无（SSM 新增） | 仅训练子集拟合 | ✅ | S2/S2_concat 用固定 seed0 训练子集拟合 |

## 5. 代码修订（把候选默认对齐历史）

1. `ssm_exp/data.py`：新增 `max_abs` 归一化（联合分量绝对最大值）。
2. `ssm_exp/model.py`：新增 `sinusoidal_positions`；`Classifier` 对 `variant='G'` 强制使用历史正弦位置编码，SSM 变体用 `position='none'`。
3. `ssm_exp/engine.py`：scheduler 从 epoch 级改为 **step 级** warmup_fraction + cosine（与历史 `train.py` 一致），`scheduler.step()` 移到每 batch。
4. `configs/base.json`：`warmup_epochs:0` → `warmup_fraction:0.05`。

这些是对「共同外围结构」的修订，未改变研究问题、数据划分或 SSM 模块定义；G 仍为历史 Global 结构的适配版（不称复现 0.3869，因为种子不同）。

## 6. 环境

- conda `SFM`：python 3.11.16、torch 2.14.0+cu126、cuda 12.6（驱动 610.43.02）、numpy 2.4.6、h5py 3.16.0。
- GPU：3×A100-40GB（卡0 PCIe，卡1/2 SXM4），审计时全部空闲（0 MiB used，0% util）。
- 完整依赖版本在正式训练时另存 `reports/environment.txt`。

## 7. 结论

历史 Global 结构、cls_dev 划分与预处理均已核对清楚，无关键事实缺失。新工程候选默认中「输入归一化、位置编码、scheduler/warmup」三项与历史不符，已按历史修订；种子按任务书 17/42/2026（与 0.3869 背景数的 2024/2025/2026 不同，已明确不混用）。可进入验证与正式实验阶段。
