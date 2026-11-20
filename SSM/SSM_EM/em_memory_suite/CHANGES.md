# 代码与适配变更记录

本包（em_memory_suite）核心代码（`emmem/*.py`、`tools/*.py`、`tests/*.py`）**未修改**。
全部变更集中在 `server_prep/` 与 `configs/`，均为服务器数据接入/调度适配。非 git 仓库，以文件清单代替 diff。

## 新增文件

| 文件 | 作用 |
|---|---|
| `configs/registry.server.json` | 真实数据 registry（F1 预训练 + AMC/雷达下游 + 扰动），batch 128 |
| `configs/registry.server.cuda1.json` / `cuda2.json` | 主矩阵按变体拆 2 卡的 registry（device 分开） |
| `server_prep/prepare_data.py` | RML2018A + RadChar hdf5 → emmem npy manifest（复用 SSM_L cls_dev/f1/radchar 划分） |
| `server_prep/finalize_manifests.py` | 补 class_names、sample_rate=null、radar 任务改名 |
| `server_prep/fix_f1_val.py` | 修复 F1 预训练 val 泄漏（原复用下游 cls_dev_val → 独立 3120 样本） |
| `server_prep/audit_correct.py` | 正确审计：区分协议允许的 train↔train 复用与 val/test 泄漏 |
| `server_prep/prepare_techrec.py` | TECHRec（LTE/WiFi/DVB-T）bin → npy manifest，跨位置划分 |
| `server_prep/split_and_launch.py` | 主矩阵 34 变体按 seed 数均衡拆 2 卡 |
| `server_prep/gen_extended.py` | 扩展 campaign（native_width/scale/data/long）按卡生成 plan |
| `server_prep/historical_baseline.py` | 历史 G/Q（SSM_L round5 F1-MAE）接本包数据+线性头 |
| `server_prep/techrec_downstream.py` | 主矩阵 checkpoint → TECHRec 跨域下游 |
| `server_prep/status.py` | 一次性进度汇总 |
| `CHANGES.md` | 本文件 |

## 关键决策与协议差异（记录在案）

1. **batch 128**（registry 默认 32 → 128）：dim=64 模型在 A100 40G 上 batch 32 单 epoch 103s 过慢，
   128 为 36.5s（提速 2.8×）。与历史 round5/round6 协议一致。
2. **F1 val 泄漏修复**：SSM_L 原 F1 manifest 的 val 复用下游 cls_dev_val（预训练 checkpoint 选择会泄漏
   下游 val）。改为从 F1 内"不在任何下游 split"的 80032 样本中取独立 3120 作 val。预训练 train 仍含
   下游 cls_dev_train 19968（DATA_AND_PROTOCOL #6 允许的无标签复用）。
3. **groups 退化为样本级唯一 ID**：RML2018A/RadChar/TECHRec 均为独立模拟/独立采集样本，无同源
   capture group；在 provenance 明确记录，不伪造 group 元数据。
4. **历史基线协议差异**：历史 G/Q 输入 [B,2,L]+max_abs_norm，本包 [B,L,2] 无归一化；历史 dim=176/
   depth=6 vs 本包 dim=64/depth=3（参数不匹配，分别报告）。历史 probe/finetune 用本包协议
   （线性 head 50ep AdamW），在 adapter 内完成输入适配。
5. **TECHRec 采样率**：仅用 1Msps 文件（10Msps 排除，采样率不一致会引入采样率分类混淆）；
   按采集位置划分 train/val/test 实现跨位置泛化测试。

## 未修改的核心实现边界

- 未实现 DDP（runner 单进程串行 per-plan）；并行方式=按变体拆 2 个 plan 各占 1 卡。
- 未实现多标签/多源分离/点级分割（本包文档已声明）。
- 官方 Radio-FM/SpectrumFM/EMind 未接入（无官方代码/checkpoint，见 `metadata/blocked.json`）。
