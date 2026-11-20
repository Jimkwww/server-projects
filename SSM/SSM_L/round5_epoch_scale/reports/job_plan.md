# Round5 固定 50-epoch 数据扩量实验 —— 新增/复用清单与 job 数

> 目标：F0/F1/F2 历史均按固定 7,800 步（F0≈50ep、F1/F2≈10ep）训练；本实验将 F1/F2 统一到
> **50 个完整 epoch**（F1/F2 = 50 × 782 = 39,100 步），验证「数据扩量 + 更多训练」能否提升
> comm 表征与雷达迁移。与历史结果并存、不覆盖。**全部新预训练与下游无 CFO 增强。**

## 数据 / 协议（复用，不重新生成）

| 项 | 来源 |
|---|---|
| F0（19,968 comm IQ，50ep） | round3 `pretrain_scale/f0`，已 50ep，复用 |
| F1（100,000 comm IQ，(mod,SNR) 分层修正版） | round3 `configs/manifest_f1.json` + `data/f1_indices.npy` |
| F2（50,000 comm + 50,000 OOD EM-Radar） | round3 `configs/manifest_f2.json` + `data/f2_*.npy` |
| RML 划分 train 19968/val 3120/test 8112（封存） | `configs/data.json`（cls_dev） |
| RadChar 划分 train 30000/dev 10000/eval 10000 | `round2_overnight/configs/radchar.json` |
| 随机初始化（shared_init，逐字节与 round3 一致） | 拷贝自 round3，`runs/pretrain/{f1,f2}/shared_init/` |

## 新增 job（共 270）

### 1. 预训练 50ep：18
- 3 骨干（G / G_Qlearn / Q）× 2 语料（F1 / F2）× 3 种子（17 / 42 / 2026）。
- 50 完整 epoch = 39,100 步；warmup+cosine 按新总步数重设；从随机初始化（shared_init）训练，
  不续训 7800 步。
- 中间 checkpoint：第 7,800 步 + epoch 20/30/40/50（50ep = 主评估对象）。记录
  train/val recon loss、lr、累计样本、时间、峰值显存、init hash。

### 2. 冻结线性读出（主终点）A：180
- 90 个 checkpoint 对象（18 run × 5 checkpoint）× 2 任务（RML val、雷达 dev+eval）。
- 协议：冻结 → mean-pool → StandardScaler(train) → L2 LR(C=1, lbfgs)，无 eval 选超参
  （round3/revision1 修正版，非旧 AdamW head）。
- 趋势用 RML val + 雷达 dev；雷达 eval 仅对最终（50ep）checkpoint 报告；RML test 封存。

### 3. 全量微调（无 CFO）B：36
- 18 个 50ep checkpoint × 2 任务（RML 报告 val、RadChar dev 选模报告 eval）。

### 4. 历史 7,800 步无 CFO 微调（对比用）：36
- round3 的 F1/F2 微调用了 CFO 0.002（`finetune_s6` train.augment_cfo_max=0.002），
  **不可放入无 CFO 比较表** → 在 18 个 7800 步 checkpoint 上补跑无 CFO 微调（18 × 2）。

## 复用（不新增计算）

| 项 | 用途 |
|---|---|
| F0 50ep checkpoint（3 骨干 × 3 种子 = 9） | 历史对比 |
| F1/F2 7800 步 checkpoint（18） | 历史对比 + 补跑无 CFO 微调 |
| F0 无 CFO 下游（round4 P0/P1：scratch/F0-MAE/F0-denoise × RML/雷达） | 无 CFO 有监督基线 |
| F0/F1/F2 冻结读出（round3 统一读出，StandardScaler+L2 LR） | 历史对比 |
| round4 scratch 基线（G/Q/S1plain/G_Qlearn） | 无 CFO 有监督基线 |

## 总计

- 新增 job：**270**（18 预训练 + 180 冻结读出 + 36 微调[50ep] + 36 微调[7800步无CFO]）
- 复用（不新增计算）：F0 50ep、F1/F2 7800步、F0 无CFO下游、统一冻结读出、scratch 基线、
  shared_init 初始化、全部 manifest/split。
