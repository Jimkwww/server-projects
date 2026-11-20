# Round6 目标对比 + 双向 SSM 实验 —— 协议

## 1. 目的

固定 F1 语料（100k comm IQ）与 50 epoch 预训练预算，回答三个问题：

1. F1 规模训练下，**去噪（denoise）是否优于 MAE**？
2. 同一骨干下，**MAE+去噪联合训练（joint）是否优于最优单目标**？
3. **双向 SSM 是否降低 MAE 重构误差并改善下游/微调**？

> 注：「单向因果 → MAE 高 loss」只是待验证的假设，不作结论预设。

## 2. 命名

- **G** = 原 Global Transformer（patch=8/dim=176/depth=6/heads=4）。
- **Q** = 原 Q 复数相关滤波前端（ComplexRecurBranch, 8 通道复状态）+ **单向** S1 SSM（ComplexSSM, state=128）。
- **Q-bi** = 同一个 Q 前端 + **双向** S1 SSM（BidirectionalSSM, 正反各 state=64）。
  - 「64」= 每方向的状态维度（正 64 + 反 64 = 总状态 128），**不是截断序列**。
  - 双向：正向扫全序列 + 反向扫全序列（`u.flip(1)` 扫描后翻转回来），两方向参数独立，
    输出仍 176 维，加一次 `direct` 跳连；block/residual/norm/FFN/pooling 不变。
  - **Q 前端保持单向**（不做双向化）；只有 SSM 主干扫描方向 + 状态分配改变。
  - 实际参数量以 `model_info.json` 为准（正反两方向各有独立 selector，比 Q 单向多一层
    selector 输入投影的参数量，**不假设 64+64 与 128 参数完全相同**）。

## 3. 数据与隔离

- 复用 round3 F1 manifest（100,000 comm IQ，分层修正版 `round3_scale_transfer/configs/manifest_f1.json`）。
- RML：train 19,968 / val 3,120 / test 8,112（test 封存）。RadChar：train 30,000 / dev 10,000（manifest `val`）/ eval 10,000（manifest `test`）。
- 预训练语料与 RML val/test、RadChar 无泄漏（round3 已核验）。

## 4. 预训练协议（21 个新 run）

矩阵（每项 3 seed：17/42/2026）：

| 组 | 骨干 | 目标 |
|---|---|---|
| G-denoise | G | denoise |
| G-joint | G | joint |
| Q-denoise | Q | denoise |
| Q-joint | Q | joint |
| Q-bi-MAE | Q-bi | mae |
| Q-bi-denoise | Q-bi | denoise |
| Q-bi-joint | Q-bi | joint |

- **复用** round5 的 G-MAE、Q-MAE（F1、50ep、3 seed），本轮不重训。无 G_Qlearn。
- 通用协议：50 完整 epoch = 39,100 优化步（batch 128, drop_last=False, ceil(100000/128)=782 步/epoch）；
  AdamW(lr 3e-4, wd 0.01, betas (0.9,0.999), eps 1e-8)，cosine + 5% warmup，grad_clip 1.0，fp32；
  patch=8/dim=176/depth=6/heads=4（G）/单向 SSM state=128（Q）；无 CFO。
- **随机初始化**（非从 MAE 训练权重续训）：G/Q 复用 round5 的 shared-init 缓存（逐字节一致）；
  Q-bi 的 front/aux_branch/aux_fusion/norm/aux 统计复用 Q 的 shared-init，双向 A64 blocks 随机初始化。
- 解码头初始化、数据顺序、mask、噪声分别独立 RNG（见 §5），避免新增模块改变编码器初值或数据顺序。

### Denoise 任务

- 目标 = 原始观测 x；输入 = x + 合成复高斯噪声，噪声级 **0/5/10/15 dB**（附加噪声 SNR，非真实 SNR）。
- sigma 仅进入 denoise 解码头（ReconstructHead 的额外标量槽），不进入编码器。
- MSE 覆盖全部 IQ 元素（`denoise_loss` = 每个 (batch, time) 位置的 (Î−I)²+(Q̂−Q)² 均值）。

### Joint 目标

- 共享编码器 + 两个独立解码头（MAE 头 + denoise 头）。
- 每个 batch 拆成约两个等大子集（128→64 各；尾批按实际数量近似均分）。
- 每样本每次通过只做一个任务，**每 epoch 重新随机分配**（独立 split RNG 流），禁止永久指定。
- 两个子集各自前向，梯度累加，**每个原始 batch 只做一次 optimizer update**。
- `L_joint = 0.5·L_MAE + 0.5·L_DN`；无动态加权/梯度冲突修正。
- 两个损失各自按有效 IQ 元素数归一化：MAE 只统计遮挡位置，去噪统计全部位置。
- 记录两任务各自的 train/val loss + 加权总 loss + 各自样本数。

### Bidirectional SSM（Q-bi）

- Q 单向 = 128 维状态；Q-bi = 正 64 + 反 64（两方向都扫全序列，反向输出翻转回来再融合）。
- 方向参数独立；输出仍 176；保留 block/residual/norm/FFN/pooling。
- RML 仍 128 token（1024/8），RadChar 64 token（512/8）。
- MAE mask 在 **raw IQ 上先施加**，再算 Q 分支特征（两方向都不能通过 aux 分支泄漏被 mask 的 IQ）。
- 本轮无 A128。双向 denoise/joint（Q-bi-denoise / Q-bi-joint）为 **round6 扩展**（6 个新 run，3 卡并行），
  其余协议与 Q-bi-MAE 完全一致（双向只改 SSM 主干扫描方向，denoise/joint 目标定义见上）。

## 5. RNG 流

沿用 round2/round5 约定，新增 joint split 流（不扰动既有流）：

| 用途 | seed 偏移 |
|---|---|
| 数据顺序 | +0 |
| denoise 噪声 | +100000 |
| MAE mask | +200000 |
| val 噪声 | +300000 |
| val mask | +400000 |
| joint batch split | +500000 |

G/Q 复用 round5 shared-init ⇒ 数据顺序 + mask + 编码器初值与 round5 G-MAE/Q-MAE 逐字节一致。

## 6. Checkpoint

每 run 保存 epoch **10/20/30/40/50** 末的 encoder + resume 状态（`last.pt`）。epoch50 = 主评估对象，
同时写 `encoder_final.pt`。每 epoch 记录 MAE/denoise train+val loss（joint 记录两项 + 加权总），
lr/步数/各任务样本数/时间/峰值显存。

## 7. Smoke test（9 项，全量前必过）

1. Q_bi 构建：front/aux/norm 与 Q shared-init 逐字节一致，A64 blocks 为双向且参数独立。
2. Q_bi 参数量 = 记录值（不假设与 Q 相等），且输出 176 维。
3. 双向输出 token 对齐正确（正向/反向翻转回原位），每样本状态独立、正确清零。
4. MAE mask 先于 Q 分支特征施加（aux 分支不泄漏）。
5. denoise：0/5/10/15 dB 附加噪声，sigma 仅进 denoise 头。
6. joint：batch 拆分子集、两损失各自归一化、单次 optimizer update、加权 0.5/0.5。
7. RNG 流独立（数据/噪声/mask/split 互不干扰）。
8. 预训练 smoke（每任务各骨干若干步）loss 有限且下降。
9. 下游 smoke（readout + finetune 各 1）跑通、协议不变（StandardScaler+L2 LR / no-CFO 50ep）。

## 8. 下游评估

21 个新 encoder × 2 任务 = **42 冻结读出 + 42 微调**。

### A. 冻结线性读出（主终点）

冻结 encoder → mean-pool → StandardScaler(train) + L2 LogisticRegression(C=1, lbfgs, max_iter=2000)。
**无 eval 选超参**；禁用旧 AdamW 分类头。RML 报 val（test 封存）；雷达报 dev + eval。

### B. 全量微调（无 CFO）

50 epoch，AdamW(lr 3e-4)+cosine+5% warmup，无 CFO（in-loop max_abs_norm）；dev/val 选模（OA 最早平局）；
RML 报 val，雷达报 eval（--eval-split test）。Q-bi 在微调/推理中保持双向。

## 9. 效率

记录编码器参数量、预训练总参数量（含解码头）、GPU 小时、训练吞吐、峰值显存、编码器推理延迟。

## 10. 复用（不重跑）

round5 G-MAE / Q-MAE 的冻结读出 + 无 CFO 微调结果（round5_epoch_scale/runs/downstream/）。

## 11. 分析

- OA / Macro-F1，3 seed 均值±std；逐 seed 配对差值；RML 分 SNR。
- 主表（9 行）：G-MAE / G-denoise / G-joint / Q-MAE / Q-denoise / Q-joint / Q-bi-MAE / Q-bi-denoise / Q-bi-joint。
- 回答 §1 三问（含统计口径）。

## 12. 交付物

`reports/`：protocol / run commands / run inventory / configs+code hashes / pretrain curves /
per-task losses / efficiency / readout_main / finetune_main / per-seed / paired_deltas /
per-sample predictions / checkpoint SHA256 / conclusion_zh（一页）+ PPT 表格。
PPT 表 = 9 行 × 4 列（comm 冻结 OA、radar 冻结 OA、comm 微调 OA、radar 微调 OA）。
权重留在服务器；报告/配置/代码/预测打包。
