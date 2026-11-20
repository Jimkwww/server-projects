# Round5 固定 50-epoch 数据扩量实验 —— 协议

## 1. 目的

历史 SSM_L 扩量实验（round3）里，F0/F1/F2 都用**固定 7,800 优化步**的算力预算
（F0=19,968 comm IQ ≈ 50 epoch；F1/F2=100,000 ≈ 10 epoch）。这混淆了「数据扩量」与
「训练充分度」两个变量。本实验把 F1/F2 统一到 **50 个完整 epoch**（= 50 × 782 = 39,100 步），
验证「数据扩量 + 更多训练」能否提升 comm 表征与雷达迁移。与历史结果并存、不覆盖。

## 2. 模型与语料矩阵

- 3 骨干（沿用 round2 命名，内部 `build_encoder_r2` 映射）：
  - **G** = 普通 Global Transformer（patch=8/dim=176/depth=6/heads=4，无位置编码）
  - **G_Qlearn** = Q 复数相关滤波前端 + Global 主干
  - **Q** = Q 复数相关滤波前端 + S1 SSM 主干
- 2 语料 × 3 种子（17/42/2026）= 18 个新预训练 run。
  - **F1** = 100,000 comm IQ，(mod,SNR) 分层修正版（round3 `data/f1_indices.npy`，24 类各 4160–4186）。
  - **F2** = 50,000 comm + 50,000 OOD EM-Radar（round3 `data/f2_*.npy`）。
  - F0（19,968 comm IQ）历史已 50 epoch → 复用，不重训。

## 3. 数据与隔离

- 复用 round3 F0/F1/F2 的 manifest / 划分 / 去重，**必须用修正后的 F1**（不能用早期按索引
  顺序扫出的调制-major 错误语料）。
- RML：train 19,968 / val 3,120 / test 8,112（**test 封存**）。RadChar：train 30,000 /
  dev 10,000（manifest `val`）/ eval 10,000（manifest `test`）。
- 预训练语料与 RML val/test、RadChar 无泄漏（round3 已核验）。

## 4. 预训练协议（新 18 run）

- MAE，mask_ratio 0.5，batch 128，AdamW(lr 3e-4, wd 0.01, betas (0.9,0.999), eps 1e-8)，
  cosine + 5% warmup，grad_clip 1.0，fp32，**augment_cfo_max=0（无 CFO）**。
- RNG 流（同 round2/3）：数据顺序 seed；mask seed+200000；val 噪声 seed+300000；
  val mask seed+400000。
- 从**随机初始化**训练（复用 round3 的 shared_init 缓存，逐字节一致），**不从 7,800 步续训**。
  同模型同种子 F1/F2 用同一初始化权重（init hash 已入 provenance）。
- 50 完整 epoch：`total_steps = 50 × ceil(100000/128=782) = 39,100`（由 DataLoader drop_last=False
  决定，记录而非硬编码）。warmup+cosine 按新总步数重设（同 5% 比例）。
- Q/G_Qlearn 的描述子统计量（aux mean/std）按「训练集估计」规则（shared_init 在各自语料上
  fit），从不碰 val/test。

## 5. 中间 checkpoint

每 run 保存：第 **7,800** 个优化步、epoch **20/30/40/50** 末的 encoder（50ep = 主评估对象，
同时写 `encoder_final.pt`）。每 epoch 记录 train/val recon loss、lr、累计样本、时间、峰值显存。

> 注意：新 50ep 调度下第 7,800 步的 LR 轨迹与历史 7,800 步不同（后者第 7,800 步即训练终点），
> 故「新 50ep 的 step7800」与「历史 7800 步」不是严格同轨迹对照，仅作趋势参考。

## 6. 下游评估

### A. 冻结线性读出（主终点）
冻结 encoder → mean-pool → StandardScaler(train) + L2 LogisticRegression(C=1, lbfgs, max_iter=2000)。
**无 eval 选超参**；用 round3/revision1 修正流程，**禁用旧 AdamW 分类头**。
- 所有中间 checkpoint（step7800/ep20/30/40/50）→ RML val + 雷达 dev 作趋势。
- 雷达 eval 仅对最终 50ep checkpoint 报告；RML test 封存。

### B. 全量微调（无 CFO）
50 epoch，AdamW(lr 3e-4)+cosine+5% warmup，无 CFO（in-loop max_abs_norm）；dev/val 选模
（OA 最早平局）；RML 报告 val，雷达报告 eval（--eval-split test）。
- 18 个 50ep checkpoint × 2 任务 = 36 run；另在 18 个历史 7800 步 checkpoint 补跑无 CFO
  微调 = 36 run（round3 原微调用 CFO 0.002，不可入无 CFO 表）。

## 7. 历史对比（复用）

- F0 50ep（round3 shared 读出 + round4 无 CFO 微调）
- 历史 F1/F2 7800 步冻结读出（round3 `section6_probe.json`）
- 无 CFO 有监督 scratch 基线（round4 P0）

## 8. 交付物

`reports/`：job_plan / protocol / commands / hashes / readout_trend / readout_main /
finetune_main / paired_deltas / recon_trend / conclusion_zh / tables_ppt / analysis_raw.json；
`figs/`：训练曲线 + 读出趋势图；checkpoint SHA256；逐样本预测 npz；打包（无权重）。
