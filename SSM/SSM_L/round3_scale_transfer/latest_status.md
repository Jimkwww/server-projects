# SSM_L 第三轮实验（round3_scale_transfer）— 阶段状态

更新时间：2026-11-11（服务器时间）— ✅ 全部完成

## 第 1 节：correction.md ✅ 完成
4 处更正（随机 Q 支路零初始化、弱信号检测措辞、低标签 scaler 口径、EM-Radar 重审）已写入 `reports/correction.md`，原 round-2 报告保留。

## 第 2 节：G_Qlearn 微调 vs 从头训练 ✅ 完成
- G_Qlearn-scratch ×3：均值 **0.4598**（比 G-scratch **+1.80pp**）。
- G_Qlearn-MAE 微调 **0.5347**（+7.49pp over scratch）；denoise 0.4980（+3.81pp）。
- **问题 1 答案**：53.47% = Global 主干微调能力 + MAE 预训练（+7.5pp）+ Q 前端（+1.8pp）。

## 第 3 节：统一冻结读出排名 ✅ 完成
统一 StandardScaler+L2 LR 全标签 RML val OA（均值，3 seed）：

| 结构 | random | MAE | denoise |
|---|---|---|---|
| G | 0.1989 | 0.3169 | 0.2916 |
| G_Qlearn | 0.1993 | 0.2843 | 0.2968 |
| S1plain | 0.2232 | 0.3281 | 0.2873 |
| Q(=Q_corr) | 0.2230 | **0.3370** | 0.3019 |

- **问题 2 答案**：MAE 排名 Q > S1plain > G > G_Qlearn；Q 前端在 Global 内 −3.26pp、在 S1 内 +0.89pp。

## 第 4 节：雷达冻结迁移 ✅ 完成
统一 LR RadChar eval OA（均值）：G_Qlearn-MAE 0.8042 vs G-MAE 0.7641（**+4.01pp**）；Q-MAE 0.8107 vs S1plain-MAE 0.7883（**+2.24pp**）。
- **问题 3 答案**：Global+Q 表征能迁移到雷达且提升；与 RML 冻结读出（为负）相反。

## 第 5 节：语料重审 ✅ 完成
- F0=19968；F1=100000（(mod,SNR) 分层、含全部 F0、去重）；F2=100000（50k comm + 50k OOD EM-Radar）。
- **关键修正**：RML2018A 调制-major 排列，round-2 索引顺序扫 100k = 纯调制 0 语料（已废弃），round-3 改分层采样。

## 第 6 节：扩量预训练 ✅ 完成
- 18 个 MAE 预训练（G/G_Qlearn/Q_corr × F1/F2 × 3 seed，7800 步）+ 18 个 RML 微调 + 统一 LR 探测 + CFO±0.005。
- 见 `reports/section6_probe.json` / `section6_finetune_cfo.json` / `section6_predictions/`。

### 问题 4（同域扩量）
F1（100k，固定 7800 步）vs F0（19968，50 epoch）：三项下游基本持平（±1pp，无一致方向）→ **数据量不是瓶颈**。

### 问题 5（OOD 是否两域通吃）
F2（50k comm + 50k OOD）**非两域通吃，依赖主干**：
- G（Global）：微调 +2.30pp、雷达 +1.71pp —— **唯一两域都涨**；
- Q_corr：雷达 +0.57pp 但微调 −2.83pp、冻结读出 −1.49pp；
- G_Qlearn：微调 −2.70pp、雷达 −1.20pp（冻结读出 +1.82pp）。
→ OOD 让 Global 两域受益，但让复数分支模型在 RML 微调退化（复数分支向 OOD 统计过拟合）。

## 第 7 节：验收 ✅ 完成
`reports/section7_acceptance.json`：**56 ok / 0 fail / 0 pending**（权重加载、Q 参数可训、probe 冻结、数据 ID 隔离、断点恢复、MAE 掩码先于 Q 支路）。

## 第 8 节：打包 ✅ 完成
`deliverables/SSM_L_round3_results.zip`（1.07MB，62 文件）：报告/脚本/configs/data 索引/逐样本预测 + `weights_sha256.json`（90 个 .pt 权重 SHA256，共 1.7GB，权重本体留服务器）。
zip SHA256：`4deff3e1a8a9e559d88948d09feceea2adbbc3f139ffbfa0be1fa91ac4c6b8c8`

## 五个问题总答
1. 53.47% = Global 主干 + MAE（+7.5pp）+ Q 前端（+1.8pp）。
2. 冻结读出 MAE：Q > S1plain > G > G_Qlearn。
3. Global+Q 雷达冻结迁移 +4.01pp（与 RML 冻结读出相反）。
4. 同域扩量无增益，数据量不是瓶颈。
5. OOD 非两域通吃：Global 两域都涨，复数分支模型 RML 微调退化。
