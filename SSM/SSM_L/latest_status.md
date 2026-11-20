# SSM_L 第二轮夜间实验 — 阶段状态

更新时间：2026-11-10 17:32（服务器时间）— **全部完成**

## 第 0 节：核对与验收 ✅
- ✅ 第一轮 12 份 encoder_final.pt、6 份 last.pt（含 decoder.proj）、24 份 best.pt、36 份 DONE.json 均在。
- ✅ split 哈希、shared_init 缓存（G/Q × 3 seed）核验通过。
- ✅ 3×A100、/data1 9.6T、/home 已告警（97% 满）。
- ✅ 位同一性测试 PASS（G/Q 第二轮 == 第一轮；G_Qlearn 主干==G、aux==Q；S1plain 主干==Q）。
- ✅ 冻结模型复现第一轮 G_mae_seed17 val OA=0.214744（diff 0.00e+00）。

## A. 随机冻结 probe ✅ 完成
| 模型 | seed17 | seed42 | seed2026 | 均值 |
|---|---|---|---|---|
| G random OA | 0.0987 | 0.1135 | 0.1154 | 0.1092 |
| Q random OA | 0.1442 | 0.1494 | 0.1465 | 0.1467 |

- Q_random − G_random = **+3.75pp**；第一轮 Q−G MAE 冻结差距 ≈ 5.27pp → 随机初始化解释 ~71%，
  预训练增益差 (Q_MAE−Q_rand)−(G_MAE−G_rand) ≈ +1.52pp。

## B. 修正去噪评估 ✅ 完成
- ✅ 严格加载 `last.pt['model']`（含训练好的 decoder），核验 6/6 通过；第一轮 buggy 结果标记 superseded。
- 域内 MSE vs x：去噪相对恒等降低 2–6×（0 dB 档 G 0.0528/Q 0.0598 vs 恒等 0.3021）；G 略优于 Q。
- 弱信号检测：**负结果**（弱幅度下所有方法 Pd≈Pfa）；幅度扫描 boxcar 是最佳弱脉冲检测器
  （amp 1.6/3.2：boxcar 0.90/1.00 > G去噪 0.43/0.87 > Q去噪 0.36/0.84 > 恒等 0.11/0.54）。

## D. 2×2 结构消融 ✅ 完成（含 CFO）
新增 G_Qlearn / S1plain × {MAE, denoise} × 3 seed；随机 6/6、预训练 12/12、probe 18/18(+6 随机)、微调 12/12。

冻结 probe（MAE 2×2）：Global 无 Q=21.13（首轮 G）、Global+Q=20.50（G_Qlearn）、S1 无 Q=24.35（S1plain）、S1+Q=26.40（首轮 Q）。
→ Q 增益 Global −0.63pp / S1 +2.05pp。
微调 clean-val OA（CFO 评估，3 seed 均值）：

| 结构 | MAE | denoise |
|---|---|---|
| G_Qlearn | 0.5347（0.5317/0.5349/0.5375） | 0.4980（0.5151/0.5077/0.4712） |
| S1plain | 0.4694（0.4673/0.4657/0.4753） | 0.4233（0.4202/0.4253/0.4244） |

→ 微调下 Q 增益（MAE）：Global +5.13pp（0.5347−0.4834）、S1 +1.68pp（0.4862−0.4694）；与冻结方向相反。
- CFO ±0.005 衰减：G_Qlearn_mae 0.5317→0.4936/0.4894；S1plain 结构 CFO 衰减更大（见 finetune_cfo_round2.json）。

## C. 低标签冻结 probe ✅ 完成
18 表征 × 预算 {1,10,50,100,full}/类 × 5 标签种子，LR(L2+标准化)。Q-MAE > G-MAE、MAE > denoise 在所有预算均成立。
全标签参照（LR，3 seed 均值）：Q_mae 33.70 / G_mae 31.69 / Q_denoise 30.19 / G_denoise 29.16 / Q_random 22.30 / G_random 19.89。

## E. 独立雷达迁移 ✅ 完成
- 数据：RadChar-Tiny（纯雷达、复 IQ、5 类、512 点、30000/10000/10000 分层无泄漏、50000 条 IQ 无重复）。
  RadarCommDataset（实值）❌、EM-Radar-mixed（盲源分离）❌。
- 全标签 eval OA（3 seed 均值，chance=20%）：

| 表征 | G | Q |
|---|---|---|
| random | 0.5488 | 0.5812 |
| MAE | 0.6388 | **0.7060** |
| denoise | 0.6325 | 0.6921 |

- 每类 10 条低标签均值：Q-MAE 0.5627 > Q-denoise 0.5422 > G-MAE 0.5095 > G-denoise 0.4946 > G-random 0.4554 > Q-random 0.4380。

## F. 扩大量（条件）✅ eligibility 完成、预训练未启动
- F0=19968 条去重 IQ；F1=100000 条独立 IQ（上限，与 F0 重叠仅 790）；F2 不适用（无合格异域原始 IQ）。
- 夜间资源被 D/C/E 占用，12 次 F 预训练未启动 → **(f) 无直接实验证据**（仅语料可用性结论）。

## 交付物
- ✅ `round2_overnight/reports/`：denoise_eval_round2.json、detection_sweep_round2.json、lowlabel_probe_round2.json、
  radar_transfer_round2.json、finetune_cfo_round2.json、aggregate_round2.json、split_hashes.json、checkpoint_sha256.json、
  eligibility.csv、radar_inventory.md、conclusion.md、curves/*.png。
- ✅ 逐样本预测：probe/finetune val_predictions.npz（A/D）+ radar_eval_preds（E，18 份）。
- ✅ `round2_overnight/deliverables/SSM_L_round2_overnight_results.zip`（权重留服务器，仅 SHA256 清单）。
- tmux 会话已全部退出；重跑/恢复命令见 zip 内 commands.md。
