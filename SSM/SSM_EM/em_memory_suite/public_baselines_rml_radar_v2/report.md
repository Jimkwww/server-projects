# public_baselines_rml_radar_v2：四模型跨 seed / 学习率 / 训练预算对比（修正版）

> 硬预算 8 小时，训练 00:55–06:42 CST 全部完成；随后一次修正补跑（Q seed42/2026 + 效率重测，
> 01:19–01:50 CST）。四模型 Q / structured / EMind / WavesFM，两任务 RML / radar，~10M 参数，FP32，
> effective batch 128。v1 seed17 八组复用。本版修正内容见 `DIFF.md`。

## 结论（先给结论，均限本协议范围）

1. **structured 在 RML 上三 seed 一致领先 Q**：test OA 0.4503±0.0069 vs 0.3872±0.0103（**+6.3pp**），
   三个 seed（17/42/2026）方向一致（+4.9~+8.2pp）。标准差为样本标准差（N=3，ddof=1）。
2. **Q 在 radar 上三 seed 一致但幅度弱**：test OA 0.8707±0.0061 vs structured 0.8622±0.0017（**+0.9pp**），
   三 seed 一致（+0.4~+1.2pp），但幅度接近 Q 自身 std，属"稳定但弱"，不足以作细排序定论。
3. **本协议下，两个 SSM 类实现（structured、Q）在 RML/radar 两任务的排名均高于两个 ViT 类实现
   （EMind、WavesFM）**，且 10 个 (task×pair) 差值的符号在三 seed 上完全一致。**这仅是这两个具体
   实现、本协议、3 seed 下的实测排序，不构成"SSM 普遍优于 ViT"或"物理分支有效"的泛化结论**。
4. **排序不是学习率或训练预算造成的**：lr∈{3e-4,1e-4}×ep∈{30,60} 的 2×2 里四模型排序不变；60ep 无系统
   提升（Q/rml 60ep 反降 0.391→0.380，过拟合）。
5. **精度—效率代价**：RML 上 structured 以最慢速度（batch1 18.9ms、吞吐 1395 samples/s）换最高精度；
   WavesFM 最快（1.76ms、7780 samples/s）但精度最低（0.358）；Q 居中（4.4ms、2627 samples/s）。

## 一、三 seed 主表（30 epoch，lr3e-4；样本标准差 ddof=1）

| 模型 | RML OA (mean±std) | RML F1 (mean±std) | radar OA (mean±std) | radar F1 (mean±std) |
|---|---|---|---|---|
| **structured** | **0.4503±0.0069** | **0.4442±0.0054** | 0.8622±0.0017 | 0.8639±0.0016 |
| **Q** | 0.3872±0.0103 | 0.3867±0.0106 | **0.8707±0.0061** | **0.8714±0.0058** |
| EMind | 0.3556±0.0049 | 0.3469±0.0064 | 0.8428±0.0029 | 0.8432±0.0036 |
| WavesFM | 0.3579±0.0039 | 0.3623±0.0017 | 0.8402±0.0022 | 0.8409±0.0015 |

随机水平：RML 4.17%，radar 20%。各 seed 原始 OA：RML structured [.4491,.4577,.4440]、Q [.3907,.3756,.3953]；
radar Q [.8758,.8639,.8724]、structured [.8634,.8602,.8629]（完整见 `analysis.json`）。

## 二、配对差值（test OA，每 seed）

| 对比 | RML mean | RML 三 seed | radar mean | radar 三 seed |
|---|---|---|---|---|
| structured − Q | **+0.063** | [.058, .082, .049] | **−0.009** | [−.012,−.004,−.010] |
| Q − EMind | +0.032 | [.036, .024, .034] | +0.028 | [.031, .024, .028] |
| Q − WavesFM | +0.029 | [.034, .013, .041] | +0.031 | [.033, .025, .033] |
| structured − EMind | +0.095 | [.095, .106, .083] | +0.019 | [.019, .021, .019] |
| structured − WavesFM | +0.092 | [.093, .095, .089] | +0.022 | [.021, .022, .024] |

全部 10 个 (task×pair) 差值三 seed 符号一致（不翻转）。RML 上 structured 领先所有对手 +6.3~+9.5pp；
radar 上 Q > structured > EMind > WavesFM，但相邻差仅 ~1–2pp。

## 三、学习率 / 预算 2×2（seed17，val-F1 选点）

| 任务 | 模型 | 30 预算选 lr | 30 test OA | 60 预算选 lr | 60 test OA |
|---|---|---|---|---|---|
| RML | Q | 3e-4 | 0.3907 | 3e-4 | 0.3804 |
| RML | structured | 3e-4 | 0.4491 | 3e-4 | 0.4506 |
| RML | EMind | 1e-4 | 0.3482 | 3e-4 | 0.3597 |
| RML | WavesFM | 3e-4 | 0.3565 | 3e-4 | 0.3559 |
| radar | Q | 3e-4 | 0.8758 | 3e-4 | 0.8737 |
| radar | structured | 1e-4 | 0.8649 | 1e-4 | 0.8633 |
| radar | EMind | 3e-4 | 0.8445 | 1e-4 | 0.8424 |
| radar | WavesFM | 3e-4 | 0.8427 | 1e-4 | 0.8469 |

lr 1e-4 与 3e-4 差 <1pp；60ep 相对 30ep 无系统提升。**四模型排序在 4 个 lr×预算格子内不变。**
调参成本 = 24 组新训 ≈ 21 GPU-h；单 run = 30ep 15–53min / 60ep 30–106min。

## 四、效率（A100-PCIE-40GB 独占，FP32，每模型独立进程实测）

| 任务 | 模型 | token | batch1 延迟 | 吞吐 (samples/s) | 峰值显存 (batch64) |
|---|---|---|---|---|---|
| RML | Q | 128 | 4.39 ms | 2627 | 157 MB |
| RML | structured | 128 | 18.86 ms | 1395 | 153 MB |
| RML | EMind | 257 | 4.21 ms | 1527 | 462 MB |
| RML | WavesFM | 65 | 1.76 ms | 7780 | 114 MB |
| radar | Q | 64 | 4.04 ms | 4986 | 104 MB |
| radar | structured | 64 | 17.63 ms | 2128 | 102 MB |
| radar | EMind | 129 | 4.55 ms | 3252 | 179 MB |
| radar | WavesFM | 33 | 1.53 ms | 14655 | 81 MB |

- **token 数含 cls token**：EMind (L+4)//4=257/129，WavesFM L//16+1=65/33（均含 1 个 cls）；structured/Q 无 cls。
- **吞吐 = samples/s**（batch64 × forward 次数/秒），已修正（此前误标为 batches/s）。
- 记录：torch 2.14.0+cu126，A100-PCIE-40GB，attention sdpa（EMind），FP32（TF32 matmul/cudnn 状态见
  `runs/efficiency.json`）。structured 的慢主要来自 Hillis-Steele 参考扫描（O(T log T)，非融合 kernel），
  不代表理论下限。

## 五、分 SNR / 混淆矩阵 / 训练曲线

- `figs/fig_snr.png`：RML 上 structured 优势集中在 **SNR>0** 区（~+10pp），SNR≤0 四模型都接近随机；
  radar 上 Q 优势集中在 SNR≤0 区（SNR>0 四模型 ~1.0）。
- `figs/fig_confusion_rml.png` / `fig_confusion_radar.png`：逐类混淆（seed17）。
- `figs/fig_learning_curves.png`：seed17 val-F1 曲线（完整 history 在各 result.json）。
- 逐类 F1 / 混淆矩阵 / 分SNR 数值：`eval/{task}/{model}/s{seed}.json`。

## 六、协议与可复现

- 数据集：**RML2018.01A（Deepsig 2018.01a）cls_dev 划分**（24 类，train 19968/val 3120/test 8112，L=1024）；
  **RadChar-Tiny**（5 类，train 30000/**dev** 10000/**eval** 10000，L=512——本数据集 val 即 dev、test 即 eval）。
- 参数量（trainable）：Q 10.46M / structured 10.40M / EMind 9.68M / WavesFM 9.88M（±10% 内）。
- 统一协议：AdamW(lr, wd0.01, betas[0.9,0.999])、cosine + 5% warmup、grad_clip1.0、fp32、effective batch 128
  （batch64 × grad-accum2）、完整训练集、per-sample max-abs 归一化、val macro-F1 选 best、test 重载 best。
- seed 仅改变初始化 + 训练随机性，数据划分固定。Q 的 aux 分支（ComplexRecurBranch）已随 --seed 独立初始化
  （`Encoder(cfg,'Q',seed=seed)`；v2 初版曾硬编码 seed=17，已修复并补跑 Q seed42/2026，见 `DIFF.md`）。
- 数据 hash：RML train_x `2fec9871…` val_x `101daeaa…` test_x `5c297dea…`；radar train_x `50a1d62a…`
  val_x `02972b76…` test_x `c30412e1…`。
- 脚本：`train.py`（--seed/--lr/--epochs/--resume，last.pt 含 optimizer/scheduler/RNG）、`scheduler.py`
  （三卡队列、tmux 持久、8h 截止保护）、`efficiency.py`（每模型独立进程）。

## 七、限制

1. **三 seed 是初步稳定性判断，不是统计显著**；radar 上 Q 领先仅 ~0.9pp（接近 Q 自身 std），不能据此下
   细排序结论。
2. 30ep 未严格收敛（60ep 亦无提升）；不能据此宣称"充分收敛"，但排序对预算/lr 稳健。
3. lr 只搜 {3e-4, 1e-4} 两个值。
4. 不把"structured 领先"解释为"物理分支有效"，不把"SSM>ViT"泛化为普遍结论——本报告只陈述本协议下
   四模型的实测排名。
5. structured 的慢速度部分来自未优化的参考扫描实现，不代表理论下限。

## 八、交付物

- `report.md`（本文，修正版）、`DIFF.md`（差异清单）、`summary.csv`（48 run）、`summary_runs.json`、
  `analysis.json`（汇总）
- `runs/{task}/{model}/s{seed}_lr{lr}_ep{ep}/`：`result.json`（含 history）、`best.pt`、`last.pt`、
  `test_pred.npz`；60ep run 另有 `mid30.pt`。**Q 的 seed42/2026 四组为修正后产物**（覆盖了原错误版本）。
- `runs/efficiency.json`（修正版效率）、`figs/*.png`（4 图）、`eval/**/*.json`（逐类/SNR/混淆）、
  `checksums_best_pt.txt`（best.pt 路径+hash）
- 脚本：`train.py`、`scheduler.py`、`efficiency.py`、`merge_efficiency.py`、`eval_from_preds.py`、
  `analyze.py`、`plot.py`、`gen_csv.py`、`fix_runner.sh`
- v1 结果在 `public_baselines_rml_radar_v1/`，未覆盖。

**未完成清单：无**（41 job + 4 组 Q 修正补跑 + 效率重测全部完成）。
