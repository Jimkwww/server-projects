# Revision-1 config（本轮实际使用的配置与 checkpoint）

## 模型配置（无 CFO，全轮统一）

```
patch=8, dim=176, depth=6, heads=4, state=128, ffn_ratio=4, dropout=0,
position=none, direction=forward, scan=parallel, ssm_gain=1.0
input_length: RML=1024 (128 token) / RadChar=512 (64 token)
classes: RML=24 / RadChar=5
```

来源：`round4_no_cfo_qstudy/configs/downstream_nocfo_{rml,radar}.json` 的 `model` 段。

## 数据

- RML2018A：`configs/data.json`（X [N,1024,2]，train 19968 / val 3120 / test 8112 **封存**）。
- RadChar-Tiny：`round2_overnight/configs/radchar.json`（512 IQ，train 30000 / val=dev 10000 /
  test=eval 10000）。
- 归一化：`max_abs_norm` in-loop（`normalization=none`）。

## 复用的 checkpoint（无新训练）

### Section 3 统一读出（encoder_final，冻结）

| 对象 | checkpoint |
|---|---|
| G_Qlearn fixed-Q MAE | `round4_no_cfo_qstudy/runs/pretrain/G_Qlearn_fixedq_mae_seed{17,42,2026}/encoder_final.pt` |
| G_Qlearn learn-Q MAE | `round2_overnight/runs/pretrain/G_Qlearn_mae_seed{17,42,2026}/encoder_final.pt` |
| G_corr MAE | `round4_no_cfo_qstudy/runs/pretrain/G_corr_mae_seed{17,42,2026}/encoder_final.pt` |
| G MAE | `runs/pretrain/G_mae_seed{17,42,2026}/encoder_final.pt` |

### Section 4 记忆诊断

- A 组（无监督编码器）：random → `round2_overnight/runs/random_ckpt/S1plain_seed*.pt`；
  mae/denoise → `round2_overnight/runs/pretrain/S1plain_{mae,denoise}_seed*/encoder_final.pt`。
- B 组（下游微调）：RML scratch → 旧 SSM 工程 `SSM/runs/clean/S1_seed*/best.pt`（old_classifier
  权重映射，已核实与原始 Classifier 前向逐位一致）；其余 →
  `round4_no_cfo_qstudy/runs/{rml,radar}/{scratch,mae,denoise}/S1plain_seed*/best.pt`。

### Section 5 状态统计

18 个 S1plain 模型 = 上述 A 组（9 个，random/mae/denoise × 3 seed）+ B 组 RML（9 个）。

## 统一读出协议（round-3 协议）

- 冻结 encoder → mean-pool 特征 → `StandardScaler`（fit train）→ L2 `LogisticRegression`
  （C=1, solver=lbfgs, max_iter=2000）。
- 不做基于 eval 的超参选择；RML main=val（test 封存），radar main=eval（dev 只做选模侧表）。
- 逐样本预测：`runs/predictions/*_pred.npz`（pred/y/snr/idx），并记录 encoder sha256。
