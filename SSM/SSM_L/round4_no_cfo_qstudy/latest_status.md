# Round-4 状态（2026-11-11 完成）

**状态：全部完成。** 105/105 训练 job（P0+P1）成功，0 失败；P0 附项雷达基线、P2 SSM 记忆
诊断、全部报告与打包均已产出。

## 结果速览

- **P0**（无 CFO 下游）：RML 上预训练收益 1.8–3.1pp（Global 主干吃去噪、SSM 主干吃 MAE）；
  雷达已近上限（~0.87）仅小增益。最优 RML = Q+MAE **0.4191**。
- **P1**（Q 学习阶段 2×2）：固定 Q 最优（A fixed-fixed 0.4083 > D learn-learn 0.3949）。
  Q 滤波器不需要学习。
- **P1 参照**：Q 滤波相关 > 原始复相关（G_Qlearn 0.3949 vs G_corr 0.3785，Global 主干）。
- **P2**（SSM 记忆）：有效记忆 ~16–64 token；K=1 全崩到随机，K=64 雷达无损失。状态双峰
  （多数短记忆 + 少量超长记忆）。
- **雷达简单基线**：flattened-IQ+LR 0.630/0.634；物理特征+LR 0.442/0.439。

## 交付物

- 结论 `reports/conclusion.md`；协议 `reports/protocol.md`；命令 `reports/commands.md`
- 表格 `reports/tables_{main,p1,baselines,p2}.md`；验收 `reports/acceptance.md`
- `reports/reuse_inventory.csv`、`reports/run_manifest.csv`、`reports/checkpoint_sha256.csv`
- `reports/results_summary.json`、`reports/p2_ssm_memory_{rml,radar}.json`、曲线 `reports/curves/`
- 逐样本预测 `runs/**/val_predictions.npz`、`runs/radar/**/eval_predictions.npz`
- 打包 `round4_package.tar.gz`（代码+配置+报告+预测，**不含权重**，权重留服务器）

## 复现

`reports/commands.md` 第 1/2/3 节（launch / resume / rebuild）。
