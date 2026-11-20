# Round-2 代码改动与验证

> 项目非 git 仓库，此处以人工 diff 说明代替 `git diff`；完整源码见交付物 `source/`。

## 改动（相对 round-1）

### ssm_exp/model.py（唯一实质改动）

新增 3 个变体，全部增量、向后兼容（round-1 五个变体行为不变，clean 复现已验证）：

1. `S1_energy`：S1 基础 + token 层融合，仅 `log-energy` 维（`p[...,1:]=0` 置零两个相关维）。
2. `S1_corr`：S1 基础 + token 层融合，仅 `Re(r),Im(r)` 维（`p[...,0]=0` 置零能量维）。
3. `G_full`：G（注意力）基础 + token 层融合完整 3 维描述，保留 G 的正弦位置编码。

具体改动点：
- `Classifier.__init__`：`variant` 白名单加入 3 个新名字；`self.fusion` 条件从 `=='S2_concat'` 扩展为 `in ('S2_concat','S1_energy','S1_corr','G_full')`；`pos_mode` 正弦条件从 `=='G'` 扩展为 `in ('G','G_full')`。
- `Block.__init__`：Attention/SSM 分支条件从 `=='G'` 扩展为 `in ('G','G_full')`。
- `ComplexSSM.__init__`：selector 创建条件加入 `'S1_energy','S1_corr'`（selector 仍接收零物理槽位，与 S1/S2_concat 一致；`forward` 里 `aux = p if variant=='S2' else zeros` 无需改，新变体自然走 zeros 分支）。
- `Classifier.forward`：`p` 计算条件扩展为 `in ('S2','S2_concat','S1_energy','S1_corr','G_full')`；`S1_energy` 置零 `p[...,1:]`、`S1_corr` 置零 `p[...,0]`（保持 3 维接口与 fusion 层不变，零输入权重不参与）。

### ssm_exp/engine.py
- `--variant` choices 加入 3 个新名字。
- `fit_physics` 触发条件加入 3 个新变体（它们都需要 phys_mean/phys_std 标准化缓冲）。

### 新增文件
- `configs/resolved_round2.json`：协议名 `RML2018A_cls_dev_round2`，其余与 round-1 完全一致。
- `scripts/recheck_order.py` / `run_recheck.sh` / `run_recheck2.sh`：order A/B 冻结协议评估。
- `scripts/supervise_round2.py` / `launch_tmux_round2.sh`：训练调度。
- `scripts/aggregate_round2.py`：汇总表生成。

## 验证

- 前向/反向冒烟：3 个新变体 logits 形状正确、梯度有限；参数计数严格吻合预期（S1_energy=S1_corr=2,217,464=S2_concat；G_full=2,252,120=G+528）。
- 1-epoch `--smoke` 训练端到端通过（3/3）。
- 冻结协议评估中 order A clean 与各 run `best_val.json` 逐 run 精确相等（复现校验通过）。
- 训练 9 run 全部 exit 0，best_epoch 15–41 区间内、无 NaN/塌缩。

## Round-1 勘误

第一轮 `FINAL_REPORT.md` 曾将物理能量特征描述为「未标准化 / log 压缩」——错误。`fit_physics` 实际对 3 维描述做了标准化（phys_mean=[-1.495,0.577,0.005]，phys_std=[0.830,0.470,0.180]，8192 训练子集、rng(0) 固定）。本轮 `recheck_summary.md` 已更正。
