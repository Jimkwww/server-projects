# Revision-1 修正说明（correction.md）

本轮不覆盖原报告，仅在 `revision1/` 下对原 `reports/conclusion.md` / `tables_*.md` /
`latest_status.md` 中的错误表述逐条给出「原表述 → 修正表述 → 证据」。编号对应任务书第 1 节
的六条。所有补正均未重跑预训练 / 全量微调；仅复用既有 checkpoint 重提取特征、重训线性读出、
重跑状态诊断。

---

## 1. Token 长度与「有效记忆 16–64 token」的撤销

**原表述**（conclusion.md 问题 4）：
> 「RML（24 类，1024 token）」「RadChar（5 类，512 token）」，以及结论
> 「→ 有效记忆窗口在 16–64 token」。

**修正表述**：
- RML 每条 1024 个复采样点，patch=8，故 **128 个 token**；RadChar 每条 512 点，patch=8，故
  **64 个 token**。原文「1024 token / 512 token」是把「采样点」误写为「token」。
- RadChar 的 K=64 **等于整条序列长度**（64 token），段内**没有发生任何中途 reset**
  （reset 只发生在 n=0 的起始点）。因此「K=64 雷达完全无损失」只能说明「整条序列本身就是
  一个窗口」，**不能作为「64 token 的有效记忆足够」的证据**。
- 据此**撤销「有效记忆 16–64 token」这一结论**；修正后的有效记忆尺度见第 6 条（状态统计）
  与第 4 节（segment-reset 重做）。

**证据**：
- `forward_tokens` 输出形状：RML → `[B,128,176]`；RadChar → `[B,64,176]`
  （`revision1/scripts/` 内核实，config `patch=8, input_length=1024/512`）。
- B 组 segment-reset（`revision1/reports/memory_finetuned_original_head.csv`）：RadChar 全部
  9 个模型 K=64 与 K=none 的 OA **逐位相等**（如 scratch_seed17 0.8718 == 0.8718），且
  `reset_counts` K=64 = 1（仅起始点 reset），确认 K=64 在段内无截断。

---

## 2. 雷达主结果应以 eval（test）为准，dev 仅用于选模

**原表述**（tables_p1.md / conclusion.md 问题 2）：
> RADAR 主表用 val(=dev) OA；据此说「A 最优」「下游把 Q 放开学习反而轻微有害」「雷达四设置
> 学习阶段无差别」「基本饱和在 ~0.87」。

**修正表述**：
- 雷达**主结果 = eval（test）**，dev 只用于 checkpoint 选模。重新聚合后（3 seed 均值）：
  A=86.42%、B=86.48%、C=86.76%、D=86.43%（详见 `tables_radar_eval.md`）。
- eval 上的排序为 **C > B > D ≈ A**，与 dev 上的排序（A > B > C > D）**相反**。因此不能再说
  「A 最优」或「学习 Q 有害」；dev 上的差异在 eval 上未复现，四设置 eval 差异 ≤0.34pp、且
  配对差异符号不稳定（A vs D：+0.9/−0.36/−0.55pp）。
- ~87% 只是「该任务在该结构下的当前水平」，**不能断言「任务上限/已饱和」**（无外部饱和
  参照，且 P0 各结构之间仍有 0.85–0.87 的可分辨差异）。

**证据**：`revision1/reports/tables_radar_eval.md`（P0/P1 eval OA 与 Macro-F1、per-seed、配对
差值），数据源为各 run 的 `eval_predictions.npz`（y/pred 直接计算，未重训）。

---

## 3. 预训练目标结论应限定到任务

**原表述**（conclusion.md 问题 1 / 一句话总结）：
> 「预训练任务与主干存在匹配：Global 主干更吃去噪、SSM 主干更吃 MAE」（未限定任务）。

**修正表述**：
- 「Global→denoise、SSM→MAE」**仅对当前 no-CFO RML 结果成立**。雷达上 Global 反而是
  **MAE > denoise**（G eval：mae 0.8659 vs denoise 0.8605），方向相反。
- 因此该「匹配」是任务相关的，不是主干的普遍性质。

**证据**：`tables_main.md` RML val（G denoise 0.4075 > mae 0.3902；S1plain mae 0.4101 >
denoise 0.3881）；`revision1/reports/tables_radar_eval.md` Radar eval（G mae 0.8659 >
denoise 0.8605）。

---

## 4. Q 学习阶段结论应限定到该设置

**原表述**（conclusion.md 问题 2 / latest_status.md）：
> 「Q 滤波器不需要学习；固定 Q 反而更好」「固定最优」。

**修正表述**：
- 2×2 只在 **Global+Q 主干 × F0-MAE** 这一个设置上做了。RML 上 A（固定-固定）≈ C
  （固定-学习）（val 0.4083 vs 0.4082），雷达 eval 上反而是 C（86.76）> B（86.48）> D（86.43）
  ≈ A（86.42）。**该设置下未证明学习滤波器有必要，固定 Q 具有竞争力**——但不能外推为
  「Q 不需要学习」的普遍结论。
- 「Q+MAE=41.91%」是主表中 **Q 骨干（Q_corr，可学习 Q 滤波器）** 的 RML val 结果，用的是
  **可学习 Q**，不是固定 Q。不能把它当作「固定 Q 的竞争力」的证据；固定 Q 的证据只来自
  A/C 共享的 `G_Qlearn_fixedq_mae` 权重。

**证据**：`tables_p1.md`（RML val A/B/C/D）；`revision1/reports/tables_radar_eval.md`（雷达
eval A/B/C/D）；`tables_main.md`（Q|mae = 0.4191 为 Q_corr 可学习滤波器）；`protocol.md`
第 1 节（P1 仅 Global+Q × F0-MAE）。

---

## 5. G_corr vs Q 不只是「滤波器」差异

**原表述**（conclusion.md 问题 3）：
> 「Q 滤波相关是否优于原始复相关（G_corr）」，把差异归因于「滤波器/描述子」。

**修正表述**：G_corr 与 Q 之间除了「是否经过 Q 滤波器」外，还同时存在多处**未隔离**的差异，
差值不能单独归因于滤波器：

| 属性 | G_corr | Q（G_Qlearn） |
|---|---|---|
| 描述子维度 | 3 维（log-能量、r.real、r.imag；能量槽在 forward 中被置 0） | 16 维滤波后复相关描述子 |
| 融合层 | `Linear(3→dim)`，bias=False，**非零默认初始化** | `Linear(16→dim)`，**零初始化** |
| 统计量 | 依赖 phys 统计（energy 归一） | 依赖 aux 统计（StandardScaler 风格 mean/std） |
| 描述子内容 | 原始 IQ 局部复相关 | 滤波后复相关（8 通道） |

**证据**：`round4_no_cfo_qstudy/models_r4.py`（GcorrEncoder 的 fusion 为 `Linear(3,dim)` 非零
初始化、physical_features 3 维且 energy 槽置 0；Q/G_Qlearn 的 aux_fusion 为 `Linear(16,dim)`
零初始化、16 维滤波相关描述子）。

---

## 6. 状态统计不能证明「双峰」或「有效记忆」，tau 均值对 rho→1 敏感

**原表述**（conclusion.md 问题 4 / tables_p2.md）：
> 「状态是双峰结构：多数通道短记忆 + 少量超长记忆（τ 均值可到数千 token）」，
> 并把「K=1 全部崩到接近随机」当作「模型真的在读序列状态」的证据。

**修正表述**：
- 在 18 个 S1plain 模型的固定可复现子集（rng(0) 选 512 条 RML val）上重做状态统计
  （`state_statistics.json` / `figs/`）：逐层 rho 中位数 0.79–0.83（对应 τ 中位 6–8 token），
  rho>0.999 的通道占比仅 **0.2–0.4%**（极薄长尾），log-R(t) 随 t **近线性衰减**（R(16) 已
  衰减到 ~0.05–0.4%），**未见清晰的双峰/双时间尺度结构**。
- **tau 均值对 rho→1 极敏感**（rho→1 时 τ→∞），是长尾伪影，不能作为有效记忆尺度；应以
  τ 中位数（6–8 token）与 log-R(t) 衰减为准。原「τ 均值可到数千 token」不构成「超长记忆
  通道」的证据。
- RadChar K=1（切断全部历史）OA 为 scratch 20–23%、mae 26–35%、denoise 24–38%，**并非全部
  落到随机（20%）**——denoise 的 K=1 仍有 0.38，说明其分类有一部分来自**单 token 特征**，
  「K=1 全部崩到随机」不成立。RML 的 K=1（1/24≈4.2%）落在 4.4–6.7%，接近随机。
- 据此**不再推断「下一步应显式拉长/分段化记忆」**——在更长记忆被（A 组固定+重拟合读出、
  B 组原始 head 的 segment-reset）独立验证之前，不做该建议。

**证据**：`revision1/reports/state_statistics.json` + `figs/state_stats_*.png`（18 模型聚合）；
`revision1/reports/memory_finetuned_original_head.csv`（B 组 RadChar K=1 各 seed）；
`revision1/reports/memory_pretrain.csv`（A 组固定/重拟合读出，见第 4 节）。
