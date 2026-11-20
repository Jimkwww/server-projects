# Round6 目标对比 + 双向 SSM —— 结论（一页）
> 口径：均值 ± 样本std（3 种子）。comm 冻结 = RML val OA；radar 冻结 = RadChar eval OA；comm 微调 = RML val OA；radar 微调 = RadChar eval OA。G-MAE/Q-MAE 复用 round5。

## 问题 1：F1 规模训练下，去噪是否优于 MAE？
- G-denoise vs G-MAE（冻结 comm OA）：0.3908 ± 0.0038 vs 0.4015 ± 0.0045，Δ -0.0107 ± 0.0024
- G-denoise vs G-MAE（冻结 radar OA）：0.8075 ± 0.0057 vs 0.7849 ± 0.0161，Δ 0.0226 ± 0.0215
- G-denoise vs G-MAE（微调 comm OA）：0.4298 ± 0.0067 vs 0.4838 ± 0.0168，Δ -0.0540 ± 0.0232
- G-denoise vs G-MAE（微调 radar OA）：0.8623 ± 0.0007 vs 0.8656 ± 0.0006，Δ -0.0033 ± 0.0001
- Q-denoise vs Q-MAE（冻结 comm OA）：0.3556 ± 0.0027 vs 0.3875 ± 0.0056，Δ -0.0319 ± 0.0033
- Q-denoise vs Q-MAE（冻结 radar OA）：0.8275 ± 0.0011 vs 0.8213 ± 0.0041，Δ 0.0062 ± 0.0051
- Q-denoise vs Q-MAE（微调 comm OA）：0.4400 ± 0.0139 vs 0.4644 ± 0.0125，Δ -0.0245 ± 0.0045
- Q-denoise vs Q-MAE（微调 radar OA）：0.8742 ± 0.0040 vs 0.8714 ± 0.0040，Δ 0.0028 ± 0.0080

## 问题 2：MAE+去噪联合训练是否优于同骨干最优单目标？
- G-joint（冻结 comm）：0.3944 ± 0.0035；vs MAE Δ -0.0071 ± 0.0080，vs denoise Δ 0.0036 ± 0.0068
- G-joint（冻结 radar）：0.7960 ± 0.0091；vs MAE Δ 0.0111 ± 0.0251，vs denoise Δ -0.0115 ± 0.0049
- G-joint（微调 comm）：0.4524 ± 0.0309；vs MAE Δ -0.0314 ± 0.0182，vs denoise Δ 0.0225 ± 0.0356
- G-joint（微调 radar）：0.8689 ± 0.0044；vs MAE Δ 0.0032 ± 0.0050，vs denoise Δ 0.0066 ± 0.0050
- Q-joint（冻结 comm）：0.3647 ± 0.0042；vs MAE Δ -0.0228 ± 0.0029，vs denoise Δ 0.0092 ± 0.0015
- Q-joint（冻结 radar）：0.8227 ± 0.0058；vs MAE Δ 0.0014 ± 0.0095，vs denoise Δ -0.0049 ± 0.0048
- Q-joint（微调 comm）：0.4611 ± 0.0155；vs MAE Δ -0.0033 ± 0.0210，vs denoise Δ 0.0212 ± 0.0249
- Q-joint（微调 radar）：0.8698 ± 0.0030；vs MAE Δ -0.0016 ± 0.0029，vs denoise Δ -0.0044 ± 0.0061

## 问题 3：双向 SSM 是否降低 MAE 重构误差并改善下游/微调？
- MAE 重构 val loss：Q-MAE 0.1299 ± 0.0001 → Q-bi-MAE 0.0924 ± 0.0002
- Q-bi-MAE vs Q-MAE（冻结 comm OA）：0.3572 ± 0.0026 vs 0.3875 ± 0.0056，Δ -0.0303 ± 0.0072
- Q-bi-MAE vs Q-MAE（冻结 radar OA）：0.8051 ± 0.0080 vs 0.8213 ± 0.0041，Δ -0.0162 ± 0.0114
- Q-bi-MAE vs Q-MAE（微调 comm OA）：0.4815 ± 0.0023 vs 0.4644 ± 0.0125，Δ 0.0171 ± 0.0104
- Q-bi-MAE vs Q-MAE（微调 radar OA）：0.8757 ± 0.0048 vs 0.8714 ± 0.0040，Δ 0.0043 ± 0.0032
- （注：双向是否改善需看 Δ 的方向与幅度，正值为 Q-bi 更优。）

## 问题 4（补充）：双向 SSM 在去噪 / 联合目标上是否也优于单向 Q？

### denoise 目标
- Q-bi-denoise vs Q-denoise（冻结 comm OA）：0.3526 ± 0.0050 vs 0.3556 ± 0.0027，Δ -0.0030 ± 0.0076
- Q-bi-denoise vs Q-denoise（冻结 radar OA）：0.8188 ± 0.0020 vs 0.8275 ± 0.0011，Δ -0.0087 ± 0.0011
- Q-bi-denoise vs Q-denoise（微调 comm OA）：0.4441 ± 0.0230 vs 0.4400 ± 0.0139，Δ 0.0042 ± 0.0369
- Q-bi-denoise vs Q-denoise（微调 radar OA）：0.8737 ± 0.0020 vs 0.8742 ± 0.0040，Δ -0.0005 ± 0.0060

### joint 目标
- Q-bi-joint vs Q-joint（冻结 comm OA）：0.3518 ± 0.0125 vs 0.3647 ± 0.0042，Δ -0.0129 ± 0.0098
- Q-bi-joint vs Q-joint（冻结 radar OA）：0.8116 ± 0.0052 vs 0.8227 ± 0.0058，Δ -0.0111 ± 0.0019
- Q-bi-joint vs Q-joint（微调 comm OA）：0.4561 ± 0.0087 vs 0.4611 ± 0.0155，Δ -0.0050 ± 0.0197
- Q-bi-joint vs Q-joint（微调 radar OA）：0.8732 ± 0.0005 vs 0.8698 ± 0.0030，Δ 0.0035 ± 0.0034

## 一句话总结
- F1 规模（10 万样本、50 epoch）下：① 合成去噪预训练整体不优于 MAE——comm 冻结（G -1.1pt、Q -3.2pt）与微调（G -5.4pt、Q -2.4pt）均更差，仅 radar 冻结略优；② MAE+去噪联合训练是两单目标的折中，未超过最优单目标；③ 双向 SSM 使 MAE 重构 val loss 从 0.130 降至 0.092（−28.8%），证实「单向因果 → 高重构误差」假设，但下游迁移仅在微调后略优（comm +1.7pt、radar +0.4pt），冻结读出反而更差（comm -3.0pt、radar -1.6pt）。
