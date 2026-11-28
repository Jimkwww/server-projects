# 公开基线对比：四模型在 RML 调制识别 + RadChar 雷达分类

> `public_baselines_rml_radar_v1` —— 单 seed、统一协议、~10M 参数的四模型监督训练对比。
> 本报告为第一阶段（seed17、无 lr 搜索）结果，用于识别架构排序，不做最终统计定论。

## 一、主结果（test，seed17，30 epoch，val macro-F1 选 best 后重载）

| 模型 | RML test OA | RML test macro-F1 | radar test OA | radar test macro-F1 |
|---|---|---|---|---|
| **structured**（多分支物理 SSM） | **0.4491** | **0.4439** | 0.8634 | 0.8646 |
| **Q**（复相关 + S1 SSM） | 0.3907 | 0.3920 | **0.8758** | **0.8758** |
| EMind（ViT-MAE） | 0.3543 | 0.3483 | 0.8445 | 0.8457 |
| WavesFM（ViT） | 0.3565 | 0.3623 | 0.8427 | 0.8416 |

随机水平：RML 24 类 = 4.17%，radar 5 类 = 20%。

**核心观察**：
1. 两个 SSM 类模型（structured、Q）在两个任务上都**优于两个 ViT 类模型**（EMind、WavesFM），在匹配参数量下。
2. 更难的 RML 任务（24 类，1024）上 **structured 显著领先**（+5.8pp OA / +5.2pp F1 于第二名 Q）。
3. radar（5 类，512）任务上四模型差距小（OA 0.843~0.876，~3pp），Q 略胜；该差距在单 seed 下可能为噪声。

## 二、收敛与过拟合诊断（末 epoch train_acc vs val_acc）

| 模型 | RML tr/va | radar tr/va | 判断 |
|---|---|---|---|
| Q | 0.765 / 0.389 | 1.000 / 0.861 | RML 中等过拟合 |
| structured | 0.588 / 0.457 | 0.997 / 0.857 | **RML 过拟合最小** |
| EMind | 0.585 / 0.346 | 0.983 / 0.833 | RML 中等过拟合 |
| WavesFM | 0.968 / 0.358 | 1.000 / 0.829 | **RML 严重过拟合**（tr 96.8%→va 35.8%） |

Transformer 类在 RML 上过拟合更重；structured 的 SSM 主干泛化最好。

## 三、配置与公平性

- **参数量**（trainable，±10% 内）：Q 10.46M / structured 10.40M / EMind 9.68M / WavesFM 9.88M。
- **统一训练协议**（沿用 Q 已验证配置）：seed17、AdamW(lr=3e-4, wd=0.01, betas[0.9,0.999])、cosine + 5% warmup、grad_clip=1.0、fp32、effective batch 128、完整训练集（RML train 19968/val 3120/test 8112；radar 30000/10000/10000）。
- **归一化**：四模型统一 per-sample max-abs（Q 用 `max_abs_norm`，其余 `max-abs + clip(-5,5)`，clip 在归一化后为恒等，等价）。
- **选点**：val macro-F1 最高 checkpoint，重载后评 test。
- **模型配置**：Q dim=288/depth=12/state=144；structured dim=152/depth=12/5分支；EMind h=256/depth=12/inter=1024（patch4）；WavesFM embed=320/depth=8（segment16）。

## 四、效率（同一 A100，3 job/卡并行实测，30 epoch 单 job 耗时）

| 模型 | rml | radar | token 数 |
|---|---|---|---|
| WavesFM | 15.3 min | 15.8 min | 64/32（segment16） |
| Q | 36.6 min | 32.6 min | 128/64（patch8） |
| EMind | 44.2 min | 39.7 min | 256/128（patch4） |
| structured | 48.6 min | 53.1 min | 128/64（patch8） |

8 组 round-robin 排 3 卡，总墙钟 ~54 min（23:02 → 23:56 CST）。

## 五、已知限制

1. **单 seed**（17）：radar 上四模型差距 ~3pp，不足以宣称 Q>structured 等细排序。
2. **30 epoch 未完全收敛**：RML 四模型 val 仍在上行或过拟合区，更长时间/更强正则可能改变排序（但 structured 的领先幅度较大，较稳健）。
3. 无独立 lr 搜索：统一用 Q 的 lr=3e-4，对 ViT 类未必最优（可能加剧其过拟合）。
4. 参数量在 9.6~10.5M 内但非严格相等（EMind/WavesFM 略小）。

## 六、交付物

- `runs/{rml,radar}/{q,structured,emind,wavesfm}/result.json`（test 指标 + 训练曲线 history）
- `runs/{rml,radar}/{q,structured,emind,wavesfm}/test_pred.npz`（test 逐样本预测）
- `runs/{rml,radar}/{q,structured,emind,wavesfm}/best.pt`（val best checkpoint）
- `runs/summary.json`（汇总）
- `train.py`（训练脚本）、`train_all.sh`（tmux 编排）、`summary.py`、`match10.py`（参数匹配）
