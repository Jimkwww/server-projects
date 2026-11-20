# 服务器 agent 任务书 · SSM_L 第一轮无标签 IQ 预训练

请在服务器完成以下实验，包括实现、验收、训练、评估和回传。先做数据清单和 CPU 侧准备。原工程 /home/user_wzk/wzk/projects/SSM/SSM 正在运行 Q_corr 物理门控实验，保持只读。只在 /home/user_wzk/wzk/projects/SSM/SSM_L 下创建本轮代码、配置和结果。检查队列、运行进程和 GPU 分配，在原实验释放资源或已有独立分配后启动新训练。不要终止原任务。重复执行时识别已完成 run，核对配置哈希并从对应 checkpoint 恢复。

## 1. 研究问题和对照

本轮做同域、相同数据暴露量的筛选。G 为现有 Global，Q 为可学习 Q_corr。每个骨干分别做 MAE 和去噪预训练。主终点是冻结骨干、全量训练标签线性探测的 clean val OA。clean 表示无额外人工扰动，原始观测可以含噪。低 SNR OA、Macro-F1 为重点次指标。比较下游表现，不用两种预训练 loss 的大小排名。

| 配对差值（同 seed） | 回答的问题 |
|---|---|
| G-denoise − G-MAE，Q-denoise − Q-MAE | 固定骨干时，去噪是否优于 MAE |
| Q-denoise − G-denoise | 哪种完整模型更适合去噪预训练后迁移 |
| (Q-denoise − Q-MAE) − (G-denoise − G-MAE) | 换预训练任务时，Q 的收益是否比 G 大 |

Q 与 G 同时在前端和主干上有区别。二者的差值不能单独归因于滤波器。Q 获益时，再提出 S1 去噪消融，本轮先完成核心矩阵。

## 2. 数据与隔离

- 用旧工程 RML2018A cls_dev 的原始划分。预期 train 19,968、val 3,120、24 类、每条 1,024 个复采样点。实际不符先追踪配置、索引版本并记录，不能悄悄重切。预训练仅用 train 的 IQ，训练代码不读取类别或 SNR 作为模型输入。val 用于预定义的探测与微调评价；本轮保持 test 封存。
- 固化 train/val 样本 ID、来源和哈希。检查 ID、已知采集记录和精确重复样本跨划分的重叠；若是从长录波切窗，核对录波或场景 ID 及时间区间。发现泄漏先修正后再统一生成对照；无法核实的来源字段单列。
- 并行完成服务器数据盘点，不把其他语料自动加入本轮。在项目使用的数据根目录、现有配置和索引范围内收集元数据，不扫描整个系统或拷贝原始大文件。输出 dataset_inventory.csv，字段为名称/版本、路径、信号域、任务、仿真或实采、原始复 IQ 或其他表示、通道/天线数、采样率、典型长度、样本数或时长、类别/SNR/噪声元数据、采集/场景ID、已有划分及与下游评估集可能重叠的来源。不知道写 unknown。清单准备好即单独回报。待筛选后设计第二轮更大语料，保留本轮下游划分。

## 3. 复用源代码并固定模型

记录旧工程 commit、实际使用文件的未提交改动及 SHA256。复制必要源码或用独立 checkout，检查 Python 导入路径确实指向 SSM_L；数据路径可只读复用。不要整目录复制 run、日志或权重。以旧工程当前可验证实现为准，先列出实际模型构造入口、token 输出位置、MAE 掩码与解码实现和数据归一化顺序。

- G：普通 Global Transformer，patch=8、dim=176、depth=6、heads=4。
- Q：原始实数 Patch 投影＋8 路可学习复数递归滤波＋16→176 滤波后复相关融合＋6 层 S1。核实 Q 滤波器的 32 个实数参数参与梯度及优化器更新。沿用原初始化。不要引入 frozen 或正在运行的 gate_c/d/cd。
- 给两个编码器暴露池化前的 [B,128,176] Token。预训练和下游之间完整传递前端、可学习滤波器、融合及 normalization buffers。Q 相关描述需要的统计量只从同一组 train IQ 拟合一次并冻结，两种任务使用相同初始化和共同统计量。损失对照使用同样的原始观测预处理。按 seed 保存相同骨干的起始共享权重；扰动随机流与模型初始化、数据顺序分离。

## 4. 两种预训练具体定义

### 去噪

原观测 x 做旧工程 joint max_abs 归一化一次。对归一化 x 加噪得到 y，不再分别按峰值归一化 y 和目标。两个输入分支都从 y 计算。人工加噪强度每条样本等概率取 {0,5,10,15} dB，定义为原观测总功率与新增噪声期望功率之比，不是数据集真实 SNR。每次访问重新采样独立 I/Q 高斯噪声。零功率样本计数，输出不产生 NaN。

下面是可以直接落到 SSM_L/pretrain/iq_noise.py 的 PyTorch 实现。约定 [B,2,T]，通道顺序 I、Q，浮点数据。同设备 generator 由训练入口管理和存档。若原 loader 为 [B,T,2]，只在接口边界显式转置。

```python
import math
import torch


def corrupt_iq(x: torch.Tensor, generator: torch.Generator):
    """Return noisy observation, per-real-channel sigma, and injected level."""
    if x.ndim != 3 or x.shape[1] != 2 or not x.is_floating_point():
        raise ValueError("Expected floating IQ with shape [B, 2, T]")
    levels = x.new_tensor([0., 5., 10., 15.])
    indices = torch.randint(len(levels), (x.shape[0],), device=x.device,
                            generator=generator)
    level_db = levels[indices]
    power = x.square().sum(dim=1).mean(dim=-1)
    sigma = (power / 2).sqrt() * torch.exp(-math.log(10) * level_db / 20)
    epsilon = torch.randn(x.shape, device=x.device, dtype=x.dtype,
                          generator=generator)
    return x + sigma[:, None, None] * epsilon, sigma, level_db
```

解码器逐 Token 输出 16 个实数，严格按原切块顺序还原为 [B,2,1024]。为与 MAE 共用等容量线性头，第一轮两组都把一个附加标量送入 Linear(177,16)：去噪用该样本已知的 sigma 广播至 128 个位置，MAE 该维恒置零。若历史 MAE 有必要的 mask-token 解码适配，则保留适配并记录差异。在 x 与预测之间计算 mean((Ihat-I)^2+(Qhat-Q)^2)，以时间位置计均值。这个任务恢复原观测；报告不得将输出称为真实无噪信号。下游完全移除解码器和 sigma 输入。

### MAE

先核查既有 Global/Swin 实验实际掩码比例、形状、填充值及索引恢复方式。若定位到原配置，两骨干统一使用其遮挡比例和形状。若没有可复用配置，预先规定随机遮住每条 128 个 Patch 中的 64 个，记录为本轮新参照。每条至少保留一个可见和一个目标 Patch。G/Q 的 mask 按样本与步骤配对。

Q 的滤波递推和相关描述必须只看已遮挡的原始 IQ，任何基于完整波形的样本级统计都要检查潜在旁路。严格验收在已统一尺度的输入上进行：只改被遮位置的目标值，编码器输入、Q 描述和预测应保持一致。G/Q 优先采用同一种编码器掩码策略。如果 Q 保留全部时间位置并使用 mask token，则主对照 G 同样保留这些位置，记录为密集掩码建模；若与历史 MAE 有别，历史分数不可作为本轮对照。对有序递推不能删除位置后假装时间跨度不变。缺失位置须获得可见上下文，尤其检查因果 Q 的序列首部。只在被遮位置计算与去噪同定义的复数平方误差，分母为实际目标位置数。

## 5. 预算、下游和选模

以下为第一轮预设，若硬件无法运行，先用一致的梯度累积保持有效 batch，记录实现偏差。

| 阶段 | 训练协议 |
|---|---|
| 预训练 | 四组，seeds 17/42/2026。每组 50 epoch、有效 batch 128、fp32/complex64、AdamW、lr 3e-4、wd 0.01、5% step warmup＋cosine、grad clip 1。无额外 CFO 增强，避免与加噪交织。每个 seed 四组 train ID 顺序相同。主结果统一使用第 50 轮权重；保存训练曲线、总样本数、有效步数、耗时和显存。 |
| 冻结线性探测 | 四组各从第 50 轮独立启动。encoder.eval()，冻结所有参数及 buffers。对未经额外扰动的 IQ 做均值池化＋同初始化线性分类头。全量 train 标签，batch 128，30 epoch，AdamW、lr 1e-3、wd 0、cosine，无增强。按 clean val OA 最佳、并列最早选轮，同报最后一轮。 |
| 全量微调 | 四组从同一份各自的预训练权重重新启动，不继承 probe 的头或优化器。复用旧工程已验证的监督协议，预期 50 epoch、batch 128、lr 3e-4、wd 0.01、5% warmup＋cosine、clip 1，按样本 CFO 增强 ±0.002 cycles/sample。核对实际增强、归一化、精度与选模并统一。 |
| 监督基线 | G-scratch、Q-scratch 三种子，同下游微调协议。历史结果仅在实现、数据划分、增强和选模均对齐且能取得逐种子数据时复用，否则补跑。 |

共 12 个预训练、12 个 probe、12 个微调，另有 6 个监督基线可复用或补跑。先做 CPU 验收及小 batch 过拟合，再完整跑 seed 17 四组，确认四组均可训练、下游权重加载正确后补齐余下 seeds。不依据 seed 17 的排名淘汰模型。训练中断后恢复模型、优化器、scheduler、采样器和噪声/遮挡 RNG 状态。

## 6. 正式运行前的验收

1. 对纯 I、纯 Q、已知位置冲激测试 Patch 切块和解码顺序，验证 [B,2,1024] 往返一致。与旧工程对齐同输入、同共享权重下的干净 token 或 logits。
2. 验证新增噪声 I/Q 均值接近零、方差相等、相关接近零，对各档用大批样本统计总功率比接近目标值（可用 ±0.3 dB）。不要强迫每条随机高斯噪声的实测功率精确相等。记录零输入、极小幅度处理以及随机种子可复现性。
3. 执行上述 MAE 掩码泄漏测试。检查 Q-denoise 改变注入噪声时，编码器的两条分支随之重算。Q 滤波器在预训练和微调时获得有限梯度并发生更新；probe 前后核对整个编码器 state_dict 哈希一致。
4. 所有组检查无 NaN、同 seed 共享参数初值及样本顺序、小批量 loss 可下降、checkpoint 重载预测一致。输出实际训练命令和参数量。验收失败先修复并重做，不能用已知有泄漏的 run 进入比较。

## 7. 指标、解释与回传

- 主表分 probe 和微调两张。报告 val clean OA、Macro-F1、SNR≤0 dB 与 >0 dB 分组 OA、各 SNR 样本数和 OA、逐类别召回。给出三种子均值±样本标准差及每个配对差值（百分点）。同一主终点同时给出各 seed 的两个任务增益及交互差值，不仅给最佳 run。val 结果注明用于开发；勿称为独立 test。
- 微调的 CFO ±0.005 cycles/sample 可作为次指标，正负分开，用同一个 clean 选出的权重。clean 输入含数据集原有噪声，低 SNR 用真实元数据分组，不用人工注入强度代替。
- 去噪检查：固定一小批有干净参考的可控通信或脉冲波形，在已知 s 上先加底噪得到 x，再加人工噪声得到 y。分开报告输入、模型输出、恒等映射以及一个注明带宽与设计依据的简单滤波器对 x 和 s 的误差。记录脉冲峰值衰减及出现位置偏差。诊断样本独立于训练，此项不等于雷达任务迁移。
- 返回 dataset_inventory.csv、source_manifest、split hashes、完整配置、可运行的新增代码/差异、启动及恢复命令、验收、曲线、配对表和结论。保存每组评估的 sample_id/true_label/pred_label/SNR，按 ID 对齐并复算 OA/F1；权重留服务器，提供路径与 SHA256。打包至 /home/user_wzk/wzk/projects/SSM/SSM_L/deliverables/SSM_L_round1_results.zip。汇报完成/复用/失败的 run 数与遗漏项。

本轮完成条件是同域四组对照、三种子下游结果和可复核回传包。关于“大规模多域预训练是否更好”，本轮只提交数据盘点和候选，不使用尚未审查的数据启动第二轮。
