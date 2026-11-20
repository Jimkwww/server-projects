# 电磁通用结构记忆：完整实验设计

版本 1.0；2026-11-19。执行对象：服务器 agent。所有研究结论待实验，不把概念合理性当成有效性。

## 0. 本轮研究定位

目标是可预训练、可迁移、多任务可用的电磁信号编码器，不是 24 类调制专用分类器。
输入以 IQ 为主；保留能量、相位/相干、频谱/时频、周期/重复及可选联合统计；保留通用 IQ 通路。
不统一消除 CFO、幅度、硬件失真，不在编码器内假设所有变化都应不变。
输出时间序列，含分支状态、完整拼接和融合表征；下游分类可池化，时间定位不能强制池化。
只对可观测属性提出可解释性主张：不知道采样率时输出 cycles/sample；不知道校准时不声称绝对 dBm；不把总频偏分解成 CFO 与 Doppler 真值。

本轮是对“物理结构指导记忆”的一次完整实验系列：可以自动连续完成全部作业，但依赖关系仍要求先有 checkpoint 再迁移。
“实验尽可能全”采用预注册核心矩阵 + 有目的的扩展，而非把所有超参数无限笛卡尔积。

## 1. 预注册假设与反证

| ID | 假设 | 必须对照 | 支持证据 | 反证/限制 |
|---|---|---|---|---|
| H1 | 结构化记忆改善通用迁移 | 参数匹配 generic_ssm、generic_ssm_recon、Transformer | 多任务配对 F1/误差改善，非仅 AMC | 仅一个任务提升不得称通用优势 |
| H2 | 收益不仅是物理前端 | pointwise_physical、multi_iq、固定前端 | 递推优于无递推，尤其长历史目标 | 仅瞬时目标好意味着更好的特征工程 |
| H3 | 局部分支监督建立职责 | local vs fused_aux vs 无辅助 vs 打乱目标 | 指定结构可从指定状态读出；干预响应正确 | 只看 attention/TSNE 不构成证据 |
| H4 | 模型实际使用时间记忆 | reset、history=1/4/8/16/32、短/长观测 | 长历史任务对 reset 更敏感，probe恢复退化 | 物理观测本身已有窗口，必须单独承认 |
| H5 | 编码器保存多任务信息 | modulation/emitter/sensing/regression/segmentation | 不同任务依赖不同分支，绝对变化仍可读出 | 不变性导致 emitter/参数任务损失为负面结果 |
| H6 | 复数/选择性/残差连接有独立贡献 | real_state、fixed_state、no_residual | 匹配资源条件下改善 | 不预设复数一定优于实数 |
| H7 | 变换关系监督有额外价值 | structured_equiv vs aug_without_equiv | 相同扰动曝光下迁移/transport误差改善 | 不能把增强收益全归因等变约束 |
| H8 | 保留通用分支利于覆盖未知结构 | no_general | 未见波形/失真保持性能 | 若物理分支被全绕过需如实报告 |
| H9 | 扩容与扩数据保持收益 | 多规模/数据量/长训练 | 同数据预算和资源下收益维持 | 小模型优势不能外推基础模型规模 |
| H10 | 辅助误差改善对应任务价值 | 所有目标的下游矩阵 | seed配对关联、多任务结果一致 | 不因重构更低判定编码器更好 |

主比较（预注册）：structured vs generic_ssm（同结构辅助任务但融合读出）、structured vs generic_ssm_recon（仅重构）、structured vs pointwise_physical（无学习递推）、structured vs fused_aux（同分支但无局部职责约束）。
主要结果：全数据冻结线性探测与全微调的各任务指标；不按最优测试成绩挑模型。辅助扩展均标 exploratory。

## 2. 模型结构与物理目标

基础参考：patch=8；4 个结构分支 + 通用分支；每支 dim=64、complex states=32、depth=3；仅为新方案初始尺度，不冒充既往 dim176/depth6/历史 S1。
每个分支：观测映射→多层归一化/SSM/FFN/残差连接→隐藏序列。
局部分支头从融合前本分支读出；没有当前 IQ 或其它分支的旁路；融合头看到所有分支，辅助约束与最终融合不冲突。

| 家族 | 参考输入 | 监督目标 |
|---|---|---|
| 能量 | patch log(1+均值功率/功率std/峰值功率/二阶功率矩) | 过去 H patch 同类统计的滚动平均 |
| 相干 | lag=1,2,4,8 归一化复相关实虚部 | 过去 H patch 的相干结构 |
| 频谱 | trailing64 Hann FFT，16 频带 log-power | 过去 H patch 频带分布 |
| 周期 | lag=8,16,24,32,48,64,96,128 的归一化相关模 | 过去 H patch 的重复证据 |
| 联合统计（扩展） | 归一化二阶/四阶复矩实虚部、功率矩、幅度三阶矩 | 同类滚动统计 |
| 通用 | 原始 IQ patch | 参与重构和下游，不强制人为语义 |

这些是可重复的观测统计，不是完备物理参数，也不是最优估计器。周期头当前保留“相关模的历史均值”，不是完整循环谱或任意长周期估计；联合统计是矩，不冒称已实现无偏高阶累积量。
特征的有效时域范围由数据长度和 lags 限制。为真正长周期追加长度/lag设计时必须单独版本化，不能仅靠网络更深宣称解决。
本轮先证明统计记忆价值；LFM 物理参数递推不进入主路线，不把整个模型改成波形残差学习。

## 3. 主消融矩阵（由代码自动生成）

- structured：4物理分支+general，局部辅助，重构。
- structured_equiv：相同结构，添加有限变换关系监督及相同增强曝光。
- aug_without_equiv：匹配增强曝光、不加关系损失。
- generic_ssm：一个通用 SSM，接相同全部辅助目标；检验结构分工是否必要。
- generic_transformer：因果注意力+位置编码，接同辅助目标；资源匹配参考。
- generic_pointwise：无时间交互 MLP，接辅助目标。
- multi_iq：多个分支均接 IQ，局部目标；隔离输入先验。
- fused_aux：保留物理分支，所有辅助头从融合表征读出；隔离监督定位。
- pointwise_physical：相同物理观测、无学习递推；注意观测窗口本身有历史。
- real_state：实数状态；重新匹配参数。
- fixed_state：无输入选择门；重新匹配参数。
- no_residual：去掉层间残差连接；仍是特征学习，非物理残差学习。
- no_general：仅物理分支，检查未知信息丢失。
- fixed_front：冻结输入投影随机权重；不等于冻结整个特征提取器。
- with_statistics：增加联合统计分支，检查辐射源/失真任务。
- uncertainty：为结构目标预测对数方差，使用异方差 Gaussian NLL；是观测误差可靠性，不是物理参数后验。
- bidirectional：共享参数双向扫描后平均；不是历史 bi64；方向开销单列。
- bidirectional_transformer：全局注意力对照；不将因果/非因果性能混合。
- shuffled_aux：在 batch 中打乱结构目标，检查一般辅助正则效应；小batch有弱化，记录。
- structured_recon / generic_ssm_recon / generic_transformer_recon / multi_iq_recon：只重构。
- without_energy / without_coherence / without_spectrum / without_period：从训练开始删除某支并匹配参数。
- history_1 / history_4 / history_16 / history_32：与 H=8 基础对照。
- aux_0.1 / aux_0.3 / aux_3.0：与辅助权重1对照。

总计 34 个命名变体（以 plan.json 实际枚举与 counts 为准；若代码清单调整须更新版本，不手工掩盖差异）。
另有测试时置零任意分支和全部学习状态 reset；这与重新训练删除分支是两种实验，不混同。

## 4. 预训练与迁移

固定主协议：FP32/complex64，AdamW lr=3e-4, wd=.01, clip=1，cosine LR；100 epoch 预训练，50 epoch下游；seeds17/42/2026；样本和每个任务划分完全相同。
预训练任务：随机遮挡50% patch，重构隐藏 IQ；从受损输入预测干净观测的过去H窗口结构；可选关系监督。
本实现为固定网格 causal masked reconstruction，所有 mask patch 保留占位，不冒称原论文 MAE token-dropping实现。
先掩码 IQ，再从受损 IQ 计算所有物理输入。不能先算干净特征再掩 token，否则跨patch统计泄露。
预训练 target 从干净观测生成；“干净”指未人为mask的接收IQ，不是无噪发射波形。
等变损失用输出差值匹配实际变换前后观测目标的差值，承认有限窗效应；不是严格群等变架构证明。

每个 checkpoint×每个下游×每个seed：冻结线性探测、全微调、同架构从头训练；标签1%/10%/100%，训练子集固定并分层。
标签比例指样本比例，若有相关分组则服务器改成组级采样并复用同一索引，不允许同源样本跨划分。
冻结主干时保持eval，仍可训练任务头；from-scratch禁用init。下游主协议不继续辅助loss，另可加 retain_aux 作扩展。
每个run按验证指标选best；不同预训练目标的val loss不可直接横比。统一比较 best checkpoint，另补 last checkpoint 稳健性（服务器可用相同命令改 init=last.pt）。

## 5. 多任务矩阵

至少目标覆盖：
1. AMC：现有 RML2018A 24 类，沿用 cls_dev/已有独立test，不将历史val数据当test重新解释；OA、macroF1、每类、SNR≤0/>0。
2. 雷达波形分类：现有雷达数据；按来源/参数组合划分；OA、macroF1。
3. 干扰识别/频谱占用：分类、多类时间分割，保留时间定位；macroF1、mIoU、每类召回。
4. 辐射源/设备识别：同设备不同采集会话/日期测试；重点统计/相干/通用分支；防止用信道/场景识别设备。
5. 连续参数估计：带宽、能量、可辨识周期、信号存在时间等；MAE/RMSE及物理单位。CFO/Doppler未可分时只评价可观测总偏移；合成注入量仅合成标签。
6. 异常/开放集：正常训练和验证，未知结构test；AUROC、固定val正常5%FPR阈值的test TPR/FPR。
7. 跨域/未见波形：通信预训→雷达、雷达→通信，留出某波形家族、参数范围外、设备或数据源外测试。

已实现接口：单标签分类、连续向量回归、patch级单标签分割、正常样本嵌入异常分数。多标签重叠源与多源分离需要另外的数据和输出形式，不把单标签头冒充这些任务。
序列分割默认每8点一个标签，不能宣称采样点级定位。真实边界标注按预注册规则量化。
缺某真实数据任务列 BLOCKED_DATA；完成其余全部。不因缺数据中断整体，也不以12类合成波形替代24类RML。

## 6. 跨规模、跨数据与训练时长

自动 campaigns：主规模64×3；small32×2；medium128×6；large256×12（分支宽度×层数），states约dim/2。
重点变体：structured / structured_equiv / generic_ssm / generic_ssm_recon / generic_transformer_recon / multi_iq / pointwise_physical。
每个规模重新参数匹配（总参数±5%），报告 trainable/total、辅助头、融合、任务头。若历史baseline无法匹配，分别给参数匹配与原配置结果。
数据10%/30%/100%，训练来源固定；长预训300epoch与100epoch对照。数据缩小同时改变更新步数，当前系列是epoch匹配；需额外通过epochs调整或固定总步数做等更新量配对，写入 extra_campaigns。
main/native_width 作为同原生宽度对照，不作参数公平性结论。
额外 registration：长度256/512/1024/2048，采样率变化、来源均衡混合、每域单独预训、留一域预训；数据按来源重构 manifest，不重复用test作预训。
长度8192基准测速不等于训练过8192，更不等于长序列表征能力。

## 7. 分支职责/真实记忆验证

A. 辅助头：每个目标的MSE，按SNR和结构存在性分组。当前函数为观测统计，零能量段不能解释为真实相位估计。
B. frozen probe矩阵：每个分支→全部结构目标，线性ridge用train拟合、val选alpha、test一次。不能用非线性大头从弱信息重建结构并夸大显式性。
C. reset：每patch取消学习状态传递；原probe不重训与重训probe可分别报告。当前实现提供不重训。
D. 删除分支：测试置零与从头删支分别报告，不将置零分布偏移当因果证明。
E. 输入干预：phase/gain/CFO/noise/IQ imbalance，记录状态变化与有意义读出的transport误差。隐藏状态变化大不等于物理解释正确。
F. 前缀因果性、样本独立性、mask泄漏由单元测试硬约束。
G. 数据外结构：合成可控参数和真实接收观测分开，参数目标不能写成能恢复所有真实生成因素。
H. 可靠性扩展：误差与预测方差分箱一致性、区间覆盖、误差排序相关；只在持出集报告。Gaussian近似覆盖差时不得宣称可靠不确定性。

成功标准不是各分支完全正交，也不是对角probe必然优于全部其他分支；相关物理结构允许冗余。需要“指定信息可读出+实际用历史+响应正确+对迁移有贡献”的证据链。

## 8. 鲁棒性与任务标签一致性

无增强预训与相同增强曝光的对照分开。AMC注册phase/gain/CFO/AWGN/IQimbalance/multipath/clipping；强度见registry。
noise参数是新增噪声相对于原信号功率的SNR，不是扰动后真实总SNR；必须保持命名。
CFO为cycles/sample，扰动不越Nyquist/数据已知频带约束；超界单列，不当一般鲁棒性。
频带占用、频率回归、增益回归等任务受扰动可能变标签，必须同步变换target。默认代码对此拒绝自动运行，避免错误结论。
下游硬件指纹任务不可默认所有硬件扰动都label-preserving；研究对象定义须先固定。

## 9. 统计、效率与结论

每个任务单列3seed均值/标准差及配对差值，bootstrap重采样seed对，不对高度相关的token重采样冒充独立样本。
3seed区间较粗，核心结果追加到5seed（如7、123）后再讨论稳定性；不因单次+0.2pp宣称突破。
跨任务同时报告平均rank、正/负迁移任务数；不同单位的MAE与F1不直接平均。多比较不报告未经校正的显著性。
基准含物理前端开销：B1/16/64，L256…8192，warmup10，repeat50，CUDA同步，median/p95/吞吐/峰值allocated；同卡同精度。
参考parallel scan是O(T logT) work，不是融合线性时间内核。报告算法层面和当前实现层面分开；不把Python serial慢解释为SSM理论慢。
历史模型和官方foundation checkpoints需同接口测速，并区分推理带辅助头与删辅助头两种部署形态。

最终可能结论包括：结构约束有收益；仅物理前端有效；只改善可解释性但无迁移收益；任务间冲突；计算成本不划算；需要更丰富目标。所有负结果保留。

## 10. 文献定位（方法来源，不代表本方案已证明新颖）

- Model-Based Deep Learning, Proceedings of the IEEE 2023: https://arxiv.org/abs/2012.08405
- DDSP, ICLR 2020: https://openreview.net/forum?id=B1x1ma4tDr
- Physics-Integrated VAE, NeurIPS 2021: https://arxiv.org/abs/2102.13156
- SINDy, PNAS 2016: https://doi.org/10.1073/pnas.1517384113
- SINDy Autoencoder, PNAS 2019: https://doi.org/10.1073/pnas.1906995116
- Deep Koopman, Nature Communications 2018: https://doi.org/10.1038/s41467-018-07210-0
- KalmanNet, IEEE TSP 2022: https://arxiv.org/abs/2107.10043
- Deep IMM, ACC 2022: https://doi.org/10.23919/ACC53348.2022.9867791
- RKN, ICML 2019: https://proceedings.mlr.press/v97/becker19a.html
- KVAE, NeurIPS 2017: https://proceedings.neurips.cc/paper/2017/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html
- DPN, 2021预印本: https://arxiv.org/abs/2106.10543
- ViterbiNet, IEEE TWC 2020: https://arxiv.org/abs/1905.10750
- SCTMR-Net, IEEE TII 2025: https://doi.org/10.1109/TII.2024.3452274

当前参考实现不是这些论文的复现。研究创新需在最终文献查新中检验“共享结构记忆+多任务迁移”的具体组合，而非把物理先验/辅助头本身当创新。

## 自动化补充说明

核心8变体默认额外7/123种子自动登记；其全数据probe/finetune自动登记last-init对照，finetune自动登记retain_aux=0.1/1.0对照。数据量系列同时生成按1/fraction增加epoch的近似等更新量系列，实际step须核验ceil和batch差异。与主协议混合比较时按init checkpoint和retain_aux字段分组。
