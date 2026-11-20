# 实现说明与可审查边界

## SSM

u为实token；B映射到复数状态；delta=softplus(W_delta u)+epsilon；g=sigmoid(W_g u)；
a=exp(-softplus(alpha)*delta) exp(j theta)；h_n=a_n*h_(n-1)+g_n*B u_n。
B与C为共享可学习映射，不是输入动态生成；theta不随输入变化。读出实虚拼接后线性映射，与某个复C读出实部可对应，但不声称与旧代码初始化/参数化完全一致。
real_state取消复数表示，通过参数匹配重新分配宽度；不能只比较“状态数相同”后宣称参数公平。
fixed_state令delta和g为1，仍可学习固定衰减与旋转。

parallel affine scan递推组合(a2,b2)∘(a1,b1)=(a2*a1,b2+a2*b1)。实现用log-depth doubling，不使用除累计乘积的易下溢公式。
有串行参考用于数值/梯度测试；当前kernel无Triton/TileLang依赖。
双向扩展共享方向参数、正反scan平均；加倍扫描工作，非历史正64反64方案。

## 表征与读出

每分支维度d，拼接K*d；融合为 joined+MLP(LayerNorm(joined))；分支序列仍在返回字典中。
辅助头取融合前分支状态。aux_route=fused取融合表征作为不定位监督对照。
任务默认线性头：分类/回归mean-pool时间，分割逐token。没有隐藏的更强分类头帮助新模型。
通用分支直接接IQ，叫补充分支，不叫数值上x-xhat的残差分支。
无时间全连接模块kind=pointwise，用同残差FFN以分离递推作用。
Transformer含正弦位置编码，主协议causal；bidirectional_transformer不加causal mask。

## 预训练安全点

mask原始IQ后再算全部物理观测；mask占位token不携带干净IQ信息。
teacher descriptors在loss侧no_grad计算，不作为student输入。
history_average包含当前patch和之前H-1个patch；启动窗口按实际可用长度平均，不填充虚假统计。
频谱窗只使用patch结束前64点；period最大lag128；统计读出必须按这个定义解释。
mask会造成物理输入观测偏差，这是任务难度，不应偷用干净值修复。

## 不确定性

uncertainty变体在本分支预测log variance，截断[-8,6]。对应观测统计目标的Gaussian NLL，非物理生成参数的真实Bayesian后验。
诊断给分箱pred variance vs observed MSE、1sigma/2sigma覆盖、NLL。通用下游默认未读取logvar，验证可靠性有用后可增加任务端读取，另列对照。

## 训练、恢复与公平性

训练保存config hash，恢复不得改配置。epoch边界保存model/optimizer/scheduler/Python/NumPy/Torch/CUDA/DataLoader RNG。
中途batch中断会重放该epoch，不是任意step精确恢复；epoch边界已做逐参数exact恢复测试。
梯度累积按窗口真实样本数加权；修改microbatch/accum保持effective batch后需新run目录和变更记录，不能改已跑run配置。
num_workers>0的数据加载不含随机变换，随机遮挡/增强发生在主进程；未来若加入worker随机增强需扩展worker_init与恢复测试。
val masking固定种子且恢复训练随机状态；模型选择不用test。
评估默认batch64；超大模型OOM需调评估batch并记录，不能静默遗漏test样本。

## 代码已实现 vs 需要服务器数据/项目适配

已实现：特征/目标、SSM及消融、预训/迁移、3类任务接口、异常分数、鲁棒扰动、分支诊断、参数配额搜索、系列调度、汇总、基础HDF5转换。
服务器接入：实际数据manifest、历史S1/Q精确baseline、官方Radio-FM/SpectrumFM/EMind接口、任务特定标签变换、未登记的新型数据任务。
本包不含这些未提供的私有数据/历史代码；不能伪称已经适配完成。
采样点级分割、多标签检测、多源分离、在线stream缓存、多GPU DDP、production融合kernel未实现；若服务器扩展需明确新增实现与独立测试。

## 建议的扩展适配契约

外部编码器wrapper统一 `forward(iq)->{'sequence': Tensor[B,T,D]}`。
若只输出全局embedding，就只参加分类/回归，不伪造时间分割输出。
官方checkpoint参数和预处理原样保留为一个结果组；同语料预训练再另一个结果组。
历史S1/Q原分类头应替换成与本包相同简单头再做公平迁移组；历史完整pipeline另列复现组。
所有wrapper记录原repo URL、commit、checkpoint SHA256、license、patch/token长度、归一化、是否bidirectional、最大支持长度。

## 数值与性能边界

FP32/complex64主实验。AMP/compile/fused kernel仅后续独立工程实验，不能混入结构对比。
参数量只是一种控制；真实FLOPs、内存和延迟可能不同，必须报告。
按参数匹配会改变通道数/表征维度，故另做native-width系列。若结论只在一种匹配法成立，写清楚。
所有统计目标是可观测结构，不证明每个hidden channel物理可解释；分支之间物理信息可以重叠。

## 配对增强控制实现细节

structured_equiv与aug_without_equiv均执行同一原观测/变换观测两视图辅助监督（默认权重0.2），具有相同输入曝光；前者再添加读出差分关系损失（0.2），后者该项为0。主mask重构仍以原观测进行。
