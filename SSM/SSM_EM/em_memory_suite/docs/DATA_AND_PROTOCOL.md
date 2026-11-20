# 数据、划分与接入协议

## 实际文件格式

manifest.json 的 paths 相对于 manifest 所在目录；registry 中的 manifest 路径用绝对路径。

- x.npy: float32 [N,L,2]，最后维严格[I,Q]，不是[N,2,L]。L可整除patch8；不自动截断。
- ids.npy: numpy Unicode [N]；稳定样本ID。
- groups.npy: numpy Unicode [N]；同一次发射、原始长记录、设备采集session、同一母波形增强必须共用来源group。
- snr.npy: 可选 float [N]，单位dB；未知时不填，不从标签猜。
- 分类target: int64 [N]，0…C-1，全数据统一类别字典。
- 回归target: float32 [N,K]；task.loss_scale是固定物理尺度或仅train拟合的std，不从test拟合。
- 分割target: int64 [N,L/8]；每patch边界归属规则预先固定。多源重叠须另做多标签接口，不强行argmax丢失标签。
- splits: train/val/test，另可有ood。无标签预训dataset可只含x/ids/groups，train/val仍须分开。
- provenance: 版本、采集条件、生成种子、license、原分割文件SHA256、预处理说明。
- sample_rate_hz/center_frequency_hz: 未知写null。不得推测。

多个预训练manifest会串联，采样与样本量成比例；源均衡可预先构造等量按组采样manifest并登记extra_campaign。不将实现未有的均衡调度声称已启用。
当前batch必须同长度；长度实验用独立manifest或服务器增加bucket collator并测试。不得在collator中静默随机裁剪。

## 防泄漏

1. 先按源group/发射/会话/设备采集日期划分，再裁剪、加噪、增强。
2. 同一干净母波形不同SNR版本原则上保持同split。保留来源追踪。
3. 真实下游test/val不得参加预训练；若比较transductive方案，另命名并单列，不能混进主结论。
4. 分类任务设备标签可跨split，但采集session必须隔离；跨设备泛化与设备分类需区分任务定义。
5. 运行 audit_data 检查ID、groups、跨split与跨dataset逐样本hash。哈希只能识别完全相同的IQ；重采样、裁剪、相位旋转后的同源泄漏需元数据审计。
6. 原始训练集与预训练集重合是否允许：允许使用下游TRAIN的无标签数据，必须在全部比较方法一致；不能使用下游TEST/VAL。
7. 同一物理文件登记到多个路径会造成重复，应统一manifest。新pretrain联合manifest需保留原source_id/group_namespace，不能仅重命名规避审计。
8. synthetic split的RNG流、sample group独立；不能把生成器上的类分布泛化当真实部署能力。

## 输入与物理单位

默认不做逐样本RMS归一化，以保留强度信息。若旧数据已归一化，这部分信息已经不可恢复；报告只能解释相对幅度结构。
可用固定全数据集标定尺度，但标定参数仅train决定，训练/验证/test共用。
若为数值稳定做sample RMS归一化，必须额外保留log RMS通道，并给出不归一化对照。本包未默认启用此步骤。
CFO参数cycles/sample，乘exp(j2πνm)。每样点角频率rad/sample=2πν。
AWGN注入用当前观测功率定新增噪声，不覆盖原SNR metadata；结果同时保存原SNR及新增噪声参数。

## 现有项目接入

- 核验既有RML2018A类别字典、cls_dev划分、样本长度、训练数、验证数、独立test情况。
- 核验已有雷达任务的类别、单位、仿真参数范围和数据扩量方式。
- 原项目0.3869等历史数字仅背景，不能直接填入本次统一协议结果表。
- HDF5转换工具必须传已有split索引JSON和源groups。若数据不提供group元数据，标注UNKNOWN，不制造假的“独立group”；保留官方split和该限制。
- 对外部模型分别记录“官方checkpoint迁移”和“同语料重预训”。后者才更能隔离架构因素，仍需匹配训练目标/更新预算。

## 真实任务登记模板

服务器填写 `audit/dataset_inventory.csv`：
name,version,path,task,labels,N_train,N_val,N_test,length,fs,groups_available,split_hash,license,pretrain_allowed,missing_fields。
至少覆盖现有AMC和雷达任务。其它任务先搜索已授权服务器项目和已下载数据；必要的合法公开数据可按服务器网络权限下载，不虚构已获得的数据。
缺数据记录BLOCKED_DATA，继续所有不受影响作业。最终明确“已验证任务数”，不要用目标数替代。
