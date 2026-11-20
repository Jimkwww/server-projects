# 小规模掩码预训练与下游验证

本轮范围：RML2018A单源、IQ输入、保留位置的掩码重建。先验证实现，再做10万条开发规模预训练。不是官方MAE丢token实现，不代表已完成多源基础模型训练。

当前代码已写好，本地只做静态检查；本文所有模型测试、预训练、微调命令均需在服务器执行。不要把代码交付理解为训练已通过。

## 1. 首版设置

| 项目 | 设定 |
|---|---|
| 原始数据 | RML2018A，X[N,1024,2]，仅IQ参与预训练 |
| 训练子集 | train_smoke 10,000条，train_dev 100,000条，前者为后者前缀 |
| 重建验证集 | val_smoke 2,000条，val_dev 10,000条，前者为后者前缀 |
| 抽样 | seed2024，无放回均匀抽样，报告类别/SNR分布，不按分类成绩选样 |
| 隔离 | 预训练train排除下游val/test；预训练val排除下游全部cls_dev和预训练train |
| 重复检查 | 所选候选与cls_dev的归一化float32 IQ SHA256，排除完全相同输入，不能保证没有近重复/相同采集记录 |
| 归一化 | 与监督阶段一致，逐样本I/Q共同最大绝对分量缩放，epsilon=1e-8 |
| 掩码 | patch8，128个token，随机掩64个，即50%；保留所有位置，嵌入后用可学习mask token替换 |
| 重建 | 全局与Swin均用单层线性重建头；每个最终token还原其覆盖的连续IQ，裁掉padding |
| 损失 | 只计算被掩有效IQ分量的MSE；I/Q各计一个分量，不计算可见点损失 |
| 优化 | AdamW，lr3e-4，wd.01，batch128，5%warmup+cosine，clip1，fp32 |
| 初始预算 | smoke 2轮/158次更新；dev 13轮/10,166次更新（默认样本数） |
| 选权重 | 固定重建验证集、固定逐样本mask，以验证masked MSE保存best，另存last |
| 下游 | 同一cls_dev划分，重建头丢弃，严格载入encoder，新分类头，50轮微调 |

类别只在子集生成时核对数据与统计分布。正式PretrainIQ读取器不读取Y，也不向训练器返回label。两模型必须使用同一个dataset.json和清单、相同种子/数据顺序/掩码预算。

训练与验证的mask随机数独立于模型初始化；验证mask由sample_id和固定seed确定，不随batch大小、顺序或epoch变化。训练mask动态变化并随checkpoint保存随机状态。

全局模型最终128token，每token重建8点；Swin最终32token，每token重建32点。重建头使用相同规则但参数量不同，metadata单独报告encoder、重建头和mask token参数。编码器没有原始IQ旁路。

掩码前按完整样本做共同幅度缩放，与监督预处理一致；这不是可见点统计归一化，也不是物理绝对幅度重建。若未来研究幅度回归需另定目标契约。

## 2. 同步代码与环境

本发布包部署于独立实验目录`/home/user_wzk/wzk/projects/SwinE/pretrain_v1`，将本地整个`codes/emfm`文件夹放在其中。原分类manifest仍引用现有SwinE目录。不要在正在运行50轮实验的代码目录中途替换源码。

```bash
conda activate SFM
cd /home/user_wzk/wzk/projects/SwinE/pretrain_v1
export EMFM_H5=/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5
export EMFM_MANIFEST=/home/user_wzk/wzk/projects/SwinE/manifest/rml2018a_manifest.csv
export EMFM_PRETRAIN=data/pretrain_rml_v1/dataset.json
```

使用现有torch/numpy/h5py，无需pyarrow。两张A100可以一张跑global一张跑Swin，等GPU空闲再启动；本轮不需要DDP。

## 3. 服务器正确性检查

```bash
python -m unittest emfm.tests.test_correctness emfm.tests.test_pretrain -v
```

新增测试：固定mask批次无关性、仅被掩点产生重建梯度、修改隐藏IQ不影响模型预测、编码器严格迁移且分类头保持新初始化、预训练数据隔离与缓存一致性、两轮连续训练和中断恢复的权重一致性。

测试内仅构造极小临时数据并在服务器CPU运行。任一测试失败就停止，回传traceback，不跳过后直接长训。

## 4. 生成清单（只做一次，两模型共享）

```bash
python -m emfm.prepare_pretrain --h5 "$EMFM_H5" --cls-manifest "$EMFM_MANIFEST" --output data/pretrain_rml_v1
```

读取原始数据但不改写、不复制全部波形。只扫描所需候选和固定cls_dev内容，生成四份JSONL及dataset.json。dataset.json最后写出，存在才表示生成完成；包含清单SHA256、实际数量、分布、交集审计及源schema。目录已存在时拒绝覆盖，失败续做使用新输出目录。

train_dev允许包含原下游train中的部分样本，实际交集写入审计；不能称为未见数据源的迁移。val_dev额外与下游train隔离，避免以后微调时影响预训练选模解释。没有采集session标识，当前只能核验ID及完全相同归一化波形，不声称完全消除相关性泄漏。

生成后将dataset.json回传留档；文件虽记录服务器绝对路径，训练仍以显式--h5参数为准，并逐样本核对归一化内容哈希。

## 5. 1万条smoke预训练

```bash
python -m emfm.pretrain --config emfm/configs/pretrain_global_matched.json --h5 "$EMFM_H5" --bundle "$EMFM_PRETRAIN" --stage smoke --device cuda:0 --output runs/pre_global_smoke_s2024
python -m emfm.pretrain --config emfm/configs/pretrain_swin_small.json --h5 "$EMFM_H5" --bundle "$EMFM_PRETRAIN" --stage smoke --device cuda:1 --output runs/pre_swin_smoke_s2024
```

默认只把选中的归一化IQ缓存到CPU内存，不缓存全部250余万条。10万条IQ约0.82GB，另有验证集与元数据；结合用户服务器内存条件可使用。缓存模式workers=0，启动时逐条核对内容哈希。内存不允许时用`--no-cache --workers 2`按需读取。

smoke是工程检查，不用其权重作正式微调比较。观察有限梯度、masked MSE和零预测基线，以及initial_validation到各轮的变化；两轮内MSE未明显下降不自动判失败或通过，需要结合数值与目标信号诊断。

## 6. 10万条dev预训练

smoke数值正常且代码检查通过后：

```bash
python -m emfm.pretrain --config emfm/configs/pretrain_global_matched.json --h5 "$EMFM_H5" --bundle "$EMFM_PRETRAIN" --stage dev --device cuda:0 --output runs/pre_global_dev_s2024
python -m emfm.pretrain --config emfm/configs/pretrain_swin_small.json --h5 "$EMFM_H5" --bundle "$EMFM_PRETRAIN" --stage dev --device cuda:1 --output runs/pre_swin_dev_s2024
```

dev默认从随机初始化开始，不续接smoke，以保持明确的10,166更新预算。不加载之前分类权重。

输出metadata.json、initial_validation.json、metrics.jsonl、best_metrics.json、best.pt、last.pt。验证记录固定mask MSE、恒零预测MSE及二者比值、分SNR MSE。恒零基线用于发现“只预测零”的退化，低重建损失本身不能证明表示质量好。

恢复时保持代码/配置/数据/设备等不变，在同一输出目录添加`--resume runs/对应目录/last.pt`。完整epoch保存状态，未完成epoch重跑。不要为延长预算直接更改epochs再resume；改变调度需作为独立实验。CPU恢复一致性由上述测试验证，CUDA仍需实际检查。

## 7. 预训练后的分类微调

编码器结构、归一化和下游清单必须与预训练契约匹配，加载使用strict=True；分类头重新初始化。50轮分类配置已附在本次代码中，旧13轮配置保留。

```bash
python -m emfm.train --config emfm/configs/global_matched_50ep.json --h5 "$EMFM_H5" --manifest "$EMFM_MANIFEST" --mode train --init-encoder runs/pre_global_dev_s2024/best.pt --device cuda:0 --output runs/ft_global_50_s2024
python -m emfm.train --config emfm/configs/swin_small_50ep.json --h5 "$EMFM_H5" --manifest "$EMFM_MANIFEST" --mode train --init-encoder runs/pre_swin_dev_s2024/best.pt --device cuda:1 --output runs/ft_swin_50_s2024
```

与已经计划的50轮从零分类进行对照，数据/seed/batch/lr/调度相同。若服务器50轮基线改过其他设置，先以其实际metadata对齐后再比较，不与13轮基线当成同预算结果。

冻结编码器线性评估：在对应微调命令上追加`--linear-probe`并换输出目录，编码器eval且requires_grad=False。首版沿用分类优化设置用于流程对照，不声称是线性评估最优超参。先完成全量微调比较也可以。

测试集仍不进入训练/选型。正式配置冻结后，用既有evaluate入口和对应50轮config单独评估best.pt。

比较表：global从零50轮、Swin从零50轮、global预训练后微调50轮、Swin预训练后微调50轮。报告增加的预训练计算量，不把“同微调预算”误称为“总训练成本相同”。

## 8. 回传内容

先回传测试输出、dataset.json、两个smoke的metadata/initial_validation/metrics/best_metrics。随后回传两个dev预训练与微调的同类文件。无需波形或大权重。

当前不缺少启动这轮所必需的信息。后续多源预训练才需要RadChar/BSS下游保留集落地、单位和源身份规则等额外信息。第一轮不加入这些未冻结数据。
