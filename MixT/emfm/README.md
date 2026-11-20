# EMFM 首轮监督实验

更新：小规模掩码预训练、子集生成及预训练权重微调入口已补充，见 [PRETRAIN.md](PRETRAIN.md)。本页保留第一阶段监督实验说明。新增代码尚待服务器测试，不能以历史监督训练成功代替新功能验证。

状态：代码首版已完成，本地仅做Python语法/JSON/清单哈希检查；模型前后向、单元测试和训练必须在服务器执行，目前尚未运行。使用现有SFM环境，不需要在本地安装PyTorch。

## 本轮范围

- RML2018A HDF5 + 已冻结manifest，按subset与split双重过滤。
- 自实现Pre-LN全局Transformer、三阶段1D Swin、共享分类头形式和训练器。
- CPU正确性测试、真实数据多worker核验、单GPUsmoke、overfit、监督分类、独立评估。
- 全局与窗口均为显式attention，使用相同绝对正余弦位置编码，不引入不同加速后端。模型是Swin思想的一维实现，并非官方二维Swin的逐行移植。
- 掩码预训练已有独立入口。后续再加入相对位置偏置、IQ/AP、多源、回归与分离。本轮不代表完整EMind复现或基础模型通用性验证。

## 输入与结构

输入为`[B,L,2]`，有效点mask中True表示有效。逐样本共同缩放I/Q：`scale=max(max(abs(I)),max(abs(Q)),1e-8)`；保存scale。注意这与服务器reader_v2示意函数中的最大复包络不同，首版统一采用本文件公式。全零输入有限。fs缺失通过`fs_valid=False`表示，数值占位0不送入编码器。

patch=8，将连续8个IQ点展平成16维再线性投影。位置编码只在初始patch序列加入。partial patch中无效点清零；合并有效性取相邻token的OR。当前RML2018A等长1024，不存在partial patch。

Swin每阶段交替不移位/移位窗口，window=8 token、shift=4；使用左右padding构造窗口，禁止首尾循环相连。窗口网格不随批内其他样本长度变化；即使短序列也保持该网格，这一点与部分标准Swin实现短序列关闭移位的处理不同。相邻token合并后线性投影，末端LN和有效token均值池化。变长支持用于正确性与后续适配，尚未做变长性能实验。

| 配置 | 深度 | 通道 | heads | 参数量（按层公式计算，服务器check复核） |
|---|---|---|---|---:|
| global_small | 6 | 128 | 4 | 1,195,160 |
| global_matched | 6 | 176 | 4 | 2,251,592 |
| swin_small | 2/2/2 | 64/128/256 | 2/4/8 | 2,166,488 |

正式首轮对比使用global_matched与swin_small，总参数差约3.93%（相对Swin）。global_small是工程调试备选，不能作为等参对照。两者head维度与每头维度不同，后续还需固定分辨率窗口等消融才能分离层级与注意力机制贡献。

Swin在L1024/P8时token数128→64→32、通道64→128→256。256是特征维度，不是最大时间感知范围；window单位是各阶段token。

## 服务器准备

将本地整个`codes/emfm/`复制到服务器项目目录，例如`/home/user_wzk/wzk/projects/SwinE/emfm/`。保留服务器原有`manifest/`。在包含emfm目录的项目根目录执行下述命令。Python使用已配置好的SFM环境。

```bash
conda activate SFM
cd /home/user_wzk/wzk/projects/SwinE
export EMFM_H5=/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5
export EMFM_MANIFEST=/home/user_wzk/wzk/projects/SwinE/manifest/rml2018a_manifest.csv
```

不复制server_feedback到数据根，不重新生成或抽取manifest。HDF5路径和manifest路径都由命令传入。

## 第一步：正确性与真实数据接口（CPU，在服务器）

```bash
python -m unittest emfm.tests.test_correctness -v
python -m emfm.check --h5 "$EMFM_H5" --manifest "$EMFM_MANIFEST" --output runs/p1_checks.json
```

测试覆盖attention与独立参考计算、移位窗口分组/无首尾泄漏、奇数token合并、padding不影响预测、有限梯度、权重保存恢复、原始IQ归一化恢复、不同索引、spawn worker一致性。

check复核三个模型参数量和CPU前后向，并对真实smoke子集比较workers=0/2下样本ID、IQ、标签及其余字段，而不是只比较类别多重集。任一失败即停止，回传完整traceback，不能跳过后直接训练。

## 第二步：单GPU smoke

指定当前空闲GPU；下面cuda:0仅是示例。每次运行使用新的输出目录，防止覆盖。

```bash
python -m emfm.train --config emfm/configs/global_matched.json --h5 "$EMFM_H5" --manifest "$EMFM_MANIFEST" --mode smoke --device cuda:0 --output runs/global_smoke_s2024
python -m emfm.train --config emfm/configs/swin_small.json --h5 "$EMFM_H5" --manifest "$EMFM_MANIFEST" --mode smoke --device cuda:0 --output runs/swin_smoke_s2024
```

只用256条、一轮，验证损失/梯度有限、checkpoint可写。训练和评估是同一个工程子集，不是泛化结果。顺序运行即可，不需要三卡并行。

## 第三步：高SNR小样本拟合

```bash
python -m emfm.train --config emfm/configs/global_matched.json --h5 "$EMFM_H5" --manifest "$EMFM_MANIFEST" --mode overfit --device cuda:0 --output runs/global_overfit_s2024
python -m emfm.train --config emfm/configs/swin_small.json --h5 "$EMFM_H5" --manifest "$EMFM_MANIFEST" --mode overfit --device cuda:0 --output runs/swin_overfit_s2024
```

240条、每类10条、30dB；默认300轮（每轮2个更新），关闭weight decay和dropout。观察loss下降、同集合OA向接近100%发展；若明显停滞，先诊断，不直接判架构无效。这一阶段不使用正式val/test。

## 第四步：首轮监督开发对照

小样本拟合通过后再运行：

```bash
python -m emfm.train --config emfm/configs/global_matched.json --h5 "$EMFM_H5" --manifest "$EMFM_MANIFEST" --mode train --device cuda:0 --output runs/global_cls_s2024
python -m emfm.train --config emfm/configs/swin_small.json --h5 "$EMFM_H5" --manifest "$EMFM_MANIFEST" --mode train --device cuda:0 --output runs/swin_cls_s2024
```

两者相同seed2024、batch128、AdamW(lr3e-4,wd.01)、5%warmup+cosine、grad clip1、fp32、不增强。13轮×156批=2028更新步，每轮验证一次。按验证OA选best.pt，并保存宏F1、分SNR准确率和混淆矩阵。这里只是短程诊断预算，不承诺充分收敛；若仍明显欠拟合，双方同等延长预算并作为新实验记录。

首版不做梯度累积/DDP；若显存不够，可先双方一致降低batch重新运行，但需说明样本暴露与更新次数的改变，不冒充相同更新预算。bf16可在两配置同时显式改为bf16后另开实验，不与fp32混比。

## 输出、恢复和评估

每个run包含metadata.json（配置、清单与源码哈希、环境、参数）、metrics.jsonl、last.pt、best.pt、best_metrics.json。训练吞吐含数据读取/拷贝/反向/优化但不含验证和checkpoint写盘；CUDA显存是allocated峰值，不是系统总占用。不能据此宣称纯推理加速。

中断后，从同一输出目录的last.pt恢复；命令须保持配置、seed、mode、路径、workers和device一致，添加：

```text
--resume runs/对应运行目录/last.pt
```

恢复边界是完成的epoch；未保存epoch重跑。保存优化器、scheduler、训练/验证loader生成器及Python/NumPy/Torch随机状态。首版不保证不同设备/库版本的逐位一致；恢复功能也必须在服务器验证后再视为通过。checkpoint仅加载本项目自己生成的可信文件。

配置冻结后，单独评估测试集：

```bash
python -m emfm.train --config emfm/configs/swin_small.json --h5 "$EMFM_H5" --manifest "$EMFM_MANIFEST" --mode evaluate --checkpoint runs/swin_cls_s2024/best.pt --eval-split test --device cuda:0 --output runs/swin_cls_s2024_test
```

全局模型同理。默认evaluate只评估val；测试集不在训练循环中读取。不要利用测试成绩继续挑模型。

## 首次回传

先回传正确性测试输出、p1_checks.json和两个smoke/overfit的metadata.json、metrics.jsonl、best_metrics.json及异常日志。无需上传大权重或数据。确认后再看监督曲线与决定预训练，不一次性启动全部研究阶段。
