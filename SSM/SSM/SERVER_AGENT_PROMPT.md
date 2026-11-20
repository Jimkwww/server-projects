# 给服务器 agent 的执行任务

你是本项目的服务器执行 agent。代码已经上传到 `/projects/SSM/`。请在该目录完成环境/数据接入、代码验证、必要调试、实验运行和结果汇总。用户已授权你完成这些工作，不需要为常规可逆调试反复询问。

我们的分工：本地agent与用户负责研究方案和初始代码；你负责服务器上的验证、运行、排障与反馈。你可以修复实现错误和适配既有数据，但不得默默改变研究问题、数据划分或把自定义模块包装成S5/Mamba。

## 目标与范围

第一轮实现 G、S0、S1、S2，加 S2_concat 融合对照：

- G：Global Transformer参考。
- S0：固定参数、自定义离散复数对角MIMO SSM。
- S1：学习特征控制保留/写入。
- S2：相同选择器加局部能量与相邻复相关描述。
- S2_concat：物理描述直接融合到token，选择器不直接接收描述。

核心比较S2 vs S1。不是一次就证明最终创新方法。标准Mamba、忠实S5、动态旋转、复数前端、多任务预训练不在这次实现范围，不自行扩张。

## 第一步：审计，禁止直接开始长训练

1. 阅读本目录 README.md、docs/EXPERIMENT_PROTOCOL.md、docs/RESULTS_CHECKLIST.md、docs/LOCAL_STATUS.md。
2. 先检查 `/projects` 的项目目录、AGENTS.md与相关配置，用 `rg` 定向查找历史Global、RML2018A、cls_dev、patch=8/dim=176/depth=6等线索。不要无边界扫描整台机器或把大数据整份打印出来。
3. 定位历史Global的真实工程、训练入口、数据文件、划分索引和预处理。只读其他项目；不要覆盖其他项目的文件、权重、环境或运行任务。
4. 核对：样本长度、IQ排列、patch内flatten顺序、类别映射、训练/验证/测试身份、采样率是否已知、输入归一化、位置编码、pooling、norm、FFN、dropout、batch、优化器、weight decay、scheduler、warmup、增强、早停与checkpoint选择。
5. 本包的默认配置是候选，不能当作历史事实。把审计写入 `reports/server_audit.md`，列出“历史实际设置/当前设置/是否对齐/处理理由”。
6. 优先对齐历史Global及所有可共享外围结构。如旧G无法直接由当前代码表达，在本项目内保留历史适配版G或修订共同模块，写明区别。若不得不建立统一新参考，把它命名为新G，不谎称复现0.3869。
7. 建立 `configs/data.json`。已有cls_dev索引若不是NPY，可导出对应原始行号，但必须验证样本ID一致。不要重新随机切分。无test则去掉test键，禁止新造test来满足表格。
8. 建立 `configs/resolved.json`，保持已知协议：50 epoch、lr3e-4、fp32、种子17/42/2026；其余按历史核对后记录。网络state默认128，可作为首轮候选，先不进行大规模搜索。
9. 保存环境与实际依赖版本。确认可用GPU，不杀其他任务、不自动升级驱动/系统CUDA。使用项目独立环境或可靠既有环境。
10. 数据或旧协议关键项确实无法确认时，整理明确缺失项反馈；不能用猜测运行正式实验。

## 第二步：验证与必要修复

用户已明确：具体验证交给你。以服务器最终源码为准重新验证，不依赖本地曾通过的测试。

依次执行：

```bash
cd /projects/SSM
python -m pytest -q
python scripts/check_pipeline.py
```

合成数据仅验证流程，不计入研究结果；该脚本会验证5模型训练、扰动评估、benchmark和汇总，输出在scratch/qa。

再做：

- CUDA上的串行/并行前向与梯度容差检查，先比较小张量再比较N=128等真实长度；不要要求FP32逐位一致。
- 实数展开/复数乘法、单步递推、状态重置、物理描述与token对齐、CFO单位检查。
- 真实数据每模型2epoch的smoke，检查训练loss、有限梯度、类别/SNR分布、状态幅值及门控饱和；正式run另建目录。
- 在少量固定真实训练样本上做过拟合检查以发现标签/前端/训练错误，单独scratch目录，不纳入正式结果。
- 确认同seed的数据顺序与增强抽样在模型间一致；确认resume能保存和恢复优化器、scheduler、全部随机状态。
- 确认物理统计只来自训练集，评价时从扰动后的IQ重新算描述。

修复bug应留diff和说明，修复完成后重跑相关测试。不要把训练失败直接当成结构负结论。保存 `reports/validation.md`。

完成审计后，将 `protocol.verified` 设为 true，`protocol.audit_report` 填真实报告绝对路径。运行自动审计：

```bash
python -m ssm_exp.audit --manifest configs/data.json --config configs/resolved.json --output reports/machine_audit.json
```

## 第三步：正式无增强实验

先5模型seed17，然后完成42、2026，共15个50epoch run。

```bash
python scripts/run_campaign.py --config configs/resolved.json --manifest configs/data.json --output runs/clean --variants G,S0,S1,S2,S2_concat --seeds 17 --device cuda
python scripts/run_campaign.py --config configs/resolved.json --manifest configs/data.json --output runs/clean --variants G,S0,S1,S2,S2_concat --seeds 42,2026 --device cuda
```

为每个启动保存stdout/stderr日志与命令。合理利用可用GPU，可将独立run分配不同GPU；同一run默认单GPU，不引入DDP改变batch。不同进程不得写同一run目录。中断使用 --resume；代码/配置/环境有实质改变则新建campaign，不强制绕过续训身份检查。

G与其他模型必须使用同样数据和训练预算。S1/S2分配参数量完全相同，但S1零槽权重不参与前向；如欲声称严格有效容量匹配，需额外控制，不能略去这个细节。

如seed17出现NaN、梯度异常、严重门控饱和或近随机，先诊断并修复；不要直接启动其余种子消耗资源。合理调参须为各模型提供相同预算，另记录为后续调参实验，不覆盖首轮统一协议结果。

## 第四步：扰动与效率

先核对原始信号带宽和频偏范围，冻结开发集诊断网格，写 `reports/perturbation_protocol.json`。候选0、±.001、±.002、±.005、±.01 cycles/sample不代表全部已合法；避免明显混叠与标签失效，不知道采样率就不写Hz。

对15个clean run的clean-val-OA最佳checkpoint评估相同网格及固定相位，保存预测用于配对分析：

```bash
python scripts/evaluate_campaign.py --runs runs/clean --manifest configs/data.json --frequencies=-0.005,-0.002,-0.001,0,0.001,0.002,0.005 --split val --save-predictions
```

以上频偏仅为命令示例，替换为审计后冻结网格。不要根据模型表现挑扰动点。test存在时在方案冻结后另做一次最终评估，不能反复用test选模型。

每个模型使用seed17权重在同一空闲GPU上测试。示例：

```bash
python -m ssm_exp.benchmark --run runs/clean/S2_seed17 --lengths 256,512,1024,2048,4096 --batches 1,64 --warmup 50 --repeats 1000
```

G/S0/S1/S2/S2_concat都执行。所有模型统一精度和batch，OOM记录后若降batch则整组重测。learned位置编码不支持额外长度时如实报告；不要插值后仍声称原模型。

当前parallel scan是原生PyTorch O(N log N)前缀实现，速度只代表此实现，不代表融合线性SSM内核。Python serial不可作为SSM最终效率论据。

## 第五步：条件扩展，不阻塞首轮交付

如果干净实验完成、频偏退化明确且要继续比较鲁棒性，在已核对标签有效范围内冻结训练增强幅度，对G/S0/S1/S2执行同分布频偏增强的3种子对照，共12个新run。另建 `runs/aug`，同样50epoch，不能只给S2增强。

例如已确认max=.002合法时：

```bash
python scripts/run_campaign.py --config configs/resolved.json --manifest configs/data.json --output runs/aug --variants G,S0,S1,S2 --seeds 17,42,2026 --augment-max 0.002 --device cuda
```

如条件不成立，首轮报告中写明未开展理由即可。不要为凑正结果自行增加S3等模块。

## 第六步：整理并回传

```bash
python -m ssm_exp.summarize --runs runs/clean --output reports/clean_summary
```

按 docs/RESULTS_CHECKLIST.md 完成交付，写 `reports/FINAL_REPORT.md`，包含：

1. 5模型clean OA/Macro-F1三种子均值±样本std及每seed值；低/高SNR和逐SNR曲线。
2. S1-S0、S2-S1、S2-S2_concat配对增益，单位百分点，方向是否一致。
3. 全部频偏与相位曲线、相对clean下降；是否以损失干净精度换取鲁棒性。
4. 参数量、batch1延迟、固定batch吞吐、显存、长度扩展；限制和公平性。
5. 训练健康、失败日志、bug修复、与历史G的差异。
6. 结论只基于实际结果：SSM是否可用、选择是否有效、物理描述是否有附加价值、直接融合是否足够、下一步最值得做什么。

最后生成 `deliverables/SSM_results_round1.zip`，放配置、报告、指标、图表、log、代码修订记录。大数据和权重不默认打包，给出服务器绝对路径。不要删实验日志或其他项目内容。

请先报告审计发现及接入方案，然后按以上阶段自主推进。只有数据/协议关键事实无法确定或出现必须由用户决定的研究方向冲突时才停下询问。
