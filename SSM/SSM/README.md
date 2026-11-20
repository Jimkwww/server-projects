# 电磁复数 SSM：第一轮机制实验

目标服务器目录：`/projects/SSM/`。本包为可独立运行的研究工程；不含数据、旧项目代码或旧权重。

**先读 `SERVER_AGENT_PROMPT.md`，再由服务器完成验证和正式训练。** 本包不是已验证的 RML2018A 实验结果，也不声称已复现历史 Global。

## 实现范围

| 名称 | 定义 |
|---|---|
| G | 同外围结构的 Global Transformer 候选参考，服务器核对历史实现后定案 |
| S0 | 自定义离散复数对角 MIMO SSM，固定时不变系数 |
| S1 | S0 加学习特征驱动的衰减与写入选择 |
| S2 | S1 的选择器额外接收局部能量和归一化相邻复相关 |
| S2_concat | 物理描述直接映射并加到 Token，选择器不直接接收描述的融合对照 |

本轮不实现标准 Mamba、忠实 S5、动态旋转 S3、复数前端、多任务预训练。这些属于后续研究，不能把当前自定义模块称为它们。

## 文件入口

- `ssm_exp/model.py`：前端、物理描述、复数状态、选择器、网络块和分类头。
- `ssm_exp/data.py`：HDF5/NPY 数据读取、固定索引划分、数据审计。
- `ssm_exp/engine.py`：训练、物理统计拟合、checkpoint、续训。
- `ssm_exp/eval.py`：干净、固定相位、归一化频偏评估。
- `ssm_exp/benchmark.py`：推理延迟、吞吐、峰值显存、长度扩展。
- `ssm_exp/summarize.py`：多种子表、配对差值、曲线、效率表。
- `configs/base.json`：待服务器核对的候选配置，不直接等同于历史配置。
- `configs/data.example.json`：数据 manifest 示例；不能直接用于正式训练。
- `scripts/run_campaign.py`：顺序执行多模型多种子，不抢占或自动选择 GPU。
- `scripts/make_synthetic.py` / `scripts/check_pipeline.py`：纯合成数据验证。
- `tests/`：数学、数据和续训测试。
- `docs/EXPERIMENT_PROTOCOL.md`：实现定义、比较口径、实验矩阵。
- `docs/RESULTS_CHECKLIST.md`：服务器需要整理的交付物。
- `docs/LOCAL_STATUS.md`：本地验证范围及未验证事项。

## 安装与目录

解压后的目录应直接是 `/projects/SSM/ssm_exp/`、`/projects/SSM/configs/` 等，避免 `/projects/SSM/SSM/` 多套一层。

在服务器选用已有合适环境，或建立项目独立环境。不要修改其他项目的全局依赖。

```bash
cd /projects/SSM
python -m pip install -r requirements.txt
```

服务器需先确认 CUDA PyTorch 已正确安装，再执行上述命令。`requirements.txt` 为最低兼容范围；正式实验记录实际 `pip freeze`，不擅自升级服务器 CUDA/驱动。

数据支持：

1. 单个 HDF5 文件，默认数组键 `X/Y/Z`，可配置；
2. 三个独立 NPY 文件，分别是 IQ、标签、SNR，支持 mmap；
3. 每个划分为一维整数 NPY 原始行号数组。索引必须来自既有 `cls_dev`，本包不会生成真实数据划分。

如果旧项目采用其他存储，服务器可实现最小数据适配或导出索引，保持样本身份和预处理一致。不要用本包合成数据替代真实实验。

## 服务器基本顺序

```bash
cd /projects/SSM
python -m pytest -q
python scripts/check_pipeline.py
python -m ssm_exp.audit --manifest configs/data.json --config configs/resolved.json --output reports/machine_audit.json
```

`check_pipeline.py` 只在 CPU 上使用合成数据验证流程，不进行科研实验；默认目录 `scratch/qa` 已存在时会拒绝覆盖。

核对历史协议后，服务器在 `configs/resolved.json` 中设置 `protocol.verified=true`，并让 `protocol.audit_report` 指向真实核对报告。然后进行真实数据 smoke：

```bash
python -m ssm_exp.engine --config configs/resolved.json --manifest configs/data.json --out runs/smoke/S2 --variant S2 --device cuda --smoke --epochs 2
```

所有模型 smoke 正常后，执行主实验：

```bash
python scripts/run_campaign.py --config configs/resolved.json --manifest configs/data.json --output runs/clean --variants G,S0,S1,S2,S2_concat --seeds 17,42,2026 --device cuda
```

先只运行 seed17 可将 `--seeds` 改成 `17`，完成检查后再运行 `42,2026`。中断续训添加 `--resume`。不要在续训期间改变代码、环境、数据或训练配置；需要变更时建立新 campaign。

单个完成的 run 评估示例：

```bash
python -m ssm_exp.eval --run runs/clean/S2_seed17 --manifest configs/data.json --split val --save-predictions
python -m ssm_exp.benchmark --run runs/clean/S2_seed17 --lengths 256,512,1024,2048,4096 --batches 1,64 --warmup 50 --repeats 1000
python -m ssm_exp.summarize --runs runs/clean --output reports/clean_summary
```

负频偏列表须使用等号，如 `--frequencies=-0.005,-0.002,0,0.002,0.005`。该网格仅作示例，正式范围必须在服务器核对带宽、原数据频偏与标签有效性后冻结。

## 重要限制

- 当前只有单向 SSM，G 为全局双向交互。比较反映整体骨干差异，不能将差异全部归因于复数状态。
- 并行扫描为原生 PyTorch affine prefix，工作量 O(N log N)，没有专用融合内核。必须将实测速度与“理论上 SSM 可线性实现”区分。
- 权重和 checkpoint 只加载自己生成、可信的文件（使用 `torch.load(..., weights_only=False)` 保存完整优化器及随机状态）。
- 全流程 fp32/complex64，关闭 TF32，不使用 AMP；确定性设置为 warn-only，CUDA 精确重现仍须服务器验证。
- 数据 provenance 对索引与 manifest 做 SHA256，原始大数据仅记录路径、尺寸和修改时间。不是对全量数据内容的密码学保证；服务器可额外记录原始数据校验和。
- 若已有历史预处理超出 `none/sample_rms`，必须适配并测试，不能用近似替换。
