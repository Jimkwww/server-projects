# 任务书：在端侧设备上复现 SpectrumFM AMC 实验并完成算力对比

> 这是一份交给**端侧设备上的 Claude agent** 的任务书（可直接作为项目记忆/任务 prompt 使用）。
> 你的目标不是改代码，而是**把同一套实验在端侧设备上跑通、计时、记录结果**，用于和服务器算力对比。

---

## 0. 背景（一句话）

SpectrumFM（IEEE JSAC 2025，频谱基础模型）的 AMC 任务复现实验已在服务器跑通（3×A100，F1=0.6474）。
现在要在**本设备**上跑**完全相同的流程**，重点记录**训练耗时、推理延迟/吞吐**，与服务器做初步算力对比。

## 1. 你的设备与工具

- 先**自行检测并报告设备硬件**：CPU 型号/核数、内存、是否有 GPU（`nvidia-smi`）、是否 Jetson 平台、操作系统。
- 报告 Python 环境：`python --version`、`pip --version`；是否已有 conda。

## 2. 拿到的东西

本设备上应已解压一个目录 `SpectrumFM_edge/`，结构：

```
SpectrumFM_edge/
├── code/                          # 实验代码（工作目录）
│   ├── amc.py                     # AMC 微调入口
│   ├── eval.py                    # 评估入口
│   ├── profile_model.py           # 效率测量（需 CUDA）
│   ├── Checkpoint/                # 预训练权重 + 已微调模型（已含，勿删）
│   └── Data/README.txt            # 数据集放这里
├── requirements.txt
└── 复现与算力对比指南.md          # 详细操作文档，按它执行
```

## 3. 执行步骤

1. **读一遍** `复现与算力对比指南.md`（第 3、4、5 节是操作重点）。
2. **装环境**：按设备类型装 torch（CPU / NVIDIA GPU / Jetson 三种，见指南 3.1），再 `pip install -r requirements.txt`。
3. **取数据**：优先从服务器 scp `RML2016.10a_dict.pkl` 到 `code/Data/`；否则 DeepSig 下载；**必须校验 md5 = `61bf35ac7f0b7d8843613453447ea290`**（不一致就换来源，否则 train/test 划分对不上）。
4. **冒烟自检**：跑指南 3.3 的两条 `python -c`，确认能加载模型（应打印 `OK, 参数 18.06M`）。
5. **跑微调**：`cd code && python amc.py`。全程记录墙钟时间（建议 `time` 或 tmux 挂后台 + 时间戳日志）。端侧若太慢可先 `--epochs 5` 估时间再决定跑满。
6. **跑评估**：`cd code && python eval.py`，记录 OA/P/R/F1 与耗时。
7. **跑效率**（有 CUDA 才做）：`cd code && python profile_model.py --name <设备名> --f1 <实测F1>`；纯 CPU 设备跳过，改用 `time python eval.py` 作为推理计时。
8. **回填结果表**（指南第 6 节的表），连同训练/评估日志一起整理。

## 4. 交付物（回传服务器）

1. 设备硬件/软件信息（第 1 节那几项）。
2. 结果对比表（指南第 6 节模板填好）。
3. `amc.py` 完整训练日志（含每 epoch 行 + 时间戳）、`eval.py` 输出、`profile_model.py` 输出或 `benchmark.csv`。
4. 若中途遇到报错，附完整 traceback 和你的判断。

## 5. 必须注意（避免白跑）

- 所有命令**先 `cd code/`** 再执行（相对路径）。
- 微调会**覆盖 `Checkpoint/amc/Our.pt`**（服务器版本），如需保留先备份。
- `process_data()` 预处理阶段**长时间无输出是正常的**（Python 循环归一化），不要以为卡死。
- 打包代码已处理 `map_location="cpu"`，纯 CPU 也能加载权重；`profile_model.py` 除外（它硬编码 CUDA，CPU 设备跳过）。
- 数据 md5 校验**不可省略**，这是保证「端侧和服务器结果可比」的关键。
