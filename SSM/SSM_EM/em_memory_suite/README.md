# 电磁通用结构记忆编码器：完整实验包 v1.0

本包是可执行的独立参考实现与服务器任务书，不含训练结果、真实数据或外部模型权重。
目标：IQ 输入，结构化多分支状态记忆保留电磁通用属性，任务解码器选择使用。
不是先做 LFM 分类，不把频偏/硬件失真统一消除，不预设每个调制类别一个专家。

## 先读

1. `docs/EXPERIMENT_DESIGN.md`：研究问题、所有实验、判定与局限。
2. `docs/DATA_AND_PROTOCOL.md`：真实数据接入、划分、单位与防泄漏。
3. `docs/IMPLEMENTATION.md`：实际实现、公式、接口和没有冒充实现的部分。
4. `SERVER_PROMPT.md`：可直接复制给服务器 agent 的完整 prompt。
5. `docs/ACCEPTANCE.md`：最终交付与验收清单。

## 最简使用（在服务器）

使用独立环境或已有验证过的 CUDA PyTorch 环境。不要盲目覆盖服务器已有 CUDA 依赖。

```bash
cd em_memory_suite
python -c "import torch,numpy; print(torch.__version__,torch.cuda.is_available(),numpy.__version__)"
python -m unittest discover -s tests -v
python -m tools.server_smoke
```

`server_smoke` 是很小的合成集端到端功能测试，包含独立参数匹配和下游接口；其结果不能写进科研结论。

接入服务器真实数据，复制并编辑 `configs/registry.example.json` 为 `configs/registry.server.json`。
路径全部写绝对路径，至少加入现有调制和雷达数据。更多真实任务按任务书接入。

```bash
python -m tools.audit_data --registry configs/registry.server.json --out audit/data.json
python -m tools.campaigns --registry configs/registry.server.json --out results/full --run
```

该命令自动生成并顺序执行：主消融矩阵、原生宽度对照、3 个额外规模、数据量、长预训练、已登记单域和附加实验。
不需要完成一轮后再询问用户是否继续。执行顺序只是 checkpoint 依赖，并非分阶段审批。

只生成计划、先检查总作业数：省略 `--run`。单独执行或继续某个计划：

```bash
python -m emmem.runner --plan results/full/main/plan/plan.json --retry-failed
python -m emmem.report --root results/full/main/plan
```

每个计划有排他锁、状态清单、依赖关系、日志、epoch 边界断点恢复。多 GPU 可分配不同 campaign 的独立目录；不允许多个进程同时写同一 plan。
本包默认单卡 FP32/complex64，不宣称已经实现 DDP、跨调用流式状态缓存或融合 CUDA kernel。

## 文件结构

- `emmem/features.py`：4 类因果物理观测、滚动历史目标、受控扰动。
- `emmem/models.py`：复数选择性 SSM、参考扫描、分组状态、融合、辅助头、任务头。
- `emmem/data.py`：mmap 数据读取、标签比例、数据审计。
- `emmem/train.py`：预训练/冻结线性探测/全微调/从头训练、梯度累积、恢复。
- `emmem/evaluate.py`：分类、回归、序列分割、扰动和分支干预。
- `emmem/diagnostics.py`：跨分支探测矩阵、记忆重置、受控变化响应。
- `emmem/novelty.py`：训练正常数据拟合的异常分数基线。
- `emmem/benchmark.py`：包含物理特征计算的端到端延迟/吞吐/显存。
- `emmem/plan.py`, `runner.py`, `report.py`：全量配置、依赖调度、聚合与配对差异。
- `tools/convert_rml_h5.py`：根据明确索引转换 HDF5；需要服务器安装 h5py。
- `tools/campaigns.py`：完整实验系列自动生成与执行。
- `tests/`：小张量机制测试和微型保存恢复集成测试。

## 重要边界

- 本实现是 **S1-style 新参考模型**，不是用户历史 S1/Q 的逐行复现；历史代码需服务器接入。
- Radio-FM / SpectrumFM / EMind 等官方基线必须使用核实的官方代码、版本与协议，不能用本包 Transformer 代替并冠名。
- 真实多任务结论取决于真实数据。没有某任务数据时，列为 BLOCKED_DATA，不伪造，不以合成数据替代。
- `registry.example.json` 只示范一个真实任务接口；服务器应按任务书扩充，而不是把一个任务跑完就宣称完成通用模型验证。
- 所有本地检查为代码正确性检查；正式训练、全量数据、长序列 GPU 基准均由服务器执行。
