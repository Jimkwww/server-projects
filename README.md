# server-projects

服务器上的深度学习实验代码与结果存档（Jimkwww）。只存**代码 / 报告 / 配置 / 图表 / 结果汇总**，不存权重、数据集等大文件——大文件留在服务器原路径，位置写进各项目的 md/README。

## 目录

| 目录 | 内容 |
| --- | --- |
| `SSM/` | 电磁信号分类主线。`SSM/` 复对角 MIMO SSM（G_corr 强基线、Global Q_corr、物理门控轮）；`SSM_L/` 无标签 IQ 预训练（MAE/去噪/双向 SSM）；`SSM_EM/` EM 通用 IQ 编码器多分支状态记忆（em_memory_suite，当前主线） |
| `ELM/` | SpectrumFM 复现、SpectrumFM_compression 量化/剪枝压缩、端侧对比 |
| `MultToken/` | 多尺度 token 化实验（T0~T5，T2 胜出，含稳定失败诊断） |
| `MixT/` | 不等长窗口 + 跨层循环轮换注意力 |
| `AlignAtt/` | 复相关关系注意力偏置实验（结论负结果，已归档） |
| `SwinE/` | 符号链接 → `/data1/user_wzk/archived_projects/SwinE`（已归档暂停） |
| `LMO/` | LLM 推理基准测试（compile / cudagraph / decode / lora / int4 等） |
| `DL_test/` | DL 实操练习（resnet 推理等） |

## 存 / 不存规则

**存**：`.py` 源码、`.sh` 脚本、`configs/` 配置、`reports/` 报告、结果汇总 `.csv/.json`、图表 `.png`、任务书/README `.md`、manifest `.csv`

**不存**（见 `.gitignore`）：权重 `.pt/.pth/.ckpt`、数据 `.npy/.npz/.h5/.parquet/.bin`、日志 `.log`、`__pycache__`、打包产物 `.zip/.tar.gz`、逐样本预测 `.npz`

## 维护方式

每次实验推进后，把改动 commit + push：

```bash
cd /home/user_wzk/wzk/projects
git add -A
git status              # 确认没有误加的大文件
git commit -m "<实验名>: <一句话改动>"
git push origin main
```

大文件（权重/数据/打包结果）不提交，路径写进对应 md。
