# MixT 项目归档（2026-09-21）

> **已冻结**：本目录是阶段一实验的最终状态，不再原地修改。若继续这条线，另建 `v2/`。

## 这是什么

MixT = 一个独立小实验：在 Global 主干（dim=176/depth=6/heads=4/128 token 无 merge）上，把注意力改成「同一时间轴的不等长连续分区 + 跨层循环轮换窗口长度」，验证其是否有价值。任务书见 [unequal_window_experiment_brief.md](unequal_window_experiment_brief.md)。

## 结果

RML2018A 调制识别（24 类），从头监督 50ep，seed 2024，四模型同参数量同初始权重：

| run | 注意力 | best_ep | val OA | macroF1 |
|---|---|---|---|---|
| G | 全局 | 21 | **0.3869** | 0.3766 |
| U-F | 不等长·固定 W_A=[4,8,16,24,32,44] | 22 | 0.3577 | 0.3451 |
| E32-S | 等长 32·移 16 | 18 | 0.3500 | 0.3245 |
| U-C | 不等长·循环轮换 | 27 | 0.3462 | 0.3484 |

## 结论

在 RML2018A 24 类分类任务、128 token、固定主干、seed 2024 和 50 epoch 训练条件下，全局注意力在验证集 OA 和 macro-F1 上均优于所测试的窗口变体。不等长窗口循环轮换相较固定分区未改善 OA，虽略微提高 macro-F1，但整体未显示优于 Global 的趋势。当前暂停该方向的多种子扩展和 MAE 实验，保留实现、权重及连通性分析供后续研究使用。

（补充：循环轮换虽通过连通性分析证明能跨窗传播，而固定分区是永久边界零跨窗流，但 U-C 仍低于 U-F——说明跨窗信息混合未带来增益。）

## 目录结构

- `emfm/` — 自包含代码（拷贝自 SwinE `pretrain_v1/emfm` + 新增 `unequal_window_models.py`/`train_unequal_window.py`/`connectivity.py`/configs/tests，全部通过 5+5 单测）
- `manifest/rml2018a_manifest.csv` — 冻结分类清单（sha256 `3c76d726…`）
- `run_phase1.sh` — 四 run 顺序训练脚本（注意：G 用 `train.py` 必须带 `--mode train`）
- `experiments/unequal_window_cyclic/` — 结果与报告
  - `PHASE1_REPORT.md` — 完整报告（结论 + 诊断 + 建议）
  - `summary.csv` — 四 run 全字段汇总
  - `connectivity_{cyclic,fixed,cyclic_d12}.json` — 连通性/可达性分析
  - `runs/*_50_s2024/` — metrics.jsonl / metadata.json（含源码 sha256）/ best.pt / last.pt
  - `runs/*_smoke_s2024/` — 冒烟记录（可忽略）

## 复现命令

```bash
cd /home/user_wzk/wzk/projects/MixT
bash run_phase1.sh          # G/U-F/U-C/E32-S × 50ep，GPU2
```
单个 run 见 `run_phase1.sh` 内 `run()` 调用；`G` 需 `--mode train`（`train.py` 默认 smoke）。
