# Round 3 复核：本轮源码/流程改动记录

本轮为**收尾复核**：不新增训练、不改模型结构、不使用 test 调参。以下为新增加/修改的文件与原因。

## 新增脚本

- `scripts/eval_extended_cfo.py`
  - 统一冻结 order-B 扩展 CFO 评估：raw IQ → 注入 CFO → max_abs 归一化 → 重算物理描述 → 模型。
  - 在原有密集网格（±0.0001…±0.005）之外新增 ±0.003 / ±0.004 / ±0.0075 / ±0.01 / ±0.02。
  - 每个频偏点保存完整指标（overall OA/F1、逐类 precision/recall/F1、逐 SNR、low/high）与逐样本预测（id/label/pred/snr）。
  - 输出：`predictions_round3_review/*.npz`、`reports/round3_review/extended_cases.jsonl`。
- `scripts/aggregate_round3_review.py`
  - 读取 24 个 checkpoint 的 best_val.json / DONE.json / history.jsonl + extended_cases.jsonl，产出逐种子主表、完整 CFO 曲线、类别/SNR 分解（`reports/round3_review/*.md` 与 `per_seed_main.json`）。
- `scripts/run_extended_eval.sh`
  - 在 tmux 中批量运行上述评估的启动脚本。
- `scripts/make_deliverable_round3_review.py`
  - 打包 `SSM_results_round3_review.zip`。

## 修复的 bug

- **预测 npz 文件名冲突**：原 `eval_extended_cfo.py` 用 `run.name`（仅 basename）命名预测文件，导致 `runs/clean/G_seed17` 与 `runs/round3_aug/G_seed17` 的逐样本预测互相覆盖（276 个文件 = 12×23，无增强组的 npz 被增强组覆盖）。
  - 修复：预测文件名加入父目录前缀（`{parent}_{name}`），例如 `clean_G_seed17_clean.npz` vs `round3_aug_G_seed17_clean.npz`。
  - 冒烟测试确认 jsonl 行数 == npz 文件数（552 == 552），且 clean OA 与 best_val.json 逐位一致、CFO 值与 round-3 recheck 完全一致。

## 数据一致性核对

- 统一评估的 clean OA 与各 run `best_val.json` 完全一致（e.g. S1_corr_seed42 aug clean = 0.444551282051282）。
- 统一评估的 CFO（±0.001/±0.005）与 round-3 `aug_cases.jsonl` order B 完全一致。
- 全部指标用样本标准差（ddof=1）。
