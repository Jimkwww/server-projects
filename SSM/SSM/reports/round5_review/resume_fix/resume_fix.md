# Round 5 补充审计：恢复训练时重算辅助统计量问题修复说明

## 1. 问题

`ssm_exp/engine.py` 的 `train()` 在进入训练循环前**无条件**调用 `fit_aux()`（以及
`fit_physics()`），即使传入了 `--resume`。对于 Q_linear / Q_corr / R_corr 等 B 线变体，
`fit_aux()` 会在固定的 8192 样本子集上重算 16 维辅助描述符的 `aux_mean` / `aux_std`，
并**覆盖**从 checkpoint 恢复的、本应冻结的标准化 buffer。

后果：任何用 `--resume` 续训的 B 线变体，其辅助描述符标准化统计量会被悄悄重估，
续训前后描述符尺度不一致，破坏"训练期冻结统计量"的协议约束。

## 2. 修复内容（engine.py / model.py）

1. `fit_physics()` / `fit_aux()` 只在**首次训练**（非 `--resume`）分支执行；续训不再重估。
2. checkpoint 字典新增 `aux`（首次 fit 的 mean/std/样本数/ids_sha256）与
   `frozen_hash`（Q_corr_frozen 冻结支路参数 SHA256）。
3. 续训时从 `ck['aux']` 恢复统计量；若 checkpoint 缺少 `aux_mean`/`aux_std` 或 `aux`，
   **显式报错**（"refusing to silently refit"），不再静默重估。
4. Q_corr_frozen 每个 epoch 结束时校验冻结支路 hash，一旦被梯度/权重衰减改动立即终止。
5. 优化器改为只接收 `requires_grad=True` 的参数，冻结支路彻底排除在 AdamW 之外
   （此前 `model.parameters()` 会把冻结参数也传进优化器，虽因 `grad is None` 而不会被
   更新，但语义不干净）。
6. `model_info.json` 增加 `trainable_real_scalar_parameters`，与总量并记。
7. 新增变体 `Q_corr_frozen`（见 [model.py](../../../ssm_exp/model.py)）：
   - 与 Q_corr 完全相同的初始化（同一 seed+70002 的 `ComplexRecurBranch`，同一 S1 主路，
     W_f 零初始化），唯一区别是支路 `raw_alpha/theta/b` 从初始值冻结（`requires_grad_(False)`）。
   - 修正 `ComplexSSM` 的 selector 变体列表遗漏（原列表缺 `Q_corr_frozen`，会导致主路
     selector 缺失、结构与 Q_corr 不一致）。

## 3. 修复前后代码指纹

| 项 | SHA256 |
|---|---|
| 修复前 `ssm_exp/*.py` | `c58df87f3dc72a5d6bd7696ec29e9e67e1fab7bea0e486538393791e330781a0` |
| 修复后 `ssm_exp/*.py` | `6e7486f7373a90c6e61e47bf7603c68ad4cab5674366853e1e74caab222f506a` |

完整 diff 见同目录 `engine.py.diff`（108 行）与 `model.py.diff`（47 行）。

## 4. 恢复一致性测试

`tests/test_round5_resume.py`（smoke、CPU、Q_corr_frozen、3 epoch）：

- **逐位一致**：从头连续训练 3 epoch 与"从头训练 1 epoch → 中断 → `--resume` 续训 2 epoch"
  的最终 checkpoint 完全一致，逐位比较以下字段：
  - 模型参数 + `aux_mean`/`aux_std` buffer；
  - optimizer state、scheduler/lr；
  - 数据 RNG 流（`rng`）与 CFO 增广 RNG 流（`augment_rng`）；
  - `aux`（首次 fit 的 mean/std 记录）、`frozen_hash`。
- **显式报错**：删除 checkpoint 中的 `aux_mean`/`aux_std` 后 resume，进程报错
  "lacks frozen buffer ... refusing to silently refit"，而不是静默重估。

测试结果：`2 passed`；全测试套件 `75 passed`。

## 5. 对 21 个历史 Round-5 运行的影响审计

逐一核查 `runs/round5_iq/` 下 21 个运行的启动命令（`.log` 中的 `# cmd:`）与日志：

- **全部 21 个运行的 `# cmd:` 均不含 `--resume`**；
- 全部 21 个日志无 "resume config/data/code/environment differs" 或 "use --resume" 痕迹；
- 全部 21 个运行的 `provenance.code_sha256` 均为 `c58df87f…`（修复前代码），且无任何
  `FileExistsError`（该错误只在非 resume 命中已存在 checkpoint 时出现）。

结论：**21 个历史运行全部"未受影响"**——该 bug 只在 `--resume` 续训路径触发，而历史上
没有任何运行使用了 `--resume`（训练中断后均从头重跑，而非续训）。因此原始 Round-5
结果无需重训，可继续作为对照组。
