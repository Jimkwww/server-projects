# Q_corr 物理门控轮 — 训练前验收摘要

验收脚本 `scripts/accept_qcorr_gates.py`（3 seed） + `tests/test_qcorr_gates*.py`（pytest）全部通过。

## Check 1 — 手算描述子
常量复指数（`s[m]=e^{jωm}`）→ `c≈1`（最大偏差 1.19e-7）、`d≈0`、首块 `d[:,0]` 精确为 0；两相（增量 0 与 π/2）→ `r0=1, r1=j, d1=√2`。✅

## Check 2 — CFO 变换
`perturb` 逐样本乘 `e^{j2πf m}` → `r_n` 整体共旋 `e^{j2πf}`（最大误差 2.5e-7），`c=|r|`、`d=|r_n−r_{n−1}|` 精确保持。✅

## Check 3 — 零初始化等价 + 参数
6 个新变体在初始化时与 Q_corr 在同一 batch 上 logits 完全一致（max diff = 0.0）。实测参数增量与设计一致：gate_c/d = +768，gate_cd = +1536，token_c/d = +176，token_cd = +352；token 形状 [batch,128,176]。✅

## Check 4 — 门控接线
写入门：`g = σ(logit(g_base) + w·c̃)` 精确成立（误差 6e-8），只改 g 不动 ρ；衰减门：`ρ = exp(−α·softplus(softplus⁻¹(−log ρ₀/α) + v·d̃))` 精确成立（误差 6e-8），只改 ρ 不动 g；`_cd` 无串扰。符号/幅度均正确。✅

## Check 5 — 梯度 + resume
新参数（write_w/retention_v/token_proj）均收到非零梯度并在一步后更新；token 形状正确；save/restore 位一致（模型/优化器/调度器/RNG/冻结统计 aux + gate 标量全部逐位相等），缺失冻结缓冲时 resume 拒绝并报错（不静默重拟合）。✅

验收证据：`reports/qcorr_gates/acceptance/seed{17,42,2026}.json` + `check5b_resume.json`。
