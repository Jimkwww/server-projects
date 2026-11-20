# Round-4 正确性验证报告

全部为 `tests/test_round4.py`（12 项）与既有 `tests/test_core.py` 回归。运行：`python -m pytest tests/ -q` → **37 passed**。

## 逐项结论

| # | 验证项 | 方法 | 结果 |
|---|---|---|---|
| 1 | 复数前端 == 手工实数展开 | `ComplexLinear(3,4)` 输出 vs `W_R I − W_I Q`, `W_I I + W_R Q` | ✅ rtol/atol 1e-5 |
| 2 | 输入形状 + Re/Im 打包顺序 | `ComplexPatchEmbedding` 偶数位=Re、奇数位=Im；形状 [B,N,2·cd] | ✅ |
| 3 | 正/反向串行 == 并行扫描 | `BidirectionalSSM`/`ComplexSelectiveSSM` 两种 scan 同种子对比 | ✅ rtol/atol 1e-4 |
| 4 | 反向输出翻转回后位置正确 | `_Direction` 翻转-扫描-翻转回 vs 直接手工反向递推 | ✅ rtol/atol 1e-4 |
| 5 | 双向依赖前后文；单向无未来泄漏 | 扰动末 token：双向 y[0] 改变、单向 y[0] 不变 | ✅ |
| 6 | 复数权重/状态/选择器梯度回传 | 双向/复数 SSM/D_complex 全参数 backward | ✅ 全部有限非零 |
| 7 | 无跨样本隐状态泄漏 | 批内输出 == 逐样本拼接 | ✅ rtol/atol 1e-5 |
| 8 | CFO 单位 + 顺序不变 | `perturb` 角 = 2π·f·t+p；`max_abs_norm` 幂等 | ✅ |
| 9 | 复数前端无偏：全局相位旋转为线性映射 | `ComplexLinear(e^{jφ}s) == e^{jφ} ComplexLinear(s)` | ✅ 相位等变 |
| 10 | D_complex 残差/norm/FFN 保持复数（无误取实部） | 各部件输出 `is_complex()`；仅最终打包为实 | ✅ |
| 11 | NaN/状态范数/gate 饱和监控 | `collect_stats` 生成 last_stats 全有限 | ✅ |
| 12 | 旧模型路径未受影响 | 旧变体前向 + S1==S2 参数量不变；全量回归 | ✅ |

## 补充验证

- **过拟合检查**：A64/A128/B_front/C_bi_front/D_complex 各取 16 样本训练 400 步 → 全部 `train_oa=1.000`、`train_loss<0.001`，端到端梯度流完整。
- **smoke 训练**：5 配置 2 epoch（128 样本）无 NaN、loss 单调下降、checkpoint/history 正常写出。
- **前端初始化输出尺度**（对比 `Linear(16,176)`）：`ComplexLinear(8,88)` 的复数权重由 `torch.randn(cfloat)/sqrt(in)` 生成；`torch.randn(cfloat)` 的实/虚部**各自** N(0,1/2)（PyTorch 复数标准正态总方差 E|z|²=1），除以 `sqrt(8)` 后**实/虚部各自方差 = 1/(2·in) = 1/16**；复数随机变量的**总方差** `E|W|² = 1/in = 1/8`（实部方差+虚部方差）。其**等效实数矩阵**（展开为 `M=[[W_R,−W_I],[W_I,W_R]]`）各元素方差 = 1/16，是 `nn.Linear(16,176)` 默认 uniform 方差 1/48 的 **3 倍**。这是 Round-4 有意沿用 SSM 内部约定的差异，但意味着复数前端与实数前端的初始尺度失配；Round-5 将改为匹配 `nn.Linear` 方差的初始化。前端后紧跟 LayerNorm，尺度差异大部分被吸收（但非全部）。

## 记录在案的限制

- 并行扫描为**非融合** eager PyTorch O(N log N)，与生产级 fused SSM 效率不同（benchmark 已注明）。
- D_complex 改变 norm+FFN+读取三处，收益不可归因到单一部件；选择器读 Re/Im，不声称严格等变性。
- 3 种子均值按样本标准差（ddof=1），不把 3 种子当作 3 个独立测试样本。
