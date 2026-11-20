# AlignAtt — 对齐关系参与电磁信号注意力（实验 02）

实验方案见 `02_alignment_attention_experiment.md`（本目录）。本目录是**可执行实现**，
在 SwinE 固定 Global 基线（从头 0.3835±0.0034，3 种子）上做**单因素**验证：只在注意力
logits 上追加一项 `g_h · r_ij^(h)`（原始 I/Q 片段间的复相关描述经小 MLP 映射），其余
（切块、token 数、主干、分类头、训练协议）全部不动。

## 目录结构

```
align/
  correlation.py        # 复相关 rho_ij(tau)（向量化 + 朴素参考），支持逐样本边界掩码
  relation_bias.py      # 描述子 -> 小 MLP -> 每头偏置；门控 g、shuffle(A5)、real_mlp(A4)
  align_models.py       # 带偏置的全局 encoder + AlignClassifier（A0-A6 全在此）
  train_align.py        # 训练入口（复用 emfm/train.py 协议，与基线逐项可比）
  diagnose_correlation.py  # 阶段1：合成配对关系诊断（不训练）
  test_correlation.py   # 阶段0：代数/数值/梯度正确性（不训练）
  synthetic.py          # 合成复基带波形（与真实训练样本隔离）
configs/                # a0_global … a6_relation_only 七个配置
run.sh                  # 单实验 detached tmux 启动器
run_screening.sh        # 阶段2 筛选 A0-A4 一键入队
runs/                   # 输出（每个实验一个目录 + .log）
```

依赖 SwinE 的 `emfm` 包（数据加载 + 训练协议），通过 `PYTHONPATH` 引用，
**不改动 SwinE 冻结源码**。

## 实验矩阵

| ID | relation.mode | 含义 | 参数量(关系部分) |
|----|---------------|------|-----------------|
| A0 | none | 主基线（新代码路径复现冻结基线） | 0 |
| A1 | zero_delay | 标准QK + 零时延复相关偏置 | 168 |
| A2 | magnitude | 标准QK + 多时延幅度描述 | 328 |
| A3 | full | 标准QK + 多时延完整复描述（主候选） | 488 |
| A4 | real_mlp | 标准QK + 同R上下文普通实值MLP关系（控制） | 1112 |
| A5 | full + shuffle | A3 但打乱配对 j 索引（控制） | 488 |
| A6 | full + use_qk=False | 只用关系项、移除QK（诊断） | 488 |

共用：patch=8、dim=176、depth=6、heads=4、batch=128、lr=3e-4、50ep、fp32。
关系上下文默认 `R=16`（=2P，见下「诊断结论」）；`d=min(2, floor(R/4))=2`。

## 关键设计决策（对应文档条款）

- **门控 g 初始化 0**，关系 MLP **非零初始化**（避免门控与映射互相零梯度）。g=0 时
  新路径与基线**逐位一致**（已测，max diff = 0）。
- **复相关相位约定**：若 z_j 是 z_i 延后 τ_true、旋转 e^{jφ} 的副本，峰在 τ=τ_true、
  `arg(rho)=-φ`（由公式+合成真值校准，见测试）。
- **R 上下文长度**：默认 = patch（8），可扩到 2P/4P。边界 padding 用逐样本掩码排除、
  不参与统计；低能量/无有效候选时偏置置 0 并返回有效标志。
- **注入范围**：默认只注入第一层（`inject=first`），可 `first_half`/`all` 消融。
- **不降低 O(T²) 复杂度**，关系是额外计算；首版全局关系为机制验证服务。

## 如何运行

```bash
cd /home/user_wzk/wzk/projects/AlignAtt
export PYTHONPATH=/home/user_wzk/wzk/projects/SwinE/pretrain_v1
PY=~/miniconda3/envs/SFM/bin/python

# 阶段0 正确性（CPU，秒级）
$PY -m align.test_correlation

# 阶段1 诊断（CPU，秒级；R 扫描看真假配对区分度）
$PY -m align.diagnose_correlation --snr inf --P 8 --out runs/diagnose_P8.json

# 阶段2 训练（detached tmux，GPU1）
./run.sh a3_full 17          # 主候选，seed=17，50ep
./run_screening.sh           # A0-A4 一键入队（seed=17）
```

冒烟（先 `--epochs 1` 验证全流程再正式跑）：
`./run.sh a3_full 17 1` → 输出到 `runs/a3_full_smoke/`。

## 阶段1 诊断结论（已跑）

合成 QPSK（sps=4）配对，noiseless，max|ρ|：

| R(=片段长) | 受控变换(真) | 同类独立符号 | 异类 | 噪声 | 真假差距 |
|---|---|---|---|---|---|
| 8   | 1.00 | 0.68 | 0.71 | 0.56 | 0.32 |
| 16  | 1.00 | 0.50 | 0.51 | 0.38 | 0.50 |
| 32  | 1.00 | 0.35 | 0.35 | 0.26 | 0.65 |
| 64  | 1.00 | 0.25 | 0.25 | 0.19 | 0.75 |

**R=patch=8 太短**：假配对 max|ρ| 高达 0.55–0.71（文档 §5 预判的「P 太短相关估计不稳定」）。
故主实验关系上下文默认 `R=16`，`R=32/64` 作为消融轴。另注：矩形脉冲 sps=4 下
子符号级时延不可辨识（延时不大于符号周期即无法靠相关分辨），时延精度 ~0.33 是合成
信号的脉冲形状所致，非实现问题——阶段0 测试已用随机波形证明整数时延+相位精确可恢复。

## 最终结果（结论：负结果，已归档）

完整结论见 [conclusion.md](conclusion.md)，原始数据见 [results.csv](results.csv)。
主候选 A3（完整复相关偏置）在 3 个种子下未复现 phase-2 的收益，平均 OA 反比基线
低 0.21 点且逐种子方向发散。结论**收窄**为：当前这版复相关偏置，在当前数据和训练
配置下，没有稳定提高验证集 OA —— 不据此推断"该数据不需要此机制"或"物理启发注意力
这一类方法无效"。

### 阶段2 筛选（seed=17）

| 配置 | OA | macro-F1 | best epoch |
|---|---|---|---|
| A0 基线 | 0.3933 | 0.3573 | 16 |
| A1 零延迟相关 | 0.3753 | 0.3435 | 21 |
| A2 幅度描述 | 0.3804 | 0.3828 | 31 |
| A3 完整复相关 | **0.3958** | 0.3975 | 37 |
| A4 实数MLP对照 | 0.3853 | 0.3692 | 20 |

### 阶段3 多种子（A0 vs A3，17/42/2026）

| 配置 | s17 | s42 | s2026 | 均值 | std |
|---|---|---|---|---|---|
| A0 | 0.3933 | 0.3837 | 0.3837 | **0.3869** | 0.0056 |
| A3 | 0.3958 | 0.3779 | 0.3804 | **0.3847** | 0.0097 |

逐种子 A3−A0 = **+0.26 / −0.58 / −0.32** 点：phase-2 的 +0.26 是种子噪声，非机制收益。

## 状态

- [x] 阶段0 正确性（8 项测试全过）
- [x] 阶段1 诊断（R 扫描完成，结论：需 R≥16）
- [x] 7 条模型路径 1-epoch 冒烟全过
- [x] 阶段2 筛选 A0-A4（seed=17，50ep）
- [x] 阶段3 多种子确认（17/42/2026）→ 负结果，归档
- [~] 阶段4 预训练(MAE)兼容性（因阶段3负结果，不再继续）
