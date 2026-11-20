# Revision-1 source diff

项目不在 git 管理下（无 `.git`），故以「新增/未改动」清单代替 diff。

## 未改动（原 round4 源码保持原样，只读复用）

- `round4_no_cfo_qstudy/models_r4.py`、`pretrain_r4.py`、`engine_r4.py`、`*.py`
- `round4_no_cfo_qstudy/reports/**`（conclusion/tables/protocol/acceptance/… 一律未覆盖）
- `round4_no_cfo_qstudy/runs/**`（权重 / best.pt / 预测 / 历史 json 一律未覆盖）
- `round2_overnight/models.py`、`pretrain/engine.py`、`ssm_exp/**`

## 新增（revision1 独立目录，不改原工程）

| 文件 | 用途 |
|---|---|
| `scripts/common_r1.py` | 共享：加载 5 骨干 encoder、特征提取、统一 LR 读出、S1plain 下游加载（含旧 SSM S1 权重映射） |
| `scripts/unified_readout.py` | Section 3 统一冻结读出（4 编码器 × 2 任务 × 3 seed） |
| `scripts/memory_diagnosis.py` | Section 4 记忆诊断 A/B 组（段内 reset、偏移、一致性校验） |
| `scripts/state_stats.py` | Section 5 状态统计（18 模型、固定子集、log 域保留积） |
| `scripts/fairness_audit.py` | Section 6 Q 2D 公平性审计（权重级，无重跑） |
| `scripts/radar_eval_agg.py` | Section 1.2 雷达 eval 主表 + P1 配对表 |
| `reports/*` | correction.md、completion_audit.csv、revised_conclusion.md、unified_probe.*、memory_*.csv、state_statistics.*、fairness_audit.*、tables_radar_eval.md、commands.md、config.md、acceptance_revision1.md、source_diff.md、latest_status.md |
| `runs/features/*.pt` | 统一读出特征缓存（可复算，不打包） |
| `runs/memory_features/*.pt` | A 组记忆特征缓存（可复算，不打包） |
| `runs/predictions/*_pred.npz` | 统一读出逐样本预测 |

## 关键实现差异（原 P2 记忆诊断 vs revision1 记忆诊断）

原 `round4` 的 P2 段内 reset：仅 A 组 mae/denoise、K∈{1,4,16,64}；revision1 重做：

1. A 组补 random init，B 组用**下游微调模型的原始 head**（原 P2 B 组未含 scratch 的旧 SSM
   权重映射）。
2. K 集合扩为 {1,4,8,16,32,64,full}，并新增 K=16/32 的**半段偏移**（offset=8/16），以排除
   「reset 位置恰好对齐」的伪影。
3. 每编码器跑一致性校验：`K=none` 与原始 `forward_tokens` 逐位一致（maxdiff=0）、segment 向量
   化与串行参考一致（~1e-6）、`K>=N`(零偏移) == full、`K=1` == 直接写入项 b。
4. A 组双读出：① 固定（full-state train 上 fit 的 scaler+LR，逐 K 评测）；② 重拟合（每 K 在
   该 K 的 train 特征上重 fit）。用于区分「表征确实丢失记忆」vs「固定读出不适配截断分布」。
