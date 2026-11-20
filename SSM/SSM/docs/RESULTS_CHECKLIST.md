# 服务器结果交付清单

## 1. 实验前记录

- `reports/server_audit.md`：旧项目具体路径、代码版本、训练入口、现有Global结构与预处理、旧cls_dev身份与索引来源、与新工程的逐项差异及处理。
- `reports/machine_audit.json`：新工程自动审计、样本数量/形状、索引SHA256、参数量。
- `configs/resolved.json`、`configs/data.json`、完整实际运行命令与GPU分配。
- `reports/environment.txt`：Python/PyTorch/CUDA/驱动/GPU型号、依赖版本、tf32、精度、是否有其他负载。
- `reports/validation.md`：pytest、CPU合成集、真实数据smoke、CUDA串并行误差/梯度检查、发现的问题与修复。
- `reports/perturbation_protocol.json`：冻结后的相位、频偏、增强幅度（如有）、单位、合法性依据。

## 2. 每个正式run必须保留

`config.json`、`provenance.json`、`model_info.json`、`history.jsonl`、`best.pt`、`last.pt`、`best_val.json`、`val_predictions.npz`、`DONE.json`、完整控制台log。

- history包括逐epoch train loss/OA、val loss/OA/Macro-F1、逐SNR OA/F1与混淆矩阵、学习率、训练时间、clip前最大梯度范数、训练峰值显存、抽样层级状态/门控诊断。
- best_val为val OA选出的同一个checkpoint的全部指标，epoch为0-based；报告给人时可同时注明第epoch+1轮。
- 失败run保留traceback与最后checkpoint；不能只提交成功run。
- `eval_val/results.json` 和可选逐扰动预测；有独立test时冻结方案后另存 `eval_test/`。
- 效率只需每模型一个代表seed；保留 `benchmark.json`，不能把每种子重复测量当独立硬件样本。

## 3. 自动汇总

运行 `python -m ssm_exp.summarize --runs runs/clean --output reports/clean_summary`：

- `runs.csv`：每模型每seed的clean OA、Macro-F1、低/高SNR OA、参数量、选中epoch。
- `summary.csv`：均值、样本标准差（ddof=1）、实际种子数。
- `paired_deltas.csv`：S0-G、S1-S0、S2-S1、S2-S2_concat，各同seed的OA/F1百分点差值。
- `curves.csv`：逐SNR、频偏、相位指标。
- `efficiency.csv`：延迟、吞吐、显存及长度；缺少benchmark的run不会伪造数值。
- PNG：逐SNR、频偏、相位OA，均值±样本标准差。F1原始数据在CSV，可补绘。
- `incomplete_runs.json`：未完成run。

汇总按协议hash分组；不允许同一协议/模型/seed重复计数。增强条件独立汇总。报告必须检查n_seeds，不把一个seed输出当成三种子结论。

## 4. 人工整理的最终报告

写 `reports/FINAL_REPORT.md`，按下列顺序：

1. 本次实际完成哪些实验，哪些失败/未做，是否复现原Global。
2. clean主表：5模型×3种子原始值、均值±std、参数量、低/高SNR结果。
3. 机制主表：S1-S0、S2-S1、S2-S2_concat，配对差值与方向一致性；不要只展示最优seed。
4. 鲁棒性：全部合法频偏强度的OA/F1及相对clean下降，固定相位结果，代表性混淆矩阵；区分val诊断与test。
5. 效率：batch1平均/p50/p95延迟，固定batch吞吐，峰值显存，长度曲线；说明当前并行扫描O(N log N)、非融合实现。
6. 训练健康：loss曲线、状态RMS/max、rho接近1比例、gate饱和比例、梯度范数；是否存在明显优化差异。
7. 明确回答：固定SSM可用吗？选择有价值吗？物理描述有额外价值吗？直接融合是否同样有效？鲁棒收益是否牺牲clean？证据是否跨seed稳定？
8. 下一步仅选有证据支持的一项：保留S0、保留一般选择、深入物理控制、补公平调参、或停止该分支。

不要宣称统计显著、严格等变、真实物理状态、已经频偏补偿或理论线性复杂度已经落地。

## 5. 回传给本地agent/用户

制作 `/projects/SSM/deliverables/SSM_results_round1.zip`，包括 configs、reports、每run的JSON/CSV/log、历史曲线及代码修订diff/最终源码清单。通常不放数据与巨大checkpoint；权重留服务器并给出绝对路径。若用户另要权重，再单独打包。

同时提供一段短摘要：完成run数、主表、关键配对增益、最重要限制、建议下一步。用户无需逐个打开日志才能知道结论。
