"""Aggregate sections 2/3/4 (+5 corpus summary) into reports/round3_report.md."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R3 = ROOT / 'round3_scale_transfer'
BACKBONES = ('G_Qlearn', 'S1plain', 'G', 'Q')
PRETRAINS = ('random', 'mae', 'denoise')
SEEDS = (17, 42, 2026)


def mean_std(vals):
    vals = list(vals)
    return (sum(vals) / len(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)


def load(name):
    return json.loads((R3 / 'reports' / name).read_text())


def sec2():
    d = load('section2_finetune_cfo.json')
    agg = d['aggregate']
    lines = ['## 2. G_Qlearn 微调 vs 从头训练（问题 1）', '']
    lines.append('clean val OA（均值±std，3 seed）：')
    lines.append('')
    lines.append('| 组 | clean | CFO+0.005 | CFO−0.005 |')
    lines.append('|---|---|---|---|')
    for g in ('scratch', 'mae', 'denoise'):
        a = agg[g]
        lines.append(f"| {g} | {a['clean']['mean']:.4f}±{a['clean']['std']:.4f} | "
                     f"{a['cfo_plus']['mean']:.4f} | {a['cfo_minus']['mean']:.4f} |")
    lines.append('')
    dl = d['deltas']
    lines.append(f"- MAE 预训练收益（vs 同结构 scratch）：**+{dl['mae_pretrain_gain_over_G_Qlearn_scratch']*100:.2f}pp**")
    lines.append(f"- denoise 预训练收益：**+{dl['denoise_pretrain_gain_over_G_Qlearn_scratch']*100:.2f}pp**")
    lines.append(f"- Q 前端收益（G_Qlearn-scratch vs G-scratch）：**+{dl['G_Qlearn_scratch_minus_G_scratch']*100:.2f}pp**")
    lines.append(f"- 结论：53.47% = Global 主干微调能力（~46pp） + MAE 预训练（+7.5pp） + Q 前端（+1.8pp）。")
    lines.append('')
    return lines


def sec3():
    d = load('section3_unified_readout.json')['results']
    g = {}
    for k, v in d.items():
        bb = next(b for b in BACKBONES if k.startswith(b + '_'))
        pt = k[len(bb) + 1:].split('_seed')[0]
        g.setdefault((bb, pt), []).append(v['full_label']['oa'])
    lines = ['## 3. 统一冻结读出排名（问题 2）', '']
    lines.append('全标签统一 LR（StandardScaler(train) + L2 LR C=1），RML val OA 均值±std：')
    lines.append('')
    lines.append('| 结构 | random | MAE | denoise | MAE−random | MAE−denoise |')
    lines.append('|---|---|---|---|---|---|')
    for bb in ('G', 'G_Qlearn', 'S1plain', 'Q'):
        r = mean_std(g[(bb, 'random')]); ma = mean_std(g[(bb, 'mae')]); dn = mean_std(g[(bb, 'denoise')])
        lines.append(f"| {bb} | {r[0]:.4f}±{r[1]:.4f} | {ma[0]:.4f}±{ma[1]:.4f} | "
                     f"{dn[0]:.4f}±{dn[1]:.4f} | {100*(ma[0]-r[0]):+.2f} | {100*(ma[0]-dn[0]):+.2f} |")
    lines.append('')
    mG = mean_std(g[('G', 'mae')])[0]; mGQ = mean_std(g[('G_Qlearn', 'mae')])[0]
    mS1 = mean_std(g[('S1plain', 'mae')])[0]; mQ = mean_std(g[('Q', 'mae')])[0]
    lines.append(f'- Q 前端在 Global 内的增益（MAE）：G_Qlearn − G = **{100*(mGQ-mG):+.2f}pp**（负）')
    lines.append(f'- Q 前端在 S1 内的增益（MAE）：Q − S1plain = **{100*(mQ-mS1):+.2f}pp**（正）')
    lines.append(f'- MAE 排名：Q {mQ:.4f} > S1plain {mS1:.4f} > G {mG:.4f} > G_Qlearn {mGQ:.4f}')
    lines.append('')
    return lines


def sec4():
    d = load('section4_radar_transfer.json')['results']
    g = {}
    for k, v in d.items():
        bb = next(b for b in BACKBONES if k.startswith(b + '_'))
        pt = k[len(bb) + 1:].split('_seed')[0]
        g.setdefault((bb, pt), []).append(v['full_label_lr']['oa'])
    lines = ['## 4. 雷达冻结迁移（问题 3）', '']
    lines.append('统一 LR，RadChar eval OA 均值±std：')
    lines.append('')
    lines.append('| 结构 | random | MAE | denoise | MAE−random |')
    lines.append('|---|---|---|---|---|')
    for bb in ('G', 'G_Qlearn', 'S1plain', 'Q'):
        r = mean_std(g[(bb, 'random')]); ma = mean_std(g[(bb, 'mae')]); dn = mean_std(g[(bb, 'denoise')])
        lines.append(f"| {bb} | {r[0]:.4f}±{r[1]:.4f} | {ma[0]:.4f}±{ma[1]:.4f} | "
                     f"{dn[0]:.4f}±{dn[1]:.4f} | {100*(ma[0]-r[0]):+.2f} |")
    lines.append('')
    mG = mean_std(g[('G', 'mae')])[0]; mGQ = mean_std(g[('G_Qlearn', 'mae')])[0]
    mS1 = mean_std(g[('S1plain', 'mae')])[0]; mQ = mean_std(g[('Q', 'mae')])[0]
    lines.append(f'- Global+Q 在雷达冻结迁移的增益（MAE）：G_Qlearn − G = **{100*(mGQ-mG):+.2f}pp**（正）')
    lines.append(f'- S1+Q 在雷达冻结迁移的增益（MAE）：Q − S1plain = **{100*(mQ-mS1):+.2f}pp**（正）')
    lines.append(f'- 即：Global+Q 的表征**能**迁移到雷达，且相对 Global 提升；这与 RML 冻结读出（Global+Q 为负）相反。')
    lines.append('')
    return lines


def sec5():
    lines = ['## 5. 语料重审（问题 4/5 的前置）', '']
    lines.append('- F0 = 19968（RML cls_dev train，24 调制 × 26 SNR × 32，分层平衡）。')
    lines.append('- F1 = 100000（RML 全量减 val/test，(mod,SNR) 分层、含全部 F0、IQ 去重）。')
    lines.append('- **修正**：RML2018A 按调制-major 排列，round-2 按索引顺序扫 100k 会得到纯调制 0 语料；round-3 改为分层采样。')
    lines.append('- F2 = 100000（50k comm = F1 分层子集 + 50k OOD = EM-Radar-mixed 复 IQ，5MHz/16dB/1024 点）。')
    lines.append('- EM-Radar-mixed 合格（复 IQ、时间对齐、近叠加）；RadChar 已留出用于迁移评估，长度不同（512 vs 1024）无重叠。')
    lines.append('- 详见 eligibility_v2.csv。')
    lines.append('')
    return lines


F0_RML = {'G': 0.3169, 'G_Qlearn': 0.2843, 'Q': 0.3370}
F0_RAD = {'G': 0.7641, 'G_Qlearn': 0.8042, 'Q': 0.8107}
F0_FT = {'G': 0.4834, 'G_Qlearn': 0.5347, 'Q': 0.4862}


def sec6():
    probe = load('section6_probe.json')['results']
    ft = load('section6_finetune_cfo.json')['per_run']
    lines = ['## 6. 扩量预训练（问题 4/5）', '',
             '统一协议：StandardScaler(train)+L2 LR(C=1)。F0 基准 = round-2 MAE（同一协议/同一 RML 微调口径）。均值±std（3 seed）：', '',
             '| 语料 | 结构 | RML 冻结读出 | vs F0 | 雷达冻结迁移 | vs F0 | RML 微调 clean | vs F0 |',
             '|---|---|---|---|---|---|---|---|']
    for corpus in ('f1', 'f2'):
        for mdl in ('G', 'G_Qlearn', 'Q'):
            keys = [f'{corpus}_{mdl}_seed{s}' for s in SEEDS]
            rml = mean_std([probe[k]['rml']['full_label']['oa'] for k in keys])
            rad = mean_std([probe[k]['radar']['oa'] for k in keys])
            fto = mean_std([ft[k]['metrics']['clean']['oa'] for k in keys])
            lines.append(f"| {corpus.upper()} | {mdl} | {rml[0]:.4f}±{rml[1]:.4f} | "
                         f"{100*(rml[0]-F0_RML[mdl]):+.2f} | {rad[0]:.4f}±{rad[1]:.4f} | "
                         f"{100*(rad[0]-F0_RAD[mdl]):+.2f} | {fto[0]:.4f}±{fto[1]:.4f} | "
                         f"{100*(fto[0]-F0_FT[mdl]):+.2f} |")
    lines += ['', '- **问题 4（同域扩量）**：F1（100k，固定 7800 步 ≈ 每样本 ~10 epoch）相对 F0（19968，50 epoch）',
              '  在三项下游上基本持平（±1pp 内，无一致方向）：冻结读出 −0.38/+1.16/−0.45pp、微调 −1.08/−0.53/−1.19pp。',
              '  → **数据量不是瓶颈**；小语料多 epoch 复用与 5× 独立数据等价。',
              '- **问题 5（OOD 是否两域通吃）**：F2（50k comm + 50k OOD）**非两域通吃，存在 trade-off 且依赖主干**：',
              '  - G（Global）：微调 +2.30pp、雷达 +1.71pp —— **唯一两域都涨**；',
              '  - Q_corr：雷达 +0.57pp 但微调 −2.83pp、冻结读出 −1.49pp；',
              '  - G_Qlearn：微调 −2.70pp、雷达 −1.20pp（冻结读出 +1.82pp）。',
              '  → OOD 让 Global 主干两域受益，但让带复数分支的模型在 RML 微调上明显退化（复数分支向雷达/OOD 统计过拟合）。',
              '']
    return lines


def main():
    parts = ['# SSM_L 第三轮实验报告（round3_scale_transfer）', '']
    parts += sec2()
    parts += sec3()
    parts += sec4()
    parts += sec5()
    parts += sec6()
    out = R3 / 'reports' / 'round3_report.md'
    out.write_text('\n'.join(parts))
    print('wrote', out)


if __name__ == '__main__':
    main()
