#!/usr/bin/env python
"""Render reports/qcorr_gates/aggregate.json into markdown tables + CFO curve PNG."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
REPORT = ROOT / 'reports' / 'qcorr_gates'
NEW = {'Q_corr_gate_c', 'Q_corr_gate_d', 'Q_corr_gate_cd',
       'Q_corr_token_c', 'Q_corr_token_d', 'Q_corr_token_cd'}
ORDER = ['Q_corr', 'Q_corr_gate_c', 'Q_corr_gate_d', 'Q_corr_gate_cd',
         'Q_corr_token_c', 'Q_corr_token_d', 'Q_corr_token_cd',
         'G', 'G_corr', 'G_Q_corr_frozen']
CLASS_NAMES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
               'FM', 'GMSK', 'OQPSK']


def f4(x):
    return '—' if x is None else f'{x:.4f}'


def f3(x):
    return '—' if x is None else f'{x:.3f}'


def pm(m, s):
    if m is None:
        return '—'
    return f'{m:.4f}±{s:.4f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agg', default=str(REPORT / 'aggregate.json'))
    p.add_argument('--out', default=str(REPORT))
    a = p.parse_args()
    out = Path(a.out)
    agg = json.loads(open(a.agg).read())
    main_rows = agg['main']
    paired = agg['paired']
    curves = agg['cfo_curves']
    snr = agg['per_snr']
    class_rows = agg['per_class_f1']
    class_delta = agg['class_delta_gate_cd_vs_Q_corr']
    bench = agg['benchmark']

    L = []
    W = L.append

    W('# Q_corr 物理门控轮结果报告')
    W('')
    W('来源标记：`*` = 本轮新增；其余为历史基线（同一 11 条件 order-B 协议复用，已验证对齐）。'
      'OA 差异均以百分点 (pp) 表示。c=|r_n|（写入门），d=|r_n−r_{n−1}|（历史衰减门），'
      '`_gate_*` 直接调制 SSM 门，`_token_*` 把同样的标量零初始化线性加入输入 token（同信息对照）。')
    W('')

    # ---- 主表 ---------------------------------------------------------------------
    W('## 1. 10 模型主表（clean / ±0.005，3 seed 均值±std）')
    W('')
    W('| variant | clean OA | clean Macro-F1 | +0.005 OA | −0.005 OA | params |')
    W('|---|---|---|---|---|---|')
    for v in ORDER:
        r = next(x for x in main_rows if x['variant'] == v)
        mark = '*' if v in NEW else ''
        W(f"| {v}{mark} | {pm(r['clean_oa_mean'], r['clean_oa_std'])} | "
          f"{pm(r['clean_macro_f1_mean'], r['clean_macro_f1_std'])} | "
          f"{pm(r['cfo0p005_oa_mean'], r['cfo0p005_oa_std'])} | "
          f"{pm(r['cfom0p005_oa_mean'], r['cfom0p005_oa_std'])} | {r['params']:,} |")
    W('')
    W('逐 seed clean OA：')
    W('')
    W('| variant | seed 17 | seed 42 | seed 2026 |')
    W('|---|---|---|---|')
    for v in ORDER:
        r = next(x for x in main_rows if x['variant'] == v)
        mark = '*' if v in NEW else ''
        W(f"| {v}{mark} | {f4(r['seed17_clean_oa'])} | {f4(r['seed42_clean_oa'])} | {f4(r['seed2026_clean_oa'])} |")
    W('')

    # ---- 配对差值 -----------------------------------------------------------------
    W('## 2. 逐 seed 配对差值（pp）')
    W('')
    W('| 对照 | condition | seed 17 | seed 42 | seed 2026 | 均值±std |')
    W('|---|---|---|---|---|---|')
    for r in paired:
        W(f"| {r['pair']} | {r['condition']} | {f3(r['seed17'])} | {f3(r['seed42'])} | {f3(r['seed2026'])} | "
          f"{f3(r['mean_pp'])}±{f3(r['std_pp'])} |")
    W('')

    # ---- CFO 曲线 -----------------------------------------------------------------
    W('## 3. CFO 曲线（绝对 OA，保留符号；3 seed 均值）')
    W('')
    conds = ['clean', 'cfo0p001', 'cfom0p001', 'cfo0p002', 'cfom0p002',
             'cfo0p005', 'cfom0p005', 'cfo0p01', 'cfom0p01', 'cfo0p02', 'cfom0p02']
    header = '| variant | ' + ' | '.join(conds) + ' |'
    W(header)
    W('|' + '---|' * (len(conds) + 1))
    by_variant = {}
    for c in curves:
        by_variant.setdefault(c['variant'], {})[c['condition']] = c['oa_mean']
    for v in ORDER:
        vals = ' | '.join(f4(by_variant.get(v, {}).get(c)) for c in conds)
        mark = '*' if v in NEW else ''
        W(f'| {v}{mark} | {vals} |')
    W('')
    W('相对下降（pp，越低越好）：')
    W('')
    W('| variant | +0.005 drop | −0.005 drop |')
    W('|---|---|---|')
    for v in ORDER:
        b = by_variant.get(v, {})
        c0 = b.get('clean')
        dp = (c0 - b.get('cfo0p005')) * 100 if c0 is not None and b.get('cfo0p005') is not None else None
        dm = (c0 - b.get('cfom0p005')) * 100 if c0 is not None and b.get('cfom0p005') is not None else None
        mark = '*' if v in NEW else ''
        W(f'| {v}{mark} | {f3(dp)} | {f3(dm)} |')
    W('')

    # ---- per-SNR ------------------------------------------------------------------
    W('## 4. 每 SNR 分组（clean）')
    W('')
    W('| variant | ≤0 dB OA | >0 dB OA | ≤0 dB Macro-F1 | >0 dB Macro-F1 |')
    W('|---|---|---|---|---|')
    for v in ORDER:
        r = next(x for x in snr if x['variant'] == v)
        mark = '*' if v in NEW else ''
        W(f"| {v}{mark} | {pm(r['low_oa_mean'], r['low_oa_std'])} | {pm(r['high_oa_mean'], r['high_oa_std'])} | "
          f"{f4(r['low_f1_mean'])} | {f4(r['high_f1_mean'])} |")
    W('')

    # ---- per-class (gate_cd vs Q_corr) --------------------------------------------
    W('## 5. 逐类分析：`Q_corr_gate_cd` − `Q_corr` 的 Macro-F1 差值（pp，3 seed 均值）')
    W('')
    W('| class | clean | +0.005 | −0.005 |')
    W('|---|---|---|---|')
    by_key = {(r['condition'], r['class']): r for r in class_delta}
    for c in range(24):
        clean = by_key.get(('clean', c), {}).get('gate_cd_minus_Q_corr_pp')
        p = by_key.get(('cfo0p005', c), {}).get('gate_cd_minus_Q_corr_pp')
        m = by_key.get(('cfom0p005', c), {}).get('gate_cd_minus_Q_corr_pp')
        W(f"| {CLASS_NAMES[c]} | {f3(clean)} | {f3(p)} | {f3(m)} |")
    W('')

    # ---- efficiency ---------------------------------------------------------------
    if bench:
        W('## 6. 效率（L=1024）')
        W('')
        keys = list(bench[0].keys()) if bench else []
        W('| ' + ' | '.join(keys) + ' |')
        W('|' + '---|' * len(keys))
        for r in bench:
            W('| ' + ' | '.join(str(r.get(k, '')) for k in keys) + ' |')
        W('')

    W('---')
    W('*逐样本预测见 `predictions_qcorr_gates/`（198 份新）与 `predictions_global_qcorr/`（历史基线），'
      'OA/Macro-F1 已从预测文件重算核对。逐类 F1 表见 `tables/per_class_f1.csv`。*')
    W('')

    (out / 'report.md').write_text('\n'.join(L))

    # ---- CFO curve PNG ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        freq_to_cond = {0.0: 'clean', 0.001: 'cfo0p001', -0.001: 'cfom0p001',
                        0.002: 'cfo0p002', -0.002: 'cfom0p002',
                        0.005: 'cfo0p005', -0.005: 'cfom0p005',
                        0.01: 'cfo0p01', -0.01: 'cfom0p01',
                        0.02: 'cfo0p02', -0.02: 'cfom0p02'}
        freqs = [-0.02, -0.01, -0.005, -0.002, -0.001, 0.0,
                 0.001, 0.002, 0.005, 0.01, 0.02]
        colors = {'Q_corr': 'tab:green',
                  'Q_corr_gate_c': 'tab:red', 'Q_corr_gate_d': 'tab:orange',
                  'Q_corr_gate_cd': 'tab:purple',
                  'Q_corr_token_c': 'tab:pink', 'Q_corr_token_d': 'tab:brown',
                  'Q_corr_token_cd': 'tab:olive',
                  'G': 'k', 'G_corr': 'tab:gray', 'G_Q_corr_frozen': 'tab:cyan'}
        plt.figure(figsize=(9, 5.5))
        for v in ORDER:
            b = by_variant.get(v, {})
            pts = [(f, b.get(freq_to_cond[f])) for f in freqs]
            xs = [f for f, y in pts if y is not None]
            yy = [y for f, y in pts if y is not None]
            ls = '--' if v in ('G', 'G_corr', 'G_Q_corr_frozen') else '-'
            plt.plot(xs, yy, label=v + ('*' if v in NEW else ''), color=colors[v], linestyle=ls, marker='o', ms=3)
        plt.xlabel('CFO (cycles/sample)')
        plt.ylabel('OA')
        plt.title('CFO robustness (3-seed mean)')
        plt.legend(fontsize=7, ncol=2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / 'cfo_curves.png', dpi=150)
    except Exception as e:
        print('PNG skipped:', e)


if __name__ == '__main__':
    main()
