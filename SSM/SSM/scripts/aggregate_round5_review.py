#!/usr/bin/env python
"""Round-5 Review Section 五: Q_corr_frozen evaluation aggregation.

Core comparisons (all order-B val, NO test, 3 seeds, mean ± sample std, per-seed paired):
  1. Q_corr − Q_corr_frozen   (does learning the filters add value?)
  2. Q_corr_frozen − S1       (do fixed filters + corr readout already help?)
  3. Q_corr − S1_corr         (re-confirm the original Round-5 head-to-head)

Sources:
  reports/round5_review/eval_cases.jsonl  (Q_corr_frozen, produced by eval_round5.py)
  reports/round5_iq/eval_cases.jsonl       (Q_corr, S1_corr)
  reports/round4_structure/eval_cases.jsonl (S1)
"""
import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [17, 42, 2026]
MAIN_MAGS = [0.001, 0.002, 0.005]
PHASES = [('p_pi4', math.pi / 4), ('p_pi2', math.pi / 2),
          ('p_pi', math.pi), ('p_3pi2', 3 * math.pi / 2)]
COMPARISONS = [('Q_corr', 'Q_corr_frozen'), ('Q_corr_frozen', 'S1'), ('Q_corr', 'S1_corr')]
VARIANTS = ['S1', 'S1_corr', 'Q_corr', 'Q_corr_frozen']


def load_cases(path):
    cases = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cases.setdefault((r['variant'], r['seed']), []).append(r)
    return cases


def get_metric(cases, variant, seed, kind, value):
    for r in cases.get((variant, seed), []):
        if r['kind'] == kind and abs(r['value'] - value) < 1e-9:
            return r['metrics']
    return None


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def fmt_ms(m, s):
    if m is None:
        return 'n/a'
    return f'{m:.4f} ± {s:.4f}' if s else f'{m:.4f}'


def fmt_pp(m, s):
    if m is None:
        return 'n/a'
    return f'{100*m:+.2f} ± {100*s:.2f}' if s else f'{100*m:+.2f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen-cases', default=str(ROOT / 'reports/round5_review/eval_cases.jsonl'))
    ap.add_argument('--round5-cases', default=str(ROOT / 'reports/round5_iq/eval_cases.jsonl'))
    ap.add_argument('--round4-cases', default=str(ROOT / 'reports/round4_structure/eval_cases.jsonl'))
    ap.add_argument('--out', default=str(ROOT / 'reports/round5_review/eval_comparison'))
    a = ap.parse_args()

    cases = load_cases(a.frozen_cases)
    for k, v in load_cases(a.round5_cases).items():
        cases.setdefault(k, []).extend(v)
    for k, v in load_cases(a.round4_cases).items():
        cases.setdefault(k, []).extend(v)

    lines = ['# Round-5 Review 评价：Q_corr_frozen 对照（order-B val，未用 test）', '',
             '3 个种子 mean ± 样本标准差(ddof=1)；差值单位为百分点(pp)=OA 小数差×100。',
             '全文不构成统计显著表述（见 errata.md）。', '']

    # 1. per-variant clean
    lines += ['## 1. clean val 主表', '',
              '| variant | seed17 | seed42 | seed2026 | OA mean ± std | Macro-F1 mean ± std |',
              '|---|---|---|---|---|---|']
    main_json = {}
    for v in VARIANTS:
        oas, f1s = [], []
        for s in SEEDS:
            m = get_metric(cases, v, s, 'clean', 0.0)
            oas.append(m['oa'] if m else None)
            f1s.append(m['macro_f1'] if m else None)
        om, os_ = mean_std(oas); fm, fs_ = mean_std(f1s)
        main_json[v] = {'per_seed_oa': oas, 'oa_mean': om, 'oa_sample_std': os_,
                        'macro_f1_mean': fm, 'macro_f1_sample_std': fs_}
        lines.append(f'| {v} | ' + ' | '.join(f'{x:.4f}' if x is not None else 'n/a' for x in oas) +
                     f' | {fmt_ms(om, os_)} | {fmt_ms(fm, fs_)} |')
    lines.append('')

    # 2. paired diffs (per-seed, mean ± std) across conditions
    lines += ['## 2. 三个核心比较（per-seed 配对差，mean ± std）', '',
              '正号 = 前者更高。条件列：clean / CFO 主网格 / 全局相位。', '',
              '| comparison | clean | CFO+0.001 | CFO−0.001 | CFO+0.002 | CFO−0.002 | CFO+0.005 | CFO−0.005 | phase π/4 | phase π/2 | phase π | phase 3π/2 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    diff_json = {}

    def diff_at(va, vb, kind, value):
        ds = []
        for s in SEEDS:
            am = (get_metric(cases, va, s, kind, value) or {}).get('oa')
            bm = (get_metric(cases, vb, s, kind, value) or {}).get('oa')
            if am is not None and bm is not None:
                ds.append(am - bm)
        m, sd = mean_std(ds)
        return {'mean': m, 'std': sd, 'per_seed': ds}

    for va, vb in COMPARISONS:
        cells = []
        grid = [('clean', 0.0)] + [('cfo', f) for m in MAIN_MAGS for f in (m, -m)] + \
               [('phase', pv) for _, pv in PHASES]
        for kind, value in grid:
            d = diff_at(va, vb, kind, value)
            diff_json.setdefault(f'{va}-{vb}', {})[f'{kind}:{value}'] = d
            cells.append(fmt_pp(d['mean'], d['std']))
        lines.append(f'| {va} − {vb} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    # 3. per-SNR (clean)
    lines += ['## 3. clean per-SNR 分组（低 SNR ≤0 / 高 SNR >0）', '',
              '| variant | low OA (SNR≤0) | high OA (SNR>0) |',
              '|---|---|---|']
    for v in VARIANTS:
        lows, highs = [], []
        for s in SEEDS:
            m = get_metric(cases, v, s, 'clean', 0.0)
            if m:
                if m.get('low'): lows.append(m['low']['oa'])
                if m.get('high'): highs.append(m['high']['oa'])
        lm, ls = mean_std(lows); hm, hs = mean_std(highs)
        lines.append(f'| {v} | {fmt_ms(lm, ls)} | {fmt_ms(hm, hs)} |')
    lines.append('')

    # 4. absolute CFO/phase OA curves for the four variants
    lines += ['## 4. CFO / 相位绝对 OA（mean ± std）', '',
              '| frequency | ' + ' | '.join(VARIANTS) + ' |', '|---|' + '---|' * len(VARIANTS)]
    grid = [0.0] + [f for m in MAIN_MAGS for f in (m, -m)]
    for f in grid:
        row = []
        for v in VARIANTS:
            m, s = mean_std([(get_metric(cases, v, sd, 'clean' if f == 0.0 else 'cfo', f) or {}).get('oa')
                             for sd in SEEDS])
            row.append(fmt_ms(m, s))
        lines.append(f'| {f:+.4f} | ' + ' | '.join(row) + ' |')
    lines.append('')
    lines.append('| phase | ' + ' | '.join(VARIANTS) + ' |')
    lines.append('|---|' + '---|' * len(VARIANTS))
    for _, pv in [(None, 0.0)] + PHASES:
        row = []
        for v in VARIANTS:
            m, s = mean_std([(get_metric(cases, v, sd, 'clean' if pv == 0.0 else 'phase', pv) or {}).get('oa')
                             for sd in SEEDS])
            row.append(fmt_ms(m, s))
        lines.append(f'| {pv:.4f} | ' + ' | '.join(row) + ' |')

    Path(a.out + '.md').write_text('\n'.join(lines) + '\n')
    Path(a.out + '.json').write_text(json.dumps({'main': main_json, 'diffs': diff_json}, indent=2))
    print('comparison written to', a.out + '.md / .json')


if __name__ == '__main__':
    main()
