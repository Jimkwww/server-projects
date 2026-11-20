#!/usr/bin/env python
"""Aggregate round-2 part-1 (recheck) + part-2 (new models) into deliverable tables.

Reads:
  reports/round2/recheck_cases.jsonl    (round-1 15 runs, frozen protocol)
  reports/round2/newmodel_cases.jsonl   (round-2 9 new runs, frozen protocol)
  runs/*/best_val.json                  (clean low/high SNR)

Writes reports/round2/{order_comparison,six_model_table,paired_diffs}.{json,md}
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
REPORT = ROOT / 'reports' / 'round2'
SEEDS = [17, 42, 2026]
FREQS = [0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.005,
         -0.0001, -0.00025, -0.0005, -0.001, -0.002, -0.005]
KEY_MODELS = ['G', 'G_full', 'S1', 'S1_energy', 'S1_corr', 'S2_concat']


def load_rows():
    rows = []
    for name in ('recheck_cases.jsonl', 'newmodel_cases.jsonl'):
        p = REPORT / name
        if p.exists():
            rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    # Sample standard deviation (n-1 denominator), matching the round-1 convention.
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0


def main():
    rows = load_rows()
    # index: (variant,seed) -> {clean:{A,B}, cfo:{freq:{A,B}}, phase:{ph:{A,B}}}
    clean = defaultdict(dict)
    cfo = defaultdict(lambda: defaultdict(dict))
    phase = defaultdict(lambda: defaultdict(dict))
    runpath = {}
    for r in rows:
        k = (r['variant'], r['seed'])
        runpath[k] = r['run']
        if r['kind'] == 'clean':
            clean[k][r['order']] = r
        elif r['kind'] == 'cfo':
            cfo[k][r['value']][r['order']] = r
        elif r['kind'] == 'phase':
            phase[k][r['value']][r['order']] = r

    # ---- item 1: order comparison (per variant/seed/CFO, A/B + diff) ----
    order_lines = []
    order_json = []
    variants = sorted({k[0] for k in clean})
    for v in variants:
        for seed in SEEDS:
            k = (v, seed)
            if k not in clean:
                continue
            c = clean[k]['A']['oa']
            for f in FREQS:
                a = cfo[k].get(f, {}).get('A')
                b = cfo[k].get(f, {}).get('B')
                if a is None or b is None:
                    continue
                rec = {'variant': v, 'seed': seed, 'freq': f,
                       'A_oa': a['oa'], 'A_macro_f1': a['macro_f1'], 'A_drop': c - a['oa'],
                       'B_oa': b['oa'], 'B_macro_f1': b['macro_f1'], 'B_drop': c - b['oa'],
                       'diff_B_minus_A_drop': (c - b['oa']) - (c - a['oa'])}
                order_json.append(rec)
    order_lines.append('# Order comparison (A = normalize→perturb, B = perturb→normalize)\n')
    order_lines.append('| variant | seed | freq | A_OA | A_drop | B_OA | B_drop | B−A drop |')
    order_lines.append('|---|---|---|---|---|---|---|---|')
    for rec in order_json:
        order_lines.append(f"| {rec['variant']} | {rec['seed']} | {rec['freq']:+.5f} | "
                           f"{rec['A_oa']:.4f} | {rec['A_drop']:.4f} | {rec['B_oa']:.4f} | "
                           f"{rec['B_drop']:.4f} | {rec['diff_B_minus_A_drop']:+.4f} |")
    (REPORT / 'order_comparison.json').write_text(json.dumps(order_json, indent=2))
    (REPORT / 'order_comparison.md').write_text('\n'.join(order_lines) + '\n')

    # ---- item 2: six-model table ----
    low_high = {}
    for v in KEY_MODELS:
        for seed in SEEDS:
            k = (v, seed)
            p = Path(runpath.get(k, '')) / 'best_val.json'
            if p.is_file():
                m = json.loads(p.read_text())['metrics']
                low_high[k] = {'low': m['low']['oa'], 'high': m['high']['oa'],
                               'per_snr': {s: d['oa'] for s, d in m['snr'].items()}}
    md = ['# Six-model main table (clean + CFO robustness)\n']
    json_out = {}
    md.append('## Clean OA / Macro-F1 per seed (mean±std over 3 seeds)\n')
    md.append('| model | seed17 | seed42 | seed2026 | OA mean±std | F1 mean±std | low-SNR | high-SNR |')
    md.append('|---|---|---|---|---|---|---|---|')
    for v in KEY_MODELS:
        oas = [clean[(v, s)]['A']['oa'] for s in SEEDS if (v, s) in clean]
        f1s = [clean[(v, s)]['A']['macro_f1'] for s in SEEDS if (v, s) in clean]
        lows = [low_high[(v, s)]['low'] for s in SEEDS if (v, s) in low_high]
        highs = [low_high[(v, s)]['high'] for s in SEEDS if (v, s) in low_high]
        row = [f'{clean[(v,s)]["A"]["oa"]:.4f}' if (v, s) in clean else '-' for s in SEEDS]
        md.append(f"| {v} | " + ' | '.join(row) +
                  f" | {mean(oas):.4f}±{std(oas):.4f} | {mean(f1s):.4f}±{std(f1s):.4f} | "
                  f"{mean(lows):.4f} | {mean(highs):.4f} |")
        json_out.setdefault(v, {})['clean'] = {
            'oa_per_seed': {s: clean[(v, s)]['A']['oa'] for s in SEEDS if (v, s) in clean},
            'macro_f1_per_seed': {s: clean[(v, s)]['A']['macro_f1'] for s in SEEDS if (v, s) in clean},
            'oa_mean': mean(oas), 'oa_std': std(oas),
            'f1_mean': mean(f1s), 'f1_std': std(f1s),
            'low_snr_mean': mean(lows), 'high_snr_mean': mean(highs),
            'per_snr': low_high[(v, SEEDS[0])]['per_snr'] if (v, SEEDS[0]) in low_high else None}

    # CFO drop table (mean over seeds, order B as the audited correct order)
    md.append('\n## CFO robustness (clean→CFO drop, order B, mean over 3 seeds)\n')
    md.append('| model | ' + ' | '.join(f'{f:+.5f}' for f in FREQS) + ' |')
    md.append('|' + '---|' * (len(FREQS) + 1))
    for v in KEY_MODELS:
        drops = []
        for f in FREQS:
            ds = []
            for s in SEEDS:
                if (v, s) in clean and f in cfo[(v, s)]:
                    ds.append(clean[(v, s)]['A']['oa'] - cfo[(v, s)][f]['B']['oa'])
            drops.append(mean(ds) if ds else None)
        md.append('| ' + v + ' | ' + ' | '.join(f'{d:.4f}' if d is not None else '-' for d in drops) + ' |')
        json_out.setdefault(v, {})['cfo_drop_orderB'] = {f: mean([clean[(v, s)]['A']['oa'] - cfo[(v, s)][f]['B']['oa']
                                                                  for s in SEEDS if (v, s) in clean and f in cfo[(v, s)]])
                                                        for f in FREQS}
    (REPORT / 'six_model_table.json').write_text(json.dumps(json_out, indent=2))
    (REPORT / 'six_model_table.md').write_text('\n'.join(md) + '\n')

    # ---- item 3: same-seed paired diffs ----
    pairs = [('S1_energy', 'S1', 'energy−S1'),
             ('S1_corr', 'S1', 'corr−S1'),
             ('S2_concat', 'S1_energy', 'full−energy (corr add-on)'),
             ('S2_concat', 'S1_corr', 'full−corr (energy add-on)'),
             ('G_full', 'G', 'G_full−G'),
             ('S2_concat', 'G_full', 'S2_concat−G_full')]
    pd_md = ['# Same-seed paired differences (clean OA)\n']
    pd_md.append('| pair | seed17 | seed42 | seed2026 | mean±std |')
    pd_md.append('|---|---|---|---|---|')
    pd_json = []
    for a, b, label in pairs:
        diffs = []
        for s in SEEDS:
            if (a, s) in clean and (b, s) in clean:
                diffs.append(clean[(a, s)]['A']['oa'] - clean[(b, s)]['A']['oa'])
        cells = [f'{clean[(a,s)]["A"]["oa"] - clean[(b,s)]["A"]["oa"]:+.4f}'
                 if (a, s) in clean and (b, s) in clean else '-' for s in SEEDS]
        pd_md.append(f'| {label} | ' + ' | '.join(cells) + f' | {mean(diffs):+.4f}±{std(diffs):.4f} |')
        pd_json.append({'pair': label, 'a': a, 'b': b,
                        'diffs_per_seed': {s: clean[(a, s)]['A']['oa'] - clean[(b, s)]['A']['oa']
                                           for s in SEEDS if (a, s) in clean and (b, s) in clean},
                        'mean': mean(diffs), 'std': std(diffs)})
    (REPORT / 'paired_diffs.json').write_text(json.dumps(pd_json, indent=2))
    (REPORT / 'paired_diffs.md').write_text('\n'.join(pd_md) + '\n')

    print('wrote order_comparison, six_model_table, paired_diffs (.json/.md)')


if __name__ == '__main__':
    main()
