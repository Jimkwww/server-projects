#!/usr/bin/env python
"""Final test aggregation: per-seed + mean +/- sample std (ddof=1), per-SNR,
and honest test-vs-val ranking comparison. Reads the frozen test eval and the
val-side round3_review eval (same order-B protocol) for the matching grid.
"""
import json
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
SEEDS = (17, 42, 2026)
VARIANTS = ('G', 'G_corr', 'S1', 'S1_corr')
GRID = (-0.001, 0.001, -0.002, 0.002, -0.005, 0.005, -0.01, 0.01, -0.02, 0.02)
KEY_CFO = (-0.005, 0.005, -0.01, 0.01, -0.02, 0.02)
CLASS_NAMES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
               'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']


def pct(x):
    return x * 100.0


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(v)


def load_cases(path):
    cases = defaultdict(dict)
    for l in Path(path).read_text().splitlines():
        r = json.loads(l)
        key = (r['kind'], r['value'])
        rel = r['run'].replace(str(ROOT) + '/', '')
        cases[rel][key] = r['metrics']
    return cases


def run_key(v, s):
    return f'runs/round3_aug/{v}_seed{s}'


def oa(cases, v, s, key):
    return cases[run_key(v, s)][key]['oa']


def f1(cases, v, s, key):
    return cases[run_key(v, s)][key]['macro_f1']


def emit_md(path, lines):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(lines) + '\n')


def main():
    test = load_cases(ROOT / 'reports/final_test/test_cases.jsonl')
    val = load_cases(ROOT / 'reports/round3_review/extended_cases.jsonl')

    # per-model per-seed + mean
    def model_tables(cases):
        out = {}
        for v in VARIANTS:
            e = {'seeds': []}
            for s in SEEDS:
                e['seeds'].append({
                    'seed': s,
                    'clean_oa': oa(cases, v, s, ('clean', 0.0)),
                    'clean_f1': f1(cases, v, s, ('clean', 0.0)),
                    'cfo_oa': {str(g): oa(cases, v, s, ('cfo', g)) for g in GRID},
                    'cfo_f1': {str(g): f1(cases, v, s, ('cfo', g)) for g in GRID},
                })
            e['clean_oa_mean'], e['clean_oa_std'] = mean_std([x['clean_oa'] for x in e['seeds']])
            e['clean_f1_mean'], e['clean_f1_std'] = mean_std([x['clean_f1'] for x in e['seeds']])
            e['cfo_oa_mean'] = {str(g): mean_std([oa(cases, v, s, ('cfo', g)) for s in SEEDS]) for g in GRID}
            e['cfo_f1_mean'] = {str(g): mean_std([f1(cases, v, s, ('cfo', g)) for s in SEEDS]) for g in GRID}
            out[v] = e
        return out

    test_t = model_tables(test)
    val_t = model_tables(val)

    (ROOT / 'reports/final_test/test_main.json').write_text(json.dumps({
        'sample_std': True, 'split': 'test', 'grid': list(GRID),
        'test': test_t,
    }, indent=2))

    # ---------- markdown ----------
    md = []
    md.append('# 最终 test 评估（冻结协议，独立测试集）\n')
    md.append('> 12 个增强后 checkpoint（G/G_corr/S1/S1_corr × 3 seed），best.pt 按 clean val OA 最早 tie 冻结，'
              'test 仅作最终评估、未参与任何选择/调参。order B（raw→CFO→max_abs→重算描述→模型），'
              '均值±样本标准差（ddof=1）。')

    md.append('\n## clean（逐种子 + 均值±样本标准差，pp）\n')
    md.append('| 模型 | seed17 OA/F1 | seed42 OA/F1 | seed2026 OA/F1 | 均值 OA | ±std | 均值 F1 | ±std |')
    md.append('|---|---|---|---|---|---|---|---|')
    for v in VARIANTS:
        e = test_t[v]; s = e['seeds']
        md.append('| %s | %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | %.1f | ±%.1f | %.1f | ±%.1f |' % (
            v, pct(s[0]['clean_oa']), pct(s[0]['clean_f1']), pct(s[1]['clean_oa']), pct(s[1]['clean_f1']),
            pct(s[2]['clean_oa']), pct(s[2]['clean_f1']),
            pct(e['clean_oa_mean']), pct(e['clean_oa_std']), pct(e['clean_f1_mean']), pct(e['clean_f1_std'])))

    md.append('\n## 各频偏绝对 OA（3 seed 均值±样本标准差，pp）\n')
    md.append('| CFO | G | G_corr | S1 | S1_corr |')
    md.append('|---|---|---|---|---|')
    for g in GRID:
        md.append('| %s | %s | %s | %s | %s |' % (
            g,
            '%.1f±%.1f' % (pct(test_t['G']['cfo_oa_mean'][str(g)][0]), pct(test_t['G']['cfo_oa_mean'][str(g)][1])),
            '%.1f±%.1f' % (pct(test_t['G_corr']['cfo_oa_mean'][str(g)][0]), pct(test_t['G_corr']['cfo_oa_mean'][str(g)][1])),
            '%.1f±%.1f' % (pct(test_t['S1']['cfo_oa_mean'][str(g)][0]), pct(test_t['S1']['cfo_oa_mean'][str(g)][1])),
            '%.1f±%.1f' % (pct(test_t['S1_corr']['cfo_oa_mean'][str(g)][0]), pct(test_t['S1_corr']['cfo_oa_mean'][str(g)][1])),
        ))

    # per-SNR (clean and +-0.005), mean over seeds
    md.append('\n## 逐 SNR 绝对 OA（3 seed 均值，pp）\n')
    for key_name, key in (('clean', ('clean', 0.0)), ('CFO=+0.005', ('cfo', 0.005)), ('CFO=−0.005', ('cfo', -0.005))):
        md.append(f'\n### {key_name}\n')
        snrs = sorted({float(z) for z in test[run_key('G', 17)][key]['per_snr']})
        md.append('| SNR | G | G_corr | S1 | S1_corr |')
        md.append('|---|---|---|---|---|')
        for z in snrs:
            cells = []
            for v in VARIANTS:
                xs = [test[run_key(v, s)][key]['per_snr'][str(z)]['oa'] for s in SEEDS]
                cells.append('%.1f' % pct(sum(xs) / len(xs)))
            md.append('| %d | %s |' % (int(z), ' | '.join(cells)))
    emit_md('reports/final_test/test_main.md', md)

    # ---------- test vs val ranking ----------
    md2 = []
    md2.append('# 最终 test 结论：test vs val 排序核对\n')
    md2.append('> 区分验证集开发结果与独立测试集结果。以下对比同一 order-B 协议下 val 与 test 的 3-seed 均值。')

    def rank(models, key):
        # models: {v: mean oa}
        return sorted(models, key=models.get, reverse=True)

    md2.append('\n## 排序对比（OA 均值，pp；从高到低）\n')
    md2.append('| 条件 | val 排序 | test 排序 | 是否一致 |')
    md2.append('|---|---|---|---|')
    for key_name, key in (('clean', ('clean', 0.0)),
                          ('CFO=+0.005', ('cfo', 0.005)), ('CFO=−0.005', ('cfo', -0.005)),
                          ('CFO=+0.01', ('cfo', 0.01)), ('CFO=−0.01', ('cfo', -0.01)),
                          ('CFO=+0.02', ('cfo', 0.02)), ('CFO=−0.02', ('cfo', -0.02))):
        v_means = {v: val_t[v]['cfo_oa_mean'][str(key[1])][0] if key[0] == 'cfo' else val_t[v]['clean_oa_mean'] for v in VARIANTS}
        t_means = {v: test_t[v]['cfo_oa_mean'][str(key[1])][0] if key[0] == 'cfo' else test_t[v]['clean_oa_mean'] for v in VARIANTS}
        vr = rank(v_means, key)
        tr = rank(t_means, key)
        same = vr == tr
        md2.append('| %s | %s | %s | %s |' % (
            key_name, ' > '.join(vr), ' > '.join(tr), '一致' if same else '**不一致**'))

    md2.append('\n## 结论\n')
    md2.append('（由 aggregate_test.py 汇总；最终文字结论见 test_conclusion.md）')
    emit_md('reports/final_test/test_vs_val.md', md2)

    print('wrote test_main.json/.md, test_vs_val.md')


if __name__ == '__main__':
    main()
