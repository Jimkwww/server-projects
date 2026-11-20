#!/usr/bin/env python
"""Round-3 review aggregation.

Reads best_val.json / DONE.json / history.jsonl of all 24 existing checkpoints and
the unified extended-CFO eval (reports/round3_review/extended_cases.jsonl), then
emits: per-seed main tables (clean OA/F1, per-CFO absolute OA/F1, mean +/- sample
std ddof=1, seed-paired diffs, best/last-epoch train metrics), full CFO curves,
per-class concentration and low/high-SNR breakdown. No test-set use anywhere.
"""
import json
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
SEEDS = (17, 42, 2026)
CLASS_NAMES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
               'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

# regime -> variant -> [run dir per seed 17/42/2026]
RUNMAP = {
    'pre': {
        'G':      [f'runs/clean/G_seed{s}' for s in SEEDS],
        'S1':     [f'runs/clean/S1_seed{s}' for s in SEEDS],
        'S1_corr':[f'runs/round2/S1_corr_seed{s}' for s in SEEDS],
        'G_corr': [f'runs/round3/G_corr_seed{s}' for s in SEEDS],
    },
    'post': {
        'G':      [f'runs/round3_aug/G_seed{s}' for s in SEEDS],
        'G_corr': [f'runs/round3_aug/G_corr_seed{s}' for s in SEEDS],
        'S1':     [f'runs/round3_aug/S1_seed{s}' for s in SEEDS],
        'S1_corr':[f'runs/round3_aug/S1_corr_seed{s}' for s in SEEDS],
    },
}
VARIANTS = ('G', 'G_corr', 'S1', 'S1_corr')


def pct(x):
    return x * 100.0


def mean_std(xs):
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var)


def load_run(run_dir):
    run = ROOT / run_dir
    bv = json.loads((run / 'best_val.json').read_text())
    done = json.loads((run / 'DONE.json').read_text())
    hist = [json.loads(l) for l in (run / 'history.jsonl').read_text().splitlines() if l.strip()]
    by_epoch = {r['epoch']: r for r in hist}
    best_epoch = bv['best_epoch']
    last_epoch = done['epochs'] - 1
    def row(e):
        r = by_epoch[e]
        return {'epoch': e, 'train_loss': r['train_loss'], 'train_oa': r['train_oa'],
                'val_oa': r['val']['overall']['oa'], 'val_f1': r['val']['overall']['macro_f1']}
    return {
        'dir': run_dir,
        'best_epoch': best_epoch,
        'clean_oa': bv['metrics']['overall']['oa'],
        'clean_f1': bv['metrics']['overall']['macro_f1'],
        'best_row': row(best_epoch),
        'last_row': row(last_epoch),
    }


def load_cases():
    path = ROOT / 'reports/round3_review/extended_cases.jsonl'
    cases = defaultdict(dict)  # (run_dir) -> {('clean',0.0) or ('cfo',f): metrics}
    for l in path.read_text().splitlines():
        r = json.loads(l)
        key = (r['kind'], r['value'])
        cases[r['run'].replace(str(ROOT) + '/', '')][key] = r['metrics']
    return cases


def emit_md(path, lines):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(lines) + '\n')


def main():
    runs = {}
    for regime, vm in RUNMAP.items():
        runs[regime] = {}
        for v, dirs in vm.items():
            runs[regime][v] = [load_run(d) for d in dirs]

    cases = load_cases()
    grid = sorted({k[1] for kk in cases.values() for k in kk if k[0] == 'cfo'})

    def cfo_oa(regime, v, seed_i, f):
        d = RUNMAP[regime][v][seed_i]
        return cases[d][('cfo', f)]['oa']

    def cfo_f1(regime, v, seed_i, f):
        d = RUNMAP[regime][v][seed_i]
        return cases[d][('cfo', f)]['macro_f1']

    # ---------------- per-seed main + paired diffs ----------------
    per_seed = {}
    for regime in ('pre', 'post'):
        per_seed[regime] = {}
        for v in VARIANTS:
            entry = {'seeds': []}
            for i, s in enumerate(SEEDS):
                rr = runs[regime][v][i]
                seed_row = {
                    'seed': s, 'clean_oa': rr['clean_oa'], 'clean_f1': rr['clean_f1'],
                    'best_epoch': rr['best_epoch'],
                    'best_train': rr['best_row'], 'last_train': rr['last_row'],
                    'cfo_oa': {str(f): cfo_oa(regime, v, i, f) for f in grid},
                    'cfo_f1': {str(f): cfo_f1(regime, v, i, f) for f in grid},
                }
                entry['seeds'].append(seed_row)
            entry['clean_oa_mean'], entry['clean_oa_std'] = mean_std([s['clean_oa'] for s in entry['seeds']])
            entry['clean_f1_mean'], entry['clean_f1_std'] = mean_std([s['clean_f1'] for s in entry['seeds']])
            entry['cfo_oa_mean'] = {str(f): mean_std([cfo_oa(regime, v, i, f) for i in range(3)]) for f in grid}
            entry['cfo_f1_mean'] = {str(f): mean_std([cfo_f1(regime, v, i, f) for i in range(3)]) for f in grid}
            per_seed[regime][v] = entry

    paired = {}
    for regime in ('pre', 'post'):
        paired[regime] = {}
        for a, b in (('G_corr', 'G'), ('S1_corr', 'S1'), ('S1_corr', 'G_corr')):
            key = f'{a}-{b}'
            diffs = {'clean': [], 'f1': {}}
            for f in grid:
                diffs['f1'][str(f)] = []
            for i in range(3):
                diffs['clean'].append(runs[regime][a][i]['clean_oa'] - runs[regime][b][i]['clean_oa'])
                for f in grid:
                    diffs['f1'][str(f)].append(cfo_oa(regime, a, i, f) - cfo_oa(regime, b, i, f))
            diffs['clean_mean'], diffs['clean_std'] = mean_std(diffs['clean'])
            diffs['clean_signs'] = ['+' if d > 0 else ('-' if d < 0 else '0') for d in diffs['clean']]
            diffs['cfo_mean'] = {str(f): mean_std(diffs['f1'][str(f)]) for f in grid}
            diffs['cfo_signs'] = {str(f): ['+' if d > 0 else ('-' if d < 0 else '0') for d in diffs['f1'][str(f)]] for f in grid}
            paired[regime][key] = diffs

    (ROOT / 'reports/round3_review').mkdir(parents=True, exist_ok=True)
    (ROOT / 'reports/round3_review/per_seed_main.json').write_text(
        json.dumps({'sample_std': True, 'grid': grid, 'per_seed': per_seed, 'paired': paired}, indent=2))

    # ---------------- markdown: per-seed main ----------------
    md = []
    md.append('# Round 3 复核：逐种子主表 + 配对差\n')
    md.append('> 所有均值用**样本标准差（ddof=1）**；OA/F1 为 val 集冻结协议（raw→CFO→max_abs→重算描述→模型，order B）。'
              'best checkpoint 按 clean val OA 最早 tie 选择。')
    for regime, label in (('pre', '无增强'), ('post', 'CFO 增强 U(−0.002,0.002) order B')):
        md.append(f'\n## {label}\n')
        # clean table
        md.append('### clean（逐种子 + 均值±样本标准差）\n')
        md.append('| 模型 | seed17 OA/F1 | seed42 OA/F1 | seed2026 OA/F1 | 均值 OA | ±std OA | 均值 F1 | ±std F1 | best_epoch |')
        md.append('|---|---|---|---|---|---|---|---|---|')
        for v in VARIANTS:
            e = per_seed[regime][v]
            s = e['seeds']
            md.append('| %s | %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | %.1f | ±%.1f | %.1f | ±%.1f | %d/%d/%d |' % (
                v, pct(s[0]['clean_oa']), pct(s[0]['clean_f1']), pct(s[1]['clean_oa']), pct(s[1]['clean_f1']),
                pct(s[2]['clean_oa']), pct(s[2]['clean_f1']),
                pct(e['clean_oa_mean']), pct(e['clean_oa_std']), pct(e['clean_f1_mean']), pct(e['clean_f1_std']),
                s[0]['best_epoch'], s[1]['best_epoch'], s[2]['best_epoch']))
        # CFO absolute OA table (mean +- std), key points
        key_cfo = [f for f in grid if abs(f) in (0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.02)]
        md.append('\n### 各频偏绝对 OA（3 seed 均值 ± 样本标准差，pp）\n')
        md.append('| 模型 | ' + ' | '.join(f'CFO={f}' for f in key_cfo) + ' |')
        md.append('|---|' + '---|' * len(key_cfo))
        for v in VARIANTS:
            e = per_seed[regime][v]
            cells = []
            for f in key_cfo:
                m, s = e['cfo_oa_mean'][str(f)]
                cells.append('%.1f±%.1f' % (pct(m), pct(s)))
            md.append('| %s | %s |' % (v, ' | '.join(cells)))
        # best/last train metrics
        md.append('\n### 训练指标（best epoch / 末 epoch）\n')
        md.append('| 模型 | seed | best_epoch train_loss/train_OA→val_OA | last_epoch train_loss/train_OA→val_OA |')
        md.append('|---|---|---|---|')
        for v in VARIANTS:
            for srow in per_seed[regime][v]['seeds']:
                br, lr = srow['best_train'], srow['last_train']
                md.append('| %s | %d | %.4f / %.1f → %.1f | %.4f / %.1f → %.1f |' % (
                    v, srow['seed'], br['train_loss'], pct(br['train_oa']), pct(br['val_oa']),
                    lr['train_loss'], pct(lr['train_oa']), pct(lr['val_oa'])))

    # paired diffs markdown
    md.append('\n\n## 同种子配对差（pp；括号为 3 seed 符号）\n')
    for regime, label in (('pre', '无增强'), ('post', 'CFO 增强')):
        md.append(f'\n### {label}\n')
        md.append('| 配对 | clean 逐种子 | clean 均值±std | CFO=+0.005 逐种子 | CFO=−0.005 逐种子 |')
        md.append('|---|---|---|---|---|')
        for key in ('G_corr-G', 'S1_corr-S1', 'S1_corr-G_corr'):
            d = paired[regime][key]
            cl = ' / '.join('%.2f' % pct(x) for x in d['clean'])
            pm = ' / '.join('%.2f' % pct(x) for x in d['f1']['0.005'])
            nm = ' / '.join('%.2f' % pct(x) for x in d['f1']['-0.005'])
            md.append('| %s | %s | %.2f±%.2f | %s | %s |' % (
                key, cl, pct(d['clean_mean']), pct(d['clean_std']), pm, nm))

    emit_md('reports/round3_review/per_seed_main.md', md)

    # ---------------- full CFO curves ----------------
    cm = []
    cm.append('# Round 3 复核：完整 CFO 曲线（绝对 OA，3 seed 均值±样本标准差）\n')
    cm.append('> 单位 pp；order B。绝对性能为主，同时给相对 clean 的下降。')
    for regime, label in (('pre', '无增强'), ('post', 'CFO 增强 U(−0.002,0.002) order B')):
        cm.append(f'\n## {label}\n')
        cm.append('| CFO | ' + ' | '.join(VARIANTS) + ' |')
        cm.append('|---|' + '---|' * len(VARIANTS))
        for f in grid:
            cells = []
            for v in VARIANTS:
                m, s = per_seed[regime][v]['cfo_oa_mean'][str(f)]
                cells.append('%.1f±%.1f' % (pct(m), pct(s)))
            cm.append('| %s | %s |' % (f, ' | '.join(cells)))
        cm.append('| clean | ' + ' | '.join('%.1f±%.1f' % (pct(per_seed[regime][v]['clean_oa_mean']), pct(per_seed[regime][v]['clean_oa_std'])) for v in VARIANTS) + ' |')
        cm.append('\n相对 clean 的下降（pp）：')
        cm.append('| CFO | ' + ' | '.join(VARIANTS) + ' |')
        cm.append('|---|' + '---|' * len(VARIANTS))
        for f in grid:
            cells = []
            for v in VARIANTS:
                m, _ = per_seed[regime][v]['cfo_oa_mean'][str(f)]
                drop = per_seed[regime][v]['clean_oa_mean'] - m
                cells.append('%.1f' % pct(drop))
            cm.append('| %s | %s |' % (f, ' | '.join(cells)))
    emit_md('reports/round3_review/cfo_curves.md', cm)

    # ---------------- per-class concentration ----------------
    def class_f1(regime, v, seed_i, key):
        d = RUNMAP[regime][v][seed_i]
        return cases[d][key]['class_f1']

    def class_recall(regime, v, seed_i, key):
        d = RUNMAP[regime][v][seed_i]
        return cases[d][key]['class_recall']

    cl = []
    cl.append('# Round 3 复核：类别集中度与 SNR 分解\n')
    cl.append('> 3 seed 均值（未挑种子）。class 序号对应 RML2018.01a 标准顺序。')

    for regime, label in (('pre', '无增强'), ('post', 'CFO 增强')):
        cl.append(f'\n## {label}\n')
        for key_name, key in (('clean', ('clean', 0.0)), ('CFO=+0.002', ('cfo', 0.002)),
                              ('CFO=−0.002', ('cfo', -0.002)), ('CFO=+0.005', ('cfo', 0.005)),
                              ('CFO=−0.005', ('cfo', -0.005))):
            cl.append(f'\n### {key_name} —— 复相关增益（F1，pp，S1_corr−S1 与 G_corr−G）\n')
            rows = []
            for ci in range(24):
                d_ssm = mean_std([class_f1(regime, 'S1_corr', i, key)[ci] - class_f1(regime, 'S1', i, key)[ci] for i in range(3)])
                d_g = mean_std([class_f1(regime, 'G_corr', i, key)[ci] - class_f1(regime, 'G', i, key)[ci] for i in range(3)])
                rows.append((ci, d_ssm, d_g))
            rows_sorted = sorted(rows, key=lambda r: r[1][0], reverse=True)
            cl.append('| 类 | 名称 | S1_corr−S1 (均值±std) | G_corr−G (均值±std) |')
            cl.append('|---|---|---|---|')
            for ci, d_ssm, d_g in rows_sorted:
                cl.append('| %d | %s | %+.1f±%.1f | %+.1f±%.1f |' % (
                    ci, CLASS_NAMES[ci], pct(d_ssm[0]), pct(d_ssm[1]), pct(d_g[0]), pct(d_g[1])))
            # top gains summary
            top_gain = rows_sorted[:6]
            top_loss = rows_sorted[-6:]
            cl.append('- 增益最大（S1_corr−S1）：' + '；'.join(f'{CLASS_NAMES[ci]}({ci}) {pct(d_ssm[0]):+.1f}' for ci, d_ssm, _ in top_gain))
            cl.append('- 受损最大（S1_corr−S1）：' + '；'.join(f'{CLASS_NAMES[ci]}({ci}) {pct(d_ssm[0]):+.1f}' for ci, d_ssm, _ in top_loss))

        # low/high SNR decomposition for S1_corr vs G_corr and S1_corr vs S1
        cl.append(f'\n### {label} —— 低/高 SNR 分解（OA，pp）\n')
        cl.append('| 条件 | 配对 | low(≤0dB) 逐种子 | high(>0dB) 逐种子 |')
        cl.append('|---|---|---|---|')
        for key_name, key in (('clean', ('clean', 0.0)), ('CFO=+0.002', ('cfo', 0.002)),
                              ('CFO=−0.002', ('cfo', -0.002)), ('CFO=+0.005', ('cfo', 0.005)),
                              ('CFO=−0.005', ('cfo', -0.005))):
            for a, b in (('S1_corr', 'G_corr'), ('S1_corr', 'S1')):
                lows = [cases[RUNMAP[regime][a][i]][key]['low']['oa'] - cases[RUNMAP[regime][b][i]][key]['low']['oa'] for i in range(3)]
                highs = [cases[RUNMAP[regime][a][i]][key]['high']['oa'] - cases[RUNMAP[regime][b][i]][key]['high']['oa'] for i in range(3)]
                cl.append('| %s | %s−%s | %s | %s |' % (
                    key_name, a, b,
                    ' / '.join('%.2f' % pct(x) for x in lows), ' / '.join('%.2f' % pct(x) for x in highs)))
    emit_md('reports/round3_review/class_analysis.md', cl)

    print('wrote per_seed_main.json/.md, cfo_curves.md, class_analysis.md')


if __name__ == '__main__':
    main()
