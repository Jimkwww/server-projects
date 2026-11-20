#!/usr/bin/env python
"""Round-3 aggregation: G_corr control + pre/post CFO-augmentation comparison.

Reads round2 recheck/newmodel, round3 gcorr and round3 aug cases; writes
reports/round3/{four_model_prepost,sample_std}.{json,md}.
All numbers order B, mean±sample std over 3 seeds.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
REPORT = ROOT / 'reports' / 'round3'
SEEDS = [17, 42, 2026]
MODELS = ['G', 'G_corr', 'S1', 'S1_corr']
FREQS = [0.001, 0.002, 0.005, -0.001, -0.002, -0.005]


def load_rows(paths):
    rows = []
    for p in paths:
        if Path(p).exists():
            rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    return rows


def mean(xs):
    return sum(xs) / len(xs)


def sstd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0


def main():
    pre_rows = load_rows(['reports/round2/recheck_cases.jsonl',
                          'reports/round2/newmodel_cases.jsonl',
                          'reports/round3/gcorr_cases.jsonl'])
    aug_rows = load_rows(['reports/round3/aug_cases.jsonl'])

    def index(rows):
        clean = defaultdict(dict); cfo = defaultdict(lambda: defaultdict(dict))
        for r in rows:
            k = (r['variant'], r['seed'])
            if r['kind'] == 'clean':
                clean[k][r['order']] = r
            elif r['kind'] == 'cfo':
                cfo[k][r['value']][r['order']] = r
        return clean, cfo

    pre_clean, pre_cfo = index(pre_rows)
    aug_clean, aug_cfo = index(aug_rows)

    out = {'sample_std': True, 'order': 'B'}

    # Four-model absolute performance, pre and post augmentation
    md = ['# Four-model pre/post CFO-augmentation (order B, mean±sample std over 3 seeds)', '']
    for tag, clean, cfo in (('pre-aug', pre_clean, pre_cfo), ('post-aug', aug_clean, aug_cfo)):
        md.append(f'## {tag} (absolute OA)')
        md.append('')
        md.append('| model | clean | CFO=+0.001 | +0.002 | +0.005 | −0.001 | −0.002 | −0.005 |')
        md.append('|---|---|---|---|---|---|---|---|')
        for v in MODELS:
            cells = []
            for f in [None] + FREQS:
                if f is None:
                    vals = [clean[(v, s)]['B']['oa'] for s in SEEDS if (v, s) in clean]
                else:
                    vals = [cfo[(v, s)][f]['B']['oa'] for s in SEEDS if (v, s) in cfo and f in cfo[(v, s)]]
                if not vals:
                    cells.append('-')
                else:
                    cells.append(f'{mean(vals):.4f}±{sstd(vals):.4f}')
            md.append('| ' + v + ' | ' + ' | '.join(cells) + ' |')
            out.setdefault(tag, {})[v] = {
                'clean': mean([clean[(v, s)]['B']['oa'] for s in SEEDS if (v, s) in clean]),
                'cfo': {f: mean([cfo[(v, s)][f]['B']['oa'] for s in SEEDS
                                 if (v, s) in cfo and f in cfo[(v, s)]]) for f in FREQS}}
        md.append('')

    # Post-aug paired diffs (corr - base, same backbone)
    md.append('## Post-aug paired diffs (order B)')
    md.append('')
    md.append('| pair | clean | CFO=+0.005 | CFO=−0.005 |')
    md.append('|---|---|---|---|')
    for a, b, label in [('G_corr', 'G', 'G_corr−G (aug)'),
                        ('S1_corr', 'S1', 'S1_corr−S1 (aug)'),
                        ('S1_corr', 'G_corr', 'S1_corr−G_corr (aug)')]:
        cd = [aug_clean[(a, s)]['B']['oa'] - aug_clean[(b, s)]['B']['oa'] for s in SEEDS]
        p1 = [aug_cfo[(a, s)][0.005]['B']['oa'] - aug_cfo[(b, s)][0.005]['B']['oa'] for s in SEEDS]
        p2 = [aug_cfo[(a, s)][-0.005]['B']['oa'] - aug_cfo[(b, s)][-0.005]['B']['oa'] for s in SEEDS]
        md.append(f'| {label} | {mean(cd)*100:+.2f}pp | {mean(p1)*100:+.2f}pp | {mean(p2)*100:+.2f}pp |')
        out.setdefault('post_aug_paired_diffs', {})[label] = {
            'clean_per_seed': cd, 'cfo_plus_005_per_seed': p1, 'cfo_minus_005_per_seed': p2}
    md.append('')

    # Augmentation delta (post - pre), absolute OA
    md.append('## Augmentation delta (post−pre absolute OA, mean over seeds)')
    md.append('')
    md.append('| model | clean | CFO=+0.005 | CFO=−0.005 |')
    md.append('|---|---|---|---|')
    for v in MODELS:
        d_clean = mean([aug_clean[(v, s)]['B']['oa'] for s in SEEDS]) - mean([pre_clean[(v, s)]['B']['oa'] for s in SEEDS])
        d_p = mean([aug_cfo[(v, s)][0.005]['B']['oa'] for s in SEEDS]) - mean([pre_cfo[(v, s)][0.005]['B']['oa'] for s in SEEDS])
        d_m = mean([aug_cfo[(v, s)][-0.005]['B']['oa'] for s in SEEDS]) - mean([pre_cfo[(v, s)][-0.005]['B']['oa'] for s in SEEDS])
        md.append(f'| {v} | {d_clean*100:+.2f}pp | {d_p*100:+.2f}pp | {d_m*100:+.2f}pp |')
        out.setdefault('aug_delta', {})[v] = {'clean': d_clean, 'cfo_plus_005': d_p, 'cfo_minus_005': d_m}
    md.append('')

    (REPORT / 'four_model_prepost.json').write_text(json.dumps(out, indent=2))
    (REPORT / 'four_model_prepost.md').write_text('\n'.join(md) + '\n')
    print('wrote four_model_prepost.{json,md}')


if __name__ == '__main__':
    main()
