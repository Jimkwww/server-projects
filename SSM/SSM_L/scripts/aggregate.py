"""SSM_L round-1 result aggregation.

Reads every completed probe/finetune run plus the reused supervised baselines, computes the
paired tables and diffs (task book section 7), and writes reports/results.json,
reports/paired_tables.md and reports/conclusion.md.

Main endpoint: frozen-backbone linear-probe clean val OA. Secondary: low/high-SNR OA, Macro-F1.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GROUPS = ['G_mae', 'G_denoise', 'Q_mae', 'Q_denoise']
SEEDS = [17, 42, 2026]
OLD = ROOT.parent / 'SSM'  # read-only reference project


def load_metrics(path):
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d


def oa(m):
    return m['metrics']['overall']['oa'] if m else None


def f1(m):
    return m['metrics']['overall']['macro_f1'] if m else None


def group_metrics(mode, group, seed):
    p = ROOT / 'runs' / mode / f'{group}_seed{seed}' / 'best_val.json'
    return load_metrics(p)


def collect(mode):
    """Return {group: {seed: metrics}} for the given mode, plus completion flags."""
    data = {}
    for g in GROUPS:
        data[g] = {}
        for s in SEEDS:
            m = group_metrics(mode, g, s)
            data[g][s] = m['metrics'] if m else None
    return data


def baseline():
    """Reused supervised baselines: {name: {seed: metrics}}."""
    data = {}
    for name, sub, prefix in [('G_scratch', 'round3_aug', 'G'),
                              ('Q_scratch', 'round5_iq', 'Q_corr')]:
        data[name] = {}
        for s in SEEDS:
            p = OLD / 'runs' / sub / f'{prefix}_seed{s}' / 'best_val.json'
            m = load_metrics(p)
            data[name][s] = m['metrics'] if m else None
    return data


def fmt(v):
    return 'n/a' if v is None else f'{100 * v:.2f}'


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def build_tables(mode, data):
    """Build a per-group table with per-seed OA + mean±std and low/high-SNR OA."""
    lines = []
    lines.append(f'## {mode} — clean val OA (%) per seed, mean±std over seeds {SEEDS}')
    lines.append('')
    lines.append('| group | seed 17 | seed 42 | seed 2026 | mean±std | Macro-F1 (mean) | low-SNR OA (mean) | high-SNR OA (mean) |')
    lines.append('|---|---|---|---|---|---|---|---|')
    summary = {}
    for g in GROUPS:
        oas = [oa({'metrics': data[g][s]}) if data[g][s] is not None else None for s in SEEDS]
        mean, std = stats(oas)
        f1s = [f1({'metrics': data[g][s]}) if data[g][s] is not None else None for s in SEEDS]
        f1m, _ = stats(f1s)
        lows = [data[g][s]['low']['oa'] if data[g][s] is not None and data[g][s].get('low') else None for s in SEEDS]
        highs = [data[g][s]['high']['oa'] if data[g][s] is not None and data[g][s].get('high') else None for s in SEEDS]
        lm, _ = stats(lows); hm, _ = stats(highs)
        cells = [fmt(v) for v in oas]
        mean_s = 'n/a' if mean is None else f'{100 * mean:.2f}±{100 * std:.2f}'
        f1_s = 'n/a' if f1m is None else f'{100 * f1m:.2f}'
        lm_s = 'n/a' if lm is None else f'{100 * lm:.2f}'
        hm_s = 'n/a' if hm is None else f'{100 * hm:.2f}'
        lines.append(f'| {g} | ' + ' | '.join(cells) + f' | {mean_s} | {f1_s} | {lm_s} | {hm_s} |')
        summary[g] = {'per_seed_oa': oas, 'mean_oa': mean, 'std_oa': std,
                      'mean_f1': f1m, 'mean_low_oa': lm, 'mean_high_oa': hm}
    return lines, summary


def paired_diffs(data, metric='oa'):
    """Compute the four paired differences (per-seed and mean) from {group:{seed:metrics}}."""
    def gv(g, s):
        m = data[g][s]
        return m['overall'][metric] if m is not None else None

    diffs = {
        'G_denoise_minus_G_mae': [],
        'Q_denoise_minus_Q_mae': [],
        'Q_denoise_minus_G_denoise': [],
        'interaction_QminusG': [],
    }
    for s in SEEDS:
        a, b = gv('G_mae', s), gv('G_denoise', s)
        c, d = gv('Q_mae', s), gv('Q_denoise', s)
        diffs['G_denoise_minus_G_mae'].append((b - a) if (a is not None and b is not None) else None)
        diffs['Q_denoise_minus_Q_mae'].append((d - c) if (c is not None and d is not None) else None)
        diffs['Q_denoise_minus_G_denoise'].append((d - b) if (b is not None and d is not None) else None)
        g_diff = (b - a) if (a is not None and b is not None) else None
        q_diff = (d - c) if (c is not None and d is not None) else None
        diffs['interaction_QminusG'].append((q_diff - g_diff) if (g_diff is not None and q_diff is not None) else None)
    return diffs


def diff_lines(diffs):
    lines = []
    lines.append('')
    lines.append('### paired differences (percentage points, per seed then mean)')
    lines.append('')
    lines.append('| difference | seed 17 | seed 42 | seed 2026 | mean |')
    lines.append('|---|---|---|---|---|')
    names = {
        'G_denoise_minus_G_mae': 'G-denoise − G-MAE',
        'Q_denoise_minus_Q_mae': 'Q-denoise − Q-MAE',
        'Q_denoise_minus_G_denoise': 'Q-denoise − G-denoise',
        'interaction_QminusG': '(Q-denoise − Q-MAE) − (G-denoise − G-MAE)',
    }
    out = {}
    for k, vals in diffs.items():
        mean, std = stats(vals)
        cells = [('n/a' if v is None else f'{100 * v:+.2f}') for v in vals]
        mean_s = 'n/a' if mean is None else f'{100 * mean:+.2f}'
        lines.append(f'| {names[k]} | ' + ' | '.join(cells) + f' | {mean_s} |')
        out[k] = {'per_seed_pp': vals, 'mean_pp': mean}
    return lines, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'reports'))
    args = ap.parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    result = {'groups': GROUPS, 'seeds': SEEDS, 'modes': {}}
    md = ['# SSM_L round-1 results', '']

    for mode in ('probe', 'finetune'):
        data = collect(mode)
        tbl, summary = build_tables(mode, data)
        diffs = paired_diffs(data)
        dlines, dout = diff_lines(diffs)
        md += tbl + dlines + ['']
        result['modes'][mode] = {'summary': summary, 'paired_diffs_oa': dout}

    base = baseline()
    md.append('## supervised baselines (reused from old project) — clean val OA (%)')
    md.append('')
    md.append('| baseline | seed 17 | seed 42 | seed 2026 | mean±std |')
    md.append('|---|---|---|---|---|')
    result['baselines'] = {}
    for name in ('G_scratch', 'Q_scratch'):
        oas = [oa({'metrics': base[name][s]}) if base[name][s] is not None else None for s in SEEDS]
        mean, std = stats(oas)
        cells = [fmt(v) for v in oas]
        mean_s = 'n/a' if mean is None else f'{100 * mean:.2f}±{100 * std:.2f}'
        md.append(f'| {name} | ' + ' | '.join(cells) + f' | {mean_s} |')
        result['baselines'][name] = {'per_seed_oa': oas, 'mean_oa': mean, 'std_oa': std}

    (outdir / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (outdir / 'paired_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md), flush=True)
    print(f'\nwrote {outdir}/results.json and {outdir}/paired_tables.md', flush=True)


if __name__ == '__main__':
    main()
