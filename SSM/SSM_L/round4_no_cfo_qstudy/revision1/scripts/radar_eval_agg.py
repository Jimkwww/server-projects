"""Revision-1 Section 1(2) + Section 7: re-aggregate RADAR **eval** (test) OA / Macro-F1
for P0 (backbone x init) and P1 (A/B/C/D 2x2), with per-seed values and P1 paired diffs vs D.

Source of truth: each run's eval_predictions.npz (y, pred). No model reload, no rerun.
Original round-4 tables_p1.md reported val(=dev) OA; this corrects it to eval OA.

Outputs: reports/radar_eval_main.json + reports/tables_radar_eval.md
"""
import json
from pathlib import Path

import numpy as np

R4 = Path(__file__).resolve().parent.parent.parent
R1 = R4 / 'revision1'
SEEDS = (17, 42, 2026)
BACKBONES = ('G', 'Q', 'S1plain', 'G_Qlearn', 'G_corr')
INITS = ('scratch', 'mae', 'denoise')
P1_SETTINGS = ('A_fixedq_fixedq', 'B_learnq_fixedq', 'C_fixedq_learnq')


def metrics(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    oa = float((pred == y).mean())
    classes = np.unique(y)
    f1s = []
    for c in classes:
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    macro_f1 = float(np.mean(f1s))
    return oa, macro_f1


def load_eval(task_dir, seed):
    d = np.load(task_dir / f'eval_predictions.npz')
    return metrics(d['y'], d['pred'])


def p0_path(init, bb, seed):
    return R4 / 'runs' / 'radar' / init / f'{bb}_seed{seed}'


def p1_path(setting, seed):
    return R4 / 'runs' / 'radar' / setting / f'G_Qlearn_seed{seed}'


def meanstd(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def main():
    p0 = {}
    for init in INITS:
        for bb in BACKBONES:
            if bb == 'G_corr' and init == 'denoise':
                continue  # G_corr only scratch/mae in round-4
            for seed in SEEDS:
                p = p0_path(init, bb, seed)
                if not (p / 'eval_predictions.npz').exists():
                    p0[f'{bb}/{init}/{seed}'] = None
                    continue
                p0[f'{bb}/{init}/{seed}'] = load_eval(p, seed)

    p1 = {}
    for setting in P1_SETTINGS:
        for seed in SEEDS:
            p = p1_path(setting, seed)
            p1[f'{setting}/{seed}'] = load_eval(p, seed)
    # D = learn-Q pretrain + learn-Q finetune (lives in radar/mae/G_Qlearn_seed)
    for seed in SEEDS:
        p = R4 / 'runs' / 'radar' / 'mae' / f'G_Qlearn_seed{seed}'
        p1[f'D_learnq_learnq/{seed}'] = load_eval(p, seed)

    # paired diffs vs D (eval OA), per-seed
    paired = {}
    for setting in ('A_fixedq_fixedq', 'B_learnq_fixedq', 'C_fixedq_learnq'):
        deltas = []
        for seed in SEEDS:
            d_oa = p1[f'D_learnq_learnq/{seed}'][0]
            s_oa = p1[f'{setting}/{seed}'][0]
            deltas.append(s_oa - d_oa)
        paired[setting] = {'per_seed': [round(float(x), 6) for x in deltas],
                           'mean': round(meanstd(deltas)[0], 6),
                           'std': round(meanstd(deltas)[1], 6)}

    out = {'p0': p0, 'p1': p1, 'p1_paired_vs_D_eval_oa': paired}
    (R1 / 'reports').mkdir(parents=True, exist_ok=True)
    (R1 / 'reports' / 'radar_eval_main.json').write_text(json.dumps(out, indent=2))
    _write_md(out)
    print('[done] radar_eval_main.json + tables_radar_eval.md')


def _write_md(out):
    L = ['# Revision-1: RADAR eval (test) re-aggregation\n']
    L.append('\nSelection is on val(=dev); these tables report **eval** (test) OA and Macro-F1.\n')

    L.append('\n## P0 (backbone x init) eval OA\n')
    L.append('| backbone | init | seed17 | seed42 | seed2026 | mean ± sample-std |\n|---|---|---|---|---|---|\n')
    for init in INITS:
        for bb in BACKBONES:
            if bb == 'G_corr' and init == 'denoise':
                continue
            row = [bb, init]
            for seed in SEEDS:
                v = out['p0'].get(f'{bb}/{init}/{seed}')
                row.append(f'{v[0]:.4f}' if v else 'n/a')
            oas = [out['p0'][f'{bb}/{init}/{seed}'][0] for seed in SEEDS
                   if out['p0'].get(f'{bb}/{init}/{seed}')]
            m, s = meanstd(oas)
            row.append(f'{m:.4f} ± {s:.4f}')
            L.append('| ' + ' | '.join(row) + ' |\n')

    L.append('\n## P0 (backbone x init) eval Macro-F1\n')
    L.append('| backbone | init | seed17 | seed42 | seed2026 | mean ± sample-std |\n|---|---|---|---|---|---|\n')
    for init in INITS:
        for bb in BACKBONES:
            if bb == 'G_corr' and init == 'denoise':
                continue
            row = [bb, init]
            for seed in SEEDS:
                v = out['p0'].get(f'{bb}/{init}/{seed}')
                row.append(f'{v[1]:.4f}' if v else 'n/a')
            f1s = [out['p0'][f'{bb}/{init}/{seed}'][1] for seed in SEEDS
                   if out['p0'].get(f'{bb}/{init}/{seed}')]
            m, s = meanstd(f1s)
            row.append(f'{m:.4f} ± {s:.4f}')
            L.append('| ' + ' | '.join(row) + ' |\n')

    L.append('\n## P1 (Q learning-phase 2x2) eval OA + Macro-F1\n')
    L.append('| setting | seed17 OA | seed42 OA | seed2026 OA | mean OA ± std | seed17 F1 | seed42 F1 | seed2026 F1 | mean F1 ± std |\n|---|---|---|---|---|---|---|---|---|\n')
    order = ('A_fixedq_fixedq', 'B_learnq_fixedq', 'C_fixedq_learnq', 'D_learnq_learnq')
    for setting in order:
        oas = [out['p1'][f'{setting}/{seed}'][0] for seed in SEEDS]
        f1s = [out['p1'][f'{setting}/{seed}'][1] for seed in SEEDS]
        mo, so = meanstd(oas); mf, sf = meanstd(f1s)
        L.append(f"| {setting} | {oas[0]:.4f} | {oas[1]:.4f} | {oas[2]:.4f} | {mo:.4f} ± {so:.4f} | "
                 f"{f1s[0]:.4f} | {f1s[1]:.4f} | {f1s[2]:.4f} | {mf:.4f} ± {sf:.4f} |\n")

    L.append('\n## P1 paired deltas vs D (eval OA)\n')
    for setting, d in out['p1_paired_vs_D_eval_oa'].items():
        L.append(f"- {setting}: per-seed {d['per_seed']}, mean {d['mean']:+.4f} ± {d['std']:.4f}\n")

    (R1 / 'reports' / 'tables_radar_eval.md').write_text('\n'.join(L))


if __name__ == '__main__':
    main()
