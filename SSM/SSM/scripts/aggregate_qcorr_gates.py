#!/usr/bin/env python
"""Aggregate the Q_corr physical-gate round: 10-model main table, per-seed paired diffs,
CFO curves, per-SNR, per-class, efficiency, and a prediction-recomputation cross-check.

Reads reports/qcorr_gates/eval_cases.jsonl (18 new runs) + reports/global_qcorr/eval_cases.jsonl
(4 historical baselines, same 11-condition order-B grid), the runs' model_info.json (param
counts), and reports/qcorr_gates/benchmark.json. OA differences are percentage points (pp);
means use sample std (ddof=1).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
REPORT = ROOT / 'reports' / 'qcorr_gates'
NEW_VARIANTS = ['Q_corr_gate_c', 'Q_corr_gate_d', 'Q_corr_gate_cd',
                'Q_corr_token_c', 'Q_corr_token_d', 'Q_corr_token_cd']
VARIANTS = ['Q_corr'] + NEW_VARIANTS + ['G', 'G_corr', 'G_Q_corr_frozen']
NEW = set(NEW_VARIANTS)
CONDITIONS = ['clean', 'cfo0p001', 'cfom0p001', 'cfo0p002', 'cfom0p002',
              'cfo0p005', 'cfom0p005', 'cfo0p01', 'cfom0p01', 'cfo0p02', 'cfom0p02']
MAIN = ['clean', 'cfo0p005', 'cfom0p005']
N_CLASSES = 24

RUN_DIRS = {
    'Q_corr': 'runs/round5_iq', 'G': 'runs/round3_aug', 'G_corr': 'runs/round3_aug',
    'G_Q_corr_frozen': 'runs/global_qcorr',
}
for v in NEW_VARIANTS:
    RUN_DIRS[v] = 'runs/qcorr_gates'
PRED_DIRS = {'Q_corr': 'predictions_global_qcorr', 'G': 'predictions_global_qcorr',
             'G_corr': 'predictions_global_qcorr', 'G_Q_corr_frozen': 'predictions_global_qcorr'}
for v in NEW_VARIANTS:
    PRED_DIRS[v] = 'predictions_qcorr_gates'


def load_eval(*paths):
    data = defaultdict(dict)   # (variant, seed) -> condition -> metrics
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            data[(r['variant'], r['seed'])][r['condition']] = r['metrics']
    return data


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def pp(delta):
    return 100.0 * delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval', default=str(REPORT / 'eval_cases.jsonl'))
    p.add_argument('--baseline-eval', default=str(ROOT / 'reports' / 'global_qcorr' / 'eval_cases.jsonl'))
    p.add_argument('--bench', default=str(REPORT / 'benchmark.json'))
    p.add_argument('--out', default=str(REPORT))
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    data = load_eval(a.eval, a.baseline_eval)
    seeds = sorted({s for (_, s) in data})
    assert seeds == [17, 42, 2026], seeds

    params = {}
    for v in VARIANTS:
        info = json.loads((ROOT / RUN_DIRS[v] / f'{v}_seed17' / 'model_info.json').read_text())
        params[v] = info['real_scalar_parameters']

    def oa(v, s, cond):
        return data.get((v, s), {}).get(cond, {}).get('oa')

    def f1(v, s, cond):
        return data.get((v, s), {}).get(cond, {}).get('macro_f1')

    def class_f1(v, s, cond):
        return data.get((v, s), {}).get(cond, {}).get('class_f1')

    # ---- Table 1: 10-model main table ------------------------------------------------
    main_rows = []
    for v in VARIANTS:
        row = {'variant': v, 'params': params[v], 'source': 'new' if v in NEW else 'historical'}
        for cond in MAIN:
            oas = [oa(v, s, cond) for s in seeds]
            m, sd = mean_std(oas)
            row[f'{cond}_oa_mean'] = m
            row[f'{cond}_oa_std'] = sd
        f1s = [f1(v, s, 'clean') for s in seeds]
        m, sd = mean_std(f1s)
        row['clean_macro_f1_mean'] = m
        row['clean_macro_f1_std'] = sd
        for s in seeds:
            row[f'seed{s}_clean_oa'] = oa(v, s, 'clean')
        main_rows.append(row)

    # ---- Table 2: per-seed paired diffs ----------------------------------------------
    pairs = []
    for v in NEW_VARIANTS:
        pairs.append((v, 'Q_corr'))                      # each new variant vs Q_corr
    pairs += [('Q_corr_gate_c', 'Q_corr_token_c'),
              ('Q_corr_gate_d', 'Q_corr_token_d'),
              ('Q_corr_gate_cd', 'Q_corr_token_cd')]      # gate vs token (same information)
    pairs += [('Q_corr_gate_cd', 'G'), ('Q_corr_gate_cd', 'G_corr'),
              ('Q_corr_gate_cd', 'G_Q_corr_frozen')]      # cross-backbone
    diff_rows = []
    for a_, b_ in pairs:
        for cond in MAIN:
            row = {'pair': f'{a_}-{b_}', 'condition': cond}
            for s in seeds:
                row[f'seed{s}'] = pp(oa(a_, s, cond) - oa(b_, s, cond))
            vals = [pp(oa(a_, s, cond) - oa(b_, s, cond)) for s in seeds]
            row['mean_pp'], row['std_pp'] = mean_std(vals)
            diff_rows.append(row)

    # ---- Table 3: full CFO curves (absolute OA, sign kept) ---------------------------
    curve_rows = []
    for v in VARIANTS:
        for cond in CONDITIONS:
            oas = [oa(v, s, cond) for s in seeds]
            m, sd = mean_std(oas)
            freq = 0.0 if cond == 'clean' else float(cond[3:].replace('m', '-').replace('p', '.'))
            curve_rows.append({'variant': v, 'condition': cond, 'frequency': freq,
                               'oa_mean': m, 'oa_std': sd})

    # ---- Table 4: per-SNR (low <=0 / high >0) on clean -------------------------------
    snr_rows = []
    for v in VARIANTS:
        lows = [data.get((v, s), {}).get('clean', {}).get('low') for s in seeds]
        highs = [data.get((v, s), {}).get('clean', {}).get('high') for s in seeds]
        low_oa = mean_std([l['oa'] if l else None for l in lows])
        high_oa = mean_std([h['oa'] if h else None for h in highs])
        low_f1 = mean_std([l['macro_f1'] if l else None for l in lows])
        high_f1 = mean_std([h['macro_f1'] if h else None for h in highs])
        snr_rows.append({'variant': v, 'low_oa_mean': low_oa[0], 'low_oa_std': low_oa[1],
                         'high_oa_mean': high_oa[0], 'high_oa_std': high_oa[1],
                         'low_f1_mean': low_f1[0], 'high_f1_mean': high_f1[0]})

    # ---- Table 5: per-class Macro-F1 (3-seed mean) on clean + +/-0.005 ----------------
    class_rows = []
    for v in VARIANTS:
        for cond in MAIN:
            for c in range(N_CLASSES):
                vals = []
                for s in seeds:
                    cf = class_f1(v, s, cond)
                    vals.append(cf[c] if cf else None)
                m, sd = mean_std(vals)
                class_rows.append({'variant': v, 'condition': cond, 'class': c,
                                   'f1_mean': m, 'f1_std': sd})

    # class-level delta (pp of F1) for gate_cd vs Q_corr (the full two-gate model)
    class_delta_rows = []
    for cond in MAIN:
        for c in range(N_CLASSES):
            dv = []
            for s in seeds:
                a_ = class_f1('Q_corr_gate_cd', s, cond)
                b_ = class_f1('Q_corr', s, cond)
                if a_ is not None and b_ is not None:
                    dv.append(100.0 * (a_[c] - b_[c]))
            m, sd = mean_std(dv)
            class_delta_rows.append({'condition': cond, 'class': c, 'gate_cd_minus_Q_corr_pp': m,
                                     'std_pp': sd})

    # ---- efficiency ------------------------------------------------------------------
    bench = json.loads(open(a.bench).read()) if Path(a.bench).exists() else []

    # ---- prediction recomputation cross-check ----------------------------------------
    verify = []
    for v in VARIANTS:
        for s in seeds:
            tag = f'{RUN_DIRS[v].split("/")[-1]}_{v}_seed{s}'
            pred_dir = Path(PRED_DIRS[v])
            for cond in MAIN:
                f = pred_dir / f'{tag}_{cond}.npz'
                if not f.exists():
                    verify.append({'tag': tag, 'cond': cond, 'status': 'missing'}); continue
                d = np.load(f)
                y, pr = d['y'].astype(int), d['pred'].astype(int)
                cm = np.bincount(y * N_CLASSES + pr, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
                tp = np.diag(cm).astype(float)
                oa_re = float(tp.sum() / max(1, cm.sum()))
                f1_re = float((np.divide(2 * tp, cm.sum(0) + cm.sum(1),
                                         out=np.zeros(N_CLASSES), where=(cm.sum(0) + cm.sum(1)) > 0)).mean())
                oa_table = oa(v, s, cond)
                verify.append({'tag': tag, 'cond': cond, 'oa_from_pred': oa_re,
                               'oa_from_table': oa_table,
                               'macro_f1_from_pred': f1_re,
                               'oa_diff': oa_re - oa_table})

    # ---- write machine-readable ------------------------------------------------------
    def csv(path, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, 'w', newline='') as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    csv(out / 'main_table.csv', main_rows)
    csv(out / 'paired_diffs.csv', diff_rows)
    csv(out / 'cfo_curves.csv', curve_rows)
    csv(out / 'per_snr.csv', snr_rows)
    csv(out / 'per_class_f1.csv', class_rows)
    csv(out / 'class_delta_gate_cd_vs_Q_corr.csv', class_delta_rows)
    if bench:
        csv(out / 'efficiency.csv', bench)
    json.dump({'main': main_rows, 'paired': diff_rows, 'cfo_curves': curve_rows,
               'per_snr': snr_rows, 'per_class_f1': class_rows,
               'class_delta_gate_cd_vs_Q_corr': class_delta_rows,
               'benchmark': bench, 'verification': verify},
              open(out / 'aggregate.json', 'w'), indent=2)

    print(json.dumps({'n_main_rows': len(main_rows), 'n_diff_rows': len(diff_rows),
                      'n_curve_rows': len(curve_rows), 'n_class_rows': len(class_rows),
                      'max_oa_diff_recompute': max((abs(x['oa_diff']) for x in verify if 'oa_diff' in x), default=0.0)},
                     indent=2))


if __name__ == '__main__':
    main()
