#!/usr/bin/env python
"""Round-5 IQ aggregation: per-seed main tables, CFO/phase curves, paired diffs,
per-SNR/class analysis, train stats, aux-branch diagnostics, benchmark.

Emits reports/round5_iq/*.md + *.json.

Data sources:
  runs/round5_iq/<variant>_seed<seed>/   (R0/C0/Rb/Cb/Q_linear/Q_corr/R_corr)
  runs/round3_aug/<variant>_seed<seed>/  (S1/S1_corr/G/G_corr baselines)
  reports/round5_iq/eval_cases.jsonl     (7 new + S1_corr, order-B)
  reports/round4_structure/eval_cases.jsonl (S1/G/G_corr, order-B, reused)
  reports/round5_iq/benchmark.json       (7 new + S1_corr)
  reports/round4_structure/benchmark.json (S1/G/G_corr, reused)

All standard deviations are SAMPLE std (ddof=1) over 3 seeds; the 3 seeds are NOT treated
as independent test samples. Paired diffs are per-seed, then mean +/- sample std.
OA 小数差 × 100 = 百分点(pp)；均值与标准差同步换算。
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NEW = ['R0', 'C0', 'Rb', 'Cb', 'Q_linear', 'Q_corr', 'R_corr']
BASELINES = ['S1', 'S1_corr', 'G', 'G_corr']
VARIANTS = BASELINES + NEW
SEEDS = [17, 42, 2026]
MAIN_MAGS = [0.001, 0.002, 0.005]
PHASES = [('p0', 0.0), ('p_pi4', math.pi / 4), ('p_pi2', math.pi / 2),
          ('p_pi', math.pi), ('p_3pi2', 3 * math.pi / 2)]
CLASS_NAMES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
               'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']


def run_dir(variant, seed):
    sub = 'round3_aug' if variant in BASELINES else 'round5_iq'
    return ROOT / 'runs' / sub / f'{variant}_seed{seed}'


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return statistics.mean(vals), statistics.stdev(vals)


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


def oa_mean_std(cases, variant, kind, value):
    return mean_std([(get_metric(cases, variant, s, kind, value) or {}).get('oa')
                     for s in SEEDS])


def fmt_ms(m, s):
    if m is None:
        return 'n/a'
    return f'{m:.4f} ± {s:.4f}' if s else f'{m:.4f}'


def fmt_pp(m, s):
    # OA 小数差 × 100 = 百分点(pp)；均值与标准差同步换算
    if m is None:
        return 'n/a'
    return f'{100*m:+.2f} ± {100*s:.2f}' if s else f'{100*m:+.2f}'


def load_best_val(variant, seed):
    d = run_dir(variant, seed) / 'best_val.json'
    if not d.exists():
        return None
    return json.loads(d.read_text())


def load_train_stats(variant, seed):
    h = run_dir(variant, seed) / 'history.jsonl'
    if not h.exists():
        return None
    rows = [json.loads(l) for l in h.read_text().splitlines() if l.strip()]
    if not rows:
        return None
    best_ep = max(rows, key=lambda r: r['val']['overall']['oa'])['epoch']
    last = rows[-1]
    rhos, gates, sat = [], [], []
    for r in rows:
        for diag in r.get('diagnostics', []):
            for lyr in diag.get('layers', []):
                if lyr and isinstance(lyr, dict):
                    if 'rho_mean' in lyr:
                        rhos.append(lyr['rho_mean'])
                    if 'gate_mean' in lyr:
                        gates.append(lyr['gate_mean'])
                    if 'gate_saturation' in lyr:
                        sat.append(lyr['gate_saturation'])
    return {
        'best_epoch': best_ep,
        'last_train_oa': last['train_oa'],
        'last_train_loss': last['train_loss'],
        'max_grad_norm': max(r['max_grad_norm_before_clip'] for r in rows),
        'rho_mean': statistics.mean(rhos) if rhos else None,
        'gate_mean': statistics.mean(gates) if gates else None,
        'gate_saturation_mean': statistics.mean(sat) if sat else None,
        'train_seconds_total': sum(r['train_seconds'] for r in rows),
    }


def load_aux_diag(variant, seed):
    """Learned aux-branch parameters from best.pt (B-line variants only)."""
    if variant not in ('Q_linear', 'Q_corr', 'R_corr'):
        return None
    pt = run_dir(variant, seed) / 'best.pt'
    if not pt.exists():
        return None
    ck = torch.load(pt, map_location='cpu', weights_only=False)
    sd = ck['model']
    out = {'aux_std': sd['aux_std'].tolist()}
    if variant == 'R_corr':
        rho = torch.sigmoid(sd['aux_branch.logit'])
        A_raw = sd['aux_branch.A_raw']
        anorm = torch.linalg.matrix_norm(A_raw, ord=2)
        out['rho'] = rho.tolist()
        out['A_raw_spectral'] = anorm.tolist()
        out['A_eff_spectral'] = (rho * anorm / anorm.clamp_min(1.0)).tolist()
        out['B_fro'] = torch.linalg.matrix_norm(sd['aux_branch.B'], ord='fro').tolist()
    else:
        alpha = F.softplus(sd['aux_branch.raw_alpha'])
        out['tau'] = (1.0 / alpha).tolist()
        out['theta'] = sd['aux_branch.theta'].tolist()
        b = sd['aux_branch.b']
        out['b_mag'] = b.abs().tolist()
        out['b_phase'] = b.angle().tolist()
    return out


def _fmt_arr(vals, nd=2):
    return '[' + ', '.join(f'{v:.{nd}f}' for v in vals) + ']'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default=str(ROOT / 'reports/round5_iq/eval_cases.jsonl'))
    ap.add_argument('--reuse-cases', default=str(ROOT / 'reports/round4_structure/eval_cases.jsonl'))
    ap.add_argument('--bench', default=str(ROOT / 'reports/round5_iq/benchmark.json'))
    ap.add_argument('--reuse-bench', default=str(ROOT / 'reports/round4_structure/benchmark.json'))
    ap.add_argument('--out-dir', default=str(ROOT / 'reports/round5_iq'))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    cases = load_cases(a.cases)
    for k, v in load_cases(a.reuse_cases).items():
        cases.setdefault(k, []).extend(v)

    # ---- 1. per-seed main (clean) --------------------------------------------------
    lines = ['# Round-5 per-seed clean val (best.pt, order-B clean)', '',
             '| variant | seed17 OA | seed42 OA | seed2026 OA | mean ± std OA | Macro-F1 mean ± std | best_epoch |',
             '|---|---|---|---|---|---|---|']
    main_json = {}
    for v in VARIANTS:
        oas, f1s, eps = [], [], []
        for s in SEEDS:
            bv = load_best_val(v, s)
            oas.append(bv['metrics']['overall']['oa'] if bv else None)
            f1s.append(bv['metrics']['overall']['macro_f1'] if bv else None)
            eps.append(bv['best_epoch'] if bv else None)
        om, os_ = mean_std(oas); fm, fs_ = mean_std(f1s)
        main_json[v] = {'per_seed_oa': oas, 'oa_mean': om, 'oa_sample_std': os_,
                        'macro_f1_mean': fm, 'macro_f1_sample_std': fs_, 'best_epoch': eps}
        lines.append(f'| {v} | ' + ' | '.join(f'{x:.4f}' if x is not None else 'n/a' for x in oas) +
                     f' | {fmt_ms(om, os_)} | {fmt_ms(fm, fs_)} | {eps} |')
    (out / 'per_seed_main.md').write_text('\n'.join(lines) + '\n')
    (out / 'per_seed_main.json').write_text(json.dumps(main_json, indent=2))

    # ---- 2. CFO curves -------------------------------------------------------------
    lines = ['# Round-5 CFO curves (order-B; absolute OA + relative drop vs clean)', '']
    lines += ['Sample std (ddof=1) over 3 seeds. Grid = 0, ±0.001, ±0.002, ±0.005.', '']
    lines.append('## Absolute OA (mean ± std)')
    hdr = '| frequency | ' + ' | '.join(VARIANTS) + ' |'
    lines.append(hdr); lines.append('|---|' + '---|' * len(VARIANTS))
    clean_mean = {v: oa_mean_std(cases, v, 'clean', 0.0)[0] for v in VARIANTS}
    grid = [0.0] + [s for m in MAIN_MAGS for s in (m, -m)]
    for f in grid:
        row = []
        for v in VARIANTS:
            m, s = oa_mean_std(cases, v, 'clean' if f == 0.0 else 'cfo', f)
            row.append(fmt_ms(m, s))
        lines.append(f'| {f:+.4f} | ' + ' | '.join(row) + ' |')
    lines.append('')
    lines.append('## Relative drop vs clean (pp, mean ± std; negative = improved)')
    lines.append(hdr); lines.append('|---|' + '---|' * len(VARIANTS))
    for f in grid:
        if f == 0.0:
            continue
        row = []
        for v in VARIANTS:
            drops = []
            for sd in SEEDS:
                cm = (get_metric(cases, v, sd, 'clean', 0.0) or {}).get('oa')
                fm = (get_metric(cases, v, sd, 'cfo', f) or {}).get('oa')
                if cm is not None and fm is not None:
                    drops.append(cm - fm)
            dm, ds = mean_std(drops)
            row.append(fmt_pp(dm, ds))
        lines.append(f'| {f:+.4f} | ' + ' | '.join(row) + ' |')
    (out / 'cfo_curves.md').write_text('\n'.join(lines) + '\n')

    # ---- 3. phase curves -----------------------------------------------------------
    lines = ['# Round-5 global phase curves (order-B; absolute OA + drop vs clean)', '',
             'Global phase = constant rotation angle on raw IQ before max_abs norm.', '']
    lines.append('## Absolute OA (mean ± std)')
    lines.append('| phase | ' + ' | '.join(VARIANTS) + ' |')
    lines.append('|---|' + '---|' * len(VARIANTS))
    for _, pv in PHASES:
        row = []
        for v in VARIANTS:
            m, s = oa_mean_std(cases, v, 'clean' if pv == 0.0 else 'phase', pv)
            row.append(fmt_ms(m, s))
        lines.append(f'| {pv:.4f} | ' + ' | '.join(row) + ' |')
    lines.append('')
    lines.append('## Drop vs clean (pp)')
    lines.append('| phase | ' + ' | '.join(VARIANTS) + ' |')
    lines.append('|---|' + '---|' * len(VARIANTS))
    for _, pv in PHASES:
        if pv == 0.0:
            continue
        row = []
        for v in VARIANTS:
            drops = []
            for sd in SEEDS:
                cm = (get_metric(cases, v, sd, 'clean', 0.0) or {}).get('oa')
                pm = (get_metric(cases, v, sd, 'phase', pv) or {}).get('oa')
                if cm is not None and pm is not None:
                    drops.append(cm - pm)
            dm, ds = mean_std(drops)
            row.append(fmt_pp(dm, ds))
        lines.append(f'| {pv:.4f} | ' + ' | '.join(row) + ' |')
    (out / 'phase_curves.md').write_text('\n'.join(lines) + '\n')

    # ---- 4. paired diffs (8 judgment comparisons) ----------------------------------
    comps = [('C0', 'R0'), ('Cb', 'Rb'), ('Rb', 'R0'), ('Cb', 'C0'),
             ('Q_linear', 'S1'), ('Q_corr', 'Q_linear'), ('Q_corr', 'S1_corr'),
             ('Q_corr', 'R_corr')]
    lines = ['# Round-5 paired diffs (per-seed, then mean ± sample std)', '',
             'Positive = first variant higher. Clean + main CFO + phase conditions.', '',
             '单位：百分点(pp)，即 OA 小数差 × 100（均值与标准差均换算）。', '']
    lines.append('| comparison | clean | CFO-0.001 | CFO-0.002 | CFO-0.005 | phase pi/2 | phase pi |')
    lines.append('|---|---|---|---|---|---|---|')
    diff_json = {}

    def diff_at(a, b, kind, value):
        ds = []
        for sd in SEEDS:
            am = (get_metric(cases, a, sd, kind, value) or {}).get('oa')
            bm = (get_metric(cases, b, sd, kind, value) or {}).get('oa')
            if am is not None and bm is not None:
                ds.append(am - bm)
        return mean_std(ds)

    for va, vb in comps:
        cells = []
        for kind, value in [('clean', 0.0), ('cfo', -0.001), ('cfo', -0.002), ('cfo', -0.005),
                            ('phase', math.pi / 2), ('phase', math.pi)]:
            m, s = diff_at(va, vb, kind, value)
            cells.append(fmt_pp(m, s))
            diff_json.setdefault(f'{va}-{vb}', {})[f'{kind}:{value}'] = {'mean': m, 'std': s}
        lines.append(f'| {va} − {vb} | ' + ' | '.join(cells) + ' |')
    (out / 'paired_diffs.md').write_text('\n'.join(lines) + '\n')
    (out / 'paired_diffs.json').write_text(json.dumps(diff_json, indent=2))

    # ---- 5. per-SNR low/high (clean) ----------------------------------------------
    lines = ['# Round-5 per-SNR groups (clean, order-B; mean ± std over seeds)', '']
    lines.append('| variant | low OA (SNR<=0) | high OA (SNR>0) |')
    lines.append('|---|---|---|')
    per_snr_json = {}
    for v in VARIANTS:
        lows, highs = [], []
        for sd in SEEDS:
            m = get_metric(cases, v, sd, 'clean', 0.0)
            if m:
                if m.get('low'): lows.append(m['low']['oa'])
                if m.get('high'): highs.append(m['high']['oa'])
        lm, ls = mean_std(lows); hm, hs = mean_std(highs)
        per_snr_json[v] = {'low': {'mean': lm, 'std': ls}, 'high': {'mean': hm, 'std': hs}}
        lines.append(f'| {v} | {fmt_ms(lm, ls)} | {fmt_ms(hm, hs)} |')
    (out / 'per_snr.md').write_text('\n'.join(lines) + '\n')
    (out / 'per_snr.json').write_text(json.dumps(per_snr_json, indent=2))

    # ---- 6. class analysis (clean, per-class F1 mean over seeds) ------------------
    lines = ['# Round-5 class analysis (clean; per-class F1, mean over 3 seeds)', '',
             '| class | ' + ' | '.join(VARIANTS) + ' |', '|---|' + '---|' * len(VARIANTS)]
    for ci in range(24):
        row = []
        for v in VARIANTS:
            vals = []
            for sd in SEEDS:
                m = get_metric(cases, v, sd, 'clean', 0.0)
                if m:
                    vals.append(m['class_f1'][ci])
            mm, _ = mean_std(vals)
            row.append(f'{mm:.4f}' if mm is not None else 'n/a')
        lines.append(f'| {CLASS_NAMES[ci]} | ' + ' | '.join(row) + ' |')
    (out / 'class_analysis.md').write_text('\n'.join(lines) + '\n')

    # ---- 7. train/state stats ------------------------------------------------------
    lines = ['# Round-5 training / state stats', '',
             '| variant | best_epoch | last_train_oa | last_train_loss | max_grad_norm | rho_mean | gate_mean | gate_sat | train_sec |',
             '|---|---|---|---|---|---|---|---|---|']
    stats_json = {}
    for v in VARIANTS:
        bests, los, ll, mg, rh, gm, gs, ts = [], [], [], [], [], [], [], []
        for sd in SEEDS:
            st = load_train_stats(v, sd)
            if st is None:
                continue
            bests.append(st['best_epoch']); los.append(st['last_train_oa'])
            ll.append(st['last_train_loss']); mg.append(st['max_grad_norm'])
            rh.append(st['rho_mean']); gm.append(st['gate_mean'])
            gs.append(st['gate_saturation_mean']); ts.append(st['train_seconds_total'])

        def clean(vals):
            return [x for x in vals if x is not None]

        def m(vals):
            vv = clean(vals)
            return None if not vv else statistics.mean(vv)

        def fmt(vals, spec):
            mm = m(vals)
            return 'n/a' if mm is None else f'{mm:{spec}}'

        stats_json[v] = {'best_epoch': bests, 'last_train_oa': los, 'last_train_loss': ll,
                         'max_grad_norm': mg, 'rho_mean': rh, 'gate_mean': gm,
                         'gate_saturation': gs, 'train_seconds': ts}
        lines.append(f'| {v} | {bests} | {fmt(los, ".4f")} | {fmt(ll, ".4f")} | '
                     f'{fmt(mg, ".3f")} | {fmt(rh, ".4f")} | {fmt(gm, ".4f")} | '
                     f'{fmt(gs, ".4f")} | {fmt(ts, ".1f")} |')
    (out / 'train_stats.md').write_text('\n'.join(lines) + '\n')
    (out / 'train_stats.json').write_text(json.dumps(stats_json, indent=2))

    # ---- 8. aux-branch diagnostics (B-line learned recurrence params) --------------
    diag_json = {}
    lines = ['# Round-5 aux-branch diagnostics (learned recurrence parameters from best.pt)', '',
             'q[m] = a·q[m-1] + b·s[m] over the full 1024-sample raw IQ; tau = 1/softplus(raw_alpha),',
             'theta is the per-channel rotation, |b| the input coupling. R_corr: 8 groups of 2-dim',
             'real state with per-step decay rho = sigmoid(logit) ∈ (0,1) (spectral norm < 1 enforced).', '']

    # complex recurrence: Q_linear / Q_corr
    lines.append('## Complex recurrence (Q_linear / Q_corr)')
    lines.append('| variant | seed | tau (min/mean/max) | \\|b\\| (min/mean/max) | theta (deg) |')
    lines.append('|---|---|---|---|---|')
    for v in ('Q_linear', 'Q_corr'):
        for sd in SEEDS:
            d = load_aux_diag(v, sd)
            if d is None:
                continue
            diag_json.setdefault(v, {})[str(sd)] = d
            tau = d['tau']; bm = d['b_mag']
            th = [round(math.degrees(t), 0) for t in d['theta']]
            lines.append(f'| {v} | {sd} | {min(tau):.2f} / {statistics.mean(tau):.2f} / {max(tau):.2f} | '
                         f'{min(bm):.3f} / {statistics.mean(bm):.3f} / {max(bm):.3f} | {th} |')

    # real recurrence: R_corr
    lines.append('')
    lines.append('## Real recurrence (R_corr)')
    lines.append('| variant | seed | rho (min/mean/max) | A_raw ‖·‖₂ (min/mean/max) | A_eff ‖·‖₂ (max) | B ‖·‖_F (min/mean/max) |')
    lines.append('|---|---|---|---|---|---|')
    for sd in SEEDS:
        d = load_aux_diag('R_corr', sd)
        if d is None:
            continue
        diag_json.setdefault('R_corr', {})[str(sd)] = d
        rho = d['rho']; an = d['A_raw_spectral']; ae = d['A_eff_spectral']; bf = d['B_fro']
        lines.append(f'| R_corr | {sd} | {min(rho):.4f} / {statistics.mean(rho):.4f} / {max(rho):.4f} | '
                     f'{min(an):.3f} / {statistics.mean(an):.3f} / {max(an):.3f} | {max(ae):.4f} | '
                     f'{min(bf):.3f} / {statistics.mean(bf):.3f} / {max(bf):.3f} |')
    (out / 'diagnostics.md').write_text('\n'.join(lines) + '\n')
    (out / 'diagnostics.json').write_text(json.dumps(diag_json, indent=2))

    # ---- 9. benchmark --------------------------------------------------------------
    bench = {}
    for bp in (a.bench, a.reuse_bench):
        p = Path(bp)
        if p.exists():
            for b in json.loads(p.read_text()):
                bench[b['variant']] = b
    if bench:
        lines = ['# Round-5 efficiency benchmark (eval mode, CUDA warmup+sync; L=1024)', '',
                 'Real-scalar params (complex ×2). Eager PyTorch O(N log N) main-path scan, not fused;',
                 'aux branch runs on the continuous 1024-sample IQ (ComplexRecurBranch via parallel_scan;',
                 'RealRecurBranch via an explicit 1024-step loop).', '']
        lines.append('| variant | params | batch1 latency (ms) | batch64 throughput (sps) | infer peak mem (MiB) |')
        lines.append('|---|---|---|---|---|')
        for v in VARIANTS:
            b = bench.get(v)
            if b is None:
                lines.append(f'| {v} | n/a | n/a | n/a | n/a |')
            else:
                lines.append(f'| {v} | {b["params"]} | {b["latency_b1_ms"]:.3f} | {b["throughput_b64_sps"]:.1f} | '
                             f'{b["inference_peak_mem_mb"]:.1f} |')
        (out / 'benchmark.md').write_text('\n'.join(lines) + '\n')

    print('aggregation written to', out)


if __name__ == '__main__':
    main()
