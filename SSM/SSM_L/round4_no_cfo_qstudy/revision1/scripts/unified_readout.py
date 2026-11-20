"""Revision-1 Section 3: unified frozen readout (StandardScaler + L2 LR C=1 lbfgs).

Objects (all encoder_final.pt, no finetuned heads):
  1. G_Qlearn_fixedq_mae  (Global + Q, fixed-Q MAE pretrain; == P1 A/C shared encoder)
  2. G_Qlearn_learnq_mae  (Global + Q, learn-Q MAE pretrain; == P1 B/D shared encoder)
  3. G_corr_mae           (Global + raw-IQ complex-correlation fusion, MAE)
  4. G_mae                (plain Global reference, MAE)

Tasks: rml (train/val, test sealed) + radar (train/dev/eval, main = eval).
Radar scaler fit on train; report dev + eval; main = eval.

Usage:
  python unified_readout.py --cache-only --device cuda:0
  python unified_readout.py --readout-only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_r1 import (load_encoder, features_for_encoder, lr_readout, sd_hash,
                       SEEDS, task_cfg, task_splits, R1)
from ssm_exp.data import IQDataset, read_manifest

ENCODERS = {
    'G_Qlearn_fixedq_mae': ('G_Qlearn', lambda s: Path(
        'round4_no_cfo_qstudy/runs/pretrain/G_Qlearn_fixedq_mae_seed%d/encoder_final.pt' % s)),
    'G_Qlearn_learnq_mae': ('G_Qlearn', lambda s: Path(
        'round2_overnight/runs/pretrain/G_Qlearn_mae_seed%d/encoder_final.pt' % s)),
    'G_corr_mae': ('G_corr', lambda s: Path(
        'round4_no_cfo_qstudy/runs/pretrain/G_corr_mae_seed%d/encoder_final.pt' % s)),
    'G_mae': ('G', lambda s: Path('runs/pretrain/G_mae_seed%d/encoder_final.pt' % s)),
}
ROOT = Path(__file__).resolve().parent.parent.parent.parent
FEAT = R1 / 'runs' / 'features'
PRED = R1 / 'runs' / 'predictions'
BATCH = 128
SMOKE_N = 512


def feat_path(ekey, task, seed):
    return FEAT / f'{ekey}__{task}_seed{seed}.pt'


def cache_features(ekey, task, seed, device, smoke=False):
    out = feat_path(ekey, task, seed)
    if out.exists():
        return out, 'skip'
    bb, pathfn = ENCODERS[ekey]
    ck = pathfn(seed)
    if not ck.exists():
        return out, f'missing:{ck}'
    cfg = task_cfg(task)
    enc, cfg_used, ckpt = load_encoder(bb, ck, device, cfg=cfg)
    manifest, splits = task_splits(task)
    m = read_manifest(str(manifest))
    payload = {'encoder_sha256': sd_hash(ckpt['encoder']), 'ckpt': str(ck.resolve())}
    for split in splits:
        ds = IQDataset(m, split)
        if smoke:
            ds.ids = ds.ids[:min(SMOKE_N, len(ds))]
        feats, ys, zs, ids = features_for_encoder(enc, ds, BATCH, device)
        payload[split] = {'feats': feats, 'y': ys, 'snr': zs, 'idx': ids}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out, 'ok'


def readout_one(ekey, task, seed):
    payload = torch.load(feat_path(ekey, task, seed), map_location='cpu', weights_only=False)
    Xtr = payload['train']['feats'].numpy().astype(np.float32)
    ytr = payload['train']['y'].numpy().astype(np.int64)
    out = {'encoder_sha256': payload['encoder_sha256'], 'ckpt': payload['ckpt']}
    for split in ('val', 'test'):
        if split not in payload:
            continue
        Xva = payload[split]['feats'].numpy().astype(np.float32)
        yva = payload[split]['y'].numpy().astype(np.int64)
        oa, f1, n_iter, conv = lr_readout(Xtr, ytr, Xva, yva)
        out[split] = {'oa': oa, 'macro_f1': f1, 'n_iter': n_iter, 'converged': conv}
    return out


def save_predictions(ekey, task, seed, device=None):
    """Re-fit LR once and write per-sample predictions/labels/snr/ids for the report split."""
    payload = torch.load(feat_path(ekey, task, seed), map_location='cpu', weights_only=False)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    Xtr = payload['train']['feats'].numpy().astype(np.float32)
    ytr = payload['train']['y'].numpy().astype(np.int64)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs').fit(scaler.transform(Xtr), ytr)
    # main report split: RML -> val; radar -> eval ('test')
    main_split = 'val' if task == 'rml' else 'test'
    for split in ('val', 'test'):
        if split not in payload:
            continue
        p = payload[split]
        X = scaler.transform(p['feats'].numpy().astype(np.float32))
        pred = clf.predict(X)
        np.savez_compressed(PRED / f'{ekey}__{task}_seed{seed}__{split}_pred.npz',
                            pred=pred.astype(np.int64), y=p['y'].numpy().astype(np.int64),
                            snr=p['snr'].numpy().astype(np.float32),
                            idx=p['idx'].numpy().astype(np.int64))
    return main_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--readout-only', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='limit each split to a few samples')
    ap.add_argument('--tasks', nargs='+', default=['rml', 'radar'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--encoders', nargs='+', default=list(ENCODERS))
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')

    FEAT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)

    if not a.readout_only:
        for task in a.tasks:
            for ekey in a.encoders:
                for s in a.seeds:
                    out, st = cache_features(ekey, task, s, device, smoke=a.smoke)
                    print(f'[cache] {ekey} {task} {s}: {st}', flush=True)

    results = {}
    for task in a.tasks:
        for ekey in a.encoders:
            for s in a.seeds:
                key = f'{ekey}__{task}_seed{s}'
                fp = feat_path(ekey, task, s)
                if not fp.exists():
                    results[key] = {'missing': True}
                    continue
                r = readout_one(ekey, task, s)
                if not a.cache_only:
                    r['main_split'] = save_predictions(ekey, task, s)
                results[key] = r
    (R1 / 'reports').mkdir(parents=True, exist_ok=True)
    (R1 / 'reports' / 'unified_probe_raw.json').write_text(json.dumps(results, indent=2))

    _write_csv(results, a.tasks)
    _write_md(results, a.tasks)
    print('[done] unified_probe.csv / unified_probe.md / unified_probe_raw.json')


def _write_csv(results, tasks):
    import csv
    rows = []
    for task in tasks:
        for ekey in ENCODERS:
            for s in SEEDS:
                key = f'{ekey}__{task}_seed{s}'
                r = results.get(key, {})
                if r.get('missing'):
                    rows.append([key, '', '', '', '', '', 'missing'])
                    continue
                for split in ('val', 'test'):
                    m = r.get(split)
                    if m is None:
                        continue
                    rows.append([key, task, ekey, s, split, f"{m['oa']:.6f}",
                                 f"{m['macro_f1']:.6f}", int(m['converged']),
                                 r.get('encoder_sha256', '')[:16], r.get('main_split', ''),
                                 r.get('ckpt', '')])
    with (R1 / 'reports' / 'unified_probe.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'task', 'encoder', 'seed', 'split', 'oa', 'macro_f1', 'converged',
                    'encoder_sha256_prefix', 'main_split', 'ckpt'])
        w.writerows(rows)


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = float(np.mean(vals)); s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s


def _write_md(results, tasks):
    lines = ['# Revision-1 Section 3: unified frozen readout\n',
             'Protocol: frozen encoder_final.pt -> mean-pool -> StandardScaler(fit train) + '
             'L2 LogisticRegression(C=1, lbfgs, max_iter=2000). No eval-based hyperparameter '
             'selection. RML main = val (test sealed); radar main = eval (dev used only for '
             'the side table).\n']
    for task in tasks:
        main = 'val' if task == 'rml' else 'test'
        lines.append(f'\n## {task.upper()} — main split = {main}\n')
        lines.append('| encoder | seed17 | seed42 | seed2026 | mean ± sample-std | macro-F1 mean |\n')
        lines.append('|---|---|---|---|---|---|\n')
        per_encoder = {}
        for ekey in ENCODERS:
            vals = []
            f1s = []
            for s in SEEDS:
                r = results.get(f'{ekey}__{task}_seed{s}', {})
                m = r.get(main)
                vals.append(m['oa'] if m else None)
                f1s.append(m['macro_f1'] if m else None)
            ms = _mean_std(vals)
            f1m = _mean_std(f1s)
            per_encoder[ekey] = vals
            if ms is None:
                lines.append(f'| {ekey} | - | - | - | missing | - |\n')
                continue
            lines.append(f"| {ekey} | " + ' | '.join(f'{v:.4f}' if v is not None else '-' for v in vals)
                         + f" | {ms[0]:.4f} ± {ms[1]:.4f} | {f1m[0]:.4f} |\n")
        # side split (dev for radar; nothing extra for rml)
        if task == 'radar':
            lines.append('\nSide table (dev, model selection split, NOT main):\n')
            lines.append('| encoder | seed17 | seed42 | seed2026 | mean ± sample-std |\n')
            lines.append('|---|---|---|---|---|\n')
            for ekey in ENCODERS:
                vals = []
                for s in SEEDS:
                    r = results.get(f'{ekey}__{task}_seed{s}', {})
                    m = r.get('val')
                    vals.append(m['oa'] if m else None)
                ms = _mean_std(vals)
                lines.append(f'| {ekey} | ' + ' | '.join(f'{v:.4f}' if v is not None else '-' for v in vals)
                             + f" | {ms[0]:.4f} ± {ms[1]:.4f} |\n" if ms else f'| {ekey} | missing |\n')
        # paired diffs (per-seed): fixed-Q vs learn-Q; G_corr vs G
        def paired(a, b):
            av = per_encoder.get(a, []); bv = per_encoder.get(b, [])
            if any(v is None for v in av + bv):
                return None
            return [av[i] - bv[i] for i in range(3)]
        lines.append('\nPaired per-seed deltas (fixed-Q − learn-Q):\n')
        d = paired('G_Qlearn_fixedq_mae', 'G_Qlearn_learnq_mae')
        if d:
            ms = _mean_std(d)
            lines.append(f"- G_Qlearn fixed-Q − learn-Q: {['%.4f' % x for x in d]}, "
                         f"mean {ms[0]:+.4f} ± {ms[1]:.4f}\n")
        d = paired('G_corr_mae', 'G_mae')
        lines.append('\nPaired per-seed deltas (G_corr − G):\n')
        if d:
            ms = _mean_std(d)
            lines.append(f"- G_corr − G: {['%.4f' % x for x in d]}, mean {ms[0]:+.4f} ± {ms[1]:.4f}\n")
    (R1 / 'reports' / 'unified_probe.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
