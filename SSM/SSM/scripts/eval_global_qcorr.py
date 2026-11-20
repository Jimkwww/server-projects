#!/usr/bin/env python
"""Unified order-B val evaluation for the G_Q_corr round (11 CFO conditions).

Grid: clean + +/-0.001, +/-0.002, +/-0.005 (main) + +/-0.01, +/-0.02 (extended diag).
Protocol: raw IQ -> perturb(CFO) -> max_abs normalize -> model (order B). Val split only
(NO test). For every condition the full metrics (OA/Macro-F1, per-class, per-SNR, low/high)
and per-sample predictions (y/pred/snr/ids) are saved with uniquely-named files.

Used for BOTH the 6 new runs and the 18 historical baselines so every number in the
8-model table comes from the identical grid + protocol. Historical G/G_corr/S1/S1_corr
checkpoints (round3_aug) lack the round-5 aux_mean/aux_std buffers; those two keys are the
only allowed missing keys on strict load (the rest must match exactly).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.engine import make_loader
from ssm_exp.model import Classifier, perturb, max_abs_norm

GRID = [0.0, 0.001, -0.001, 0.002, -0.002, 0.005, -0.005, 0.01, -0.01, 0.02, -0.02]
ALLOWED_MISSING = {'aux_mean', 'aux_std'}


def _fmt(v):
    s = f'{v:.4f}'.rstrip('0').rstrip('.')
    return s.replace('-', 'm').replace('.', 'p')


def core_metrics(y, p, classes):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    cm = np.bincount(y * classes + p, minlength=classes * classes).reshape(classes, classes)
    tp = np.diag(cm).astype(float)
    row = cm.sum(1).astype(float); col = cm.sum(0).astype(float)
    precision = np.divide(tp, col, out=np.zeros(classes), where=col > 0)
    recall = np.divide(tp, row, out=np.zeros(classes), where=row > 0)
    f1 = np.divide(2 * tp, row + col, out=np.zeros(classes), where=(row + col) > 0)
    return {'oa': float(tp.sum() / max(1, int(cm.sum()))), 'macro_f1': float(f1.mean()),
            'class_precision': [float(v) for v in precision],
            'class_recall': [float(v) for v in recall],
            'class_f1': [float(v) for v in f1]}


def full_metrics(y, p, snr, classes):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    snr = np.asarray(snr, dtype=float)

    def sub(mask):
        return None if not mask.any() else core_metrics(y[mask], p[mask], classes)

    per_snr = {str(float(z)): sub(snr == z) for z in np.unique(snr)}
    out = core_metrics(y, p, classes)
    low = sub(snr <= 0); high = sub(snr > 0)
    out['per_snr'] = per_snr
    out['low'] = None if low is None else {'oa': low['oa'], 'macro_f1': low['macro_f1']}
    out['high'] = None if high is None else {'oa': high['oa'], 'macro_f1': high['macro_f1']}
    return out


@torch.no_grad()
def ordered_eval(model, loader, device, frequency, save_predictions=None):
    model.eval()
    ys, ps, zs, ids = [], [], [], []
    for x, y, z, i in loader:
        x = x.to(device)
        if frequency != 0.0:
            x = perturb(x, frequency, 0.0)
        x = max_abs_norm(x)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite logits')
        ys.extend(y.tolist()); ps.extend(logits.argmax(-1).cpu().tolist())
        zs.extend(z.tolist()); ids.extend(i.tolist())
    if not ys:
        raise ValueError('empty eval')
    ys = np.asarray(ys, dtype=int); ps = np.asarray(ps, dtype=int)
    zs = np.asarray(zs, dtype=float); ids = np.asarray(ids)
    if save_predictions:
        np.savez_compressed(save_predictions, y=ys, pred=ps, snr=zs, ids=ids)
    return full_metrics(ys, ps, zs, model.cfg['classes'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True, help='comma-separated run dirs (each has best.pt)')
    p.add_argument('--manifest', required=True, help="raw (normalization='none') manifest")
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--pred-dir', default='predictions_global_qcorr')
    p.add_argument('--out', default='reports/global_qcorr/eval_cases.jsonl')
    a = p.parse_args()

    device = torch.device(a.device)
    m = read_manifest(a.manifest)
    if m['normalization'] != 'none':
        raise ValueError('order-B eval requires the raw manifest (normalization=none)')
    loader = make_loader(IQDataset(m, 'val'), a.batch_size, a.workers)

    pred_dir = Path(a.pred_dir); pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(s).resolve() for s in a.runs.split(',')]
    print(f'{len(run_dirs)} runs x {len(GRID)} conditions', flush=True)

    with out_path.open('a') as fh:
        for run in run_dirs:
            pt = run / 'best.pt'
            if not pt.exists():
                print(f'skip missing {pt}', flush=True)
                continue
            ck = torch.load(pt, map_location='cpu', weights_only=False)
            if ck['provenance'].get('smoke'):
                raise ValueError(f'{run}: do not evaluate smoke checkpoints')
            cfg = ck['config']['model']
            seed = ck['config']['train']['seed']
            model = Classifier(cfg, seed=seed).to(device)
            missing = set(model.state_dict().keys()) - set(ck['model'].keys())
            extra = set(ck['model'].keys()) - set(model.state_dict().keys())
            if (missing - ALLOWED_MISSING) or extra:
                raise RuntimeError(f'{run}: state_dict mismatch missing={sorted(missing - ALLOWED_MISSING)} '
                                   f'extra={sorted(extra)}')
            model.load_state_dict(ck['model'], strict=False)
            variant = cfg['variant']
            tag = f'{run.parent.name}_{run.name}'

            def emit(freq, cond_label, r):
                fh.write(json.dumps({'run': str(run), 'variant': variant, 'seed': seed,
                                     'kind': 'clean' if freq == 0.0 else 'cfo',
                                     'value': freq, 'condition': cond_label, 'metrics': r},
                                    allow_nan=False) + '\n')
                fh.flush()

            for f in GRID:
                lab = 'clean' if f == 0.0 else f'cfo{_fmt(f)}'
                r = ordered_eval(model, loader, device, f, pred_dir / f'{tag}_{lab}.npz')
                emit(f, lab, r)
            print(f'{tag} done', flush=True)

    print('ALL EVAL DONE', flush=True)


if __name__ == '__main__':
    main()
