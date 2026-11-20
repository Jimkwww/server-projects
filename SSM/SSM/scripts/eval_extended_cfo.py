#!/usr/bin/env python
"""Round-3 review: unified order-B extended CFO evaluation on existing checkpoints.

Protocol (identical to recheck_order.py order B):
  raw IQ -> perturb(CFO) -> max_abs normalize -> recompute physical desc -> model

Evaluates best.pt of every listed run on the val split at clean + a full CFO grid
(original dense grid + extended points). Saves per-point full metrics (overall
OA/F1, per-class precision/recall/F1, per-SNR OA/F1, low/high) and per-sample
predictions (id/label/pred/snr) for every point.

This is a read-only extension: no training, no model changes, no test-set use.
Outputs go to fresh directories so existing results are preserved.
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

CLASS_NAMES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
               'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']


def core_metrics(y, p, classes):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    cm = np.bincount(y * classes + p, minlength=classes * classes).reshape(classes, classes)
    tp = np.diag(cm).astype(float)
    row = cm.sum(1).astype(float)   # true count per class
    col = cm.sum(0).astype(float)   # predicted count per class
    precision = np.divide(tp, col, out=np.zeros(classes), where=col > 0)
    recall = np.divide(tp, row, out=np.zeros(classes), where=row > 0)
    f1 = np.divide(2 * tp, row + col, out=np.zeros(classes), where=(row + col) > 0)
    return {
        'oa': float(tp.sum() / max(1, int(cm.sum()))),
        'macro_f1': float(f1.mean()),
        'class_precision': [float(v) for v in precision],
        'class_recall': [float(v) for v in recall],
        'class_f1': [float(v) for v in f1],
    }


def full_metrics(y, p, snr, classes):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    snr = np.asarray(snr, dtype=float)

    def sub(mask):
        if not mask.any():
            return None
        return core_metrics(y[mask], p[mask], classes)

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
    p.add_argument('--runs', required=True, help='comma-separated run dirs (each contains best.pt)')
    p.add_argument('--manifest', required=True, help="raw (normalization='none') manifest")
    p.add_argument('--frequencies', required=True, help='comma-separated positive CFO magnitudes')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--split', choices=['val', 'test'], default='val',
                   help='test is the independent final split; never used for checkpoint selection')
    p.add_argument('--pred-dir', default='predictions_round3_review')
    p.add_argument('--out', default='reports/round3_review/extended_cases.jsonl')
    a = p.parse_args()

    device = torch.device(a.device)
    m = read_manifest(a.manifest)
    if m['normalization'] != 'none':
        raise ValueError('extended eval requires the raw manifest (normalization=none)')
    loader = make_loader(IQDataset(m, a.split), a.batch_size, a.workers)

    mags = [float(v) for v in a.frequencies.split(',')]
    grid = sorted(set([0.0] + [v for mag in mags for v in (mag, -mag) if mag > 0]))
    pred_dir = Path(a.pred_dir); pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(s).resolve() for s in a.runs.split(',')]
    print(f'{len(run_dirs)} runs, grid={grid}', flush=True)

    with out_path.open('a') as fh:
        for run in run_dirs:
            pt = run / 'best.pt'
            if not pt.exists():
                print(f'skip missing {pt}', flush=True)
                continue
            ck = torch.load(pt, map_location='cpu', weights_only=False)
            if ck['provenance'].get('smoke'):
                raise ValueError(f'{run}: do not evaluate smoke checkpoints')
            model = Classifier(ck['config']['model']).to(device)
            model.load_state_dict(ck['model'])
            variant = ck['config']['model']['variant']
            seed = ck['config']['train']['seed']
            # run.name alone collides across runs/clean vs runs/round3_aug; include parent.
            tag = f'{run.parent.name}_{run.name}'

            def emit(kind, f, r):
                fh.write(json.dumps({
                    'run': str(run), 'variant': variant, 'seed': seed,
                    'kind': kind, 'value': f, 'metrics': r,
                }, allow_nan=False) + '\n')
                fh.flush()

            # clean (f=0)
            clean = ordered_eval(model, loader, device, 0.0,
                                 pred_dir / f'{tag}_clean.npz')
            emit('clean', 0.0, clean)
            for f in grid:
                if f == 0.0:
                    continue
                r = ordered_eval(model, loader, device, f,
                                 pred_dir / f'{tag}_B_cfo{f}.npz')
                emit('cfo', f, r)
            print(f'{tag} done', flush=True)

    print('ALL EXTENDED EVAL DONE', flush=True)


if __name__ == '__main__':
    main()
