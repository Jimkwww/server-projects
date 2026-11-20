#!/usr/bin/env python
"""Round-2 part 1: preprocessing-order recheck on round-1 checkpoints.

Order A = raw IQ -> max_abs normalize -> perturb (reproduces round 1).
Order B = raw IQ -> perturb -> max_abs normalize.

Loads raw (un-normalized) IQ and applies max_abs manually so both orders share
one code path; order A must reproduce round-1 clean metrics exactly (checked).

Dense CFO grid: 0, +-0.0001, +-0.00025, +-0.0005, +-0.001, +-0.002, +-0.005.
Phase grid: round-1 fixed grid. Saves per-sample predictions (id/true/pred/snr)
for clean and CFO=+-0.001, +-0.002, +-0.005 under both orders.
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
from ssm_exp.model import Classifier, perturb
from ssm_exp.utils import grouped_metrics


def max_abs_norm(x):
    # Per-sample joint max |component| over I and Q, matching data.py __getitem__.
    m = x.abs().reshape(x.shape[0], -1).max(dim=1, keepdim=True).values  # [B,1]
    return x / m.clamp_min(1e-8).unsqueeze(-1)


@torch.no_grad()
def ordered_evaluate(model, loader, device, frequency=0.0, phase=0.0, order='A',
                     save_predictions=None):
    model.eval()
    ys, ps, zs, ids = [], [], [], []
    for x, y, z, i in loader:
        x = x.to(device)
        if order == 'A':
            x = max_abs_norm(x)
            if frequency != 0 or phase != 0:
                x = perturb(x, frequency, phase)
        else:  # order B
            if frequency != 0 or phase != 0:
                x = perturb(x, frequency, phase)
            x = max_abs_norm(x)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite logits')
        ys.extend(y.tolist()); ps.extend(logits.argmax(-1).cpu().tolist())
        zs.extend(z.tolist()); ids.extend(i.tolist())
    if not ys:
        raise ValueError('empty eval')
    ys = np.asarray(ys); ps = np.asarray(ps); zs = np.asarray(zs); ids = np.asarray(ids)
    result = grouped_metrics(ys, ps, zs, model.cfg['classes'])
    if save_predictions:
        np.savez_compressed(save_predictions, y=ys, pred=ps, snr=zs, ids=ids)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True)
    p.add_argument('--manifest', required=True)          # normalization='none'
    p.add_argument('--frequencies', required=True)        # comma-separated, no '0'
    p.add_argument('--phases', default='0.7853981634,1.5707963268,3.1415926536,4.7123889804')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--pred-dir', default='predictions_round2')
    p.add_argument('--out', default='reports/round2/recheck_cases.jsonl')
    a = p.parse_args()

    device = torch.device(a.device)
    m = read_manifest(a.manifest)
    raw_ds = IQDataset(m, 'val')
    loader = make_loader(raw_ds, a.batch_size, a.workers)

    freqs = [float(v) for v in a.frequencies.split(',')]
    phases = [float(v) for v in a.phases.split(',')]
    pred_dir = Path(a.pred_dir); pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = sorted(Path(a.runs).glob('*/best.pt'))
    print(f'{len(runs)} runs, freqs={freqs}, phases={phases}', flush=True)

    with out_path.open('a') as fh:
        for run_pt in runs:
            run = run_pt.parent
            ck = torch.load(run_pt, map_location='cpu', weights_only=False)
            model = Classifier(ck['config']['model']).to(device)
            model.load_state_dict(ck['model'])
            variant = ck['config']['model']['variant']
            seed = ck['config']['train']['seed']

            def emit(order, kind, value, result):
                fh.write(json.dumps({
                    'run': str(run), 'variant': variant, 'seed': seed,
                    'order': order, 'kind': kind, 'value': value,
                    'oa': result['overall']['oa'],
                    'macro_f1': result['overall']['macro_f1'],
                    'class_f1': result['overall']['class_f1'],
                    'per_snr_oa': {k: v['oa'] for k, v in result['snr'].items()},
                }, allow_nan=False) + '\n')
                fh.flush()

            # clean (identical for both orders; evaluate once, emit for both)
            clean = ordered_evaluate(model, loader, device, 0.0, 0.0, 'A',
                                     pred_dir / f'{run.name}_clean.npz')
            emit('A', 'clean', 0.0, clean)
            emit('B', 'clean', 0.0, clean)
            # sanity: order A clean must match round-1 best_val OA
            r1_oa = json.loads((run / 'best_val.json').read_text())['metrics']['overall']['oa']
            if abs(clean['overall']['oa'] - r1_oa) > 1e-9:
                raise AssertionError(f'{run.name}: order A clean {clean["overall"]["oa"]} != round1 {r1_oa}')

            for order in ('A', 'B'):
                for f in freqs:
                    save = pred_dir / f'{run.name}_{order}_cfo{f}.npz' if abs(f) in (0.001, 0.002, 0.005) else None
                    r = ordered_evaluate(model, loader, device, f, 0.0, order, save)
                    emit(order, 'cfo', f, r)
                for ph in phases:
                    r = ordered_evaluate(model, loader, device, 0.0, ph, order)
                    emit(order, 'phase', ph, r)
                print(f'{run.name} order={order} done', flush=True)

    print('ALL RECHECK DONE', flush=True)


if __name__ == '__main__':
    main()
