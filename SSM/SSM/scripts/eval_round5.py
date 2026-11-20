#!/usr/bin/env python
"""Round-5 IQ: unified order-B val evaluation (clean + CFO grid + global phase).

Protocol: raw IQ -> perturb(CFO or global phase) -> max_abs normalize -> model.
Best checkpoint of each listed run, val split only (NO test). For every condition the
full metrics (overall OA/Macro-F1, per-class precision/recall/F1, per-SNR, low/high)
and per-sample predictions are saved with uniquely-named files (group_model_seed_condition).

Round-5 conditions (never used for checkpoint selection):
  clean   frequency=0 phase=0
  cfo     main grid 0, +/-0.001, +/-0.002, +/-0.005
  phase   global phase 0, pi/4, pi/2, pi, 3pi/2 (frequency=0)

New round-5 variants (R0/C0/Rb/Cb/Q_linear/Q_corr/R_corr) are built with their seed so
their architecture matches the training checkpoints; the checkpoint state_dict (params
+ aux_mean/aux_std buffers) overwrites everything, so init values are irrelevant here.
Baseline S1_corr (round3_aug) is evaluated here because it was not part of round-4 eval;
S1/G/G_corr are reused from reports/round4_structure/eval_cases.jsonl.
"""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.engine import make_loader
from ssm_exp.model import Classifier, perturb, max_abs_norm

PHASE_MAP = {
    '0': (0.0, 'p0'),
    'pi/4': (math.pi / 4, 'p_pi4'),
    'pi/2': (math.pi / 2, 'p_pi2'),
    'pi': (math.pi, 'p_pi'),
    '3pi/2': (3 * math.pi / 2, 'p_3pi2'),
}


def core_metrics(y, p, classes):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    cm = np.bincount(y * classes + p, minlength=classes * classes).reshape(classes, classes)
    tp = np.diag(cm).astype(float)
    row = cm.sum(1).astype(float)
    col = cm.sum(0).astype(float)
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
        return None if not mask.any() else core_metrics(y[mask], p[mask], classes)

    per_snr = {str(float(z)): sub(snr == z) for z in np.unique(snr)}
    out = core_metrics(y, p, classes)
    low = sub(snr <= 0); high = sub(snr > 0)
    out['per_snr'] = per_snr
    out['low'] = None if low is None else {'oa': low['oa'], 'macro_f1': low['macro_f1']}
    out['high'] = None if high is None else {'oa': high['oa'], 'macro_f1': high['macro_f1']}
    return out


@torch.no_grad()
def ordered_eval(model, loader, device, frequency, phase, save_predictions=None):
    model.eval()
    ys, ps, zs, ids = [], [], [], []
    for x, y, z, i in loader:
        x = x.to(device)
        if frequency != 0.0 or phase != 0.0:
            x = perturb(x, frequency, phase)
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


def _fmt(v):
    s = f'{v:.4f}'.rstrip('0').rstrip('.')
    return s.replace('-', 'm').replace('.', 'p')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True, help='comma-separated run dirs (each has best.pt)')
    p.add_argument('--manifest', required=True, help="raw (normalization='none') manifest")
    p.add_argument('--frequencies', default='0.001,0.002,0.005', help='main CFO magnitudes')
    p.add_argument('--phases', default='0,pi/4,pi/2,pi,3pi/2', help='global phase labels')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--pred-dir', default='predictions_round5_iq')
    p.add_argument('--out', default='reports/round5_iq/eval_cases.jsonl')
    a = p.parse_args()

    device = torch.device(a.device)
    m = read_manifest(a.manifest)
    if m['normalization'] != 'none':
        raise ValueError('round5 eval requires the raw manifest (normalization=none)')
    loader = make_loader(IQDataset(m, 'val'), a.batch_size, a.workers)

    main_mags = [float(v) for v in a.frequencies.split(',')]
    main_grid = sorted(set([0.0] + [v for mag in main_mags for v in (mag, -mag) if mag > 0]))
    phases = [(PHASE_MAP[k][0], PHASE_MAP[k][1]) for k in a.phases.split(',') if k.strip()]

    pred_dir = Path(a.pred_dir); pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(s).resolve() for s in a.runs.split(',')]
    print(f'{len(run_dirs)} runs, main={main_grid}, phases={[lbl for _, lbl in phases]}', flush=True)

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
            # round-5 variants require a seed to build their architecture; the checkpoint
            # state_dict (including aux_mean/aux_std buffers) overwrites all init values.
            model = Classifier(cfg, seed=seed).to(device)
            missing = set(model.state_dict().keys()) - set(ck['model'].keys())
            extra = set(ck['model'].keys()) - set(model.state_dict().keys())
            allowed = {'aux_mean', 'aux_std'}  # pre-round-5 baselines lack these buffers
            if (missing - allowed) or extra:
                raise RuntimeError(f'{run}: unexpected state_dict mismatch '
                                   f'missing={sorted(missing - allowed)} extra={sorted(extra)}')
            model.load_state_dict(ck['model'], strict=False)
            variant = cfg['variant']
            tag = f'{run.parent.name}_{run.name}'  # round5_iq_<variant>_seed<seed>

            def emit(kind, value, cond_label, r):
                fh.write(json.dumps({
                    'run': str(run), 'variant': variant, 'seed': seed,
                    'kind': kind, 'value': value, 'condition': cond_label, 'metrics': r,
                }, allow_nan=False) + '\n')
                fh.flush()

            # clean
            r = ordered_eval(model, loader, device, 0.0, 0.0, pred_dir / f'{tag}_clean.npz')
            emit('clean', 0.0, 'clean', r)
            # main CFO grid
            for f in main_grid:
                if f == 0.0:
                    continue
                lab = f'cfo{_fmt(f)}'
                r = ordered_eval(model, loader, device, f, 0.0, pred_dir / f'{tag}_{lab}.npz')
                emit('cfo', f, lab, r)
            # global phase
            for ph, lbl in phases:
                if ph == 0.0:
                    continue  # already covered by clean
                r = ordered_eval(model, loader, device, 0.0, ph, pred_dir / f'{tag}_{lbl}.npz')
                emit('phase', ph, lbl, r)
            print(f'{tag} done', flush=True)

    print('ALL ROUND5 EVAL DONE', flush=True)


if __name__ == '__main__':
    main()
