#!/usr/bin/env python
"""Mechanism diagnosis for the Q_corr physical-gate round.

For each of the 6 new variants x 3 seeds (best.pt), on clean / +/-0.005 (order B):
  - per-layer learned gate weights (write_w for _gate_c/_cd, retention_v for _gate_d/_cd):
    L2 norm, mean, std, max |w| / |v|.
  - actual gate modulation (gate variants): RMS shift of the write gate g and the retention
    factor rho induced by the scalar modulation (paired on/off pass over the val set).
  - ablation: OA with the gate modulation zeroed (write_w/retention_v/token_proj -> 0, which
    reduces the model to Q_corr) vs OA with it active, in percentage points (pp).

Eval-mode only; nothing is written back to any checkpoint. Frozen scalar/aux stats never
change. This is the "correction magnitude + g/rho before/after zeroing" section.
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

GATE_VARIANTS = ('Q_corr_gate_c', 'Q_corr_gate_d', 'Q_corr_gate_cd')
TOKEN_VARIANTS = ('Q_corr_token_c', 'Q_corr_token_d', 'Q_corr_token_cd')
ALLOWED_MISSING = {'aux_mean', 'aux_std'}


def core_metrics(y, p, classes):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    cm = np.bincount(y * classes + p, minlength=classes * classes).reshape(classes, classes)
    tp = np.diag(cm).astype(float)
    return float(tp.sum() / max(1, int(cm.sum())))


class _Acc:
    def __init__(self):
        self.s = 0.0
        self.n = 0

    def add(self, t):
        self.s += float(t.detach().pow(2).sum())
        self.n += int(t.numel())

    def rms(self):
        return float(np.sqrt(self.s / max(1, self.n)))


def _snapshot(model):
    """Return {name: tensor} copies of the modulation params that zeroing will clear."""
    v = model.cfg['variant']
    snap = {}
    if v in GATE_VARIANTS:
        for i, b in enumerate(model.blocks):
            if getattr(b.mix, 'write_w', None) is not None:
                snap[f'{i}.write_w'] = b.mix.write_w.data.clone()
            if getattr(b.mix, 'retention_v', None) is not None:
                snap[f'{i}.retention_v'] = b.mix.retention_v.data.clone()
    if v in TOKEN_VARIANTS and model.token_proj is not None:
        snap['token_proj'] = model.token_proj.weight.data.clone()
    return snap


def _zero(model):
    v = model.cfg['variant']
    if v in GATE_VARIANTS:
        for b in model.blocks:
            if getattr(b.mix, 'write_w', None) is not None:
                b.mix.write_w.data.zero_()
            if getattr(b.mix, 'retention_v', None) is not None:
                b.mix.retention_v.data.zero_()
    if v in TOKEN_VARIANTS and model.token_proj is not None:
        model.token_proj.weight.data.zero_()


def _restore(model, snap):
    for k, val in snap.items():
        parts = k.split('.')
        if parts[0] == 'token_proj':
            model.token_proj.weight.data.copy_(val)
        else:
            getattr(model.blocks[int(parts[0])].mix, parts[1]).data.copy_(val)


@torch.no_grad()
def evaluate(model, loader, device, freq, collect_gate, snap=None):
    model.eval()
    g_shift, rho_shift = _Acc(), _Acc()
    ys, ps = [], []
    for x, y, _, _ in loader:
        x = x.to(device)
        if freq != 0.0:
            x = perturb(x, freq, 0.0)
        x = max_abs_norm(x)
        if collect_gate:
            for b in model.blocks:
                b.mix.collect_gate_detail = True
            _zero(model)                       # off pass (reduces to Q_corr)
            model(x)
            g0 = [b.mix.last_gate_detail['g'] for b in model.blocks]
            rho0 = [b.mix.last_gate_detail['rho'] for b in model.blocks]
            _restore(model, snap)              # on pass (trained modulation)
            logits = model(x)
            g1 = [b.mix.last_gate_detail['g'] for b in model.blocks]
            rho1 = [b.mix.last_gate_detail['rho'] for b in model.blocks]
            for ga, gb in zip(g0, g1):
                g_shift.add(ga - gb)
            for ra, rb in zip(rho0, rho1):
                rho_shift.add(ra - rb)
        else:
            logits = model(x)
        ys.extend(y.tolist()); ps.extend(logits.argmax(-1).cpu().tolist())
    oa = core_metrics(np.asarray(ys), np.asarray(ps), model.cfg['classes'])
    return oa, g_shift.rms(), rho_shift.rms()


def gate_weight_stats(model):
    rows = []
    for li, b in enumerate(model.blocks):
        w = getattr(b.mix, 'write_w', None)
        v = getattr(b.mix, 'retention_v', None)
        row = {'layer': li}
        if w is not None:
            wd = w.detach()
            row['write_w_l2'] = float(wd.pow(2).sum().sqrt())
            row['write_w_mean'] = float(wd.mean())
            row['write_w_std'] = float(wd.std())
            row['write_w_max_abs'] = float(wd.abs().max())
        if v is not None:
            vd = v.detach()
            row['retention_v_l2'] = float(vd.pow(2).sum().sqrt())
            row['retention_v_mean'] = float(vd.mean())
            row['retention_v_std'] = float(vd.std())
            row['retention_v_max_abs'] = float(vd.abs().max())
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True, help='comma-separated run dirs (each has best.pt)')
    p.add_argument('--manifest', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--out', default='reports/qcorr_gates/mechanism_diag.jsonl')
    a = p.parse_args()

    device = torch.device(a.device)
    m = read_manifest(a.manifest)
    if m['normalization'] != 'none':
        raise ValueError('requires raw manifest (normalization=none)')
    loader = make_loader(IQDataset(m, 'val'), a.batch_size, a.workers)
    conds = [('clean', 0.0), ('cfo0p005', 0.005), ('cfom0p005', -0.005)]

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(s).resolve() for s in a.runs.split(',')]

    with out.open('a') as fh:
        for run in run_dirs:
            pt = run / 'best.pt'
            if not pt.exists():
                print(f'skip missing {pt}', flush=True)
                continue
            ck = torch.load(pt, map_location='cpu', weights_only=False)
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
            is_gate = variant in GATE_VARIANTS
            record = {'run': str(run), 'variant': variant, 'seed': seed,
                      'gate_weight_stats': gate_weight_stats(model) if is_gate else None,
                      'conditions': {}}
            snap = _snapshot(model)
            for label, f in conds:
                _zero(model)
                oa_off, _, _ = evaluate(model, loader, device, f, collect_gate=False, snap=snap)
                _restore(model, snap)
                oa_on, g_rms, rho_rms = evaluate(model, loader, device, f, collect_gate=is_gate,
                                                 snap=snap)
                _restore(model, snap)
                record['conditions'][label] = {
                    'normal_oa': oa_on, 'zeroed_oa': oa_off,
                    'delta_pp': 100.0 * (oa_on - oa_off),
                    'g_shift_rms': g_rms if is_gate else None,
                    'rho_shift_rms': rho_rms if is_gate else None,
                }
            fh.write(json.dumps(record, allow_nan=False) + '\n')
            fh.flush()
            print(f"{run.name}: clean delta={record['conditions']['clean']['delta_pp']:+.3f}pp "
                  f"g_shift_rms={record['conditions']['clean']['g_shift_rms']}", flush=True)
    print('MECHANISM DIAG DONE', flush=True)


if __name__ == '__main__':
    main()
