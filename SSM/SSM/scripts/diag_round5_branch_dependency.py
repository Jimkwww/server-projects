#!/usr/bin/env python
"""Round-5 Review Section 三: branch-dependency diagnostic (normal vs disabled fusion).

For each Q_linear/Q_corr/R_corr x 3-seed best checkpoint, evaluate on val (NO test) at
clean / CFO=+-0.002 / +-0.005, in two modes:
  normal   u_new = u + W_f * f_std   (the trained model)
  disabled u_new = u                 (fusion contribution removed, weights/stats untouched)
Record OA, Macro-F1, per-SNR for each mode, plus per-condition RMS diagnostics:
raw aux descriptor, standardized descriptor, main-path token, fusion term, and the
fusion/main ratio (denominator protected). Eval-mode only; nothing is written back to
any checkpoint and the frozen stats are never modified.
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


def core_metrics(y, p, classes):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    cm = np.bincount(y * classes + p, minlength=classes * classes).reshape(classes, classes)
    tp = np.diag(cm).astype(float)
    row = cm.sum(1).astype(float)
    col = cm.sum(0).astype(float)
    f1 = np.divide(2 * tp, row + col, out=np.zeros(classes), where=(row + col) > 0)
    return {'oa': float(tp.sum() / max(1, int(cm.sum()))),
            'macro_f1': float(f1.mean())}


def full_metrics(y, p, snr, classes):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    snr = np.asarray(snr, dtype=float)
    per_snr = {str(float(z)): core_metrics(y[snr == z], p[snr == z], classes)
               for z in np.unique(snr)}
    out = core_metrics(y, p, classes)
    out['per_snr'] = per_snr
    return out


class _Acc:
    def __init__(self):
        self.s = 0.0
        self.n = 0

    def add(self, t):
        self.s += float(t.detach().pow(2).sum())
        self.n += int(t.numel())

    def rms(self):
        return float(np.sqrt(self.s / max(1, self.n)))


@torch.no_grad()
def evaluate_cond(model, loader, device, frequency, phase):
    """Returns (normal_metrics, disabled_metrics, diag). diag holds per-condition RMS stats."""
    model.eval()
    yn, pn, zn = [], [], []
    yd, pd = [], []
    raw = _Acc(); std = _Acc(); main = _Acc(); fus = _Acc()
    for x, y, z, _ in loader:
        x = x.to(device)
        if frequency != 0.0 or phase != 0.0:
            x = perturb(x, frequency, phase)
        x = max_abs_norm(x)
        u = model.front(x)
        f_raw = model.aux_descriptor(x)
        f = (f_raw - model.aux_mean) / model.aux_std
        fusion_term = model.aux_fusion(f)
        p = u.new_zeros(u.shape[0], u.shape[1], 3)
        u_norm = u + fusion_term          # normal
        u_dis = u                          # disabled
        for block in model.blocks:
            u_norm = block(u_norm, p)
            u_dis = block(u_dis, p)
        logits_n = model.head(model.norm(u_norm).mean(1))
        logits_d = model.head(model.norm(u_dis).mean(1))
        if not torch.isfinite(logits_n).all() or not torch.isfinite(logits_d).all():
            raise FloatingPointError('nonfinite logits')
        yn.extend(y.tolist()); pn.extend(logits_n.argmax(-1).cpu().tolist())
        yd.extend(y.tolist()); pd.extend(logits_d.argmax(-1).cpu().tolist())
        zn.extend(z.tolist())
        raw.add(f_raw); std.add(f); main.add(u); fus.add(fusion_term)

    classes = model.cfg['classes']
    m_n = full_metrics(np.asarray(yn), np.asarray(pn), np.asarray(zn), classes)
    m_d = full_metrics(np.asarray(yd), np.asarray(pd), np.asarray(zn), classes)
    main_rms = main.rms()
    fus_rms = fus.rms()
    diag = {'raw_descriptor_rms': raw.rms(),
            'standardized_descriptor_rms': std.rms(),
            'main_path_token_rms': main_rms,
            'fusion_term_rms': fus_rms,
            'fusion_over_main_ratio': fus_rms / max(main_rms, 1e-8)}
    return m_n, m_d, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True, help='comma-separated run dirs (each has best.pt)')
    p.add_argument('--manifest', required=True)
    p.add_argument('--frequencies', default='0.002,0.005')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--out', default='reports/round5_review/branch_dependency.jsonl')
    a = p.parse_args()

    device = torch.device(a.device)
    m = read_manifest(a.manifest)
    if m['normalization'] != 'none':
        raise ValueError('requires raw manifest (normalization=none)')
    loader = make_loader(IQDataset(m, 'val'), a.batch_size, a.workers)
    mags = [float(v) for v in a.frequencies.split(',')]
    conds = [('clean', 0.0)] + [(f'cfo+{v}', v) for v in mags] + [(f'cfo-{v}', -v) for v in mags]

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(s).resolve() for s in a.runs.split(',')]

    with out.open('a') as fh:
        for run in run_dirs:
            ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
            cfg = ck['config']['model']
            seed = ck['config']['train']['seed']
            model = Classifier(cfg, seed=seed).to(device)
            model.load_state_dict(ck['model'])
            variant = cfg['variant']
            for label, freq in conds:
                m_n, m_d, diag = evaluate_cond(model, loader, device, freq, 0.0)
                rec = {'run': str(run), 'variant': variant, 'seed': seed,
                       'condition': label, 'frequency': freq,
                       'normal': {'oa': m_n['oa'], 'macro_f1': m_n['macro_f1'],
                                  'per_snr': m_n['per_snr']},
                       'disabled': {'oa': m_d['oa'], 'macro_f1': m_d['macro_f1'],
                                    'per_snr': m_d['per_snr']},
                       'diag': diag}
                fh.write(json.dumps(rec, allow_nan=False) + '\n')
                fh.flush()
                print(f"{run.name} {label}: normal_oa={m_n['oa']:.4f} "
                      f"disabled_oa={m_d['oa']:.4f} ratio={diag['fusion_over_main_ratio']:.4f}",
                      flush=True)
    print('BRANCH DEPENDENCY DIAG DONE', flush=True)


if __name__ == '__main__':
    main()
