"""C. Low-label frozen probe (round-2 task book section C).

6 representations (G/Q x {random, MAE, denoise}) x 3 encoder seeds = 18 frozen encoders.
Cache mean-pooled train/val features once (CPU, resume-safe), then L2 logistic regression with
fixed standardization. Budgets 1/10/50/100/full per class, 5 label seeds. Reports OA and
macro-F1 per budget, low/high-SNR splits, and pair deltas.

Full-label reference reuses round-1 (MAE/denoise) and A (random) AdamW probe val OA; we also
compute an LR full-label for a within-protocol curve.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
R2 = Path(__file__).resolve().parent.parent
RUNS1 = ROOT / 'runs' / 'pretrain'
FEAT_DIR = R2 / 'runs' / 'features'
SEEDS = (17, 42, 2026)
BACKBONES = ('G', 'Q')
PRETRAINS = ('random', 'mae', 'denoise')
BUDGETS = (1, 10, 50, 100)
LABEL_SEEDS = (0, 1, 2, 3, 4)


def model_cfg():
    return json.loads((ROOT / 'configs' / 'pretrain.json').read_text())['model']


def ckpt_for(bb, pt, seed):
    if pt == 'random':
        return R2 / 'runs' / 'random_ckpt' / f'{bb}_seed{seed}.pt'
    return RUNS1 / f'{bb}_{pt}_seed{seed}' / 'encoder_final.pt'


def cache_features(bb, pt, seed, device, force=False):
    import sys
    sys.path.insert(0, str(ROOT))
    from ssm_exp.data import IQDataset, read_manifest
    from round2_overnight.common import load_encoder, features_for_encoder

    out = FEAT_DIR / f'{bb}_{pt}_seed{seed}.pt'
    if out.exists() and not force:
        return out
    ck = ckpt_for(bb, pt, seed)
    if pt == 'random' and not ck.exists():
        from round2_overnight.common import make_random_checkpoint
        make_random_checkpoint(model_cfg(), bb, seed, str(ck), device='cpu',
                               shared_init_dir=str(ROOT / 'runs' / 'pretrain' / 'shared_init'))
    enc = load_encoder(model_cfg(), bb, seed, str(ck), device)
    m = read_manifest(str(ROOT / 'configs' / 'data.json'))
    payload = {}
    for split in ('train', 'val'):
        ds = IQDataset(m, split)
        feats, ys, zs, ids = features_for_encoder(enc, ds, 128, device)
        payload[split] = {'feats': feats, 'y': ys, 'snr': zs, 'idx': ids}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out


def low_label_fit(X, y, budget, label_seed):
    """Sample `budget` examples per class from X/y (integer labels), return train indices."""
    rng = np.random.default_rng(label_seed)
    idx = []
    for c in range(y.max().item() + 1):
        ci = np.where(y == c)[0]
        n = min(budget, len(ci))
        idx.append(rng.choice(ci, size=n, replace=False))
    return np.concatenate(idx)


def run_lr(feat_file, budgets, label_seeds):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    payload = torch.load(feat_file, map_location='cpu', weights_only=False)
    Xtr, ytr, ztr = payload['train']['feats'].numpy(), payload['train']['y'].numpy(), payload['train']['snr'].numpy()
    Xva, yva, zva = payload['val']['feats'].numpy(), payload['val']['y'].numpy(), payload['val']['snr'].numpy()
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    rows = []
    for budget in list(budgets) + ['full']:
        for ls in label_seeds:
            if budget == 'full':
                idx = np.arange(len(ytr))
            else:
                idx = low_label_fit(Xtr, ytr, budget, ls)
            clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', n_jobs=-1)
            clf.fit(Xtr_s[idx], ytr[idx])
            pred = clf.predict(Xva_s)
            oa = float(accuracy_score(yva, pred))
            f1 = float(f1_score(yva, pred, average='macro'))
            # low/high SNR split at -20..+8 / +10..+30 (midpoint between RML integer levels)
            lo = zva <= 8
            hi = zva > 8
            oa_lo = float(accuracy_score(yva[lo], pred[lo]))
            oa_hi = float(accuracy_score(yva[hi], pred[hi]))
            rows.append({'budget': budget, 'label_seed': ls, 'oa': oa, 'macro_f1': f1,
                         'oa_low_snr': oa_lo, 'oa_high_snr': oa_hi})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=str(R2 / 'reports' / 'lowlabel_probe_round2.json'))
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--force-cache', action='store_true')
    args = ap.parse_args()
    device = torch.device(args.device)

    specs = [(bb, pt, s) for bb in BACKBONES for pt in PRETRAINS for s in SEEDS]
    results = {}
    for i, (bb, pt, s) in enumerate(specs):
        key = f'{bb}_{pt}_seed{s}'
        f = cache_features(bb, pt, s, device, force=args.force_cache)
        print(f'[{i+1}/{len(specs)}] cached {key}', flush=True)
        if not args.cache_only:
            rows = run_lr(f, BUDGETS, LABEL_SEEDS)
            results[key] = rows
            print(f'  {key}: full OA {rows[-1]["oa"]:.4f}', flush=True)
    if not args.cache_only:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'budgets': list(BUDGETS), 'label_seeds': list(LABEL_SEEDS),
                                   'results': results}, indent=2))
        print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
