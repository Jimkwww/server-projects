"""Section 3: unified frozen readout across 4 structures x {random, MAE, denoise} x 3 seeds.

4 backbones: G, Q(=Q_corr), G_Qlearn, S1plain. 36 encoders total.
G/Q features reused from round-2 cache (round2_overnight/runs/features/); G_Qlearn/S1plain are
cached here (round3_scale_transfer/runs/features/), resume-safe by weight-hash.

Readout protocol (unified): StandardScaler(fit train) + L2 LogisticRegression(C=1, lbfgs,
max_iter=2000). Full-label main table + 10/class x 5 label-seed low-label table with shared
sample IDs and TWO scaler conventions (full-pool vs strict selected-only).

Usage:
  python section3_unified_readout.py --cache-only --backbones G_Qlearn S1plain --seeds 17 42 2026 --device cuda:0
  python section3_unified_readout.py --readout-only
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
R2 = ROOT / 'round2_overnight'
R3 = ROOT / 'round3_scale_transfer'
FEAT3 = R3 / 'runs' / 'features'
FEAT2 = R2 / 'runs' / 'features'
SEEDS = (17, 42, 2026)
BACKBONES = ('G', 'Q', 'G_Qlearn', 'S1plain')
PRETRAINS = ('random', 'mae', 'denoise')
LABEL_SEEDS = (0, 1, 2, 3, 4)
BUDGET = 10  # per class


def model_cfg():
    return json.loads((ROOT / 'configs' / 'pretrain.json').read_text())['model']


def ckpt_for(bb, pt, seed):
    if pt == 'random':
        return R2 / 'runs' / 'random_ckpt' / f'{bb}_seed{seed}.pt'
    if bb in ('G', 'Q'):
        return ROOT / 'runs' / 'pretrain' / f'{bb}_{pt}_seed{seed}' / 'encoder_final.pt'
    return R2 / 'runs' / 'pretrain' / f'{bb}_{pt}_seed{seed}' / 'encoder_final.pt'


def feat_path(bb, pt, seed):
    if bb in ('G', 'Q'):
        return FEAT2 / f'{bb}_{pt}_seed{seed}.pt'
    return FEAT3 / f'{bb}_{pt}_seed{seed}.pt'


def cache_features(bb, pt, seed, device):
    from ssm_exp.data import IQDataset, read_manifest
    from round2_overnight.common import load_encoder, features_for_encoder

    out = feat_path(bb, pt, seed)
    if out.exists():
        return out, 'skip'
    ck = ckpt_for(bb, pt, seed)
    if not ck.exists():
        return out, f'missing-ckpt:{ck.name}'
    enc = load_encoder(model_cfg(), bb, seed, str(ck), device)
    m = read_manifest(str(ROOT / 'configs' / 'data.json'))
    payload = {}
    for split in ('train', 'val'):
        ds = IQDataset(m, split)
        feats, ys, zs, ids = features_for_encoder(enc, ds, 128, device)
        payload[split] = {'feats': feats, 'y': ys, 'snr': zs, 'idx': ids}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out, 'ok'


def _label_sample_idx(y, budget, label_seed):
    """Deterministic per-class sampling; same across encoders (shared sample IDs)."""
    rng = np.random.default_rng(label_seed)
    y = np.asarray(y)
    idx = []
    for c in range(int(y.max()) + 1):
        ci = np.where(y == c)[0]
        n = min(budget, len(ci))
        idx.append(rng.choice(ci, size=n, replace=False))
    return np.sort(np.concatenate(idx))


def _metrics(y, pred, snr):
    from sklearn.metrics import accuracy_score, f1_score
    y, pred, snr = np.asarray(y), np.asarray(pred), np.asarray(snr)
    oa = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, average='macro'))
    lo, hi = snr <= 0, snr > 0
    return {'oa': oa, 'macro_f1': f1,
            'low_oa': float(accuracy_score(y[lo], pred[lo])) if lo.any() else None,
            'high_oa': float(accuracy_score(y[hi], pred[hi])) if hi.any() else None}


def run_readout(feat_file):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    payload = torch.load(feat_file, map_location='cpu', weights_only=False)
    Xtr = payload['train']['feats'].numpy().astype(np.float32)
    ytr = payload['train']['y'].numpy().astype(np.int64)
    Xva = payload['val']['feats'].numpy().astype(np.float32)
    yva = payload['val']['y'].numpy().astype(np.int64)
    zva = payload['val']['snr'].numpy()

    # full-label (scaler fit on full train pool)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    clf.fit(Xtr_s, ytr)
    full = _metrics(yva, clf.predict(Xva_s), zva)

    # low-label 10/class x 5 label seeds, two scaler conventions
    low = {}
    for ls in LABEL_SEEDS:
        idx = _label_sample_idx(ytr, BUDGET, ls)
        Xsel = Xtr[idx]
        ysel = ytr[idx]
        # (a) full-pool scaler (round-2 convention)
        clf_a = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        clf_a.fit(Xtr_s[idx], ysel)
        m_a = _metrics(yva, clf_a.predict(Xva_s), zva)
        # (b) strict selected-only scaler
        scaler_b = StandardScaler().fit(Xsel)
        clf_b = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        clf_b.fit(scaler_b.transform(Xsel), ysel)
        m_b = _metrics(yva, clf_b.predict(scaler_b.transform(Xva)), zva)
        low[str(ls)] = {'full_pool_scaler': m_a, 'strict_scaler': m_b,
                        'n_train': int(len(idx)), 'sample_idx_sha256':
                            hashlib.sha256(np.sort(idx).astype(np.int64).tobytes()).hexdigest()}
    return {'full_label': full, 'low_label': low}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--readout-only', action='store_true')
    ap.add_argument('--backbones', nargs='+', default=list(BACKBONES))
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.readout_only:
        results = {}
        for bb in BACKBONES:
            for pt in PRETRAINS:
                for s in SEEDS:
                    f = feat_path(bb, pt, s)
                    key = f'{bb}_{pt}_seed{s}'
                    if not f.exists():
                        results[key] = {'missing': True}
                        print(f'{key} MISSING', flush=True)
                        continue
                    r = run_readout(f)
                    results[key] = r
                    print(f"{key}: full OA {r['full_label']['oa']:.4f} "
                          f"low10(full-pool) {statistics.mean(r['low_label'][str(ls)]['full_pool_scaler']['oa'] for ls in LABEL_SEEDS):.4f} "
                          f"strict {statistics.mean(r['low_label'][str(ls)]['strict_scaler']['oa'] for ls in LABEL_SEEDS):.4f}",
                          flush=True)
        out = R3 / 'reports' / 'section3_unified_readout.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'budget_per_class': BUDGET, 'label_seeds': list(LABEL_SEEDS),
                                   'protocol': 'StandardScaler(train) + L2 LR(C=1, lbfgs, 2000)',
                                   'results': results}, indent=2))
        print('wrote', out, flush=True)
        return

    # cache phase
    specs = [(bb, pt, s) for bb in args.backbones for pt in PRETRAINS for s in args.seeds]
    for i, (bb, pt, s) in enumerate(specs):
        f, status = cache_features(bb, pt, s, device)
        print(f'[{i+1}/{len(specs)}] {bb}_{pt}_seed{s}: {status}', flush=True)


if __name__ == '__main__':
    main()
