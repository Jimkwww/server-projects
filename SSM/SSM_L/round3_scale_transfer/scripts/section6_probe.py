"""Section 6 downstream part 1: unified LR probe (RML) + LR transfer (RadChar) for the 18 new
F1/F2 pretrained encoders (G / G_Qlearn / Q_corr x {f1,f2} x 3 seeds).

F0 baselines reuse section3/section4 unified readout results (G/Q/G_Qlearn x mae).

Protocol identical to sections 3/4: cache mean-pooled features (encoder frozen), then
StandardScaler(train-fit) + L2 LogisticRegression(C=1, lbfgs, max_iter=2000). RML full-label +
10/class x 5 label-seed low-label (both scaler conventions); RadChar full-label eval.

Outputs:
  round3_scale_transfer/runs/features_s6/           RML features for 18 encoders
  round3_scale_transfer/runs/radar_features_s6/     RadChar features for 18 encoders
  round3_scale_transfer/reports/section6_probe.json
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
FEAT6 = R3 / 'runs' / 'features_s6'
RADAR_FEAT6 = R3 / 'runs' / 'radar_features_s6'
MODELS = ('G', 'G_Qlearn', 'Q')
CORPORA = ('f1', 'f2')
SEEDS = (17, 42, 2026)
LABEL_SEEDS = (0, 1, 2, 3, 4)
BUDGET = 10


def model_cfg():
    return json.loads((ROOT / 'configs' / 'pretrain.json').read_text())['model']


def ckpt_for(corpus, model, seed):
    return R3 / 'runs' / 'pretrain_scale' / corpus / f'{model}_seed{seed}' / 'encoder_final.pt'


def cache_features(ck, model, seed, manifest, splits, device, out):
    from ssm_exp.data import IQDataset, read_manifest
    from round2_overnight.common import load_encoder, features_for_encoder

    if out.exists():
        return out, 'skip'
    if not ck.exists():
        return out, f'missing:{ck}'
    enc = load_encoder(model_cfg(), model, seed, str(ck), device)
    m = read_manifest(manifest)
    payload = {}
    for split in splits:
        ds = IQDataset(m, split)
        feats, ys, zs, ids = features_for_encoder(enc, ds, 128, device)
        payload[split] = {'feats': feats, 'y': ys, 'snr': zs, 'idx': ids}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out, 'ok'


def _metrics(y, pred, snr):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    y, pred, snr = np.asarray(y), np.asarray(pred), np.asarray(snr)
    oa = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, average='macro'))
    lo, hi = snr <= 0, snr > 0
    return {'oa': oa, 'macro_f1': f1,
            'low_oa': float(accuracy_score(y[lo], pred[lo])) if lo.any() else None,
            'high_oa': float(accuracy_score(y[hi], pred[hi])) if hi.any() else None}


def _label_sample_idx(y, budget, label_seed):
    rng = np.random.default_rng(label_seed)
    y = np.asarray(y)
    idx = []
    for c in range(int(y.max()) + 1):
        ci = np.where(y == c)[0]
        n = min(budget, len(ci))
        idx.append(rng.choice(ci, size=n, replace=False))
    return np.sort(np.concatenate(idx))


def rml_readout(feat_file):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    p = torch.load(feat_file, map_location='cpu', weights_only=False)
    Xtr = p['train']['feats'].numpy().astype(np.float32)
    ytr = p['train']['y'].numpy().astype(np.int64)
    Xva = p['val']['feats'].numpy().astype(np.float32)
    yva = p['val']['y'].numpy().astype(np.int64)
    zva = p['val']['snr'].numpy()
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    clf.fit(Xtr_s, ytr)
    full = _metrics(yva, clf.predict(Xva_s), zva)
    low = {}
    for ls in LABEL_SEEDS:
        idx = _label_sample_idx(ytr, BUDGET, ls)
        clf_a = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        clf_a.fit(Xtr_s[idx], ytr[idx])
        ma = _metrics(yva, clf_a.predict(Xva_s), zva)
        scaler_b = StandardScaler().fit(Xtr[idx])
        clf_b = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        clf_b.fit(scaler_b.transform(Xtr[idx]), ytr[idx])
        mb = _metrics(yva, clf_b.predict(scaler_b.transform(Xva)), zva)
        low[str(ls)] = {'full_pool_scaler': ma, 'strict_scaler': mb,
                        'sample_idx_sha256':
                            hashlib.sha256(np.sort(idx).astype(np.int64).tobytes()).hexdigest()}
    return {'full_label': full, 'low_label': low}


def radar_readout(feat_file):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import recall_score

    p = torch.load(feat_file, map_location='cpu', weights_only=False)
    Xtr = p['train']['feats'].numpy().astype(np.float32)
    ytr = p['train']['y'].numpy().astype(np.int64)
    Xev = p['test']['feats'].numpy().astype(np.float32)
    yev = p['test']['y'].numpy().astype(np.int64)
    zev = p['test']['snr'].numpy()
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xev))
    m = _metrics(yev, pred, zev)
    m['recall'] = recall_score(yev, pred, average=None).tolist()
    m['n'] = int(len(yev))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--readout-only', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    rml_manifest = str(ROOT / 'configs' / 'data.json')
    radar_manifest = str(R2 / 'configs' / 'radchar.json')
    specs = [(corp, mdl, s) for corp in CORPORA for mdl in MODELS for s in args.seeds]

    if args.readout_only:
        results = {}
        for corp, mdl, s in specs:
            key = f'{corp}_{mdl}_seed{s}'
            frml = FEAT6 / f'{key}.pt'
            frad = RADAR_FEAT6 / f'{key}.pt'
            if not frml.exists() or not frad.exists():
                results[key] = {'missing': True}
                print(f'{key} MISSING (rml={frml.exists()} radar={frad.exists()})', flush=True)
                continue
            rml = rml_readout(frml)
            rad = radar_readout(frad)
            results[key] = {'rml': rml, 'radar': rad}
            print(f"{key}: RML {rml['full_label']['oa']:.4f} radar {rad['oa']:.4f}", flush=True)
        out = R3 / 'reports' / 'section6_probe.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'protocol': 'StandardScaler(train)+L2 LR(C=1,lbfgs)',
                                   'results': results}, indent=2))
        print('wrote', out, flush=True)
        return

    for i, (corp, mdl, s) in enumerate(specs):
        ck = ckpt_for(corp, mdl, s)
        key = f'{corp}_{mdl}_seed{s}'
        f1, st1 = cache_features(ck, mdl, s, rml_manifest, ('train', 'val'), device,
                                 FEAT6 / f'{key}.pt')
        f2, st2 = cache_features(ck, mdl, s, radar_manifest, ('train', 'val', 'test'), device,
                                 RADAR_FEAT6 / f'{key}.pt')
        print(f'[{i+1}/{len(specs)}] {key}: rml={st1} radar={st2}', flush=True)


if __name__ == '__main__':
    main()
