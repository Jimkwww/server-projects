"""P0 appendix: RadChar-Tiny simple (non-learned-feature) baselines.

1. flattened IQ   : max_abs-normalized raw IQ [2,512] -> 1024 reals.
2. fixed features : physical_features descriptor (log-energy + lag-1 complex correlation
   Re/Im per 8-sample token) -> 64*3 = 192 reals. Hand-crafted, no learned parameters.

Both: StandardScaler (fit on radar train) + L2 LogisticRegression (C=1, lbfgs, max_iter 2000).
Train on radar 'train' (30000), report 'val' (=dev) and 'test' (=eval) once. This quantifies
how much of the radar accuracy is available to a linear model WITHOUT any learned encoder,
which explains why even a random encoder's frozen features read out well on radar.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ssm_exp.data import read_manifest, audit, IQDataset
from ssm_exp.model import max_abs_norm, physical_features
from ssm_exp.utils import save_json, seed_all


def load_split(m, split):
    ds = IQDataset(m, split)
    xs, ys = [], []
    for i in range(len(ds)):
        x, y, z, idx = ds[i]            # x: [2, L] (I row 0, Q row 1), raw (normalization=none)
        x = x / max(float(x.abs().max()), 1e-8)   # joint max-abs over I and Q (matches 'max_abs')
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), np.asarray(ys)


def metrics(y, p, n_classes):
    cm = np.bincount(y * n_classes + p, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, den, out=np.zeros(n_classes), where=den > 0)
    return {'n': int(cm.sum()), 'oa': float(tp.sum() / max(1, cm.sum())),
            'macro_f1': float(f1.mean()), 'class_f1': f1.tolist()}


def featurize(x, kind, patch):
    if kind == 'flattened_iq':
        return x.reshape(x.shape[0], -1).numpy()          # [N,1024] I-block then Q-block
    if kind == 'physical':
        return physical_features(x, patch).reshape(x.shape[0], -1).numpy()  # [N,192]
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--patch', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed)
    m = read_manifest(a.manifest)
    data_audit = audit(m)
    n_classes = m['classes']

    xtr, ytr = load_split(m, 'train')
    xval, yval = load_split(m, 'val')
    xte, yte = load_split(m, 'test')

    results = {'data': data_audit, 'patch': a.patch, 'n_classes': n_classes}
    for kind in ('flattened_iq', 'physical'):
        ftr = featurize(xtr, kind, a.patch)
        fval = featurize(xval, kind, a.patch)
        fte = featurize(xte, kind, a.patch)
        scaler = StandardScaler().fit(ftr)
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        clf.fit(scaler.transform(ftr), ytr)
        r = {
            'features': kind, 'feature_dim': int(ftr.shape[1]),
            'val': metrics(yval, clf.predict(scaler.transform(fval)), n_classes),
            'test': metrics(yte, clf.predict(scaler.transform(fte)), n_classes),
        }
        results[kind] = r
        np.savez_compressed(out / f'{kind}_predictions.npz',
                            y_val=yval, pred_val=clf.predict(scaler.transform(fval)),
                            y_test=yte, pred_test=clf.predict(scaler.transform(fte)))
        print(json.dumps({kind: {'val_oa': r['val']['oa'], 'test_oa': r['test']['oa']}}), flush=True)
    save_json(out / 'results.json', results)
    save_json(out / 'DONE.json', {'seed': a.seed, 'patch': a.patch})


if __name__ == '__main__':
    main()
