"""Compute per-class F1, confusion matrix, and per-SNR results from saved test predictions.
EVAL_ONLY: no training, reads test_pred.npz + test_snr.npy. For v1 (no ids saved, but eval
order is 0..N-1 so y/pred align with test_snr.npy) and v2 (same ordering).
"""
import argparse, json
from pathlib import Path
import numpy as np

def per_class_f1(y, p, nc):
    out = {}
    for c in range(nc):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        out[str(c)] = float(2 * tp / (2 * tp + fp + fn + 1e-9))
    return out

def macro_f1(y, p, nc):
    return float(np.mean(list(per_class_f1(y, p, nc).values())))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)   # test_pred.npz
    ap.add_argument('--snr', required=True)    # test_snr.npy
    ap.add_argument('--nc', type=int, required=True)
    ap.add_argument('--class-names', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    d = np.load(a.pred)
    y, p = d['y'].astype(int), d['pred'].astype(int)
    snr = np.load(a.snr).reshape(-1)
    assert len(y) == len(snr), f'len mismatch pred {len(y)} vs snr {len(snr)}'
    nc = a.nc
    oa = float((y == p).mean())
    mf1 = macro_f1(y, p, nc)
    pcf1 = per_class_f1(y, p, nc)
    cm = np.zeros((nc, nc), dtype=int)
    np.add.at(cm, (y, p), 1)
    # per-SNR groups
    snr_res = {}
    for gname, mask in [('all', np.ones(len(y), bool)),
                        ('le0', snr <= 0), ('gt0', snr > 0)]:
        if mask.sum() == 0:
            continue
        snr_res[gname] = {'n': int(mask.sum()), 'oa': float((y[mask] == p[mask]).mean()),
                          'macro_f1': macro_f1(y[mask], p[mask], nc)}
    result = {'oa': oa, 'macro_f1': mf1, 'per_class_f1': pcf1, 'confusion': cm.tolist(),
              'per_snr': snr_res}
    if a.class_names:
        result['class_names'] = a.class_names.split(',')
    print(json.dumps({k: v for k, v in result.items() if k != 'confusion'}, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2))
        print(f'[saved] {a.out}')

if __name__ == '__main__':
    main()
