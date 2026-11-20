"""F. Expanded-corpus eligibility check (round-2 task book section F).

Produces eligibility.csv classifying each pool as 可用/留出/不适用 with reason, independent
record count, preprocessing note and content hash. Does NOT launch pretraining (resource-gated).

F0 = round-1 RML train (cls_dev_train, 19968). F1 = enlarged same-domain RML train pool
(all RML rows minus cls_dev val/test, deduped by exact IQ hash vs held-out and within-pool,
capped at 100k). F2 = 50% comm + 50% qualified OOD raw IQ: the OOD half is unavailable because
RadChar is held out for E, RadarCommDataset is real-valued (not complex IQ) and EM-Radar-mixed
is blind source separation (not classification).
"""
import csv
import hashlib
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
R2 = Path(__file__).resolve().parent.parent
H5 = '/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5'
CAP = 100000


def sha(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def main():
    with h5py.File(H5, 'r') as f:
        n_total = f['X'].shape[0]
    train = np.load(ROOT / 'data' / 'cls_dev_train.npy')
    val = np.load(ROOT / 'data' / 'cls_dev_val.npy')
    test = np.load(ROOT / 'data' / 'cls_dev_test.npy')
    held_out = set(val.tolist()) | set(test.tolist())          # indices to exclude from F1

    # F0 = round-1 train; hash it for the record.
    with h5py.File(H5, 'r') as f:
        f0_hashes = [sha(f['X'][i]) for i in train]
    f0 = {'name': 'F0', 'pool': 'RML2018A cls_dev train', 'status': '可用',
          'reason': '第一轮 RML train', 'independent_records': int(len(np.unique(f0_hashes))),
          'preprocessing': 'none (raw IQ, max_abs_norm in-loop)',
          'sha256': hashlib.sha256(''.join(f0_hashes).encode()).hexdigest()}

    # F1: scan candidate pool (not val/test), dedup vs held-out IQ and within-pool.
    held_out_hashes = set()
    with h5py.File(H5, 'r') as f:
        for i in sorted(held_out):
            held_out_hashes.add(sha(f['X'][i]))
    # also treat F0 train hashes as "already used" only for overlap REPORTING, not exclusion.
    f0_set = set(f0_hashes)
    cand = np.setdiff1d(np.arange(n_total), np.array(sorted(held_out)), assume_unique=False)
    pool_hashes = set()
    selected = []
    with h5py.File(H5, 'r') as f:
        for start in range(0, len(cand), 100000):
            block = cand[start:start + 100000]
            xb = f['X'][block]           # [B,1024,2]
            for j, i in enumerate(block):
                h = sha(xb[j])
                if h in held_out_hashes or h in pool_hashes:
                    continue
                pool_hashes.add(h)
                selected.append(int(i))
                if len(selected) >= CAP:
                    break
            if len(selected) >= CAP:
                break
    f1_overlap_with_f0 = len(pool_hashes & f0_set)
    f1 = {'name': 'F1', 'pool': 'RML2018A (minus cls_dev val/test, deduped)',
          'status': '可用', 'reason': '同域扩大；独立 IQ 去重后池',
          'independent_records': len(selected), 'preprocessing': 'none (raw IQ)',
          'sha256': hashlib.sha256(''.join(sorted(pool_hashes)).encode()).hexdigest(),
          'overlap_with_f0_unique_iq': f1_overlap_with_f0}

    f2 = {'name': 'F2', 'pool': '50% comm + 50% OOD raw IQ',
          'status': '不适用',
          'reason': '异域原始 IQ 无合格语料：RadChar 已整体留出用于 E；RadarCommDataset 为实值'
                    '(非复 IQ)；EM-Radar-mixed 为盲源分离(非分类)。',
          'independent_records': 0, 'preprocessing': 'n/a', 'sha256': ''}

    out = R2 / 'reports' / 'eligibility.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ['name', 'pool', 'status', 'reason', 'independent_records', 'preprocessing', 'sha256']
    with out.open('w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in (f0, f1, f2):
            w.writerow(r)
    print('F0 independent', f0['independent_records'], 'unique; F1 selected', len(selected),
          'unique (cap', CAP, '); F1 overlap with F0', f1_overlap_with_f0, flush=True)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
