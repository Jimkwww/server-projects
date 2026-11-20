"""Give F1 pretrain an independent val split (no overlap with any downstream split).

Historical SSM_L F1 manifest used cls_dev_val as its pretrain val — this leaks the
downstream val into pretrain checkpoint selection. Fix: sample the pretrain val from the
F1 indices that lie OUTSIDE all cls_dev splits (i.e. never seen by downstream train/val/test).
Pretrain train stays F1 100k (reusing downstream TRAIN unlabeled data is allowed by protocol).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
SL = Path('/home/user_wzk/wzk/projects/SSM/SSM_L')

f1 = np.load(SL / 'round3_scale_transfer/data/f1_indices.npy', allow_pickle=False)
tr = np.load(SL / 'data/cls_dev_train.npy', allow_pickle=False)
va = np.load(SL / 'data/cls_dev_val.npy', allow_pickle=False)
te = np.load(SL / 'data/cls_dev_test.npy', allow_pickle=False)

downstream = set(tr.tolist()) | set(va.tolist()) | set(te.tolist())
extra = np.array([i for i in f1.tolist() if i not in downstream], dtype=np.int64)
print(f'F1 total={len(f1)}, downstream-union={len(downstream)}, extra(outside all downstream)={len(extra)}')

rng = np.random.default_rng(20261119)
val_idx = rng.choice(extra, size=3120, replace=False)
val_idx.sort()
np.save(DATA / 'f1_val_indices.npy', val_idx)
print(f'F1 pretrain val = {len(val_idx)} independent samples; overlap with downstream val/test = 0')

# Verify
assert len(set(val_idx.tolist()) & downstream) == 0
print('[fix] f1_val_indices.npy written')
