"""Expand pretrain corpus from F1 100k to 200k: F1 (100k, (mod,SNR) stratified) + an extra
100k stratified sample from RML2018A rows not used by F1 nor by any downstream split (val/test).
The extra 100k is written as a separate manifest; pretraining concatenates F1 + extra.
"""
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
SL = Path('/home/user_wzk/wzk/projects/SSM/SSM_L')

f1 = np.load(SL / 'round3_scale_transfer/data/f1_indices.npy', allow_pickle=False)
val = np.load(SL / 'data/cls_dev_val.npy', allow_pickle=False)
test = np.load(SL / 'data/cls_dev_test.npy', allow_pickle=False)
train = np.load(SL / 'data/cls_dev_train.npy', allow_pickle=False)

used = set(f1.tolist()) | set(val.tolist()) | set(test.tolist())
# RML2018A 2555904 total rows; extra pool = everything not used by F1/val/test
f = h5py.File('/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5', 'r')
y = f['Y'][:]
z = f['Z'][:]
f.close()

all_idx = np.arange(2555904)
pool = np.array([i for i in all_idx if i not in used], dtype=np.int64)
print(f'extra pool: {len(pool)} rows')

# stratified sample 100k by (mod, snr)
mod = y[pool].argmax(1)
snr = z[pool].reshape(-1).astype(int)
key = mod * 100 + (snr + 20) // 2
rng = np.random.default_rng(20261123)
extra = []
for k in np.unique(key):
    idx = pool[key == k]
    n = max(1, int(round(len(idx) / len(pool) * 100000)))
    extra.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
extra = np.sort(np.concatenate(extra))[:100000]
print(f'extra 100k: {len(extra)} rows, mod dist {np.bincount(y[extra].argmax(1), minlength=24)[:24]}')

# verify no overlap with downstream val/test (F1 train reuse is allowed)
assert len(set(extra.tolist()) & set(val.tolist())) == 0
assert len(set(extra.tolist()) & set(test.tolist())) == 0
np.save(DATA / 'f1_extra_indices.npy', extra)
print('[gen] f1_extra_indices.npy written (200k = F1 100k + extra 100k)')
