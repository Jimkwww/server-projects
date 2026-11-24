"""HisarMod 2019 (26-class modulation, cross-dataset) -> emmem npy manifest.

CSV data is 1024 complex samples stored as strings like '7.1203-7.7446i' (i not j).
We subsample (train 4000/class, val 500/class, test 2000/class) to keep it tractable, parse
complex strings in batches, and map the 26 non-contiguous label ids to 0..25.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path('/data1/user_wzk/datasets/Hisarmod_2019/HisarMod2019.1')
OUT = Path('/home/user_wzk/wzk/projects/SSM/SSM_EM/em_memory_suite/data/hisarmod')
LEN = 1024


def parse_batch(df_sub):
    """df_sub: [rows, 1024] of 'a+bi' strings -> [rows, 1024, 2] float32."""
    arr = df_sub.to_numpy().astype(str)
    # vectorized: replace i with j, then complex
    arr = np.char.replace(arr, 'i', 'j')
    z = arr.astype(np.complex64)
    return np.stack([z.real.astype(np.float32), z.imag.astype(np.float32)], axis=-1)


def load_split(data_csv, label_csv, per_class, rng):
    labels = pd.read_csv(label_csv, header=None)[0].to_numpy()
    uniq = sorted(set(labels.tolist()))
    label_map = {u: i for i, u in enumerate(uniq)}
    picks = []
    for u in uniq:
        idx = np.flatnonzero(labels == u)
        n = min(per_class, len(idx))
        picks.append(rng.choice(idx, size=n, replace=False))
    picks = np.sort(np.concatenate(picks))
    print(f'  读取 {data_csv.name} 全量 ({len(labels)} 行)...')
    df = pd.read_csv(data_csv, header=None)          # full read; server has 503GB RAM
    X = parse_batch(df.iloc[picks])                  # [n,1024,2]
    Y = np.array([label_map[labels[j]] for j in picks], dtype=np.int64)
    return X, Y, label_map, uniq


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261123)
    # train
    Xtr, Ytr, label_map, uniq = load_split(SRC / 'Train/train_data.csv', SRC / 'Train/train_labels.csv', 4000, rng)
    # val: split from train picks (500/class) — reuse a fixed subset of train
    val_mask = np.zeros(len(Ytr), dtype=bool)
    for c in range(len(uniq)):
        idx = np.flatnonzero(Ytr == c)[:500]
        val_mask[idx] = True
    Xva, Yva = Xtr[val_mask], Ytr[val_mask]
    Xtr, Ytr = Xtr[~val_mask], Ytr[~val_mask]
    # test
    Xte, Yte, _, _ = load_split(SRC / 'Test/test_data.csv', SRC / 'Test/test_labels.csv', 2000, rng)

    manifest = {
        'dataset_id': 'hisarmod_26class',
        'group_namespace': 'hisarmod',
        'provenance': ('HisarMod 2019, 26-class modulation (cross-dataset). Subsample train 4000/class, '
                       'val 500/class, test 2000/class. Source CSV complex strings parsed to [I,Q]. '
                       'sample_rate_hz varies (fs column); recorded null. groups=per-sample.'),
        'sample_rate_hz': None, 'center_frequency_hz': None,
        'tasks': {'modulation': {'kind': 'classification', 'outputs': len(uniq),
                                 'classes': [str(u) for u in uniq]}},
        'splits': {},
    }
    for split, (X, Y) in [('train', (Xtr, Ytr)), ('val', (Xva, Yva)), ('test', (Xte, Yte))]:
        n = len(X)
        ids = np.array([f'{split}_{i}' for i in range(n)], dtype=str)
        grp = np.array([f'{split}_g{i}' for i in range(n)], dtype=str)
        np.save(OUT / f'{split}_x.npy', X)
        np.save(OUT / f'{split}_y.npy', Y)
        np.save(OUT / f'{split}_ids.npy', ids)
        np.save(OUT / f'{split}_groups.npy', grp)
        manifest['splits'][split] = dict(x=f'{split}_x.npy', ids=f'{split}_ids.npy',
                                         groups=f'{split}_groups.npy', targets={'modulation': f'{split}_y.npy'})
        print(f'[{split}] {n} samples, class dist {np.bincount(Y, minlength=len(uniq))[:8]}...')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print('[hisarmod] done ->', OUT / 'manifest.json')


if __name__ == '__main__':
    main()
