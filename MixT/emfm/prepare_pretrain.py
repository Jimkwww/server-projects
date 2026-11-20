"""SERVER: deterministic nested RML2018A pretraining manifests; original data read-only.

Pretraining validation excludes ALL cls_dev samples/content. Pretraining training
excludes downstream val/test and pretraining validation content. Hash scope is
selected candidates and downstream holdouts, not the entire source dataset.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

FROZEN_CLS_SHA = '3c76d726ef0b141f213b2267f8b5933210f80776fe458d1aefd63bc9006f246d'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(x):
    # Same normalization as training. Detect raw-identical AND exactly equal
    # normalized float32 inputs. Does not prove absence of overlapping captures.
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError('Invalid IQ candidate')
    a = np.array(a / max(float(np.abs(a).max()), 1e-8), dtype='<f4', copy=True)
    a[a == 0] = 0  # canonicalize negative zero
    return hashlib.sha256(a.tobytes()).hexdigest()


def read_selected(f, ids):
    ids = np.asarray(ids, dtype=np.int64)
    unique, inverse = np.unique(ids, return_inverse=True)
    x = np.asarray(f['X'][unique], dtype=np.float32)[inverse]
    y = np.asarray(f['Y'][unique])[inverse]
    z = np.asarray(f['Z'][unique]).reshape(-1)[inverse]
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError('Nonfinite source data')
    if not (np.isin(y, [0, 1]).all() and (y.sum(1) == 1).all()):
        raise ValueError('Invalid source one-hot labels (audit only)')
    return x, y.argmax(1), z


def prepare(h5_path, cls_manifest, output, train_size=100000, val_size=10000,
            smoke_size=10000, smoke_val_size=2000, seed=2024, expected_cls_sha=FROZEN_CLS_SHA):
    if not (0 < smoke_size <= train_size and 0 < smoke_val_size <= val_size):
        raise ValueError('Invalid subset sizes')
    if file_hash(cls_manifest) != expected_cls_sha:
        raise ValueError('Classification manifest hash mismatch')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    with open(cls_manifest, newline='') as stream:
        cls_rows = [r for r in csv.DictReader(stream) if r['subset'] == 'cls_dev']
    cls = {int(r['sample_id']): r for r in cls_rows}
    if len(cls) != len(cls_rows) or {r['split'] for r in cls_rows} != {'train', 'val', 'test'}:
        raise ValueError('Classification manifest invalid')
    cls_all, cls_hold = set(cls), {sid for sid, r in cls.items() if r['split'] != 'train'}
    stats = Counter()
    with h5py.File(h5_path, 'r') as f:
        n, length, channels = f['X'].shape
        if length != 1024 or channels != 2 or f['Y'].shape[0] != n or f['Z'].shape[0] != n:
            raise ValueError('Expected RML2018A X[N,1024,2], aligned Y/Z')
        schema = {key: {'shape': list(f[key].shape), 'dtype': str(f[key].dtype)} for key in ('X','Y','Z')}
        if min(cls_all) < 0 or max(cls_all) >= n:
            raise ValueError('Classification index out of bounds')
        all_hash, hold_hash = set(), set()
        ids = sorted(cls_all)
        for start in range(0, len(ids), 512):
            batch = ids[start:start+512]
            xs, ys, zs = read_selected(f, batch)
            for sid, x, y, z in zip(batch, xs, ys, zs):
                if int(y) != int(cls[sid]['class']) or float(z) != float(cls[sid]['snr']):
                    raise ValueError(f'Classification/source mismatch: {sid}')
                digest = content_hash(x)
                all_hash.add(digest)
                if sid in cls_hold:
                    hold_hash.add(digest)
        print(f'Hashed {len(cls_all)} downstream records; holdout IDs={len(cls_hold)}', flush=True)
        order = np.random.default_rng(seed).permutation(n)
        def candidates():
            for start in range(0, n, 512):
                batch = order[start:start+512]
                xs, ys, zs = read_selected(f, batch)
                for sid, x, y, z in zip(batch, xs, ys, zs):
                    stats['candidates_examined'] += 1
                    yield int(sid), content_hash(x), int(y), float(z)
        stream = candidates()
        val, train, val_hash, train_hash = [], [], set(), set()
        distributions = {'train': Counter(), 'val': Counter()}
        # Validation chosen first and isolated from downstream TRAIN as well.
        for split, target, records, selected_hash, forbidden_ids, forbidden_hash in (
            ('val', val_size, val, val_hash, cls_all, all_hash),
            ('train', train_size, train, train_hash, cls_hold, hold_hash),
        ):
            for sid, digest, label, snr in stream:
                if sid in forbidden_ids or digest in forbidden_hash or digest in selected_hash or (split == 'train' and digest in val_hash):
                    stats[f'{split}_rejected_for_overlap'] += 1
                    continue
                selected_hash.add(digest)
                records.append({'sample_id': sid, 'source_id': 'RML2018A', 'length': length,
                                'snr': snr, 'fs_hz': None, 'normalized_iq_sha256': digest})
                distributions[split][f'class={label},snr={snr}'] += 1
                if len(records) % 10000 == 0:
                    print(f'{split}: {len(records)}/{target}', flush=True)
                if len(records) == target:
                    break
            if len(records) != target:
                raise ValueError(f'Insufficient unique candidates for {split}: {len(records)}/{target}')
    files = {}
    for key, records in [('train_smoke', train[:smoke_size]), ('train_dev', train),
                         ('val_smoke', val[:smoke_val_size]), ('val_dev', val)]:
        filename = key + '.jsonl'
        with (out/filename).open('w') as stream:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
        files[key] = {'file': filename, 'count': len(records), 'sha256': file_hash(out/filename)}
    source = Path(h5_path).resolve()
    bundle = {'version': 'rml2018a_pretrain_v1', 'seed': seed, 'normalization': 'max_abs',
              'source': {'path': str(source), 'bytes': source.stat().st_size, 'schema': schema},
              'classification_manifest_sha256': expected_cls_sha, 'files': files,
              'overlap': {'pretrain_train_downstream_train_ids': len({r['sample_id'] for r in train} & cls_all),
                          'pretrain_train_downstream_val_test_ids': 0, 'pretrain_val_downstream_all_ids': 0,
                          'pretrain_train_val_normalized_exact_duplicates': 0},
              'audit': dict(stats), 'strata': {k: dict(v) for k,v in distributions.items()},
              'sampling': 'uniform without replacement, fixed seed; class/SNR for audit only',
              'limits': 'Hashes cover downstream cls_dev and examined candidates only; no capture IDs, no near-duplicate guarantee.'}
    # Completion marker written only after all checks/manifests succeed.
    (out/'dataset.json').write_text(json.dumps(bundle, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'files': files, 'overlap': bundle['overlap']}, indent=2))
    return bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h5', required=True)
    p.add_argument('--cls-manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--train-size', type=int, default=100000)
    p.add_argument('--val-size', type=int, default=10000)
    p.add_argument('--smoke-size', type=int, default=10000)
    p.add_argument('--smoke-val-size', type=int, default=2000)
    p.add_argument('--seed', type=int, default=2024)
    a = p.parse_args()
    prepare(a.h5, a.cls_manifest, a.output, a.train_size, a.val_size, a.smoke_size, a.smoke_val_size, a.seed)


if __name__ == '__main__':
    main()
