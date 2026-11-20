"""SERVER: multi-source large-scale pretraining dataset builder (100万 train + 10万 val).

Constructs a self-supervised masked-IQ pretraining corpus from 8 RF sources. No
class labels are read for training -- only raw IQ, per-sample fs/snr metadata and
source provenance. Sources shorter than 1024 are zero-padded (valid_mask records
the true length); sources longer are split into independent 1024 segments.

Output layout (per source, train/val in separate files):
    data/pretrain_multi_v1/<SourceID>_{train,val}.h5   iq [N,1024,2] f32 + valid [N,1024] bool
    data/pretrain_multi_v1/{train,val}.jsonl            one JSON record per sample
    data/pretrain_multi_v1/dataset.json                 frozen decisions + audit

Key decisions (frozen in dataset.json):
  - normalization: per-sample max_abs over VALID points only (epsilon 1e-8), stored
    pre-normalized as float32 in [-1,1]; pad area is excluded from the max and is zero.
  - dedup: sha256 of the normalized float32 IQ bytes (negative zero canonicalized),
    enforced within-source, cross-source, and across train/val.
  - isolation: file-level -- train/val draw from disjoint source files/shards.
    RML2018A (single unsplit hdf5) excludes all cls_dev downstream holdouts and
    isolates train/val by normalized-content hash (same rule as prepare_pretrain.py).
  - DroneRFa: slice 150M-point recordings into 1024 segments, drop near-silent
    segments (max_abs < 1e-6), file-level train/val split (30/3 files, background/
    drone kept in natural ratio). fs left null (official 40MHz not re-verified).
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FROZEN_CLS_SHA = '3c76d726ef0b141f213b2267f8b5933210f80776fe458d1aefd63bc9006f246d'
DATASET_ROOT = Path('/data1/user_wzk/datasets')
LEN = 1024
EPS = 1e-8
SEED = 2024

# Per-source (train, val) targets, summing to 1_000_000 / 100_000.
ALLOC = {
    'DroneRFa': (400_000, 40_000),
    'RML2018A': (250_000, 25_000),
    'RadChar': (150_000, 15_000),
    'Hisarmod': (100_000, 10_000),
    'BSS': (50_000, 5_000),
    'PanoradioHF': (30_000, 3_000),
    'RML2016_10B': (10_000, 1_000),
    'RadarComm': (10_000, 1_000),
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def content_key(x: np.ndarray) -> str:
    """Hash of a single normalized float32 IQ array (negative zero canonicalized)."""
    a = np.asarray(x, dtype='<f4')
    a[a == 0] = 0
    return sha256_bytes(a.tobytes())


def normalize(x: np.ndarray, valid: np.ndarray):
    """max_abs over VALID points; x[N,L,2], valid[N,L]. Returns normalized, scale[N]."""
    mag = np.abs(x.astype(np.float32))
    mag = np.where(valid[..., None], mag, 0.0)
    scale = mag.reshape(len(x), -1).max(axis=1)
    scale = np.maximum(scale, EPS)
    out = x.astype(np.float32) / scale[:, None, None]
    out[out == 0] = 0  # canonicalize negative zero
    return out, scale


def valid_from_lengths(lengths: np.ndarray) -> np.ndarray:
    v = np.zeros((len(lengths), LEN), dtype=bool)
    for i, n in enumerate(lengths):
        v[i, :int(n)] = True
    return v


def write_h5(path: Path, iq: np.ndarray, valid: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(iq)
    chunk = (min(128, max(1, n)), LEN, 2)
    with h5py.File(path, 'w') as f:
        f.create_dataset('iq', data=iq.astype('<f4'), dtype='<f4', chunks=chunk)
        f.create_dataset('valid', data=valid, dtype=bool, chunks=(chunk[0], LEN))


def read_parquet_iq(path, rows=None):
    """Read the 'iq' list column as [N, L, 2] float32. rows=None reads all."""
    tbl = pq.read_table(str(path), columns=['iq'])
    if rows is not None:
        tbl = tbl.take(rows)
    col = tbl['iq'].combine_chunks()
    arr = col if isinstance(col, pa.Array) else col.chunk(0)
    if arr.null_count != 0:
        raise ValueError(f'Null IQ entries in {path}')
    n = len(arr)
    if n == 0:
        return np.empty((0, 0, 2), dtype=np.float32)
    vals = np.asarray(arr.values.values.to_numpy(), dtype=np.float32)  # flattened [n*L*2]
    if vals.shape[0] % (n * 2) != 0:
        raise ValueError(f'Ragged/invalid iq in {path}')
    L = vals.shape[0] // (n * 2)
    return vals.reshape(n, L, 2)


def read_parquet_meta(path, keys):
    tbl = pq.read_table(str(path), columns=keys)
    return {k: tbl[k].to_pylist() for k in keys}


def sample_rows(n_total, n_target, rng):
    """Indices of n_target (or all if fewer) rows, uniformly without replacement."""
    n = min(n_total, n_target)
    return rng.choice(n_total, size=n, replace=False)


def dedup_select(iq, valid, metas, n_target, seen, rng):
    """Sample up to n_target distinct (w.r.t. seen) samples; returns arrays + metas."""
    n = len(iq)
    order = rng.permutation(n)
    picked_iq, picked_valid, picked_meta, picked_key = [], [], [], []
    for idx in order:
        key = content_key(iq[idx])
        if key in seen:
            continue
        seen.add(key)
        picked_iq.append(iq[idx]); picked_valid.append(valid[idx])
        picked_meta.append(metas[idx]); picked_key.append(key)
        if len(picked_iq) == n_target:
            break
    if len(picked_iq) < n_target:
        raise ValueError(f'Insufficient distinct candidates: {len(picked_iq)}/{n_target}')
    return (np.asarray(picked_iq, dtype='<f4'), np.asarray(picked_valid, dtype=bool),
            picked_meta, picked_key)


# --------------------------------------------------------------------------- #
# Per-source extractors: return (iq[N,LEN,2] float32 raw, valid, meta list).
# meta: dict(snr, fs_hz, source_index, length)
# --------------------------------------------------------------------------- #

def extract_radchar(which, n_target, rng):
    if which == 'train':
        shards = [DATASET_ROOT / 'Radchar' / f'radchar_train_{i:04d}.parquet' for i in range(18)]
    else:
        shards = [DATASET_ROOT / 'Radchar' / f'radchar_val_{i:04d}.parquet' for i in range(2)]
    # Read just enough shards to cover n_target + margin.
    per_shard = 100_000
    n_shards = min(len(shards), math.ceil(n_target / per_shard) + 1)
    iqs, metas = [], []
    for shard in shards[:n_shards]:
        arr = read_parquet_iq(shard)                       # [100k, 512, 2]
        meta = read_parquet_meta(shard, ['fs', 'signal_to_noise_ratio', 'index'])
        for i in range(len(arr)):
            metas.append({'snr': float(meta['signal_to_noise_ratio'][i]),
                          'fs_hz': int(meta['fs'][i]),
                          'source_index': int(meta['index'][i]), 'length': arr.shape[1]})
        iqs.append(arr)
    iq = np.concatenate(iqs, axis=0)
    return iq, metas


def extract_bss(which, n_target, rng):
    shards = sorted((DATASET_ROOT / 'EM-Radar-mixed').glob(f'BSS_{which}_*.parquet'))
    iqs, metas = [], []
    for shard in shards:
        arr = read_parquet_iq(shard)                       # [N, 1024, 2]
        meta = read_parquet_meta(shard, ['fs', 'snr'])
        for i in range(len(arr)):
            metas.append({'snr': float(meta['snr'][i]), 'fs_hz': float(meta['fs'][i]),
                          'source_index': len(metas), 'length': arr.shape[1]})
        iqs.append(arr)
    return np.concatenate(iqs, axis=0), metas


def extract_simple(path, which, source_index_name, fs_fmt=int):
    arr = read_parquet_iq(path)                             # [N, L, 2]
    meta = read_parquet_meta(path, ['fs', 'snr'])
    metas = [{'snr': float(meta['snr'][i]), 'fs_hz': fs_fmt(meta['fs'][i]),
              'source_index': i, 'length': arr.shape[1]} for i in range(len(arr))]
    return arr, metas


def extract_hisarmod(which, n_target, rng):
    path = DATASET_ROOT / 'Hisarmod_2019' / 'HisarMod2019.1' / (
        'Hisarmod_train.parquet' if which == 'train' else 'Hisarmod_validation.parquet')
    return extract_simple(path, which, 'row', fs_fmt=int)


def extract_panoradio(which, n_target, rng):
    path = DATASET_ROOT / 'PanoradioHF' / (
        'PanoradioHF_train.parquet' if which == 'train' else 'PanoradioHF_validation.parquet')
    arr, metas = extract_simple(path, which, 'row', fs_fmt=int)   # [N, 2048, 2]
    # split each 2048 sample into two independent 1024 segments
    n = len(arr)
    seg = arr.reshape(n * 2, LEN, 2)
    new_metas = []
    for i in range(n):
        for seg_idx in range(2):
            new_metas.append({'snr': metas[i]['snr'], 'fs_hz': metas[i]['fs_hz'],
                              'source_index': metas[i]['source_index'] * 2 + seg_idx,
                              'length': LEN})
    return seg, new_metas


def extract_rml2016b(which, n_target, rng):
    shards = sorted((DATASET_ROOT / 'rml2016.10B' / 'RML2016B' / which).glob('part-*-rml2016_10b.parquet'))
    arr = read_parquet_iq(shards[0])                        # [102400, 128, 2]
    meta = read_parquet_meta(shards[0], ['fs', 'snr'])
    metas = [{'snr': float(meta['snr'][i]), 'fs_hz': int(meta['fs'][i]),
              'source_index': i, 'length': arr.shape[1]} for i in range(len(arr))]
    return arr, metas


def extract_radarcomm(which, n_target, rng):
    path = DATASET_ROOT / 'RadarCommDataset' / (
        'RadarCommDataset_train.parquet' if which == 'train' else 'RadarCommDataset_validation.parquet')
    return extract_simple(path, which, 'row', fs_fmt=int)


def extract_rml2018a(which, n_target, rng, cls_manifest, seen):
    """Exclude all cls_dev downstream samples; content-hash isolate train/val.

    Mirrors prepare_pretrain.py: pretrain val excludes every downstream sample
    (ID and normalized content); pretrain train excludes downstream val/test IDs
    and content but MAY reuse downstream train samples.
    """
    rows = [r for r in csv.DictReader(open(cls_manifest, newline=''))
            if r['subset'] == 'cls_dev']
    by_id = {int(r['sample_id']): r for r in rows}
    cls_all = set(by_id)
    cls_hold = {sid for sid, r in by_id.items() if r['split'] != 'train'}
    forbidden_ids = cls_all if which == 'val' else cls_hold
    h5_path = DATASET_ROOT / 'RML2018A' / 'GOLD_XYZ_OSC.0001_1024.hdf5'
    picked_iq, picked_valid, picked_meta, picked_key = [], [], [], []
    with h5py.File(h5_path, 'r') as f:
        n = f['X'].shape[0]
        # Pre-hash all downstream samples so pretrain never reuses their content.
        all_keys, hold_keys = set(), set()
        for start in range(0, len(rows), 512):
            b = sorted(int(r['sample_id']) for r in rows[start:start + 512])
            xs = np.asarray(f['X'][b], dtype=np.float32)
            for sid, x in zip(b, xs):
                norm, _ = normalize(x[None], np.ones((1, LEN), dtype=bool))
                k = content_key(norm[0])
                all_keys.add(k)
                if sid in cls_hold:
                    hold_keys.add(k)
        forbidden_keys = all_keys if which == 'val' else hold_keys
        order = rng.permutation(n)
        for start in range(0, n, 512):
            ids = order[start:start + 512]
            keep = [int(sid) for sid in ids if int(sid) not in forbidden_ids]
            if not keep:
                continue
            unique, inverse = np.unique(keep, return_inverse=True)
            xs = np.asarray(f['X'][unique], dtype=np.float32)[inverse]
            zs = np.asarray(f['Z'][unique]).reshape(-1)[inverse]
            for sid, x, z in zip(keep, xs, zs):
                norm, _ = normalize(x[None], np.ones((1, LEN), dtype=bool))
                key = content_key(norm[0])
                if key in forbidden_keys or key in seen:
                    continue
                seen.add(key)
                picked_iq.append(norm[0]); picked_valid.append(np.ones(LEN, dtype=bool))
                picked_meta.append({'snr': float(z), 'fs_hz': None,
                                    'source_index': int(sid), 'length': LEN})
                picked_key.append(key)
                if len(picked_iq) == n_target:
                    return (np.asarray(picked_iq, dtype='<f4'),
                            np.asarray(picked_valid, dtype=bool), picked_meta, picked_key)
    raise ValueError(f'Insufficient distinct RML2018A candidates for {which}')


def extract_dronerfa(which, n_target, rng, seen):
    """Slice 150M-point .mat files into 1024 segments; file-level isolation.

    Background/drone files are kept in their natural ~70/30 ratio. Each
    (file, channel) unit contributes at most ceil(n_target/n_units) segments so
    samples spread across recordings instead of concentrating in the first file.
    """
    drone_dir = DATASET_ROOT / 'DroneRFa'
    files = sorted(drone_dir.glob('*.mat'))
    bg, dr = [], []
    for p in files:
        (bg if '_D' in p.stem else dr).append(p)  # background T0000_Dxx_Sxxxx vs drone Txxxxx_Sxxxx
    if which == 'train':
        chosen = list(rng.choice(bg, size=21, replace=False)) + list(rng.choice(dr, size=9, replace=False))
    else:
        chosen = list(rng.choice(bg, size=2, replace=False)) + list(rng.choice(dr, size=1, replace=False))
    units = [(p, ch) for p in chosen for ch in ('RF0', 'RF1')]
    per_unit = math.ceil(n_target / len(units))
    picked_iq, picked_valid, picked_meta, picked_key = [], [], [], []
    for p, ch in units:
        with h5py.File(p, 'r') as f:
            i = np.asarray(f[f'{ch}_I']).ravel()
            q = np.asarray(f[f'{ch}_Q']).ravel()
            iq = np.stack([i, q], axis=1)                 # [npts, 2] f64
            nseg = iq.shape[0] // LEN
            if nseg == 0:
                continue
            seg = iq[:nseg * LEN].reshape(nseg, LEN, 2).astype(np.float32)
            valid = np.ones((nseg, LEN), dtype=bool)
            norm, scale = normalize(seg, valid)
            order = rng.permutation(nseg)
            quota = min(per_unit, n_target - len(picked_iq))
            got = 0
            for idx in order:
                if got >= quota:
                    break
                if scale[idx] < 1e-6:  # drop near-silent segments
                    continue
                key = content_key(norm[idx])
                if key in seen:
                    continue
                seen.add(key)
                picked_iq.append(norm[idx]); picked_valid.append(valid[idx])
                picked_meta.append({'snr': None, 'fs_hz': None,
                                    'source_index': len(picked_iq), 'length': LEN})
                picked_key.append(key)
                got += 1
            if len(picked_iq) >= n_target:
                break
    if len(picked_iq) < n_target:
        raise ValueError(f'Insufficient distinct DroneRFa segments for {which}: {len(picked_iq)}/{n_target}')
    return (np.asarray(picked_iq, dtype='<f4'), np.asarray(picked_valid, dtype=bool),
            picked_meta, picked_key)


EXTRACTORS = {
    'RadChar': extract_radchar,
    'BSS': extract_bss,
    'Hisarmod': extract_hisarmod,
    'PanoradioHF': extract_panoradio,
    'RML2016_10B': extract_rml2016b,
    'RadarComm': extract_radarcomm,
}


def pad_or_assert(iq, metas):
    """Zero-pad sources shorter than LEN; assert others already LEN."""
    L = iq.shape[1]
    if L == LEN:
        return iq, valid_from_lengths(np.array([m['length'] for m in metas]))
    if L > LEN:
        raise ValueError(f'Source longer than {LEN}: {L}')
    padded = np.zeros((len(iq), LEN, 2), dtype=np.float32)
    padded[:, :L] = iq
    return padded, valid_from_lengths(np.array([m['length'] for m in metas]))


def build_source(source_id, which, n_target, rng, seen, cls_manifest, out_dir):
    """Extract, pad, normalize, dedup -> write h5, return manifest rows."""
    if source_id == 'DroneRFa':
        iq, valid, metas, keys = extract_dronerfa(which, n_target, rng, seen)
    elif source_id == 'RML2018A':
        iq, valid, metas, keys = extract_rml2018a(which, n_target, rng, cls_manifest, seen)
    else:
        raw, metas = EXTRACTORS[source_id](which, n_target, rng)
        raw, valid = pad_or_assert(raw, metas)
        iq, scale = normalize(raw, valid)
        iq, valid, metas, keys = dedup_select(iq, valid, metas, n_target, seen, rng)
    write_h5(out_dir / f'{source_id}_{which}.h5', iq, valid)
    rows = []
    for i, m in enumerate(metas):
        rows.append({'source_id': source_id, 'sample_id': i,
                     'source_index': m['source_index'], 'length': int(m['length']),
                     'fs_hz': m['fs_hz'], 'snr': m['snr'],
                     'normalized_iq_sha256': keys[i]})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', help='comma-separated source ids (default all)')
    p.add_argument('--output', default=str(Path(__file__).resolve().parents[2] / 'data' / 'pretrain_multi_v1'))
    p.add_argument('--cls-manifest', default=str(Path(__file__).resolve().parents[2] / 'manifest' / 'rml2018a_manifest.csv'))
    p.add_argument('--scale', type=float, default=1.0, help='fraction of targets (smoke=small)')
    p.add_argument('--seed', type=int, default=SEED)
    a = p.parse_args()
    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if file_sha256(Path(a.cls_manifest)) != FROZEN_CLS_SHA:
        raise ValueError('Classification manifest hash mismatch')
    sources = [s for s in (a.sources.split(',') if a.sources else ALLOC)]
    missing = [s for s in sources if s not in ALLOC]
    if missing:
        raise ValueError(f'Unknown sources: {missing}')

    rng = np.random.default_rng(a.seed)
    seen = set()  # global normalized-content dedup across sources and splits
    manifest = {'train': [], 'val': []}
    audit = {}
    for which in ('val', 'train'):
        for source_id in sources:
            n_target = max(1, int(round(ALLOC[source_id][0 if which == 'train' else 1] * a.scale)))
            rows = build_source(source_id, which, n_target, rng, seen, a.cls_manifest, out_dir)
            manifest[which].extend(rows)
            audit.setdefault(source_id, {})[which] = len(rows)
            print(f'{source_id} {which}: {len(rows)}', flush=True)

    for which in ('train', 'val'):
        path = out_dir / f'{which}.jsonl'
        with path.open('w') as f:
            for row in manifest[which]:
                f.write(json.dumps(row, sort_keys=True) + '\n')

    total_train = sum(audit[s].get('train', 0) for s in sources)
    total_val = sum(audit[s].get('val', 0) for s in sources)
    # cross-split content isolation check
    t_keys = {r['normalized_iq_sha256'] for r in manifest['train']}
    v_keys = {r['normalized_iq_sha256'] for r in manifest['val']}
    overlap = len(t_keys & v_keys)
    if overlap:
        raise ValueError(f'Train/val content overlap: {overlap}')
    bundle = {
        'version': 'multisource_pretrain_v1',
        'seed': a.seed,
        'normalization': 'per_sample_max_abs_over_valid, epsilon=1e-8, stored pre-normalized f32',
        'sequence_length': LEN,
        'scale': a.scale,
        'targets': ALLOC,
        'sources': {},
        'files': {},
        'dedup': 'sha256 of normalized f32 IQ bytes, within/cross-source and train/val',
        'isolation': 'file-level for all sources; RML2018A content-hash isolated, cls_dev excluded',
        'fs_hz': 'per-sample from parquet; null where source lacks verified fs (RML2018A, DroneRFa)',
        'counts': {'train': total_train, 'val': total_val, 'per_source': audit},
        'overlap_train_val': overlap,
    }
    for source_id in sources:
        for which in ('train', 'val'):
            h5 = out_dir / f'{source_id}_{which}.h5'
            jl = out_dir / f'{which}.jsonl'
            bundle['files'][f'{source_id}_{which}.h5'] = {'bytes': h5.stat().st_size,
                                                          'sha256': file_sha256(h5)}
        bundle['sources'][source_id] = {'target': ALLOC[source_id], 'actual': audit.get(source_id)}
    bundle['files']['train.jsonl'] = {'bytes': (out_dir / 'train.jsonl').stat().st_size,
                                      'sha256': file_sha256(out_dir / 'train.jsonl')}
    bundle['files']['val.jsonl'] = {'bytes': (out_dir / 'val.jsonl').stat().st_size,
                                    'sha256': file_sha256(out_dir / 'val.jsonl')}
    (out_dir / 'dataset.json').write_text(json.dumps(bundle, indent=2) + '\n')
    print(json.dumps({'output': str(out_dir), 'counts': bundle['counts'],
                      'overlap_train_val': overlap}, indent=2))


if __name__ == '__main__':
    main()
