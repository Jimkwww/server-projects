"""Section 5: rebuild F1/F2 corpora with stratified sampling + eligibility_v2.csv.

CRITICAL FIX vs round-2 f_eligibility.py: RML2018A GOLD_XYZ_OSC is ordered modulation-major
(mod 0..23, each a 106496-row block) with SNR-minor inside (26 levels x 4096). An index-order
scan to 100k therefore yields modulation-0 ONLY. F1 is rebuilt here as a (mod,SNR)-stratified
sample of 100k, always including all F0 (cls_dev_train).

F0 = cls_dev_train (19968, 24 mods x 26 SNR x 32). F1 = 100k stratified RML minus cls_dev
val/test, deduped by IQ hash, all F0 included. F2 = 50k comm (stratified subset of F1) +
50k OOD (EM-Radar-mixed BSS_train_0000, first 50k rows).

Outputs:
  round3_scale_transfer/data/f1_indices.npy            (100k RML row indices, int64)
  round3_scale_transfer/data/f2_comm_indices.npy       (50k RML row indices, subset of F1)
  round3_scale_transfer/data/f2_comm_flags.npy         (50k booleans: which F1 rows are in F2)
  /data1/user_wzk/datasets/EM-Radar-mixed/ood_iq_50k.npy  (50000,1024,2 float32)
  round3_scale_transfer/data/emradar_ood_source_ids.npy (source ids for the 50k OOD rows)
  round3_scale_transfer/reports/eligibility_v2.csv
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
R3 = ROOT / 'round3_scale_transfer'
H5 = '/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5'
EMR = '/data1/user_wzk/datasets/EM-Radar-mixed'
N_MOD = 24
N_SNR = 26
SNR_LO = -20
CELL = N_SNR * 4096  # 106496 rows per modulation block
CAP_F1 = 100000
CAP_F2_COMM = 50000
CAP_F2_OOD = 50000
SEED = 2026  # deterministic sampling seed


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def build_f1():
    train = np.load(ROOT / 'data' / 'cls_dev_train.npy')
    val = np.load(ROOT / 'data' / 'cls_dev_val.npy')
    test = np.load(ROOT / 'data' / 'cls_dev_test.npy')
    f0_set = set(int(i) for i in train)
    held_out = set(int(i) for i in val) | set(int(i) for i in test)

    # held-out IQ hashes for exclusion (and F0 hashes for reporting / within-pool dedup)
    held_hash = set()
    f0_hash = set()
    with h5py.File(H5, 'r') as f:
        X = f['X']
        held_sorted = np.array(sorted(held_out), dtype=np.int64)
        Xh = X[held_sorted]
        held_hash = {sha(Xh[j]) for j in range(len(held_sorted))}
        f0_sorted = np.array(sorted(f0_set), dtype=np.int64)
        Xf = X[f0_sorted]
        f0_hash = {sha(Xf[j]) for j in range(len(f0_sorted))}

    rng = np.random.default_rng(SEED)
    base = CAP_F1 // (N_MOD * N_SNR)
    rem = CAP_F1 - base * (N_MOD * N_SNR)

    selected = []          # F1 indices (all F0 + sampled additional)
    f1_hashes = set(f0_hash)
    per_cell = {}
    with h5py.File(H5, 'r') as f:
        X = f['X']
        cell_id = 0
        for mod in range(N_MOD):
            blk = mod * CELL
            for snr in range(N_SNR):
                start = blk + snr * 4096
                idx = np.arange(start, start + 4096, dtype=np.int64)
                in_f0 = np.array([int(i) in f0_set for i in idx])
                in_ho = np.array([int(i) in held_out for i in idx])
                f0_cell = idx[in_f0]
                # additional candidates: not F0, not held-out
                cand = idx[~(in_f0 | in_ho)]
                target = base + (1 if cell_id < rem else 0)
                # F0 first (dedup within F0 cell)
                cell_sel = []
                cell_hash = set()
                xb = X[f0_cell]
                for j, i in enumerate(f0_cell):
                    h = sha(xb[j])
                    if h in cell_hash:
                        continue
                    cell_hash.add(h)
                    cell_sel.append(int(i))
                # sample additional up to target, dedup vs held-out + F0 + within-pool
                need = target - len(cell_sel)
                if need > 0:
                    order = rng.permutation(len(cand))
                    xb = X[cand]
                    taken = 0
                    for j in order:
                        h = sha(xb[j])
                        if h in held_hash or h in f1_hashes or h in cell_hash:
                            continue
                        cell_hash.add(h)
                        f1_hashes.add(h)
                        cell_sel.append(int(cand[j]))
                        taken += 1
                        if taken >= need:
                            break
                per_cell[cell_id] = {'mod': mod, 'snr': SNR_LO + 2 * snr,
                                     'f0': len(f0_cell), 'selected': len(cell_sel),
                                     'target': target}
                selected.extend(cell_sel)
                cell_id += 1
    selected = np.array(sorted(selected), dtype=np.int64)
    return selected, per_cell, len(f0_set), len(held_out), len(f1_hashes)


def build_f2_comm(f1_idx):
    """Stratified 50k subset of F1 (same (mod,snr) cells)."""
    with h5py.File(H5, 'r') as f:
        Z = f['Z'][:].ravel()
        Y = f['Y'][:].argmax(1)
    rng = np.random.default_rng(SEED + 1)
    base = CAP_F2_COMM // (N_MOD * N_SNR)
    rem = CAP_F2_COMM - base * (N_MOD * N_SNR)
    sel = []
    cell_id = 0
    for mod in range(N_MOD):
        blk = mod * CELL
        for snr in range(N_SNR):
            start = blk + snr * 4096
            end = start + 4096
            # F1 rows in this cell
            cell_idx = f1_idx[(f1_idx >= start) & (f1_idx < end)]
            target = base + (1 if cell_id < rem else 0)
            target = min(target, len(cell_idx))
            if target > 0:
                sel.extend(rng.choice(cell_idx, size=target, replace=False).tolist())
            cell_id += 1
    return np.array(sorted(sel), dtype=np.int64)


def extract_ood():
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(EMR + '/BSS_train_0000.parquet')
    tbl = pf.read_row_group(0)
    df = tbl.slice(0, CAP_F2_OOD).to_pandas()
    iq = np.array([np.stack(row) for row in df['iq'].values], dtype=np.float32)
    labels = df['label'].values
    # source id = file name + row index (no explicit group-id column in schema)
    src_ids = np.array([f'BSS_train_0000:r{i}' for i in range(CAP_F2_OOD)])
    return iq, labels, src_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-f1', action='store_true')
    ap.add_argument('--skip-ood', action='store_true')
    args = ap.parse_args()

    data_dir = R3 / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # F1
    f1_path = data_dir / 'f1_indices.npy'
    if not args.skip_f1 and not f1_path.exists():
        print('building F1 (stratified 100k)...', flush=True)
        f1_idx, per_cell, n_f0, n_ho, n_unique = build_f1()
        np.save(f1_path, f1_idx)
        with h5py.File(H5, 'r') as f:
            Xb = f['X'][f1_idx]
            f1_hash = hashlib.sha256(
                b''.join(sha(Xb[j]).encode() for j in range(len(f1_idx)))).hexdigest()
        meta = {'n_total': int(len(f1_idx)), 'n_unique_iq': n_unique,
                'n_f0': n_f0, 'n_held_out': n_ho,
                'per_cell': per_cell, 'sha256': f1_hash, 'seed': SEED}
        (data_dir / 'f1_meta.json').write_text(json.dumps(meta, indent=2))
        print(f'F1 built: {len(f1_idx)} rows, hash {f1_hash[:16]}...', flush=True)
    else:
        f1_idx = np.load(f1_path)

    # F2 comm (50k stratified subset of F1)
    f2c_path = data_dir / 'f2_comm_indices.npy'
    if not args.skip_f1 and not f2c_path.exists():
        print('building F2 comm subset (50k from F1)...', flush=True)
        f2c = build_f2_comm(f1_idx)
        np.save(f2c_path, f2c)
        f1_set = set(int(i) for i in f1_idx)
        flags = np.array([int(i) in set(int(i) for i in f2c) for i in f1_idx], dtype=bool)
        np.save(data_dir / 'f2_comm_flags.npy', flags)
        with h5py.File(H5, 'r') as f:
            Xb = f['X'][f2c]
            f2c_hash = hashlib.sha256(
                b''.join(sha(Xb[j]).encode() for j in range(len(f2c)))).hexdigest()
        (data_dir / 'f2_comm_meta.json').write_text(
            json.dumps({'n_total': int(len(f2c)), 'sha256': f2c_hash, 'seed': SEED + 1}))
        print(f'F2 comm built: {len(f2c)} rows', flush=True)
    else:
        f2c = np.load(f2c_path)

    # F2 OOD (50k EM-Radar rows -> /data1)
    ood_path = Path(EMR) / 'ood_iq_50k.npy'
    if not args.skip_ood and not ood_path.exists():
        print('extracting F2 OOD (50k EM-Radar rows)...', flush=True)
        iq, labels, src_ids = extract_ood()
        np.save(ood_path, iq)
        np.save(data_dir / 'emradar_ood_source_ids.npy', src_ids)
        ood_hash = hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()
        (data_dir / 'f2_ood_meta.json').write_text(json.dumps({
            'n_total': int(len(iq)), 'shape': list(iq.shape), 'dtype': str(iq.dtype),
            'sha256': ood_hash, 'source_file': 'BSS_train_0000.parquet',
            'label_counts': {str(k): int(v) for k, v in
                             zip(*np.unique(labels, return_counts=True))}}))
        print(f'F2 OOD extracted: {iq.shape} -> {ood_path}', flush=True)
    else:
        ood_meta = json.loads((data_dir / 'f2_ood_meta.json').read_text()) \
            if (data_dir / 'f2_ood_meta.json').exists() else {}

    # eligibility_v2.csv
    f1_meta = json.loads((data_dir / 'f1_meta.json').read_text())
    f2c_meta = json.loads((data_dir / 'f2_comm_meta.json').read_text())
    ood_meta = json.loads((data_dir / 'f2_ood_meta.json').read_text())
    rows = [
        {'name': 'F0', 'pool': 'RML2018A cls_dev train', 'status': '可用',
         'reason': '第一轮 RML train（24 调制 x 26 SNR x 32，分层平衡）',
         'independent_records': f1_meta['n_f0'], 'preprocessing': 'none (raw IQ)',
         'sha256': '(见 f1_meta per_cell；F0 全量并入 F1)'},
        {'name': 'F1', 'pool': 'RML2018A 全量 - cls_dev val/test（(mod,SNR) 分层，含全部 F0）',
         'status': '可用',
         'reason': '同域扩大；修正 round-2 按索引顺序扫描导致的 mod-0-only 缺陷，改 (mod,SNR) 分层',
         'independent_records': f"{f1_meta['n_total']} 行 / {f1_meta['n_unique_iq']} 唯一 IQ", 'preprocessing': 'none (raw IQ)',
         'sha256': f1_meta['sha256']},
        {'name': 'F2', 'pool': '50k comm(F1 分层子集) + 50k OOD(EM-Radar-mixed)',
         'status': '可用',
         'reason': '异域 IQ 合格（EM-Radar 为雷达+通信混合复 IQ，5MHz/16dB/1024 点，36 标签；'
                   'RadChar 已整体留出用于迁移评估，二者长度不同 512 vs 1024，无样本重叠）',
         'independent_records': f"{f2c_meta['n_total']} + {ood_meta['n_total']}",
         'preprocessing': 'comm: none(raw IQ float32)；ood: float64->float32，max_abs_norm in-loop',
         'sha256': f"comm={f2c_meta['sha256'][:16]}... ood={ood_meta['sha256'][:16]}..."},
    ]
    out = R3 / 'reports' / 'eligibility_v2.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ['name', 'pool', 'status', 'reason', 'independent_records', 'preprocessing', 'sha256']
    with out.open('w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('wrote', out, flush=True)
    print(json.dumps({'f1': f1_meta['n_total'], 'f2_comm': f2c_meta['n_total'],
                      'f2_ood': ood_meta.get('n_total')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
