"""Multi-source pretraining reader for data/pretrain_multi_v1 (100万 train + 10万 val).

Samples are stored pre-normalized (max_abs over valid points, float32 in [-1,1])
with a per-sample valid mask for zero-padded short sources. This reader indexes
the jsonl manifest, lazily opens one h5 handle per source, and verifies each
sample's normalized-content sha256 on access (same integrity rule as the single
source PretrainIQ). snr/fs are NaN/0 where the source lacks them.
"""
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

EXPECTED_VERSION = 'multisource_pretrain_v1'
EXPECTED_NORM = 'per_sample_max_abs_over_valid, epsilon=1e-8, stored pre-normalized f32'


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class MultiSourceDataset(Dataset):
    def __init__(self, data_dir, split, verify_hash=True):
        data_dir = Path(data_dir)
        bundle = json.loads((data_dir / 'dataset.json').read_text())
        if bundle['version'] != EXPECTED_VERSION or bundle['normalization'] != EXPECTED_NORM:
            raise ValueError('Unsupported multi-source data contract')
        if split not in ('train', 'val'):
            raise ValueError("split must be 'train' or 'val'")
        self.data_dir = data_dir
        self.split = split
        self.verify_hash = verify_hash
        self.rows = [json.loads(line) for line in (data_dir / f'{split}.jsonl').read_text().splitlines()]
        if len(self.rows) != bundle['counts'][split]:
            raise ValueError(f'Manifest count mismatch: {split}')
        self.sources = sorted({r['source_id'] for r in self.rows})
        for s in self.sources:
            if not (data_dir / f'{s}_{split}.h5').exists():
                raise ValueError(f'Missing h5 for {s}/{split}')
        self._files = {}
        self._pid = None

    def __len__(self):
        return len(self.rows)

    def _open(self, source_id):
        if self._pid != os.getpid() or not self._files:
            self.close()
            self._files = {}
            self._pid = os.getpid()
        if source_id not in self._files:
            self._files[source_id] = h5py.File(
                self.data_dir / f'{source_id}_{self.split}.h5', 'r')
        return self._files[source_id]

    def close(self):
        for f in self._files.values():
            f.close()
        self._files = {}
        self._pid = None

    def __getitem__(self, index):
        row = self.rows[index]
        f = self._open(row['source_id'])
        iq = np.asarray(f['iq'][row['sample_id']], dtype=np.float32)
        valid = np.asarray(f['valid'][row['sample_id']], dtype=bool)
        if iq.shape != (1024, 2) or valid.shape != (1024,):
            raise ValueError(f'Bad sample shape for {row}')
        if self.verify_hash:
            a = np.asarray(iq, dtype='<f4')
            a[a == 0] = 0
            if sha256_bytes(a.tobytes()) != row['normalized_iq_sha256']:
                raise ValueError(f'Content changed: {row["source_id"]}/{row["sample_id"]}')
        snr = row['snr']
        return {'iq': torch.from_numpy(iq), 'valid_mask': torch.from_numpy(valid),
                'snr': float('nan') if snr is None else float(snr),
                'fs_hz': 0.0 if row['fs_hz'] is None else float(row['fs_hz']),
                'source_id': row['source_id'], 'sample_id': row['sample_id'],
                'length': row['length']}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_files'] = {}
        state['_pid'] = None
        return state


def collate_multisource(records):
    iq = torch.stack([r['iq'] for r in records])
    valid = torch.stack([r['valid_mask'] for r in records])
    return {'iq': iq, 'valid_mask': valid,
            'snr': torch.tensor([r['snr'] for r in records], dtype=torch.float32),
            'fs_hz': torch.tensor([r['fs_hz'] for r in records], dtype=torch.float32),
            'fs_valid': torch.tensor([r['fs_hz'] != 0.0 for r in records], dtype=torch.bool),
            'sample_id': torch.tensor([r['sample_id'] for r in records]),
            'length': torch.tensor([r['length'] for r in records])}
