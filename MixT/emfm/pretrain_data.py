"""IQ-only pretraining reader. Y is never read by this training dataset."""
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import sha256
from .prepare_pretrain import content_hash


def load_bundle(path):
    path = Path(path)
    bundle = json.loads(path.read_text())
    if bundle['version'] != 'rml2018a_pretrain_v1' or bundle['normalization'] != 'max_abs':
        raise ValueError('Unsupported pretraining data contract')
    rows = {}
    for key in ('train_smoke', 'train_dev', 'val_smoke', 'val_dev'):
        info = bundle['files'][key]
        file = (path.parent / info['file']).resolve()
        if file.parent != path.parent.resolve() or sha256(file) != info['sha256']:
            raise ValueError(f'Manifest path/hash mismatch: {key}')
        rows[key] = [json.loads(line) for line in file.read_text().splitlines()]
        ids = [r['sample_id'] for r in rows[key]]
        if len(ids) != info['count'] or len(set(ids)) != len(ids) or not ids:
            raise ValueError(f'Invalid count/duplicates: {key}')
    for small, big in [('train_smoke','train_dev'), ('val_smoke','val_dev')]:
        if rows[small] != rows[big][:len(rows[small])]:
            raise ValueError('Subsets must be exact ordered prefixes')
    for field in ('sample_id', 'normalized_iq_sha256'):
        a = {r[field] for r in rows['train_dev']}
        b = {r[field] for r in rows['val_dev']}
        if a & b or len(a) != len(rows['train_dev']) or len(b) != len(rows['val_dev']):
            raise ValueError(f'Pretrain split conflict: {field}')
    return bundle, rows


class PretrainIQ(Dataset):
    def __init__(self, h5_path, rows, cache=False):
        self.path, self.rows = str(h5_path), rows
        self._file = None
        self._pid = None
        self.cached = None
        self.scales = None
        if cache:
            self.cached = torch.empty(len(rows), 1024, 2, dtype=torch.float32)
            self.scales = torch.empty(len(rows))
            with h5py.File(self.path, 'r') as f:
                for start in range(0, len(rows), 512):
                    records = rows[start:start+512]
                    ids, inverse = np.unique([r['sample_id'] for r in records], return_inverse=True)
                    xs = np.asarray(f['X'][ids], dtype=np.float32)[inverse]
                    for j, (x, r) in enumerate(zip(xs, records)):
                        normalized, scale = self.convert(x, r)
                        self.cached[start+j] = normalized
                        self.scales[start+j] = scale

    @staticmethod
    def convert(x, row):
        if x.shape != (1024, 2) or content_hash(x) != row['normalized_iq_sha256']:
            raise ValueError(f'IQ shape/content changed: sample {row["sample_id"]}')
        scale = max(float(np.abs(x).max()), 1e-8)
        return torch.from_numpy(np.asarray(x / scale, dtype=np.float32)), scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.cached is not None:
            x, scale = self.cached[i], float(self.scales[i])
        else:
            if self._file is None or self._pid != os.getpid():
                self.close()
                self._file = h5py.File(self.path, 'r')
                self._pid = os.getpid()
            x, scale = self.convert(np.asarray(self._file['X'][r['sample_id']], dtype=np.float32), r)
        return {'iq': x, 'sample_id': r['sample_id'], 'snr': r['snr'], 'scale': scale}

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file, self._pid = None, None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_file=None, _pid=None)
        return state


def collate_pretrain(records):
    iq = torch.stack([r['iq'] for r in records])
    return {'iq': iq, 'valid_mask': torch.ones(iq.shape[:2], dtype=torch.bool),
            'sample_id': torch.tensor([r['sample_id'] for r in records]),
            'snr': torch.tensor([r['snr'] for r in records]),
            'scale': torch.tensor([r['scale'] for r in records])}
