"""Manifest-selected RML2018A, process-local HDF5 handles, explicit IQ scaling."""
import csv
import hashlib
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RML2018A(Dataset):
    def __init__(self, h5_path, manifest, subset, split, normalization="max_abs"):
        self.path = str(h5_path)
        self.normalization = normalization
        if normalization not in ("max_abs", "none"):
            raise ValueError("normalization must be max_abs or none")
        with open(manifest, newline="") as f:
            self.rows = [r for r in csv.DictReader(f)
                         if r["subset"] == subset and r["split"] == split]
        if not self.rows:
            raise ValueError(f"Empty selection: {subset}/{split}")
        ids = [int(r["sample_id"]) for r in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate sample IDs in selection")
        self._file = None
        self._pid = None

    def __len__(self):
        return len(self.rows)

    def _open(self):
        if self._pid != os.getpid() or self._file is None:
            self.close()
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_file=None, _pid=None)
        return state

    def __getitem__(self, index):
        row = self.rows[index]
        sid = int(row["sample_id"])
        f = self._open()
        x = np.asarray(f["X"][sid], dtype=np.float32)
        y = np.asarray(f["Y"][sid])
        snr = float(np.asarray(f["Z"][sid]).reshape(-1)[0])
        label = int(row["class"])
        if x.ndim != 2 or x.shape[1] != 2 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError(f"Invalid IQ at {sid}")
        if not (np.all((y == 0) | (y == 1)) and y.sum() == 1):
            raise ValueError(f"Invalid one-hot at {sid}")
        if int(y.argmax()) != label or snr != float(row["snr"]):
            raise ValueError(f"Manifest/source mismatch at {sid}")
        # Joint max absolute COMPONENT, not max complex-envelope magnitude.
        scale = max(float(np.abs(x).max()), 1e-8) if self.normalization == "max_abs" else 1.0
        return {"iq": torch.from_numpy(x / scale), "label": label,
                "snr": snr, "sample_id": sid, "scale": scale}


def collate(records):
    lengths = torch.tensor([len(r["iq"]) for r in records], dtype=torch.long)
    x = torch.zeros(len(records), int(lengths.max()), 2)
    for i, r in enumerate(records):
        x[i, :lengths[i]] = r["iq"]
    valid = torch.arange(x.shape[1])[None, :] < lengths[:, None]
    return {"iq": x, "valid_mask": valid, "lengths": lengths,
            "label": torch.tensor([r["label"] for r in records], dtype=torch.long),
            "snr": torch.tensor([r["snr"] for r in records]),
            "sample_id": torch.tensor([r["sample_id"] for r in records]),
            "scale": torch.tensor([r["scale"] for r in records]),
            "fs_hz": torch.zeros(len(records)),
            "fs_valid": torch.zeros(len(records), dtype=torch.bool)}
