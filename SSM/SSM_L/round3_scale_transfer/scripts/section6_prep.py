"""Section 6 prep: materialize F1/F2 pretrain manifests.

F1 = 100k stratified RML rows (f1_indices.npy) -> manifest pointing at the RML h5 with the F1
train split; val = cls_dev_val (monitoring only). No new array needed.

F2 = 50k comm (RML f2_comm_indices) + 50k OOD (EM-Radar ood_iq_50k) -> one npy array
[103120,1024,2] float32 = 100k train + 3120 RML val appended (single-file manifest, dummy
labels/SNR since MAE ignores them). Written to /data1 to keep /home free.

Outputs:
  round3_scale_transfer/configs/manifest_f1.json
  round3_scale_transfer/configs/manifest_f2.json
  /data1/user_wzk/datasets/SSM_L_round3/f2_x.npy, f2_y.npy, f2_snr.npy
  round3_scale_transfer/data/f2_train_idx.npy, f2_val_idx.npy
"""
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
R3 = ROOT / 'round3_scale_transfer'
H5 = '/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5'
EMR_OOD = '/data1/user_wzk/datasets/EM-Radar-mixed/ood_iq_50k.npy'
DST = Path('/data1/user_wzk/datasets/SSM_L_round3')


def main():
    data_dir = R3 / 'data'
    cfg_dir = R3 / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    DST.mkdir(parents=True, exist_ok=True)

    # F1 manifest (train = f1_indices into RML h5, val = cls_dev_val)
    f1_manifest = {
        'format': 'h5', 'file': H5, 'x_key': 'X', 'y_key': 'Y', 'snr_key': 'Z',
        'layout': 'NLC', 'label_format': 'one_hot', 'normalization': 'none', 'classes': 24,
        'splits': {'train': str(data_dir / 'f1_indices.npy'),
                   'val': str(ROOT / 'data' / 'cls_dev_val.npy')},
    }
    (cfg_dir / 'manifest_f1.json').write_text(json.dumps(f1_manifest, indent=2))

    # F2 materialization
    f2c = np.sort(np.load(data_dir / 'f2_comm_indices.npy'))  # 50k RML comm
    ood = np.load(EMR_OOD)                                    # 50k OOD float32 [1024,2]
    assert ood.shape == (50000, 1024, 2), ood.shape
    val_idx = np.sort(np.load(ROOT / 'data' / 'cls_dev_val.npy'))  # 3120 RML val (monitoring)
    with h5py.File(H5, 'r') as f:
        comm = f['X'][f2c].astype(np.float32)                 # [50000,1024,2]
        val_x = f['X'][val_idx].astype(np.float32)            # [3120,1024,2]
    x = np.concatenate([comm, ood, val_x], axis=0)            # [103120,1024,2]
    y = np.zeros((x.shape[0], 1), dtype=np.int64)             # dummy (MAE ignores labels)
    snr = np.zeros((x.shape[0], 1), dtype=np.float32)
    np.save(DST / 'f2_x.npy', x)
    np.save(DST / 'f2_y.npy', y)
    np.save(DST / 'f2_snr.npy', snr)
    train_idx = np.arange(0, 100000, dtype=np.int64)
    val_f2_idx = np.arange(100000, x.shape[0], dtype=np.int64)
    np.save(data_dir / 'f2_train_idx.npy', train_idx)
    np.save(data_dir / 'f2_val_idx.npy', val_f2_idx)

    f2_manifest = {
        'format': 'npy', 'x': str(DST / 'f2_x.npy'), 'y': str(DST / 'f2_y.npy'),
        'snr': str(DST / 'f2_snr.npy'), 'layout': 'NLC', 'label_format': 'integer',
        'normalization': 'none', 'classes': 24,
        'splits': {'train': str(data_dir / 'f2_train_idx.npy'),
                   'val': str(data_dir / 'f2_val_idx.npy')},
    }
    (cfg_dir / 'manifest_f2.json').write_text(json.dumps(f2_manifest, indent=2))

    print('F2 x shape', x.shape, 'dtype', x.dtype, '->', DST / 'f2_x.npy')
    print('F1 manifest ->', cfg_dir / 'manifest_f1.json')
    print('F2 manifest ->', cfg_dir / 'manifest_f2.json')


if __name__ == '__main__':
    main()
