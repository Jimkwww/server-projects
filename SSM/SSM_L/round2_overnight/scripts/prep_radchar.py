"""E. Preprocess RadChar-Tiny into the RML-compatible manifest format.

RadChar-Tiny: 50k synthetic radar waveforms, 512 complex baseband IQ samples @3.2MHz, 5 signal
types (coherent_pulse_train=0, barker=1, polyphase_barker=2, frank=3, LFM=4), per-waveform
params (n_pulses/pulse_width/PRI/time_delay) + integer SNR (-20..20dB). No official split, so we
create a stratified-by-class random split (fixed seed), dedup by IQ hash, and write X/Y/Z in the
same NLC one-hot layout the RML manifest uses. The encoder runs length-agnostically (64 patches).

Outputs (h5 on /data1 to keep /home free):
  /data1/user_wzk/datasets/Radchar/radchar_tiny_rml.h5        X[N,512,2], Y[N,5], Z[N,1]
  round2_overnight/data/radchar_{train,dev,eval}.npy          split index arrays
  round2_overnight/configs/radchar.json                        manifest
  round2_overnight/configs/radar_probe.json                    probe config (5 classes, L=512)
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
R2 = Path(__file__).resolve().parent.parent
SRC = '/data1/user_wzk/datasets/Radchar/RadChar-Tiny.h5'
DST = '/data1/user_wzk/datasets/Radchar/radchar_tiny_rml.h5'
CLASS_NAMES = ['coherent_pulse_train', 'barker_code', 'polyphase_barker_code',
               'frank_code', 'linear_frequency_modulated']
N_CLASSES = 5


def main():
    with h5py.File(SRC, 'r') as f:
        iq = f['iq'][:]          # (N,512) complex128
        labels = f['labels'][:]  # structured
    n = len(iq)
    print(f'loaded {n} waveforms')

    signal_type = labels['signal_type'].astype(np.int64)
    snr = labels['signal_to_noise_ratio'].astype(np.float64)
    # class balance
    vals, counts = np.unique(signal_type, return_counts=True)
    print('class counts:', dict(zip(vals.tolist(), counts.tolist())))
    assert set(signal_type.tolist()) == set(range(N_CLASSES))

    # dedup by IQ hash
    hashes = np.array([hashlib.sha256(iq[i].tobytes()).hexdigest() for i in range(n)])
    _, uniq_idx, uniq_counts = np.unique(hashes, return_index=True, return_counts=True)
    dups = int((uniq_counts > 1).sum())
    print(f'unique waveforms: {len(uniq_idx)}, duplicated hash groups: {dups}')

    # Stratified split 60/20/20 by signal_type, fixed seed, on unique indices.
    rng = np.random.default_rng(0)
    train, dev, eval_ = [], [], []
    for c in range(N_CLASSES):
        idx = np.where(signal_type == c)[0]
        rng.shuffle(idx)
        nd = int(round(len(idx) * 0.2))
        ne = int(round(len(idx) * 0.2))
        dev.extend(idx[:nd])
        eval_.extend(idx[nd:nd + ne])
        train.extend(idx[nd + ne:])
    train = np.array(sorted(train)); dev = np.array(sorted(dev)); eval_ = np.array(sorted(eval_))
    print(f'split train/dev/eval = {len(train)}/{len(dev)}/{len(eval_)}')

    # Write X/Y/Z (NLC layout, one-hot Y).
    X = np.stack((iq.real, iq.imag), axis=-1).astype(np.float32)  # (N,512,2)
    Y = np.zeros((n, N_CLASSES), dtype=np.int64)
    Y[np.arange(n), signal_type] = 1
    Z = snr[:, None].astype(np.float32)
    with h5py.File(DST, 'w') as f:
        f.create_dataset('X', data=X, chunks=(1024, 512, 2))
        f.create_dataset('Y', data=Y)
        f.create_dataset('Z', data=Z)

    data_dir = R2 / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in [('radchar_train', train), ('radchar_dev', dev), ('radchar_eval', eval_)]:
        np.save(data_dir / f'{name}.npy', idx)

    manifest = {
        'format': 'h5', 'file': DST, 'x_key': 'X', 'y_key': 'Y', 'snr_key': 'Z',
        'layout': 'NLC', 'label_format': 'one_hot', 'normalization': 'none',
        'classes': N_CLASSES,
        'splits': {
            'train': str(data_dir / 'radchar_train.npy'),
            'val': str(data_dir / 'radchar_dev.npy'),
            'test': str(data_dir / 'radchar_eval.npy'),
        },
        'class_names': CLASS_NAMES,
    }
    cfg_dir = R2 / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / 'radchar.json').write_text(json.dumps(manifest, indent=2))

    # Radar probe config (copy probe.json, 5 classes, L=512).
    probe = json.loads((ROOT / 'configs' / 'probe.json').read_text())
    probe['model']['classes'] = N_CLASSES
    probe['model']['input_length'] = 512
    probe['protocol']['name'] = 'RadChar_cls_ssml_r2_transfer'
    (cfg_dir / 'radar_probe.json').write_text(json.dumps(probe, indent=2))

    print(f'wrote {DST}')
    print(f'wrote {cfg_dir}/radchar.json + radar_probe.json')


if __name__ == '__main__':
    main()
