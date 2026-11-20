"""TECHRec (LTE / WiFi / DVB-T technology recognition) -> emmem npy manifest.

Real captured IQ: .bin files of interleaved int16 IQ at 1 MHz (10Msps variants excluded to
keep a single sample rate). Each .bin is one capture record; we slice it into 1024-length
examples. Split by CAPTURE LOCATION (train: merelbeke/rabot/gentbrugge/igent; val: uz;
test: reep) so test measures cross-location generalization — same capture record never
crosses a split (group = file).

Note: sample_rate_hz=1e6 for the 1Msps files (from the dataset's MATLAB read script);
center_frequency_hz varies per file (not provided per-file) -> recorded as null with the
frequency range noted in provenance.
"""
import json
import os
from pathlib import Path

import numpy as np

SRC = Path('/data1/user_wzk/datasets/TECHRec/Technology_Recognition_(LTE,_Wi-Fi_and_DVB-T)')
OUT = Path('/home/user_wzk/wzk/projects/SSM/SSM_EM/em_memory_suite/data/techrec')

LENGTH = 1024
MAX_SLICES_PER_FILE = 512
SPLIT_BY_LOCATION = {
    'train': ['merelbeke', 'rabot', 'igent', 'uz'],
    'val': ['gentbrugge'],
    'test': ['reep'],
}
CLASSES = ['dvbt', 'lte', 'wf']  # alphabetical; labels = index


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # collect 1Msps files only (exclude *10Msps*)
    files = []
    for loc in SRC.iterdir():
        if not loc.is_dir() or loc.name == 'Processing_.bin_files_and_add_SNR':
            continue
        for f in loc.glob('*.bin'):
            if '10Msps' in f.name:
                continue
            tech = f.name.split('_')[0]
            if tech not in CLASSES:
                continue
            files.append((loc.name, tech, f))

    splits = {s: [] for s in SPLIT_BY_LOCATION}
    for loc, tech, f in files:
        for s, locs in SPLIT_BY_LOCATION.items():
            if loc in locs:
                splits[s].append((loc, tech, f))
                break

    manifest = {
        'dataset_id': 'techrec_3class',
        'group_namespace': 'techrec_capture',
        'provenance': ('TECHRec (Technology Recognition: LTE/WiFi/DVB-T), real captured IQ, 1 MHz '
                       '(10Msps files excluded). 1024-length slices, up to 512 per capture record. '
                       'Split by capture location (train merelbeke/rabot/gentbrugge/igent, val uz, '
                       'test reep) -> cross-location generalization. group = capture file. '
                       'sample_rate_hz=1e6; center_frequency_hz varies per file (unavailable per-file).'),
        'sample_rate_hz': 1e6,
        'center_frequency_hz': None,
        'tasks': {'technology': {'kind': 'classification', 'outputs': len(CLASSES), 'classes': CLASSES}},
        'splits': {},
    }

    for split, fl in splits.items():
        X = []
        Y = []
        IDS = []
        GRP = []
        n_kept = 0
        for loc, tech, f in fl:
            raw = np.fromfile(f, dtype=np.int16).astype(np.float32)
            n_c = len(raw) // 2  # complex samples
            raw = raw[:2 * n_c].reshape(-1, 2)  # [I,Q] interleaved
            # normalize int16 -> ~[-1,1]
            raw = raw / 32768.0
            n_slices = min(MAX_SLICES_PER_FILE, n_c // LENGTH)
            for i in range(n_slices):
                seg = raw[i * LENGTH:(i + 1) * LENGTH]
                X.append(seg)
                Y.append(CLASSES.index(tech))
                IDS.append(f'{loc}_{tech}_{f.stem}_s{i}')
                GRP.append(f'{loc}_{f.stem}')
                n_kept += 1
        X = np.stack(X).astype(np.float32)
        Y = np.array(Y, dtype=np.int64)
        IDS = np.array(IDS, dtype=str)
        GRP = np.array(GRP, dtype=str)
        np.save(OUT / f'{split}_x.npy', X)
        np.save(OUT / f'{split}_y.npy', Y)
        np.save(OUT / f'{split}_ids.npy', IDS)
        np.save(OUT / f'{split}_groups.npy', GRP)
        manifest['splits'][split] = dict(x=f'{split}_x.npy', ids=f'{split}_ids.npy',
                                         groups=f'{split}_groups.npy',
                                         targets={'technology': f'{split}_y.npy'})
        print(f'[{split}] {len(fl)} files -> {n_kept} slices (class dist: '
              f'{np.bincount(Y, minlength=3).tolist()})')

    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print('[techrec] done ->', OUT / 'manifest.json')


if __name__ == '__main__':
    main()
