"""Convert RML2018A + RadChar into emmem npy-manifest format.

Reuses SSM_L's verified splits (cls_dev / f1 / radchar); does NOT re-derive splits.
Groups degenerate to per-sample unique IDs (both datasets are independently simulated
samples with no shared source-capture grouping) — recorded explicitly in provenance.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
SL = Path('/home/user_wzk/wzk/projects/SSM/SSM_L')
RML = '/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5'
RAD = '/data1/user_wzk/datasets/Radchar/radchar_tiny_rml.h5'

PY = [sys.executable, '-m', 'tools.convert_rml_h5']


def run(cmd):
    print('+', ' '.join(cmd[:6]), '...', flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    DATA.mkdir(exist_ok=True)
    # 1. groups
    np.save(DATA / 'rml2018_groups.npy', np.arange(2555904, dtype=np.int64))
    np.save(DATA / 'radchar_groups.npy', np.arange(50000, dtype=np.int64))

    # 2. splits JSON (absolute paths)
    splits = {
        'amc': {'train': str(SL / 'data/cls_dev_train.npy'),
                'val': str(SL / 'data/cls_dev_val.npy'),
                'test': str(SL / 'data/cls_dev_test.npy')},
        'f1': {'train': str(SL / 'round3_scale_transfer/data/f1_indices.npy'),
               'val': str(SL / 'data/cls_dev_val.npy')},
        'radchar': {'train': str(SL / 'round2_overnight/data/radchar_train.npy'),
                    'val': str(SL / 'round2_overnight/data/radchar_dev.npy'),
                    'test': str(SL / 'round2_overnight/data/radchar_eval.npy')},
    }
    for k, v in splits.items():
        (ROOT / 'server_prep' / f'splits_{k}.json').write_text(json.dumps(v, indent=2))

    # 3. convert (absolute --groups/--splits/--out)
    prov_amc = ('RML2018A (Deepsig 2018.01a) 24-class AMC. Splits reused from SSM_L cls_dev '
                '(train 19968/val 3120/test 8112), (mod,SNR) stratified. Samples independently '
                'simulated; no shared source-capture grouping -> groups=per-sample unique id. '
                'normalization=none (raw IQ, amplitude retained).')
    prov_f1 = ('RML2018A F1 pretrain corpus (100k, (mod,SNR) stratified) reused from SSM_L round3 '
               'f1_indices. Verified no overlap with cls_dev val/test. Samples independently '
               'simulated; groups=per-sample unique id. normalization=none.')
    prov_rad = ('RadChar-Tiny radar waveforms (5-class) reused from SSM_L round2 split '
                '(train 30000/dev 10000/eval 10000). Samples independently simulated; '
                'groups=per-sample unique id. normalization=none.')
    for h5, spl, grp, out, did, prov in [
        (RML, 'amc', 'rml2018_groups.npy', 'rml2018_clsdev', 'rml2018a_clsdev', prov_amc),
        (RML, 'f1', 'rml2018_groups.npy', 'rml2018_f1', 'rml2018a_f1', prov_f1),
        (RAD, 'radchar', 'radchar_groups.npy', 'radchar_tiny', 'radchar_tiny', prov_rad),
    ]:
        run(PY + ['--h5', h5, '--splits', str(ROOT / 'server_prep' / f'splits_{spl}.json'),
                  '--groups', str(DATA / grp), '--out', str(DATA / out),
                  '--dataset-id', did, '--provenance', prov])
    print('[prep] ALL DATA CONVERTED')


if __name__ == '__main__':
    main()
