#!/usr/bin/env python
"""Freeze the final-test protocol: record code/config/checkpoint hashes + eval spec.

The test split is the independent hold-out; it was never used for checkpoint
selection (best.pt is selected on clean val OA, earliest tie) and is not used here
for any tuning or re-selection. This script only records a tamper-evident manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
SEEDS = (17, 42, 2026)
VARIANTS = ('G', 'G_corr', 'S1', 'S1_corr')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256()
    for p in sorted((ROOT / 'ssm_exp').glob('*.py')):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='reports/final_test/freeze_manifest.json')
    a = p.parse_args()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    for v in VARIANTS:
        for s in SEEDS:
            d = ROOT / 'runs' / 'round3_aug' / f'{v}_seed{s}'
            cfg = json.loads((d / 'config.json').read_text())
            bv = json.loads((d / 'best_val.json').read_text())
            runs.append({
                'run': str(d.relative_to(ROOT)),
                'variant': cfg['model']['variant'],
                'seed': cfg['train']['seed'],
                'protocol': cfg['protocol']['name'],
                'config_sha256': sha256_file(d / 'config.json'),
                'best_pt_sha256': sha256_file(d / 'best.pt'),
                'last_pt_sha256': sha256_file(d / 'last.pt'),
                'best_epoch': bv['best_epoch'],
                'selection': bv['selection'],
                'clean_val_oa': bv['metrics']['overall']['oa'],
                'clean_val_f1': bv['metrics']['overall']['macro_f1'],
            })

    manifest = {
        'freeze_note': ('Independent test evaluation. No re-training, no test-based tuning, '
                        'no checkpoint re-selection. best.pt is selected on clean val OA (earliest tie).'),
        'eval_protocol': {
            'order': 'B',  # raw IQ -> inject CFO -> max_abs normalize -> recompute desc -> model
            'manifest': 'configs/data_raw.json',
            'manifest_sha256': sha256_file(ROOT / 'configs' / 'data_raw.json'),
            'split': 'test',
            'test_count': 8112,
            'checkpoint': 'best.pt',
            'grid': [0.0, -0.001, 0.001, -0.002, 0.002, -0.005, 0.005, -0.01, 0.01, -0.02, 0.02],
        },
        'source_sha256': code_hash(),
        'eval_script_sha256': sha256_file(ROOT / 'scripts' / 'eval_extended_cfo.py'),
        'runs': runs,
    }
    out.write_text(json.dumps(manifest, indent=2))
    print(f'wrote {out}')
    print(f'source_sha256: {manifest["source_sha256"]}')
    print(f'{len(runs)} runs frozen')


if __name__ == '__main__':
    main()
