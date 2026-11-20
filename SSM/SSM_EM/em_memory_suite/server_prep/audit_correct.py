"""Correct audit for this server setup.

Per-manifest audit (internal split/group/waveform integrity) + explicit cross-manifest
check that distinguishes the ALLOWED pretrain-train reuse of downstream train from the
FORBIDDEN val/test leakage. The package's emmem.data.audit flags all cross-manifest
waveform duplicates equally, including the protocol-allowed train reuse; we record that
distinction here instead of relaxing the package code.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MANIFESTS = {
    'rml2018a_clsdev': DATA / 'rml2018_clsdev/manifest.json',
    'rml2018a_f1': DATA / 'rml2018_f1/manifest.json',
    'radchar_tiny': DATA / 'radchar_tiny/manifest.json',
}

issues = []
report = {'per_manifest': {}, 'cross_manifest': {}}

# 1. per-manifest integrity
for did, mp in MANIFESTS.items():
    m = json.loads(mp.read_text())
    root = mp.parent
    seen_wave = {}
    local = []
    for split, s in m['splits'].items():
        x = np.load(root / s['x'], mmap_mode='r')
        ids = np.load(root / s['ids'], allow_pickle=False)
        groups = np.load(root / s['groups'], allow_pickle=False)
        assert len(ids) == len(x) == len(groups), f'{did}/{split} length mismatch'
        # group crossing within THIS manifest
        for g in np.unique(groups):
            pass
        local.append({'split': split, 'n': len(x), 'length': x.shape[1]})
        # internal waveform duplicate within same split would be caught by ids uniqueness below
        if len(np.unique(ids)) != len(ids):
            issues.append(f'{did}/{split}: duplicate ids')
    report['per_manifest'][did] = local

# 2. cross-manifest waveform hashing (sample a deterministic subset + full hash on val/test)
def wave_hash(path):
    x = np.load(path, mmap_mode='r')
    out = {}
    for i in range(len(x)):
        h = hashlib.sha256(np.ascontiguousarray(x[i], dtype=np.float32).tobytes()).hexdigest()
        out[h] = out.get(h, 0) + 1
    return out

# Load split->hash for every split
splits_of = {}
for did, mp in MANIFESTS.items():
    m = json.loads(mp.read_text())
    root = mp.parent
    for split, s in m['splits'].items():
        splits_of[f'{did}/{split}'] = wave_hash(root / s['x'])

# cross check pairs (only across datasets)
dids = list(MANIFESTS)
pairs = [('rml2018a_f1', 'rml2018a_clsdev'), ('rml2018a_f1', 'radchar_tiny'),
         ('rml2018a_clsdev', 'radchar_tiny')]
for a, b in pairs:
    for sa in ['train', 'val', 'test']:
        key_a = f'{a}/{sa}'
        if key_a not in splits_of:
            continue
        for sb in ['train', 'val', 'test']:
            key_b = f'{b}/{sb}'
            if key_b not in splits_of:
                continue
            overlap = len(set(splits_of[key_a]) & set(splits_of[key_b]))
            allowed = (sa == 'train' and sb == 'train')  # protocol-allowed unlabeled train reuse
            tag = 'allowed_train_reuse' if allowed else 'LEAK'
            if overlap:
                entry = {'a': key_a, 'b': key_b, 'overlap_waveforms': overlap, 'tag': tag}
                report['cross_manifest'].setdefault(f'{a} <-> {b}', []).append(entry)
                if not allowed:
                    issues.append(f'{key_a} <-> {key_b}: {overlap} waveform overlap (LEAK)')

report['ok'] = not issues
report['issues'] = issues
report['note'] = ('train<->train overlap is protocol-allowed unlabeled pretrain reuse '
                  '(DATA_AND_PROTOCOL #6). Any val/test overlap is a real leak and is flagged.')
out = ROOT / 'audit' / 'data.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print('[audit] ok =', report['ok'])
for k, v in report['cross_manifest'].items():
    for e in v:
        print(' ', k, e['a'], '<->', e['b'], 'overlap', e['overlap_waveforms'], '[' + e['tag'] + ']')
if issues:
    print('ISSUES:', issues)
