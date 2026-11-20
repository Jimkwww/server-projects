"""Generate extended campaigns (native_width / scale / data / long_pretrain) per GPU,
with batch_size=128 (matching the main matrix) and the 7 selected variants split across
GPU1/GPU2. Mirrors tools/campaigns.py but skips the 'main' campaign (already running) and
splits variants across two devices.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emmem.plan import build
from emmem.train import dump
SELECTED = ['structured', 'structured_equiv', 'generic_ssm', 'generic_ssm_recon',
            'generic_transformer_recon', 'multi_iq', 'pointwise_physical']

BASE = json.loads((ROOT / 'configs/registry.server.json').read_text())
BASE['batch_size'] = 128


def split_selected(device):
    # core variants (5 seeds) vs others (3 seeds): balance by seed count
    CORE = {'structured', 'structured_equiv', 'generic_ssm', 'generic_ssm_recon',
            'generic_transformer_recon', 'pointwise_physical'}
    def sc(v):
        return 5 if v in CORE else 3
    target = sum(sc(v) for v in SELECTED) / 2
    g1, g2, acc = [], [], 0
    for v in SELECTED:
        if acc + sc(v) <= target + 1e-9:
            g1.append(v)
            acc += sc(v)
        else:
            g2.append(v)
    return (g1, g2)[device == 'cuda:2']


def main():
    for gpu in ['cuda:1', 'cuda:2']:
        vs = split_selected(gpu)
        base = dict(BASE)
        base['device'] = gpu
        base['variants'] = vs
        root = ROOT / f'results/extended_{gpu.replace(":", "")}'
        campaigns = [('native_width', dict(base, parameter_mode='native', fractions=[1.0]))]
        for size, dim, depth in [('small', 32, 2), ('medium', 128, 6), ('large', 256, 12)]:
            r = dict(base)
            r.update(variants=vs, model_overrides=dict(dim=dim, states=dim // 2, depth=depth),
                     fractions=[1.0], parameter_mode='native')
            campaigns.append(('scale_' + size, r))
        for frac in [0.1, 0.3]:
            r = dict(base)
            r.update(variants=vs, pretrain_fraction=frac, fractions=[1.0])
            campaigns.append(('data_' + str(frac), r))
            r2 = dict(r)
            r2['pretrain_epochs'] = round(base.get('pretrain_epochs', 100) / frac)
            campaigns.append(('data_equal_updates_' + str(frac), r2))
        r = dict(base)
        r.update(variants=vs, pretrain_epochs=300, fractions=[1.0])
        campaigns.append(('long_pretrain', r))

        index = []
        for name, r in campaigns:
            path = root / name
            registry = path / 'registry.json'
            dump(registry, r)
            count = build(registry, path / 'plan')
            index.append(dict(name=name, jobs=count, plan=str(path / 'plan' / 'plan.json')))
        dump(root / 'campaign_index.json', index)
        total = sum(i['jobs'] for i in index)
        print(f'[{gpu}] {len(campaigns)} campaigns, {len(vs)} variants, total {total} jobs: '
              f'{[i["jobs"] for i in index]}')


if __name__ == '__main__':
    main()
