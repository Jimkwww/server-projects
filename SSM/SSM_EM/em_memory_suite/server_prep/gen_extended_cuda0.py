"""Generate extended campaigns for GPU0 night-shift (device cuda:0, all 7 selected variants,
batch 128). Mirrors tools/campaigns.py (no 'main' campaign). Used by night_scheduler.sh.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emmem.plan import build
from emmem.train import dump

SELECTED = ['structured', 'structured_equiv', 'generic_ssm', 'generic_ssm_recon',
            'generic_transformer_recon', 'multi_iq', 'pointwise_physical']


def main():
    base = json.loads((ROOT / 'configs/registry.server.json').read_text())
    base['batch_size'] = 128
    base['device'] = 'cuda:0'
    base['variants'] = SELECTED

    root = ROOT / 'results/extended_cuda0'
    campaigns = [('native_width', dict(base, parameter_mode='native', fractions=[1.0]))]
    for size, dim, depth in [('small', 32, 2), ('medium', 128, 6), ('large', 256, 12)]:
        r = dict(base)
        r.update(variants=SELECTED, model_overrides=dict(dim=dim, states=dim // 2, depth=depth),
                 fractions=[1.0], parameter_mode='native')
        campaigns.append(('scale_' + size, r))
    for frac in [0.1, 0.3]:
        r = dict(base)
        r.update(variants=SELECTED, pretrain_fraction=frac, fractions=[1.0])
        campaigns.append(('data_' + str(frac), r))
        r2 = dict(r)
        r2['pretrain_epochs'] = round(base.get('pretrain_epochs', 100) / frac)
        campaigns.append(('data_equal_updates_' + str(frac), r2))
    r = dict(base)
    r.update(variants=SELECTED, pretrain_epochs=300, fractions=[1.0])
    campaigns.append(('long_pretrain', r))

    index = []
    for name, r in campaigns:
        path = root / name
        dump(path / 'registry.json', r)
        count = build(path / 'registry.json', path / 'plan')
        index.append(dict(name=name, jobs=count, plan=str(path / 'plan' / 'plan.json')))
    dump(root / 'campaign_index.json', index)
    total = sum(i['jobs'] for i in index)
    print(f'[cuda:0] {len(campaigns)} campaigns, total {total} jobs: '
          f'{[(i["name"], i["jobs"]) for i in index]}')


if __name__ == '__main__':
    main()
