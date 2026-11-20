"""Split the main matrix across GPU1/GPU2 and generate per-GPU plans.

The package runner executes one plan DAG serially (one process per plan dir). To use two
GPUs we split the 34 variants into two disjoint variant sets, each with its own registry
(device cuda:1 / cuda:2) and its own plan directory. Downstream jobs depend only on their
own variant's pretrain, so the two shards are fully independent; the final report merges
both eval/ trees.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / 'configs/registry.server.json'

# 34 variants in plan.py variants() order
VARIANTS = [
    'structured', 'structured_equiv', 'generic_ssm', 'generic_transformer',
    'generic_pointwise', 'multi_iq', 'fused_aux', 'pointwise_physical',
    'real_state', 'fixed_state', 'no_residual', 'no_general', 'fixed_front',
    'with_statistics', 'uncertainty', 'bidirectional', 'bidirectional_transformer',
    'aug_without_equiv', 'shuffled_aux', 'structured_recon', 'generic_ssm_recon',
    'generic_transformer_recon', 'multi_iq_recon', 'without_energy',
    'without_coherence', 'without_spectrum', 'without_period', 'history_1',
    'history_4', 'history_16', 'history_32', 'aux_0.1', 'aux_0.3', 'aux_3.0',
]
CORE = {'structured', 'structured_equiv', 'generic_ssm', 'fused_aux', 'pointwise_physical',
        'aug_without_equiv', 'generic_ssm_recon', 'generic_transformer_recon'}


def seed_count(v):
    return 5 if v in CORE else 3


def main():
    base = json.loads(REG.read_text())
    base['batch_size'] = 128  # A100 40GB: dim64 model fits batch 128 easily; matches historical protocol
    base['pretrain_epochs'] = 100
    base['downstream_epochs'] = 50

    # Balanced split by pretrain seed count
    total = sum(seed_count(v) for v in VARIANTS)
    target = total / 2
    g1, g2, acc = [], [], 0
    for v in VARIANTS:
        if acc + seed_count(v) <= target + 1e-9:
            g1.append(v)
            acc += seed_count(v)
        else:
            g2.append(v)
    print(f'split: GPU1 {len(g1)} variants ({sum(seed_count(v) for v in g1)} seeds), '
          f'GPU2 {len(g2)} variants ({sum(seed_count(v) for v in g2)} seeds)')

    for gpu, vs in [('cuda:1', g1), ('cuda:2', g2)]:
        r = dict(base)
        r['device'] = gpu
        r['variants'] = vs
        out = ROOT / f'configs/registry.server.{gpu.replace(":", "")}.json'
        out.write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(f'[registry] {out.name} (device={gpu}, {len(vs)} variants)')


if __name__ == '__main__':
    main()
