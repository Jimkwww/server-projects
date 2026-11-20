"""Split the 34-variant main matrix across 3 GPUs (GPU0 night-shift + GPU1/GPU2 24h).

GPU0 only runs ~12h/night, so it gets a smaller share (~30 seeds) while GPU1/GPU2 (~24h/day)
each get ~44 seeds, keeping GPU0 from being the critical path. Variants are assigned
contiguously (not interleaved) so downstream jobs stay with their pretrain.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / 'configs/registry.server.json'

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


def sc(v):
    return 5 if v in CORE else 3


def main():
    base = json.loads(REG.read_text())
    base['batch_size'] = 128
    base['pretrain_epochs'] = 100
    base['downstream_epochs'] = 50

    total = sum(sc(v) for v in VARIANTS)  # 118
    # GPU0 night-only (12h/day) is ~half a card's throughput, so it gets ~1/5 of the seeds;
    # GPU1/GPU2 (24h/day) each get ~2/5. This keeps all three finishing near the same time.
    target_g0 = total * 0.2
    target_g12 = (total - target_g0) / 2
    g0, g1, g2, acc = [], [], [], 0
    # assign first ~20% to g0, then split the rest between g1/g2
    for v in VARIANTS:
        if acc < target_g0:
            g0.append(v)
            acc += sc(v)
        elif acc < target_g0 + target_g12:
            g1.append(v)
            acc += sc(v)
        else:
            g2.append(v)
            acc += sc(v)
    for gpu, vs in [('cuda:0', g0), ('cuda:1', g1), ('cuda:2', g2)]:
        r = dict(base)
        r['device'] = gpu
        r['variants'] = vs
        out = ROOT / f'configs/registry.server.{gpu.replace(":", "")}.json'
        out.write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(f'{gpu}: {len(vs)} variants, {sum(sc(v) for v in vs)} seeds -> {out.name}')


if __name__ == '__main__':
    main()
