"""Verify round-2 backbone init bit-identity.

1. build_encoder_r2('G') == round-1 build_encoder('G') for all three seeds.
2. build_encoder_r2('Q') == round-1 build_encoder('Q') for all three seeds.
3. G_Qlearn trunk (front/blocks/norm) == G-MAE trunk; G_Qlearn aux_branch == Q-MAE aux_branch.
4. S1plain trunk == Q-MAE trunk (Q's S1 main path); S1plain has no aux branch.
5. Q zero-init fusion output is exactly zero at init.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ssm_exp.utils import seed_all
from pretrain.model import build_encoder
from round2_overnight.models import build_encoder_r2, Encoder2

CFG = json.loads((Path(__file__).resolve().parent.parent.parent / 'configs' / 'pretrain.json').read_text())['model']


def eq(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys(), f'key mismatch: {sa.keys() ^ sb.keys()}'
    for k in sa:
        if not torch.equal(sa[k], sb[k]):
            return False, k
    return True, None


def main():
    fails = []
    for seed in (17, 42, 2026):
        for bb in ('G', 'Q'):
            seed_all(seed)
            a = build_encoder_r2(CFG, bb, seed)
            seed_all(seed)
            b = build_encoder(CFG, bb, seed)
            ok, k = eq(a, b)
            print(f'[{"OK" if ok else "FAIL"}] r2 {bb} seed{seed} == round1 {bb}', '' if ok else f'differs at {k}')
            if not ok:
                fails.append((bb, seed, k))

    # Cross-backbone init relationships.
    for seed in (17, 42, 2026):
        seed_all(seed)
        g = build_encoder_r2(CFG, 'G', seed)
        seed_all(seed)
        q = build_encoder_r2(CFG, 'Q', seed)
        seed_all(seed)
        gql = build_encoder_r2(CFG, 'G_Qlearn', seed)
        seed_all(seed)
        s1p = build_encoder_r2(CFG, 'S1plain', seed)

        # G_Qlearn trunk == G, aux_branch == Q's aux_branch.
        for k in g.state_dict():
            if k.startswith('front.') or k.startswith('blocks.') or k.startswith('norm.'):
                assert torch.equal(g.state_dict()[k], gql.state_dict()[k]), f'G_Qlearn trunk {k} != G'
        for k in q.state_dict():
            if k.startswith('aux_branch.'):
                assert torch.equal(q.state_dict()[k], gql.state_dict()[k]), f'G_Qlearn aux {k} != Q'
        # S1plain trunk == Q's S1 main path.
        for k in q.state_dict():
            if k.startswith('front.') or k.startswith('blocks.') or k.startswith('norm.'):
                assert torch.equal(q.state_dict()[k], s1p.state_dict()[k]), f'S1plain trunk {k} != Q'
        assert s1p.aux_branch is None and s1p.aux_fusion is None
        assert isinstance(gql, Encoder2) and gql.aux_branch is not None
        print(f'[OK] seed{seed}: G_Qlearn trunk==G & aux==Q; S1plain trunk==Q (no aux)')

        # Q zero-init fusion output must be zero at init.
        seed_all(seed)
        enc = build_encoder_r2(CFG, 'Q', seed)
        x = torch.randn(2, 2, 1024)
        with torch.no_grad():
            f = enc.aux_descriptor(x)
            out = enc.aux_fusion(f)
        print(f'[{"OK" if out.abs().max() == 0 else "FAIL"}] seed{seed}: Q aux_fusion zero-init output max={out.abs().max().item():.2e}')
        if out.abs().max() != 0:
            fails.append(('Q_zero_fusion', seed, out.abs().max().item()))

    if fails:
        print('FAILURES:', fails)
        sys.exit(1)
    print('ALL BIT-IDENTITY CHECKS PASS')


if __name__ == '__main__':
    main()
