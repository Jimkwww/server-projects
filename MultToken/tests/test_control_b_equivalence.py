"""Control B (single large-kernel conv + 16-d bottleneck) implementation checks.

No training: verifies (a) parameter counts and (b) the T2 -> B weight-transfer
equivalence specified in the audit. Transfer: 8-pt branch (6 kernels) centered
at indices [12:20], 16-pt branch (5) at [8:24], 32-pt branch (5) full [0:32],
channel order 6/5/5; proj.weight/proj.bias copied verbatim. Tokenizer outputs
(and logits under an explicitly-shared backbone+head) must match exactly.
"""
import json

import torch
import pytest

from mtk.model import TokenClassifier
from mtk.tokenizers import BSingleConv, T2MultiscaleConv

CONFIG_DIR = "mtk/configs"


def _transfer(t2, b):
    with torch.no_grad():
        b.conv.weight.zero_()
        b.conv.weight[0:6, :, 12:20] = t2.convs[0].weight   # 8-pt, 6 kernels
        b.conv.weight[6:11, :, 8:24] = t2.convs[1].weight    # 16-pt, 5 kernels
        b.conv.weight[11:16, :, 0:32] = t2.convs[2].weight   # 32-pt, 5 kernels
        b.proj.weight.copy_(t2.proj.weight)
        b.proj.bias.copy_(t2.proj.bias)


def test_frontend_param_counts():
    torch.manual_seed(0)
    b = BSingleConv()
    conv = sum(v.numel() for v in b.conv.parameters())
    proj = sum(v.numel() for v in b.proj.parameters())
    assert conv == 1024        # 2 * 16 * 32, bias=False
    assert proj == 2992        # 16 * 176 + 176
    assert conv + proj == 4016


def test_full_model_params():
    cfg = json.loads(open(f"{CONFIG_DIR}/b_single_conv.json").read())
    m = TokenClassifier(cfg)
    assert sum(v.numel() for v in m.parameters()) == 2252616


@pytest.mark.parametrize("L,trail_invalid", [(1024, False), (1027, False), (1024, True)])
def test_tokenizer_equivalence(L, trail_invalid):
    torch.manual_seed(17)
    t2 = T2MultiscaleConv()
    torch.manual_seed(17)
    b = BSingleConv()
    _transfer(t2, b)

    iq = torch.randn(4, L, 2)
    valid = torch.ones(4, L, dtype=torch.bool)
    if trail_invalid:
        valid[:, L - 24:] = False

    with torch.no_grad():
        x2 = t2(iq, valid).embeddings
        xb = b(iq, valid).embeddings
    assert x2.shape == xb.shape == (4, (L + 7) // 8, 176)
    assert (xb - x2).abs().max() == 0.0


def test_logits_equivalence():
    cfg2 = json.loads(open(f"{CONFIG_DIR}/t2_multiscale.json").read())
    cfgB = json.loads(open(f"{CONFIG_DIR}/b_single_conv.json").read())

    torch.manual_seed(2024)
    mA = TokenClassifier(cfg2)
    torch.manual_seed(2024)
    mB = TokenClassifier(cfgB)
    # Tokenizer construction consumes a different RNG budget for t2 vs b, so
    # share backbone+head explicitly rather than relying on the same seed.
    mB.backbone.load_state_dict(mA.backbone.state_dict())
    mB.head.load_state_dict(mA.head.state_dict())
    _transfer(mA.tokenizer, mB.tokenizer)

    mA.eval(); mB.eval()
    iq = torch.randn(4, 1024, 2)
    valid = torch.ones(4, 1024, dtype=torch.bool)
    valid[:, 1000:] = False
    with torch.no_grad():
        lA = mA(iq, valid)
        lB = mB(iq, valid)
    assert (lB - lA).abs().max() == 0.0
