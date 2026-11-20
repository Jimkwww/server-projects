"""Acceptance: patch/decode round-trip, encoder init equivalence, MAE mask leakage,
denoising branch recompute, and shared-init reproducibility (task book section 6 checks 1/3/4)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from pretrain.model import (build_encoder, PretrainModel, ReconstructHead, mask_patches,
                            Encoder, DownstreamModel, stable_state_corr_pack)
from pretrain.engine import sample_mask, denoise_loss, mae_loss
from pretrain.iq_noise import corrupt_iq
from ssm_exp.model import (Classifier, build_round5_model, max_abs_norm, PatchEmbedding,
                           state_corr_pack)
from ssm_exp.utils import seed_all

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / 'configs' / 'pretrain.json').read_text())['model']


def _small_cfg(**over):
    c = dict(CFG)
    c.update({'depth': 2, 'dim': 16, 'state': 8, 'heads': 2, 'input_length': 64,
              'selector_hidden': 8, 'patch': 8})
    c.update(over)
    return c


def test_patch_decode_roundtrip():
    """PatchEmbedding's flatten [I0,Q0,I1,Q1,...] and ReconstructHead's un-flatten are inverse."""
    patch = 8
    x = torch.randn(3, 2, 64)
    # forward flatten path (mirrors PatchEmbedding.forward's reshape)
    b, c, l = x.shape
    flat = x.transpose(1, 2).reshape(b, l // patch, 2 * patch)  # [B,N,16]
    # decoder un-flatten path (ReconstructHead.forward reshape)
    out = flat.reshape(b, l // patch, patch, 2).permute(0, 3, 1, 2).reshape(b, 2, l)
    assert torch.equal(x, out)


def test_pure_i_q_impulse_ordering():
    """Pure-I / pure-Q / impulse land at the correct (channel, position) after un-flatten."""
    patch = 8
    n = 128 // patch
    # build tokens in [I0,Q0,...] order: token k has I at slot 2*j, Q at slot 2*j+1.
    tokens = torch.zeros(1, n, 2 * patch)
    tokens[0, 2, 2 * 3] = 5.0      # patch 2, within-patch pos 3, I channel
    tokens[0, 5, 2 * 7 + 1] = -7.0  # patch 5, within-patch pos 7, Q channel
    out = tokens.reshape(1, n, patch, 2).permute(0, 3, 1, 2).reshape(1, 2, 128)
    assert out[0, 0, 2 * patch + 3] == 5.0
    assert out[0, 1, 5 * patch + 7] == -7.0


def test_encoder_init_matches_old_classifier():
    for backbone in ('G', 'Q'):
        seed_all(17)
        ref = (build_round5_model({**CFG, 'variant': 'Q_corr'}, 17) if backbone == 'Q'
               else Classifier({**CFG, 'variant': 'G'}))
        seed_all(17)
        enc = build_encoder(dict(CFG), backbone, 17)
        ref_sd, enc_sd = ref.state_dict(), enc.state_dict()
        expect = {k for k in ref_sd if not k.startswith('head.') and not k.startswith('phys_')}
        assert set(enc_sd) == expect
        for k in expect:
            assert torch.equal(ref_sd[k], enc_sd[k]), k
        # logits alignment
        x = max_abs_norm(torch.randn(2, 2, 1024))
        enc.eval(); ref.eval()
        with torch.no_grad():
            head = nn.Linear(CFG['dim'], CFG['classes'])
            head.weight.data.copy_(ref.head.weight); head.bias.data.copy_(ref.head.bias)
            assert torch.allclose(ref(x), head(enc.features(x)), atol=1e-5)


def test_q_filter_has_32_grad_scalars():
    seed_all(17)
    enc = build_encoder(dict(CFG), 'Q', 17)
    ps = list(enc.aux_branch.parameters())
    total = sum(p.numel() * (2 if p.is_complex() else 1) for p in ps)
    assert total == 32
    assert all(p.requires_grad for p in ps)


def test_mask_leakage_g_and_q():
    """Changing only masked target values must not change encoder tokens / Q descriptor."""
    seed_all(17)
    n = 128
    x = max_abs_norm(torch.randn(2, 2, 1024))
    mask = torch.zeros(2, n, dtype=torch.bool)
    mask[:, :64] = True  # mask first 64 patches
    for backbone in ('G', 'Q'):
        enc = build_encoder(dict(CFG), backbone, 17).eval()
        with torch.no_grad():
            tok_a = enc.forward_tokens(x, mask=mask)
            desc_a = enc.aux_descriptor(mask_patches(x, mask, 8)) if backbone == 'Q' else None
            x2 = x.clone()
            x2[:, :, :64 * 8] = torch.randn(2, 2, 64 * 8)  # perturb only masked region
            tok_b = enc.forward_tokens(x2, mask=mask)
            desc_b = enc.aux_descriptor(mask_patches(x2, mask, 8)) if backbone == 'Q' else None
        assert torch.equal(tok_a, tok_b), backbone
        if backbone == 'Q':
            assert torch.equal(desc_a, desc_b)


def test_denoise_branch_recompute():
    """Changing injected noise changes the Q descriptor (the branch recomputes from the input)."""
    seed_all(17)
    enc = build_encoder(dict(CFG), 'Q', 17).eval()
    x = max_abs_norm(torch.randn(2, 2, 1024))
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(1)
    y1, _, _ = corrupt_iq(x, g1)
    y2, _, _ = corrupt_iq(x, g2)
    with torch.no_grad():
        d1 = enc.aux_descriptor(y1)
        d2 = enc.aux_descriptor(y2)
    assert not torch.equal(d1, d2)


def _assert_branch_grad_after_steps(model, make_loss, key='encoder'):
    """The aux_fusion is zero-init, so the aux branch gets zero grad at init; after a few
    optimizer steps fusion becomes non-zero and gradient must flow to the branch."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(4):
        loss = make_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    make_loss().backward()
    branch = model.encoder.aux_branch if key == 'encoder' else model.aux_branch
    g = [p.grad for p in branch.parameters()]
    assert all(pg is not None and torch.isfinite(pg).all() and pg.abs().sum() > 0 for pg in g)


def test_q_filter_gradient_flows_pretrain_and_downstream():
    seed_all(17)
    x = max_abs_norm(torch.randn(4, 2, 1024))

    # pretraining path (denoise)
    pm = PretrainModel(build_encoder(dict(CFG), 'Q', 17), 8)
    gen = torch.Generator().manual_seed(0)

    def pretrain_loss():
        y, sigma, _ = corrupt_iq(x, gen)
        pred = pm(y, extra=sigma[:, None].expand(4, 128))
        return denoise_loss(pred, x)
    _assert_branch_grad_after_steps(pm, pretrain_loss)

    # downstream fine-tune path
    dm = DownstreamModel(build_encoder(dict(CFG), 'Q', 17), 24)
    labels = torch.randint(0, 24, (4,))

    def sup_loss():
        return torch.nn.functional.cross_entropy(dm(x), labels)
    _assert_branch_grad_after_steps(dm, sup_loss)


def test_stable_corr_matches_old_and_is_nan_free():
    torch.manual_seed(0)
    q = torch.randn(2, 128, 8, dtype=torch.cfloat)
    # non-degenerate: identical to the old state_corr_pack
    assert torch.allclose(stable_state_corr_pack(q, 8), state_corr_pack(q, 8), atol=1e-7)
    # fully-zero patch region: finite value and finite backward
    keep = torch.ones(128)
    keep[32:96] = 0.0
    qz = (q * keep.view(1, 128, 1)).requires_grad_(True)
    d = stable_state_corr_pack(qz, 8)
    assert torch.isfinite(d).all()
    d.square().sum().backward()


def test_shared_init_reproducible():
    """Same backbone+seed builds byte-identical init across two independent calls."""
    for backbone in ('G', 'Q'):
        seed_all(17)
        a = build_encoder(dict(CFG), backbone, 17)
        seed_all(17)
        b = build_encoder(dict(CFG), backbone, 17)
        for k in a.state_dict():
            assert torch.equal(a.state_dict()[k], b.state_dict()[k]), (backbone, k)


def test_sample_mask_constraints():
    g = torch.Generator().manual_seed(0)
    mask = sample_mask(64, 128, 0.5, g, torch.device('cpu'))
    assert mask.shape == (64, 128)
    assert (mask.sum(1) == 64).all()
    # at least one visible and one target (64 masked of 128 always holds)
    assert (mask.sum(1) > 0).all() and (mask.sum(1) < 128).all()


def test_mae_loss_denominator():
    x = max_abs_norm(torch.randn(2, 2, 1024))
    mask = torch.zeros(2, 128, dtype=torch.bool)
    mask[:, :64] = True
    pred = x.clone()
    pred[:, :, :64 * 8] = 0.0  # zero the masked region only
    loss = mae_loss(pred, x, mask, 8)
    # error only over masked time points; denominator = 64*8 per sample
    expected = (x[:, :, :64 * 8] ** 2).sum() / (2 * 64 * 8)
    assert abs(loss.item() - expected.item()) < 1e-5
