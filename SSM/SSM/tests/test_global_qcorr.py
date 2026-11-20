"""G_Q_corr / G_Q_corr_frozen correctness tests (Global backbone + 8-channel filtered corr).

Mirrors the Round-5 B-line contract but for the Global (attention) backbone: shared-G init,
zero-fusion == G, branch == Q_corr branch, frozen boundary, gradient reaches the branch only
from step 2, no raw-IQ physical-feature fusion, param counts, data/aug stream independence.
"""
import torch
from torch import nn
from torch.nn import functional as F
import pytest
from ssm_exp.model import (Classifier, ComplexRecurBranch, state_corr_pack,
                           build_round5_model, build_global_qcorr_model, parameter_count)


def _g_cfg(variant):
    return dict(variant=variant, patch=2, dim=8, depth=2, heads=2, state=4,
                selector_hidden=8, classes=4, input_length=16, ffn_ratio=2,
                dropout=0.0, position='none', direction='forward', scan='parallel', ssm_gain=1.0)


def _build(variant, seed):
    torch.manual_seed(seed)
    if variant in ('Q_corr', 'Q_corr_frozen'):
        return build_round5_model(_g_cfg(variant), seed)
    return build_global_qcorr_model(_g_cfg(variant), seed)


# --- 1. zero fusion == same-weight G (both new variants) -------------------------------
@pytest.mark.parametrize('variant', ['G_Q_corr', 'G_Q_corr_frozen'])
def test_zero_fusion_equals_g(variant):
    x = torch.randn(2, 2, 16)
    torch.manual_seed(17)
    g = Classifier(_g_cfg('G')).eval()
    m = _build(variant, 17).eval()
    with torch.no_grad():
        torch.testing.assert_close(m(x), g(x), rtol=0, atol=0)


# --- 2. shared-G backbone byte-identical across G and both new variants ----------------
@pytest.mark.parametrize('variant', ['G_Q_corr', 'G_Q_corr_frozen'])
def test_shared_backbone_with_g(variant):
    torch.manual_seed(17)
    g = Classifier(_g_cfg('G'))
    torch.manual_seed(17)
    m = _build(variant, 17)
    for key in g.state_dict():
        if key.startswith(('front.', 'blocks.', 'norm.', 'head.')):
            torch.testing.assert_close(g.state_dict()[key], m.state_dict()[key], rtol=0, atol=0)
    # blocks are attention (Global), not SSM
    for b in m.blocks:
        assert type(b.mix).__name__ == 'Attention'
    assert m.pos_mode == 'sinusoidal'


# --- 3. branch identical to Q_corr (same seed+70002 init), no raw-IQ corr fusion --------
def test_branch_identical_to_q_corr_and_no_phys_fusion():
    torch.manual_seed(17)
    qc = _build('Q_corr', 17)
    torch.manual_seed(17)
    gqc = _build('G_Q_corr', 17)
    for k in ('raw_alpha', 'theta', 'b'):
        torch.testing.assert_close(qc.aux_branch.state_dict()[k],
                                   gqc.aux_branch.state_dict()[k], rtol=0, atol=0)
    x = torch.randn(2, 2, 16)
    with torch.no_grad():
        torch.testing.assert_close(qc.aux_branch(x), gqc.aux_branch(x), rtol=0, atol=0)
        torch.testing.assert_close(qc.aux_descriptor(x), gqc.aux_descriptor(x), rtol=0, atol=0)
    # no G_corr raw-IQ physical-feature fusion; no fusion module at all
    assert gqc.fusion is None


# --- 4. frozen boundary: only the 32 branch scalars differ in requires_grad -------------
def test_frozen_boundary_exactly_branch_params():
    torch.manual_seed(17)
    gqc = _build('G_Q_corr', 17)
    torch.manual_seed(17)
    gqcf = _build('G_Q_corr_frozen', 17)
    assert parameter_count(gqc) == parameter_count(gqcf)
    diff = [n for (n, p), (_, q) in zip(gqc.named_parameters(), gqcf.named_parameters())
            if p.requires_grad != q.requires_grad]
    assert diff == ['aux_branch.raw_alpha', 'aux_branch.theta', 'aux_branch.b']
    n_frozen = sum(p.numel() * (2 if p.is_complex() else 1)
                   for p in gqcf.aux_branch.parameters())
    assert n_frozen == 32


# --- 5. gradient reaches branch from step 2 (W_f zero init) ----------------------------
def test_gradient_reaches_branch_step2():
    torch.manual_seed(700)
    m = _build('G_Q_corr', 700)
    x = torch.randn(4, 2, 16)
    y = torch.arange(4) % 4
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    opt.zero_grad(); F.cross_entropy(m(x), y).backward()
    assert m.aux_fusion.weight.grad.abs().sum() > 0
    for k in ('raw_alpha', 'theta', 'b'):
        assert getattr(m.aux_branch, k).grad.abs().sum() == 0, k
    opt.step()
    opt.zero_grad(); F.cross_entropy(m(x), y).backward()
    for k in ('raw_alpha', 'theta', 'b'):
        assert getattr(m.aux_branch, k).grad.abs().sum() > 0, k


# --- 6. frozen branch params unchanged by an optimizer step, fusion/backbone update -----
def test_frozen_branch_unchanged_after_step():
    torch.manual_seed(700)
    m = _build('G_Q_corr_frozen', 700)
    opt = torch.optim.AdamW((p for p in m.parameters() if p.requires_grad))
    ids = {id(p) for grp in opt.param_groups for p in grp['params']}
    for name, p in m.named_parameters():
        if name.startswith('aux_branch.'):
            assert not p.requires_grad and id(p) not in ids
        elif name.startswith(('aux_fusion.', 'front.', 'blocks.', 'norm.', 'head.')):
            assert p.requires_grad and id(p) in ids
    before = {n: p.detach().clone() for n, p in m.aux_branch.named_parameters()}
    x = torch.randn(4, 2, 16); y = torch.arange(4) % 4
    opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    for n, p in m.aux_branch.named_parameters():
        torch.testing.assert_close(p, before[n], rtol=0, atol=0)


# --- 7. param counts match the spec ----------------------------------------------------
def test_param_counts():
    torch.manual_seed(17)
    g = Classifier(_g_cfg('G'))
    torch.manual_seed(17)
    gqc = _build('G_Q_corr', 17)
    torch.manual_seed(17)
    gqcf = _build('G_Q_corr_frozen', 17)
    # small config: fusion 16*8=128, branch 32
    assert parameter_count(gqc) == parameter_count(g) + 128 + 32
    assert parameter_count(gqcf) == parameter_count(gqc)
    trainable_frozen = sum(p.numel() * (2 if p.is_complex() else 1)
                           for p in gqcf.parameters() if p.requires_grad)
    assert parameter_count(gqcf) - trainable_frozen == 32


# --- 8. data + CFO augmentation stream independent of model build ----------------------
def test_data_and_aug_stream_independent_of_model_build():
    from ssm_exp.utils import seed_all
    seed = 17

    def draw(variant):
        seed_all(seed)
        gen = torch.Generator().manual_seed(seed)
        aug_gen = torch.Generator().manual_seed(seed + 100000)
        _build(variant, seed)  # consumes global RNG differently per variant
        perm = torch.randperm(64, generator=gen).tolist()
        cfo = (2 * torch.rand(8, generator=aug_gen) - 1) * 0.002
        return perm, cfo

    p0, c0 = draw('G_Q_corr')
    p1, c1 = draw('G_Q_corr_frozen')
    assert p0 == p1
    torch.testing.assert_close(c0, c1, rtol=0, atol=0)


# --- 9. readout is corr (state_corr_pack), not mean ------------------------------------
def test_readout_is_corr():
    torch.manual_seed(17)
    m = _build('G_Q_corr', 17)
    assert m.aux_readout == 'corr'
    x = torch.randn(2, 2, 16)
    with torch.no_grad():
        q = m.aux_branch(x)
        d = m.aux_descriptor(x)
    torch.testing.assert_close(d, state_corr_pack(q, 2), rtol=0, atol=0)
    assert d.shape == (2, 8, 16)
