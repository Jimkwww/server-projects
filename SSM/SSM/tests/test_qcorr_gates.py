"""Q_corr physical-gate round correctness tests.

Covers the section-5 pre-training checks: hand-calculated scalar descriptors, CFO transform
invariance, zero-init equivalence with Q_corr, gate wiring (write -> g only, retention ->
delta/rho only, with exact sign+amplitude), gradient reach/param update, param counts, and
data/CFO stream independence of the variant.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
import pytest

from ssm_exp.model import (Classifier, ComplexSSM, raw_iq_scalar_features, perturb,
                           max_abs_norm, build_round5_model, parameter_count,
                           GATE_C_VARIANTS, GATE_D_VARIANTS, SCALAR_VARIANTS)

VARIANTS = ['Q_corr_gate_c', 'Q_corr_gate_d', 'Q_corr_gate_cd',
            'Q_corr_token_c', 'Q_corr_token_d', 'Q_corr_token_cd']


def _cfg(variant):
    return dict(variant=variant, patch=2, dim=8, depth=2, heads=2, state=4,
                selector_hidden=8, classes=4, input_length=16, ffn_ratio=2,
                dropout=0.0, position='none', direction='forward', scan='parallel', ssm_gain=1.0)


def _build(variant, seed):
    torch.manual_seed(seed)
    return build_round5_model(_cfg(variant), seed)


# --- 1. hand-calculated descriptors ---------------------------------------------------
def test_constant_exponential_c1_d0():
    L, patch = 16, 2
    omega = 0.3
    t = torch.arange(L).float()
    s = torch.exp(1j * omega * t)
    x = torch.stack((s.real, s.imag), 0).unsqueeze(0)
    c, d = raw_iq_scalar_features(x, patch)
    assert torch.allclose(c, torch.ones_like(c), atol=1e-5)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)
    assert d[:, 0].abs().max() == 0  # first patch exactly zero


def test_two_phase_increments_hand_d():
    # patch 0 phase increment 0 -> r0 = 1; patch 1 increment pi/2 -> r1 = j
    s = torch.tensor([1.0 + 0j, 1.0 + 0j, 1.0 + 0j, 0.0 + 1j])
    x = torch.stack((s.real, s.imag), 0).unsqueeze(0)  # patch=2 -> 2 patches
    c, d = raw_iq_scalar_features(x, 2)
    assert torch.allclose(c, torch.ones(1, 2), atol=1e-6)   # |1|, |j|
    # d1 = |j - 1| = sqrt(2); d0 = 0
    assert d[0, 0].item() == 0.0
    assert abs(d[0, 1].item() - math.sqrt(2.0)) < 1e-5


def test_zero_input_no_nan():
    x = torch.zeros(1, 2, 16)
    c, d = raw_iq_scalar_features(x, 2)
    assert torch.isfinite(c).all() and torch.isfinite(d).all()
    assert d[:, 0].abs().max() == 0


# --- 2. CFO transform: r rotates together, c/d preserved ------------------------------
def test_cfo_rotates_r_preserves_cd():
    L, patch = 16, 2
    omega, freq = 0.3, 0.1
    t = torch.arange(L).float()
    s = torch.exp(1j * omega * t)
    x = torch.stack((s.real, s.imag), 0).unsqueeze(0)

    def r_of(sig):
        sc = torch.complex(sig[:, 0], sig[:, 1]).reshape(sig.shape[0], -1, patch)
        cur, prev = sc[..., 1:], sc[..., :-1]
        return (cur * prev.conj()).sum(-1) / (
            (cur.abs().square().sum(-1) * prev.abs().square().sum(-1)).sqrt() + 1e-8)

    r0 = r_of(x)
    xp = perturb(x, freq)
    r1 = r_of(xp)
    rot = torch.tensor(math.e ** (1j * 2 * math.pi * freq), dtype=torch.complex64)
    # every patch's r rotates by the common CFO phase
    assert torch.allclose(r1, r0 * rot, atol=1e-4)

    c0, d0 = raw_iq_scalar_features(x, patch)
    c1, d1 = raw_iq_scalar_features(xp, patch)
    assert torch.allclose(c0, c1, atol=1e-5)
    assert torch.allclose(d0, d1, atol=1e-5)


def test_cfo_low_energy_still_finite():
    L, patch = 16, 2
    omega = 0.3
    t = torch.arange(L).float()
    s = 1e-3 * torch.exp(1j * omega * t)
    x = torch.stack((s.real, s.imag), 0).unsqueeze(0)
    c, d = raw_iq_scalar_features(max_abs_norm(x), patch)
    assert torch.isfinite(c).all() and torch.isfinite(d).all()
    # normalized correlation is amplitude-invariant for a clean exponential
    assert torch.allclose(c, torch.ones_like(c), atol=1e-4)


# --- 3. zero-init equivalence with Q_corr ---------------------------------------------
@pytest.mark.parametrize('variant', VARIANTS)
def test_zero_init_equals_q_corr(variant):
    x = torch.randn(2, 2, 16)
    torch.manual_seed(17)
    qc = _build('Q_corr', 17).eval()
    torch.manual_seed(17)
    m = _build(variant, 17).eval()
    with torch.no_grad():
        torch.testing.assert_close(m(x), qc(x), rtol=0, atol=0)


def test_shared_backbone_and_branch_with_q_corr():
    torch.manual_seed(17)
    qc = _build('Q_corr', 17)
    for variant in VARIANTS:
        torch.manual_seed(17)
        m = _build(variant, 17)
        for k in qc.state_dict():
            # Q branch + backbone + head are shared with Q_corr at init
            if not k.startswith(('gate_',)) and 'token_proj' not in k and \
               not k.endswith(('.write_w', '.retention_v')):
                torch.testing.assert_close(qc.state_dict()[k], m.state_dict()[k], rtol=0, atol=0)
        for k in ('raw_alpha', 'theta', 'b'):
            torch.testing.assert_close(qc.aux_branch.state_dict()[k],
                                       m.aux_branch.state_dict()[k], rtol=0, atol=0)


# --- 4. gate wiring: write -> g only, retention -> delta/rho only ----------------------
def _ssm(variant):
    torch.manual_seed(0)
    return ComplexSSM(8, 4, variant, 8, 'parallel', 1.0)


def test_write_gate_only_changes_g():
    s = _ssm('Q_corr_gate_c')
    s.write_w.data = torch.tensor([1.0, -0.5, 2.0, 0.0])
    u = torch.randn(2, 3, 8)
    p = torch.zeros(2, 3, 3)
    c_tilde = torch.tensor([[0.5, -1.0, 0.3], [0.2, 0.8, -0.4]])
    s.collect_gate_detail = True
    s(u, p, gate=None)
    g_base, rho_base = s.last_gate_detail['g'].clone(), s.last_gate_detail['rho'].clone()
    s(u, p, gate=(c_tilde, None))
    g_mod, rho_mod = s.last_gate_detail['g'].clone(), s.last_gate_detail['rho'].clone()
    # exact sign+amplitude: g = sigmoid(logit(g_base) + w*c_tilde)
    logit_base = torch.logit(g_base.clamp(1e-7, 1 - 1e-7))
    expected = torch.sigmoid(logit_base + s.write_w * c_tilde.unsqueeze(-1))
    torch.testing.assert_close(g_mod, expected, rtol=1e-5, atol=1e-5)
    # rho untouched
    torch.testing.assert_close(rho_base, rho_mod, rtol=0, atol=0)
    # sign: w>0 & c>0 increases g
    assert (g_mod[0, 0, 0] > g_base[0, 0, 0]).item()


def test_retention_gate_only_changes_delta_rho():
    s = _ssm('Q_corr_gate_d')
    s.retention_v.data = torch.tensor([1.0, -0.5, 2.0, 0.0])
    u = torch.randn(2, 3, 8)
    p = torch.zeros(2, 3, 3)
    d_tilde = torch.tensor([[0.5, 1.0, 0.3], [0.2, 0.8, 0.4]])
    s.collect_gate_detail = True
    s(u, p, gate=None)
    g_base, rho_base = s.last_gate_detail['g'].clone(), s.last_gate_detail['rho'].clone()
    s(u, p, gate=(None, d_tilde))
    g_mod, rho_mod = s.last_gate_detail['g'].clone(), s.last_gate_detail['rho'].clone()
    # g untouched
    torch.testing.assert_close(g_base, g_mod, rtol=0, atol=0)
    # exact rho: delta_arg = softplus_inv(-log(rho_base)/alpha) + v*d_tilde
    alpha = F.softplus(s.raw_alpha).view(1, 1, -1)
    delta_base = -torch.log(rho_base.clamp_min(1e-12)) / alpha
    delta_arg_base = torch.log(torch.expm1(delta_base))
    delta_arg_mod = delta_arg_base + s.retention_v * d_tilde.unsqueeze(-1)
    expected = torch.exp(-alpha * F.softplus(delta_arg_mod))
    torch.testing.assert_close(rho_mod, expected, rtol=1e-5, atol=1e-5)
    # sign: v>0 & d>0 -> larger decay -> smaller retention (rho)
    assert (rho_mod[0, 0, 0] < rho_base[0, 0, 0]).item()


def test_cd_gate_both_and_no_cross_talk():
    s = _ssm('Q_corr_gate_cd')
    s.write_w.data = torch.tensor([1.0, 0.0, 0.0, 0.0])
    s.retention_v.data = torch.tensor([0.0, 0.0, 1.0, 0.0])
    u = torch.randn(2, 3, 8)
    p = torch.zeros(2, 3, 3)
    c_tilde = torch.full((2, 3), 1.0)
    d_tilde = torch.full((2, 3), 1.0)
    s.collect_gate_detail = True
    s(u, p, gate=(c_tilde, d_tilde))
    g, rho = s.last_gate_detail['g'].clone(), s.last_gate_detail['rho'].clone()
    s(u, p, gate=None)
    g0, rho0 = s.last_gate_detail['g'].clone(), s.last_gate_detail['rho'].clone()
    # write_w affects channel 0 (g changes, rho unchanged there); retention_v affects
    # channel 2 (rho changes, g unchanged there).
    assert not torch.allclose(g[..., 0], g0[..., 0])
    assert torch.allclose(rho[..., 0], rho0[..., 0])
    assert torch.allclose(g[..., 2], g0[..., 2])
    assert not torch.allclose(rho[..., 2], rho0[..., 2])


# --- 5. gradient + param counts + token shape -----------------------------------------
def test_gate_params_receive_grad_and_update():
    torch.manual_seed(700)
    m = _build('Q_corr_gate_cd', 700)
    x = torch.randn(4, 2, 16); y = torch.arange(4) % 4
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    opt.zero_grad(); F.cross_entropy(m(x), y).backward()
    for blk in m.blocks:
        assert blk.mix.write_w.grad.abs().sum() > 0
        assert blk.mix.retention_v.grad.abs().sum() > 0
    before = {n: p.detach().clone() for n, p in m.named_parameters()
              if n.endswith(('.write_w', '.retention_v'))}
    opt.step()
    for n, p in m.named_parameters():
        if n.endswith(('.write_w', '.retention_v')):
            assert not torch.equal(p, before[n]), n


def test_token_proj_receives_grad_and_update():
    torch.manual_seed(700)
    m = _build('Q_corr_token_cd', 700)
    x = torch.randn(4, 2, 16); y = torch.arange(4) % 4
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    opt.zero_grad(); F.cross_entropy(m(x), y).backward()
    assert m.token_proj.weight.grad.abs().sum() > 0
    before = m.token_proj.weight.detach().clone()
    opt.step()
    assert not torch.equal(m.token_proj.weight, before)


def test_param_counts():
    torch.manual_seed(17)
    qc = _build('Q_corr', 17)
    base = parameter_count(qc)
    # gate deltas = state per layer (write_w/retention_v are [state] real params); token
    # deltas = dim per scalar via the zero-init no-bias Linear (2 scalars -> 2*dim for cd).
    c = _cfg('Q_corr_gate_c')
    expected = {
        'Q_corr_gate_c': c['state'] * c['depth'],
        'Q_corr_gate_d': c['state'] * c['depth'],
        'Q_corr_gate_cd': 2 * c['state'] * c['depth'],
        'Q_corr_token_c': c['dim'],
        'Q_corr_token_d': c['dim'],
        'Q_corr_token_cd': 2 * c['dim'],
    }
    for variant, delta in expected.items():
        torch.manual_seed(17)
        m = _build(variant, 17)
        assert parameter_count(m) == base + delta, variant


def test_token_shape_full_dim():
    cfg = dict(variant='Q_corr_token_cd', patch=8, dim=176, depth=2, heads=4, state=128,
               selector_hidden=64, classes=24, input_length=1024, ffn_ratio=4, dropout=0.0,
               position='none', direction='forward', scan='parallel', ssm_gain=1.0)
    torch.manual_seed(17)
    m = build_round5_model(cfg, 17)
    x = torch.randn(2, 2, 1024)
    # hook the token tensor before blocks
    captured = {}
    def hook(module, inp):
        captured['u'] = inp[0]
    m.blocks[0].register_forward_pre_hook(hook)
    with torch.no_grad():
        m(x)
    assert tuple(captured['u'].shape) == (2, 128, 176)


# --- 6. data + CFO augmentation stream independent of variant --------------------------
def test_data_and_aug_stream_independent():
    from ssm_exp.utils import seed_all
    seed = 17

    def draw(variant):
        seed_all(seed)
        gen = torch.Generator().manual_seed(seed)
        aug_gen = torch.Generator().manual_seed(seed + 100000)
        _build(variant, seed)
        perm = torch.randperm(64, generator=gen).tolist()
        cfo = (2 * torch.rand(8, generator=aug_gen) - 1) * 0.002
        return perm, cfo

    ref_perm, ref_cfo = draw('Q_corr')
    for variant in VARIANTS:
        p, c = draw(variant)
        assert p == ref_perm
        torch.testing.assert_close(c, ref_cfo, rtol=0, atol=0)
