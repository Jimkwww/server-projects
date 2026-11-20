"""Round-4 correctness verification (12-item checklist from the task book).

Covers: complex front == real expansion; pack/unpack order; serial==parallel scan;
backward flip-back alignment; bidirectional past+future dependence / unidirectional no
future leakage; complex gradients; cross-sample isolation; CFO unit+order; front phase
equivariance; D_complex complex retention; stats monitoring; old paths unaffected.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
import pytest
from ssm_exp.model import (Classifier, ComplexSSM, BidirectionalSSM, ComplexSelectiveSSM,
                           ComplexLinear, ComplexRMSNorm, RadialAct, ComplexPatchEmbedding,
                           _Direction, parameter_count, perturb, max_abs_norm, physical_features)

torch.set_num_threads(1)
torch.manual_seed(0)


def _small_cfg(variant):
    return dict(variant=variant, patch=2, dim=8, depth=2, heads=2, state=4,
                selector_hidden=8, classes=4, input_length=16, ffn_ratio=2,
                dropout=0.0, position='none', direction='forward', scan='parallel', ssm_gain=1.0)


# --- 1. complex frontend == manual real expansion -----------------------------------
def test_complex_linear_matches_real_expansion():
    torch.manual_seed(1)
    lin = ComplexLinear(3, 4)
    s = torch.randn(2, 5, 3, dtype=torch.cfloat)
    z = lin(s)
    Wr, Wi = lin.W.real, lin.W.imag
    I, Q = s.real, s.imag
    zr = F.linear(I, Wr) - F.linear(Q, Wi)
    zi = F.linear(I, Wi) + F.linear(Q, Wr)
    torch.testing.assert_close(z.real, zr, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(z.imag, zi, rtol=1e-5, atol=1e-5)


# --- 2. input shape + real/imag pack/unpack order -----------------------------------
def test_complex_patch_embedding_pack_order():
    torch.manual_seed(2)
    emb = ComplexPatchEmbedding(patch=2, complex_dim=4)
    x = torch.randn(3, 2, 16)
    out = emb(x)  # [3, 8, 8] interleaved [Re0,Im0,Re1,Im1,...]
    s = torch.complex(x[:, 0], x[:, 1]).reshape(3, 8, 2)
    z = emb.proj(s)
    torch.testing.assert_close(out[..., 0::2], z.real, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[..., 1::2], z.imag, rtol=1e-5, atol=1e-5)
    assert out.shape == (3, 8, 8)


# --- 3. forward/backward serial == parallel scan ------------------------------------
def test_bidirectional_serial_equals_parallel():
    torch.manual_seed(3)
    u = torch.randn(2, 9, 8)
    torch.manual_seed(4)
    par = BidirectionalSSM(8, 4, 8, scan='parallel')
    torch.manual_seed(4)
    ser = BidirectionalSSM(8, 4, 8, scan='serial')
    torch.testing.assert_close(par(u, None), ser(u, None), rtol=1e-4, atol=1e-4)


def test_complex_selective_serial_equals_parallel():
    torch.manual_seed(5)
    z = torch.randn(2, 9, 8, dtype=torch.cfloat)
    torch.manual_seed(6)
    par = ComplexSelectiveSSM(8, 4, 8, scan='parallel')
    torch.manual_seed(6)
    ser = ComplexSelectiveSSM(8, 4, 8, scan='serial')
    torch.testing.assert_close(par(z), ser(z), rtol=1e-4, atol=1e-4)


# --- 4. backward output position correct after flip-back -----------------------------
def test_backward_flip_matches_direct_backward_recurrence():
    torch.manual_seed(7)
    d = _Direction(8, 4, 8, scan='parallel')
    u = torch.randn(2, 7, 8)
    # model path: flip -> scan -> flip back
    hb_model = d(u.flip(1))[0].flip(1)
    # manual backward recurrence h_b[n] = a[n]*h_b[n+1] + g[n]*(B u[n]), h_b[N]=0
    v = F.linear(u.to(d.B.dtype), d.B)
    dr, gr = d.selector(torch.cat((u, torch.zeros_like(u[..., :3])), -1)).chunk(2, -1)
    delta = F.softplus(d.raw_delta.view(1, 1, -1) + dr)
    g = torch.sigmoid(gr)
    rho = torch.exp(-F.softplus(d.raw_alpha) * delta)
    a = torch.polar(rho, d.theta.expand_as(rho))
    b = g * v
    N = u.shape[1]
    h = torch.zeros_like(b)
    h[:, N - 1] = b[:, N - 1]
    for n in range(N - 2, -1, -1):
        h[:, n] = a[:, n] * h[:, n + 1] + b[:, n]
    torch.testing.assert_close(hb_model, h, rtol=1e-4, atol=1e-4)


# --- 5. bidirectional past+future; unidirectional no future leakage ------------------
def test_bidirectional_sees_future_unidirectional_does_not():
    torch.manual_seed(8)
    bi = BidirectionalSSM(8, 4, 8)
    uni = ComplexSSM(8, 4, 'S1', 8)
    u = torch.randn(1, 6, 8)
    p = torch.zeros(1, 6, 3)
    u_pert = u.clone()
    u_pert[:, -1] += 1.0
    y0 = bi(u, p)[:, 0]
    y0p = bi(u_pert, p)[:, 0]
    assert (y0 - y0p).abs().max() > 1e-3, 'bidirectional must depend on future'
    z0 = uni(u, p)[:, 0]
    z0p = uni(u_pert, p)[:, 0]
    torch.testing.assert_close(z0, z0p, rtol=0, atol=0)


# --- 6. complex weight/state/selector gradients -------------------------------------
def test_bidirectional_gradients():
    torch.manual_seed(9)
    bi = BidirectionalSSM(8, 4, 8)
    u = torch.randn(2, 5, 8, requires_grad=True)
    bi(u, None).square().mean().backward()
    for name, param in bi.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_complex_selective_gradients():
    torch.manual_seed(10)
    m = ComplexSelectiveSSM(8, 4, 8)
    z = torch.randn(2, 5, 8, dtype=torch.cfloat, requires_grad=True)
    m(z).abs().square().mean().backward()
    for name, param in m.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_d_complex_gradients():
    m = Classifier(_small_cfg('D_complex'))
    x = torch.randn(2, 2, 16)
    F.cross_entropy(m(x), torch.tensor([0, 1])).backward()
    for name, param in m.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


# --- 7. no cross-sample hidden state leakage -----------------------------------------
def test_no_cross_sample_leakage():
    torch.manual_seed(11)
    for mod in (BidirectionalSSM(8, 4, 8), ComplexSelectiveSSM(8, 4, 8)):
        inp = torch.randn(3, 5, 8) if isinstance(mod, BidirectionalSSM) else torch.randn(3, 5, 8, dtype=torch.cfloat)
        full = mod(inp, None) if isinstance(mod, BidirectionalSSM) else mod(inp)
        parts = torch.cat([mod(inp[i:i + 1], None) if isinstance(mod, BidirectionalSSM) else mod(inp[i:i + 1]) for i in range(3)])
        torch.testing.assert_close(full, parts, rtol=1e-5, atol=1e-5)


# --- 8. CFO unit + order unchanged ---------------------------------------------------
def test_cfo_unit_and_order():
    torch.manual_seed(12)
    x = torch.randn(2, 2, 32)
    # perturb applies angle = 2*pi*f*t + p rotation on I/Q
    f, p = 0.02, 1.3
    xp = perturb(x, f, p)
    t = torch.arange(32).float()
    ang = 2 * math.pi * f * t + p
    co, si = ang.cos(), ang.sin()
    ref = torch.stack((x[:, 0] * co - x[:, 1] * si, x[:, 0] * si + x[:, 1] * co), 1)
    torch.testing.assert_close(xp, ref, rtol=1e-5, atol=1e-5)
    # max_abs_norm idempotent (order-B stable when already normalized)
    n = max_abs_norm(x)
    torch.testing.assert_close(max_abs_norm(n), n, rtol=1e-5, atol=1e-5)


# --- 9. complex frontend unbiased: global phase rotation is a linear map -------------
def test_complex_linear_phase_equivariance():
    torch.manual_seed(13)
    lin = ComplexLinear(3, 4)
    s = torch.randn(2, 5, 3, dtype=torch.cfloat)
    phi = 1.2
    z = lin(s)
    zp = lin(s * torch.exp(torch.tensor(1j * phi)))
    torch.testing.assert_close(zp, z * torch.exp(torch.tensor(1j * phi)), rtol=1e-5, atol=1e-5)


# --- 10. D_complex keeps complex (no accidental Re()) --------------------------------
def test_d_complex_keeps_complex():
    torch.manual_seed(14)
    cd = 4
    z = torch.randn(2, 5, cd, dtype=torch.cfloat)
    assert ComplexRMSNorm(cd)(z).is_complex()
    assert RadialAct()(z).is_complex()
    assert ComplexSelectiveSSM(cd, 4, 8)(z).is_complex()
    blk = nn.Module()
    # full ComplexBlock path
    from ssm_exp.model import ComplexBlock
    assert ComplexBlock(cd, 4, 8, 2, 0.0)(z).is_complex()
    m = Classifier(_small_cfg('D_complex'))
    out = m(torch.randn(2, 2, 16))
    assert not out.is_complex() and out.shape == (2, 4)


# --- 11. stats monitoring finite -----------------------------------------------------
def test_stats_monitoring():
    torch.manual_seed(15)
    bi = BidirectionalSSM(8, 4, 8)
    bi.collect_stats = True
    bi(torch.randn(2, 5, 8), None)
    for k in ('state_rms', 'state_max', 'rho_mean', 'rho_near_one', 'gate_mean', 'gate_saturation'):
        assert k in bi.last_stats and math.isfinite(bi.last_stats[k])
    cs = ComplexSelectiveSSM(8, 4, 8)
    cs.collect_stats = True
    cs(torch.randn(2, 5, 8, dtype=torch.cfloat))
    assert math.isfinite(cs.last_stats['rho_mean'])


# --- 12. old model paths unaffected --------------------------------------------------
@pytest.mark.parametrize('variant', ['G', 'S0', 'S1', 'S2', 'S2_concat', 'G_corr'])
def test_old_variants_forward_and_param_invariant(variant):
    m = Classifier(_small_cfg(variant))
    out = m(torch.randn(2, 2, 16))
    assert out.shape == (2, 4)
    assert parameter_count(ComplexSSM(8, 4, 'S1', 8)) == parameter_count(ComplexSSM(8, 4, 'S2', 8))
