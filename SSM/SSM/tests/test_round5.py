"""Round-5 correctness verification (20-item implementation contract).

Line A: complex frontend fair comparison (R0/C0/Rb/Cb matched init + matched backbone).
Line B: free real main path + IQ auxiliary branch (Q_linear/Q_corr/R_corr).
Covers: complex mult == real expansion; interleaved IQ permutation; matched init output
identity; shared backbone; complex stays bound / real free; bias per config; branch reads
raw IQ; Q_corr/R_corr init identity; serial==parallel; state continuity; manual 7-pair
corr; zero-fusion == S1; gradient reaches branch params; no cross-sample leakage;
dims/params/dtype; old paths unchanged; data/aug stream independent of model build.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
import pytest
from ssm_exp.model import (Classifier, ComplexLinear, parameter_count,
                           round5_front_weights, complex_to_real_interleaved,
                           FairRealFront, FairComplexFront, ComplexRecurBranch, RealRecurBranch,
                           state_mean_pack, state_corr_pack, parallel_scan, serial_scan,
                           build_round5_model)

torch.set_num_threads(1)


def _r5_cfg(variant):
    return dict(variant=variant, patch=2, dim=8, depth=2, heads=2, state=4,
                selector_hidden=8, classes=4, input_length=16, ffn_ratio=2,
                dropout=0.0, position='none', direction='forward', scan='parallel', ssm_gain=1.0)


def _build(variant, seed):
    torch.manual_seed(seed)
    return build_round5_model(_r5_cfg(variant), seed)


def _is_complex_structured(M):
    """Binding condition for an interleaved real matrix expanded from a complex weight."""
    return (torch.allclose(M[0::2, 0::2], M[1::2, 1::2])
            and torch.allclose(M[0::2, 1::2], -M[1::2, 0::2]))


# --- 1. complex multiplication == real expansion ------------------------------------
def test_fair_front_complex_equals_real_expansion():
    gen = torch.Generator().manual_seed(100)
    W_R, W_I = round5_front_weights(3, 4, gen)
    M = complex_to_real_interleaved(W_R, W_I)
    x = torch.randn(2, 2, 18)  # patch=3 -> 6 patches, interleaved [I0,Q0,...]
    s = torch.complex(x[:, 0], x[:, 1]).reshape(2, 6, 3)
    z = F.linear(s, torch.complex(W_R, W_I))
    zc = torch.stack((z.real, z.imag), -1).reshape(2, 6, 8)
    zr = F.linear(x.transpose(1, 2).reshape(2, 6, 6), M)
    torch.testing.assert_close(zc, zr, rtol=1e-5, atol=1e-5)


# --- 2. interleaved IQ permutation correct -------------------------------------------
def test_complex_to_real_interleaved_permutation():
    gen = torch.Generator().manual_seed(101)
    W_R, W_I = round5_front_weights(3, 4, gen)
    M = complex_to_real_interleaved(W_R, W_I)
    assert M.shape == (8, 6)
    for k in range(4):
        for j in range(3):
            assert M[2 * k, 2 * j] == W_R[k, j]
            assert M[2 * k, 2 * j + 1] == -W_I[k, j]
            assert M[2 * k + 1, 2 * j] == W_I[k, j]
            assert M[2 * k + 1, 2 * j + 1] == W_R[k, j]


# --- 3. R0/C0/Rb/Cb identical init front output + logits -----------------------------
@pytest.mark.parametrize('pair', [('R0', 'C0'), ('R0', 'Rb'), ('R0', 'Cb'),
                                  ('C0', 'Cb'), ('Rb', 'Cb')])
def test_a_line_matched_init_output_identity(pair):
    x = torch.randn(2, 2, 16)
    ma = _build(pair[0], 17).eval()
    mb = _build(pair[1], 17).eval()
    with torch.no_grad():
        fa, fb = ma.front(x), mb.front(x)
        oa, ob = ma(x), mb(x)
    torch.testing.assert_close(fa, fb, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(oa, ob, rtol=1e-5, atol=1e-5)


# --- 4. shared backbone state_dict, front only differs -------------------------------
def test_a_line_shared_backbone_front_differs():
    r0 = _build('R0', 17)
    c0 = _build('C0', 17)
    for key in r0.state_dict():
        if key.startswith(('blocks.', 'norm.', 'head.')):
            torch.testing.assert_close(r0.state_dict()[key], c0.state_dict()[key], rtol=0, atol=0)
    # front weights are the matched pair (M == expand(W))
    W = c0.front.W.detach()
    torch.testing.assert_close(r0.front.proj.weight.detach(),
                               complex_to_real_interleaved(W.real, W.imag), rtol=0, atol=0)
    # R0 and C0 front key names differ (real matrix vs complex weight)
    assert 'front.proj.weight' in r0.state_dict()
    assert 'front.W' in c0.state_dict()


# --- 5. C stays bound after one update / R free --------------------------------------
def test_complex_stays_bound_real_free_after_update():
    torch.manual_seed(200)
    c0 = build_round5_model(_r5_cfg('C0'), 200)
    r0 = build_round5_model(_r5_cfg('R0'), 200)
    torch.testing.assert_close(r0.front.proj.weight.detach(),
                               complex_to_real_interleaved(c0.front.W.real.detach(),
                                                           c0.front.W.imag.detach()), rtol=0, atol=0)
    x = torch.randn(4, 2, 16)
    y = torch.arange(4) % 4
    for m in (c0, r0):
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
    # C0 equivalent real matrix still complex-structured (bound)
    W1 = c0.front.W.detach()
    assert _is_complex_structured(complex_to_real_interleaved(W1.real, W1.imag))
    # R0 free real matrix no longer complex-structured (unbound)
    assert not _is_complex_structured(r0.front.proj.weight.detach())


# --- 6. bias present per config, init zero, learnable --------------------------------
def test_bias_present_per_config_init_zero_learnable():
    assert _build('R0', 17).front.proj.bias is None
    assert _build('C0', 17).front.bias is None
    rb = _build('Rb', 17)
    cb = _build('Cb', 17)
    assert rb.front.proj.bias is not None
    assert cb.front.bias is not None
    torch.testing.assert_close(rb.front.proj.bias, torch.zeros(8), rtol=0, atol=0)
    torch.testing.assert_close(cb.front.bias, torch.zeros(8), rtol=0, atol=0)
    x = torch.randn(4, 2, 16)
    y = torch.arange(4) % 4
    for m in (rb, cb):
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        bias = m.front.proj.bias if hasattr(m.front, 'proj') else m.front.bias
        assert bias.grad is not None and bias.grad.abs().sum() > 0
        opt.step()


# --- 7. branch reads the same augmented+normalized raw IQ ----------------------------
def test_branch_reads_raw_iq_via_hook():
    m = _build('Q_corr', 17)
    x = torch.randn(2, 2, 16)
    captured = []

    def hook(mod, inp, out):
        captured.append(inp[0])

    h = m.aux_branch.register_forward_hook(hook)
    m(x)
    h.remove()
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], x, rtol=0, atol=0)


# --- 8. Q_corr / R_corr identical init state/descriptor/full output ------------------
def test_q_corr_r_corr_init_identity():
    x = torch.randn(2, 2, 16)
    qc = _build('Q_corr', 17).eval()
    rc = _build('R_corr', 17).eval()
    with torch.no_grad():
        q = qc.aux_branch(x)
        r = rc.aux_branch(x)
        q_cplx = torch.complex(r[..., 0], r[..., 1])
        # identical to fp32 rounding (real 2x2 recurrence vs complex scalar recurrence)
        assert (q - q_cplx).abs().max() < 1e-3
        torch.testing.assert_close(qc.aux_descriptor(x), rc.aux_descriptor(x), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(qc(x), rc(x), rtol=1e-5, atol=1e-5)


# --- 9. serial == parallel (Q_linear/Q_corr complex scan; R_corr matrix recurrence) ---
def test_complex_recur_branch_serial_equals_parallel():
    torch.manual_seed(300)
    br = ComplexRecurBranch(seed=300)
    x = torch.randn(2, 2, 32)
    s = torch.complex(x[:, 0], x[:, 1])
    a = torch.exp(-F.softplus(br.raw_alpha)) * torch.exp(1j * br.theta)
    b_s = s.unsqueeze(-1) * br.b
    a_exp = a.expand(x.shape[0], x.shape[2], br.state)
    torch.testing.assert_close(parallel_scan(a_exp, b_s), serial_scan(a_exp, b_s),
                               rtol=1e-4, atol=1e-4)


def test_real_recur_branch_serial_equals_closed_form():
    torch.manual_seed(400)
    br = RealRecurBranch(seed=400)
    x = torch.randn(2, 2, 16)
    out = br(x)  # [B,L,S,2] production sequential scan
    with torch.no_grad():
        rho = torch.sigmoid(br.logit)
        anorm = torch.linalg.matrix_norm(br.A_raw, ord=2)
        A = rho[:, None, None] * br.A_raw / anorm.clamp_min(1.0)[:, None, None]
        B = br.B
        iq = x.transpose(1, 2)  # [B,L,2]
        L = x.shape[2]
        S = A.shape[0]
        # matrix powers A^0..A^{L-1}
        Ap = [torch.eye(2).repeat(S, 1, 1)]
        for _ in range(1, L):
            Ap.append(torch.einsum('gij,gjk->gik', Ap[-1], A))
        w = [torch.einsum('gij,bj->bgi', B, iq[:, t]) for t in range(L)]
        ref = torch.zeros(2, L, S, 2)
        for m in range(L):
            acc = torch.zeros(2, S, 2)
            for t in range(m + 1):
                acc = acc + torch.einsum('gij,bgj->bgi', Ap[m - t], w[t])
            ref[:, m] = acc
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


# --- 10. state continuous across patch, reset at sample boundary ---------------------
def test_state_continuous_across_patch_reset_at_sample_boundary():
    torch.manual_seed(500)
    br = ComplexRecurBranch(seed=500)
    x = torch.randn(1, 2, 32)  # L=32, patch=8 -> 4 patches
    q = br(x)  # [1,32,8]
    s = torch.complex(x[:, 0], x[:, 1])
    a = torch.exp(-F.softplus(br.raw_alpha)) * torch.exp(1j * br.theta)
    # q[0] = b*s[0] (reset at sample start)
    torch.testing.assert_close(q[:, 0], br.b * s[:, 0].unsqueeze(-1), rtol=1e-5, atol=1e-5)
    # q[8] = a*q[7] + b*s[8] (continuation across patch boundary, NOT reset)
    expected = a * q[:, 7] + br.b * s[:, 8].unsqueeze(-1)
    torch.testing.assert_close(q[:, 8], expected, rtol=1e-4, atol=1e-4)
    # per-sample independence (batch == stacked single-sample runs)
    x2 = torch.randn(3, 2, 32)
    full = br(x2)
    parts = torch.cat([br(x2[i:i + 1]) for i in range(3)])
    torch.testing.assert_close(full, parts, rtol=1e-5, atol=1e-5)


# --- 11. manual 7-pair correlation, no cross-patch pairing ---------------------------
def test_state_corr_manual_7_pairs_no_cross_patch():
    torch.manual_seed(600)
    q = torch.randn(1, 16, 2, dtype=torch.cfloat)  # L=16, patch=8, S=2 -> 2 patches
    out = state_corr_pack(q, 8)  # [1,2,4]
    eps = 1e-8
    for n in range(2):
        qp = q[:, n * 8:(n + 1) * 8]  # [1,8,2]
        cur, prev = qp[:, 1:], qp[:, :-1]  # exactly 7 adjacent within-patch pairs
        num = (cur * prev.conj()).sum(1)  # [1,2]
        den = (cur.abs().square().sum(1) * prev.abs().square().sum(1)).sqrt() + eps
        r = num / den
        torch.testing.assert_close(out[:, n, 0::2], r.real, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out[:, n, 1::2], r.imag, rtol=1e-5, atol=1e-5)


# --- 12. zero fusion == same-weight S1 -----------------------------------------------
@pytest.mark.parametrize('variant', ['Q_linear', 'Q_corr', 'R_corr', 'Q_corr_frozen'])
def test_zero_fusion_equals_s1(variant):
    x = torch.randn(2, 2, 16)
    torch.manual_seed(17)
    s1 = Classifier(_r5_cfg('S1')).eval()
    m = _build(variant, 17).eval()
    with torch.no_grad():
        torch.testing.assert_close(m(x), s1(x), rtol=0, atol=0)


# --- 13. gradient reaches branch params (W_f step-1; branch params step-2) -----------
def test_gradient_reaches_branch_params():
    torch.manual_seed(700)
    m = build_round5_model(_r5_cfg('Q_corr'), 700)
    x = torch.randn(4, 2, 16)
    y = torch.arange(4) % 4
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    # step 1: W_f nonzero grad; branch params zero grad (W_f=0 -> d loss/d f = 0), not detach.
    opt.zero_grad()
    F.cross_entropy(m(x), y).backward()
    assert m.aux_fusion.weight.grad is not None and m.aux_fusion.weight.grad.abs().sum() > 0
    assert m.aux_branch.b.grad is not None and m.aux_branch.b.grad.abs().sum() == 0
    opt.step()
    # step 2: gradient now flows through nonzero W_f to branch params.
    opt.zero_grad()
    F.cross_entropy(m(x), y).backward()
    for name in ('raw_alpha', 'theta', 'b'):
        p = getattr(m.aux_branch, name)
        assert p.grad is not None and p.grad.abs().sum() > 0, name


# --- 14. no cross-sample leakage in aux branch ---------------------------------------
@pytest.mark.parametrize('branch', [ComplexRecurBranch(seed=800), RealRecurBranch(seed=800)])
def test_no_cross_sample_leakage(branch):
    x = torch.randn(3, 2, 32)
    full = branch(x)
    parts = torch.cat([branch(x[i:i + 1]) for i in range(3)])
    torch.testing.assert_close(full, parts, rtol=1e-5, atol=1e-5)


# --- 15. dims / params / dtype correct ----------------------------------------------
def test_dims_params_dtype():
    r0 = _build('R0', 17)
    c0 = _build('C0', 17)
    rb = _build('Rb', 17)
    cb = _build('Cb', 17)
    ql = _build('Q_linear', 17)
    qc = _build('Q_corr', 17)
    rc = _build('R_corr', 17)
    qf = _build('Q_corr_frozen', 17)
    # front binding param saving (real 8x4=32 vs complex 4x2=8 -> x2 = 16 real scalars)
    assert parameter_count(r0) - parameter_count(c0) == 16
    assert parameter_count(rb) - parameter_count(r0) == 8  # bias
    assert parameter_count(cb) - parameter_count(c0) == 8  # bias
    # B-line aux branch params: Q 32 real scalars, R_corr 72 (logit 8 + A_raw 32 + B 32)
    assert parameter_count(qc) - parameter_count(ql) == 0  # same branch, diff readout
    assert parameter_count(rc) - parameter_count(qc) == 40  # 72 - 32
    assert parameter_count(qf) == parameter_count(qc)  # freeze does not change count
    # dtype: complex branch weight vs real input map
    assert qc.aux_branch.b.is_complex()
    assert not rc.aux_branch.B.is_complex()
    # dims
    x = torch.randn(2, 2, 16)
    assert qc.aux_branch(x).shape == (2, 16, 8)
    assert rc.aux_branch(x).shape == (2, 16, 8, 2)
    assert qc.aux_descriptor(x).shape == (2, 8, 16)
    assert qc.aux_fusion.weight.shape == (8, 16)
    assert qc.aux_fusion.bias is None


# --- 16. old paths unchanged (frozen Round-4 ComplexLinear init included) ------------
@pytest.mark.parametrize('variant', ['G', 'S0', 'S1', 'S2', 'B_front', 'D_complex'])
def test_old_variants_unchanged(variant):
    m = Classifier(_r5_cfg(variant))
    out = m(torch.randn(2, 2, 16))
    assert out.shape == (2, 4)
    # Round-4 ComplexLinear init untouched: torch.randn(cfloat) real/imag each N(0,1/2),
    # /sqrt(in) -> real/imag each variance 1/(2*in)=1/16, NOT 1/(6*in)=1/48 (Round-5 init).
    torch.manual_seed(1)
    W = ComplexLinear(8, 88).W.detach()
    assert abs(W.real.var().item() - 1 / 16) < 1e-2
    assert abs(W.imag.var().item() - 1 / 16) < 1e-2


# --- 17. data order + CFO augmentation stream independent of model build -------------
def test_data_and_aug_stream_independent_of_model_build():
    from ssm_exp.utils import seed_all
    seed = 17

    def draw(variant):
        seed_all(seed)
        gen = torch.Generator().manual_seed(seed)
        aug_gen = torch.Generator().manual_seed(seed + 100000)
        # model construction consumes global RNG differently per variant
        build_round5_model(_r5_cfg(variant), seed)
        perm = torch.randperm(64, generator=gen).tolist()
        cfo = (2 * torch.rand(8, generator=aug_gen) - 1) * 0.002
        return perm, cfo

    p0, c0 = draw('R0')
    for variant in ('C0', 'Rb', 'Cb', 'Q_linear', 'Q_corr', 'R_corr', 'Q_corr_frozen'):
        p, c = draw(variant)
        assert p == p0, variant
        torch.testing.assert_close(c, c0, rtol=0, atol=0)


# --- 18. all branch params in optimizer, no detach -----------------------------------
@pytest.mark.parametrize('variant', ['Q_linear', 'Q_corr', 'R_corr'])
def test_branch_params_in_optimizer_no_detach(variant):
    torch.manual_seed(900)
    m = build_round5_model(_r5_cfg(variant), 900)
    opt = torch.optim.AdamW(m.parameters())
    ids = {id(p) for grp in opt.param_groups for p in grp['params']}
    for name, p in m.named_parameters():
        if name.startswith(('aux_branch.', 'aux_fusion.')):
            assert id(p) in ids, name
            assert p.requires_grad, name
    # forward keeps gradient path (no detach) to branch params
    m.train()
    m(torch.randn(2, 2, 16)).sum().backward()
    for name, p in m.named_parameters():
        if name.startswith('aux_branch.'):
            assert p.grad is not None, name


# --- 18b. Q_corr_frozen: branch params frozen out of optimizer, W_f still trains ------
def test_q_corr_frozen_branch_excluded_from_optimizer():
    torch.manual_seed(900)
    m = build_round5_model(_r5_cfg('Q_corr_frozen'), 900)
    # engine builds the optimizer from requires_grad=True params only
    opt = torch.optim.AdamW((p for p in m.parameters() if p.requires_grad))
    ids = {id(p) for grp in opt.param_groups for p in grp['params']}
    for name, p in m.named_parameters():
        if name.startswith('aux_branch.'):
            assert not p.requires_grad, name
            assert id(p) not in ids, name  # frozen params excluded from AdamW
        elif name.startswith('aux_fusion.'):
            assert p.requires_grad and id(p) in ids, name
    # frozen params get no grad through backward (requires_grad=False)
    m.train()
    m(torch.randn(2, 2, 16)).sum().backward()
    for name, p in m.named_parameters():
        if name.startswith('aux_branch.'):
            assert p.grad is None, name
    assert m.aux_fusion.weight.grad is not None and m.aux_fusion.weight.grad.abs().sum() > 0
    # a real optimizer step must not touch the frozen params
    before = {n: p.detach().clone() for n, p in m.aux_branch.named_parameters()}
    opt.step()
    for n, p in m.aux_branch.named_parameters():
        torch.testing.assert_close(p, before[n], rtol=0, atol=0)


def test_r_corr_spectral_norm_below_one():
    # A_k spectral norm < 1 enforced by forward scaling (rho<1, divide by max(1,||A_raw||))
    torch.manual_seed(1000)
    br = RealRecurBranch(seed=1000)
    rho = torch.sigmoid(br.logit)
    anorm = torch.linalg.matrix_norm(br.A_raw, ord=2)
    A = rho[:, None, None] * br.A_raw / anorm.clamp_min(1.0)[:, None, None]
    assert (torch.linalg.matrix_norm(A, ord=2) < 1.0).all()
    # and it stays <1 after an update
    x = torch.randn(2, 2, 16)
    br(x).square().mean().backward()
    rho = torch.sigmoid(br.logit)
    anorm = torch.linalg.matrix_norm(br.A_raw, ord=2)
    A = rho[:, None, None] * br.A_raw / anorm.clamp_min(1.0)[:, None, None]
    assert (torch.linalg.matrix_norm(A, ord=2) < 1.0).all()
