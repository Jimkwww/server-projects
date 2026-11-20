"""Phase 0: algebra & implementation correctness for the relation descriptor.

Covers section 8 / stage 0 of the plan:

1. vectorized rho_all == naive double-loop reference;
2. known integer delay + phase -> peak at the right lag, arg(rho) == -phi;
3. common phase rotation leaves the descriptor invariant;
4. all-zero / low-energy / boundary-short-overlap stay finite and flag invalid;
5. gate == 0 -> relation-augmented forward equals the frozen baseline forward
   (same dtype/computation path), and gate & relation MLP have nonzero gradients.

Runnable directly:  python -m align.test_correlation
"""
import math
import torch

from .correlation import rho_all, rho_naive, iq_to_complex
from .relation_bias import RelationBias
from .align_models import RelationGlobalEncoder
from emfm.models import IQEncoder


def _complex_iq(z, pad=0):
    """Wrap complex [B,T,P] into real IQ [B,T*P,2] with optional zero padding."""
    B, T, P = z.shape
    iq = torch.view_as_real(z).reshape(B, T * P, 2)
    valid = torch.ones(B, T * P, dtype=torch.bool)
    if pad:
        iq = torch.cat([iq, torch.zeros(B, pad, 2)], dim=1)
        valid = torch.cat([valid, torch.zeros(B, pad, dtype=torch.bool)], dim=1)
    return iq, valid


def test_vectorized_matches_naive():
    torch.manual_seed(0)
    B, T, P, d = 2, 5, 8, 2
    z = torch.randn(B, T, P, dtype=torch.complex64)
    z[0, 4] = 0  # force one zero-energy token
    r1, o1, v1 = rho_all(z, d)
    r2, o2, v2 = rho_naive(z, d)
    assert torch.allclose(r1, r2, atol=1e-5), "rho mismatch"
    assert torch.equal(o1, o2) and torch.equal(v1, v2), "overlap/valid mismatch"
    print("PASS  vectorized == naive")


def test_sample_valid_masking():
    torch.manual_seed(10)
    B, T, R, d = 2, 5, 16, 2
    z = torch.randn(B, T, R, dtype=torch.complex64)
    sv = torch.ones(B, T, R, dtype=torch.bool)
    sv[:, 0, :2] = False     # boundary samples of token 0
    sv[:, -1, -3:] = False   # boundary samples of last token
    z[1, 2] = 0              # zero-energy token in batch 1
    r1, o1, v1 = rho_all(z, d, sample_valid=sv)
    r2, o2, v2 = rho_naive(z, d, sample_valid=sv)
    assert torch.allclose(r1, r2, atol=1e-5), "rho mismatch under sample mask"
    assert torch.equal(o1, o2) and torch.equal(v1, v2), "overlap/valid mismatch under sample mask"
    assert torch.isfinite(r1.real).all() and torch.isfinite(r1.imag).all()
    print("PASS  per-sample boundary mask == naive")


def test_known_delay_and_phase():
    torch.manual_seed(1)
    B, T, P, d = 1, 3, 8, 2
    base = torch.randn(P, dtype=torch.complex64)
    z = torch.stack([base, base, base]).unsqueeze(0)  # [1,3,8]
    tau_true, phi = 1, 0.7
    # j-th segment = delayed (by tau_true) + rotated copy of i-th segment
    rot = torch.exp(torch.tensor(1j * phi))
    z[0, 1, tau_true:] = z[0, 0, :P - tau_true] * rot
    rho, _, valid = rho_all(z, d)
    ri = rho[0, 0, 1]                      # pair (i=0, j=1), K lags
    k = torch.argmax(ri.abs())
    assert k == tau_true + d, f"peak at lag index {k}, expected {tau_true + d}"
    assert abs(torch.angle(ri[k]) - (-phi)) < 1e-4, "phase sign mismatch"
    assert valid[0, 0, 1, k].item(), "true match should be valid"
    print("PASS  known delay -> peak position, phase sign == -phi")


def test_common_phase_invariance():
    torch.manual_seed(2)
    B, T, P, d = 1, 4, 8, 2
    z = torch.randn(B, T, P, dtype=torch.complex64)
    r1, o1, v1 = rho_all(z, d)
    z2 = z * torch.exp(torch.tensor(1j * 0.9))
    r2, o2, v2 = rho_all(z2, d)
    assert torch.allclose(r1.abs(), r2.abs(), atol=1e-5), "common rotation changed |rho|"
    assert torch.allclose(r1, r2, atol=1e-5), "common rotation changed rho"
    assert torch.equal(o1, o2) and torch.equal(v1, v2)
    print("PASS  common phase rotation leaves rho invariant")


def test_degenerate_stays_finite():
    B, T, P, d = 2, 3, 8, 2
    for name, z in [
            ("all zero", torch.zeros(B, T, P, dtype=torch.complex64)),
            ("tiny energy", torch.full((B, T, P), 1e-9, dtype=torch.complex64))]:
        rho, overlap, valid = rho_all(z, d)
        assert torch.isfinite(rho.real).all() and torch.isfinite(rho.imag).all(), name
        assert not valid.any(), f"{name}: degenerate input flagged valid"
        print(f"PASS  {name} -> finite, all invalid")


def test_gate_zero_matches_baseline():
    torch.manual_seed(3)
    base_cfg = dict(kind="global", patch_size=8, dims=[32], depths=[2],
                    heads=[2], window=8, shifted=True, ratio=4, dropout=0.)
    rel_cfg = dict(mode="full", tau_radius=2, mlp_hidden=16, gate_init=0.0)
    base = IQEncoder(**base_cfg)
    rel = RelationGlobalEncoder({**base_cfg, "relation": rel_cfg})
    rel.load_state_dict(base.state_dict(), strict=False)  # share backbone weights
    B, T, P = 2, 16, 8
    z = torch.randn(B, T, P, dtype=torch.complex64)
    z[:, -2:, :] = 0
    iq, valid = _complex_iq(z)
    with torch.no_grad():
        y0, m0 = base(iq, valid)
        y1, m1 = rel(iq, valid)
    assert torch.equal(m0, m1)
    assert torch.allclose(y0, y1, atol=1e-6), "gate=0 must equal baseline forward"
    print("PASS  gate=0 -> identical to baseline forward")


def test_gate_and_mlp_gradients():
    B, T, P = 2, 8, 8
    z = torch.randn(B, T, P, dtype=torch.complex64)
    z[0, 7] = 0
    token_valid = torch.ones(B, T, dtype=torch.bool)
    token_valid[0, 7] = False

    # (a) gate escapes zero: a loss that is not flat at bias==0 (like the softmax
    #     path in real training) must give the gate a nonzero gradient.
    rb = RelationBias(4, "full", R=P, tau_radius=2, gate_init=0.0)
    bias = rb(z, token_valid)
    w = torch.arange(1, bias.numel() + 1, dtype=bias.dtype, device=bias.device).reshape_as(bias)
    (bias * w).sum().backward()
    assert rb.gate.grad is not None and rb.gate.grad.abs().sum() > 0, "gate stuck at zero"

    # (b) the relation MLP is not zero-initialised and, once the gate is nonzero,
    #     receives finite nonzero gradients.
    rb2 = RelationBias(4, "full", R=P, tau_radius=2, gate_init=1.0)
    bias2 = rb2(z, token_valid)
    assert bias2.abs().max() > 0, "MLP output is all zero (bad init)"
    bias2.square().mean().backward()
    grads = [p.grad for p in rb2.mlp.parameters()]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in grads), "mlp gradient problem"
    print("PASS  gate escapes zero; relation MLP is non-degenerate and trainable")


def main():
    test_vectorized_matches_naive()
    test_sample_valid_masking()
    test_known_delay_and_phase()
    test_common_phase_invariance()
    test_degenerate_stays_finite()
    test_gate_zero_matches_baseline()
    test_gate_and_mlp_gradients()
    print("\nAll phase-0 correctness tests passed.")


if __name__ == "__main__":
    main()
