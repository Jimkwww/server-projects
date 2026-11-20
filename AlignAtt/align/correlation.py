"""Complex cross-correlation descriptor between patch segments.

Implements the per-token complex correlation rho_ij(tau) defined in section 6 of
``02_alignment_attention_experiment.md``, plus a naive double-loop reference used
only to validate the vectorized path (phase 0, step 2 of the plan).

Conventions (deliberately kept unambiguous, not inferred from plots):

* ``z`` is ``[B, T, R]`` complex64: token ``i`` owns an ``R``-point complex
  context segment ``z[b, i, :] = I + j*Q`` in sample order, centred on the token.
  ``R`` defaults to the patch size and can be extended (``R = 2P / 4P``) when the
  short patch is statistically unstable (section 5).
* ``tau`` is an integer sample delay (units: raw IQ samples).
* For the pair ``(i, j)`` at lag ``tau`` the valid overlap indices ``n`` are
  those satisfying ``0 <= n < R`` and ``0 <= n+tau < R``, i.e.
  ``n in [lo, hi)`` with ``lo = max(0, -tau)``, ``hi = min(R, R-tau)``.
* ``sample_valid`` is an optional ``[B, T, R]`` bool marking which context samples
  are real (not boundary/padding). Boundary samples are excluded from every sum.

    C_ij(tau) = sum_n m_ij(n) z_i[n] conj(z_j[n+tau]),   m_ij(n) = v_i[n] v_j[n+tau]
    E_i(tau)  = sum_n m_ij(n) |z_i[n]|^2                (same overlap)
    E_j(tau)  = sum_n m_ij(n) |z_j[n+tau]|^2            (same overlap)
    rho_ij(tau) = C_ij(tau) / (sqrt(E_i E_j) + eps)

The numerator and denominator always use the *same* pair-specific overlap for
each tau; different lags use their own overlap and are never normalised by the
full-segment energy.

Sign convention: if ``z_j`` is a copy of ``z_i`` delayed by ``tau_true`` samples
and rotated by ``e^{j phi}``, the peak sits at ``tau = tau_true`` and
``arg(rho) = -phi`` (see the unit test ``test_known_delay_and_phase``).
"""
import torch


def iq_to_complex(iq):
    """Convert ``[..., 2]`` real IQ (last dim I then Q) to complex64."""
    return torch.view_as_complex(iq.float().contiguous())


def rho_all(z, tau_radius, eps=1e-6, min_valid=4, sample_valid=None):
    """Vectorized rho for every pair ``(i, j)`` and lag ``tau in [-d, d]``.

    Args:
        z: ``[B, T, R]`` complex64.
        tau_radius: integer delay radius ``d``.
        eps: denominator floor (plan default 1e-6 for normalised input).
        min_valid: minimum overlap length below which a lag is flagged invalid
            (plan default ``max(4, ceil(R/2))``).
        sample_valid: optional ``[B, T, R]`` bool; defaults to all-True.

    Returns:
        rho:      ``[B, T, T, K]`` complex64
        overlap:  ``[B, T, T, K]`` int32 (pair-specific valid overlap length)
        valid:    ``[B, T, T, K]`` bool  (enough energy AND enough overlap)
    """
    B, T, R = z.shape
    taus = list(range(-tau_radius, tau_radius + 1))
    K = len(taus)
    rho = torch.empty((B, T, T, K), dtype=torch.complex64, device=z.device)
    overlap = torch.empty((B, T, T, K), dtype=torch.int32, device=z.device)
    energy_ok = torch.empty((B, T, T, K), dtype=torch.bool, device=z.device)
    if sample_valid is None:
        sample_valid = torch.ones(B, T, R, dtype=torch.bool, device=z.device)
    for k, tau in enumerate(taus):
        lo = max(0, -tau)
        hi = min(R, R - tau)
        zi = z[:, :, lo:hi]
        zj = z[:, :, lo + tau:hi + tau]
        vi = sample_valid[:, :, lo:hi]
        vj = sample_valid[:, :, lo + tau:hi + tau]
        C = torch.bmm(zi * vi, (zj * vj).conj().transpose(-1, -2))   # [B, T, T]
        ei = zi.abs().square() * vi                                  # [B, T, L]
        ej = zj.abs().square() * vj                                  # [B, T, L]
        Ei = torch.bmm(ei, vj.float().transpose(-1, -2))            # [B, T, T]
        Ej = torch.bmm(vi.float(), ej.transpose(-1, -2))            # [B, T, T]
        denom = (Ei * Ej).sqrt() + eps
        rho[..., k] = C / denom
        overlap[..., k] = torch.bmm(vi.float(), vj.float().transpose(-1, -2)).round().to(torch.int32)
        energy_ok[..., k] = (Ei > eps) & (Ej > eps)
    valid = energy_ok & (overlap >= min_valid)
    return rho, overlap, valid


def rho_naive(z, tau_radius, eps=1e-6, min_valid=4, sample_valid=None):
    """Reference implementation with explicit loops (slow, correctness only)."""
    B, T, R = z.shape
    taus = list(range(-tau_radius, tau_radius + 1))
    K = len(taus)
    rho = torch.zeros((B, T, T, K), dtype=torch.complex64, device=z.device)
    overlap = torch.zeros((B, T, T, K), dtype=torch.int32, device=z.device)
    valid = torch.zeros((B, T, T, K), dtype=torch.bool, device=z.device)
    for b in range(B):
        for i in range(T):
            for j in range(T):
                for k, tau in enumerate(taus):
                    lo = max(0, -tau)
                    hi = min(R, R - tau)
                    zi = z[b, i, lo:hi]
                    zj = z[b, j, lo + tau:hi + tau]
                    if sample_valid is None:
                        m = torch.ones(hi - lo, device=z.device, dtype=torch.float32)
                    else:
                        m = (sample_valid[b, i, lo:hi] & sample_valid[b, j, lo + tau:hi + tau]).float()
                    C = (zi * zj.conj() * m).sum()
                    Ei = (zi.abs().square() * m).sum()
                    Ej = (zj.abs().square() * m).sum()
                    rho[b, i, j, k] = C / ((Ei * Ej).sqrt() + eps)
                    overlap[b, i, j, k] = int(m.sum())
                    valid[b, i, j, k] = (Ei > eps) and (Ej > eps) and (overlap[b, i, j, k] >= min_valid)
    return rho, overlap, valid
