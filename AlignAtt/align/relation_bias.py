"""Relation path: raw-IQ pairwise descriptor -> per-head scalar attention bias.

Turns the complex correlation descriptor of ``correlation.py`` into the additive
attention-logit term of section 4:

    L_ij^(l,h) = q_i^(l,h) . k_j^(l,h) / sqrt(d_h)  +  g_h * r_ij^(h)

The content term is left untouched; the baseline has no separate positional bias,
so the only added term is ``g_h * r_ij^(h)``. ``g`` is a per-head learnable gate
initialised to 0 (configurable), and the relation MLP is NOT zero-initialised, so
neither the gate nor the mapping gets stuck at zero gradient (section 4).

Modes (experiment matrix, section 7):

* ``full``       (A3)  per-lag [Re, Im, abs, overlap_frac, valid], K = 2d+1 lags.
* ``zero_delay`` (A1)  tau = 0 only, same features, K = 1.
* ``magnitude``  (A2)  per-lag [abs, overlap_frac, valid], K = 2d+1 lags.
* ``real_mlp``   (A4)  no correlation: plain MLP on the concatenated real I/Q of
    the two segments (same context + parameter budget, no complex/delay maths).
* ``none``       (A0)  module unused (pure baseline).

The descriptor is computed once per input and shared across layers/heads; only a
small per-head mapping (hidden width 16 -> H outputs) differs per head.
"""
import math
import torch
from torch import nn

from .correlation import rho_all


class RelationBias(nn.Module):
    def __init__(self, heads, mode, R=8, tau_radius=2, mlp_hidden=16,
                 eps=1e-6, gate_init=0.0, shuffle=False, shuffle_seed=0):
        super().__init__()
        if mode not in ("none", "full", "zero_delay", "magnitude", "real_mlp"):
            raise ValueError(f"Unknown relation mode: {mode}")
        if R < 1 or tau_radius < 0 or tau_radius >= R:
            raise ValueError("Require 0 <= tau_radius < R")
        self.mode = mode
        self.R = R
        self.tau_radius = 0 if mode == "zero_delay" else tau_radius
        self.eps = eps
        self.min_valid = max(4, (R + 1) // 2)
        self.shuffle = bool(shuffle)
        self.shuffle_seed = shuffle_seed
        self.gate = nn.Parameter(torch.full((heads,), float(gate_init)))

        K = 2 * self.tau_radius + 1
        if mode == "magnitude":
            self.feature_dim = K * 3          # abs, overlap_frac, valid
        elif mode in ("full", "zero_delay"):
            self.feature_dim = K * 5          # Re, Im, abs, overlap_frac, valid
        elif mode == "real_mlp":
            self.feature_dim = 4 * R          # Re(z_i), Im(z_i), Re(z_j), Im(z_j)
        else:
            self.feature_dim = 0
        if self.feature_dim:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, heads))
        self._perm = None
        self._perm_key = None

    # ------------------------------------------------------------------ #
    def forward(self, z, token_valid, sample_valid=None):
        """Compute the gated, per-head bias.

        Args:
            z: ``[B, T, R]`` complex64 context segments (raw IQ, before embedding).
            token_valid: ``[B, T]`` bool (True = non-padding token).
            sample_valid: optional ``[B, T, R]`` bool (boundary/padding samples).

        Returns:
            bias: ``[B, H, T, T]`` float32, ready to add to attention logits.
        """
        if self.mode == "none":
            return None
        B, T, R = z.shape
        if self.mode == "real_mlp":
            r = self._real_mlp(z)                     # [B, T, T, H]
            pair_valid = token_valid[:, :, None] & token_valid[:, None, :]
        else:
            rho, overlap, valid = rho_all(z, self.tau_radius, self.eps, self.min_valid,
                                          sample_valid=sample_valid)
            desc = self._descriptor(rho, overlap, valid)   # [B, T, T, F]
            r = self.mlp(desc)                             # [B, T, T, H]
            pair_valid = (token_valid[:, :, None] & token_valid[:, None, :]
                          & valid.any(-1))
        r = r.masked_fill(~pair_valid[..., None], 0.0)
        if self.shuffle:
            r = self._shuffle(r)                       # [B, T, T, H]
        r = r.permute(0, 3, 1, 2)                      # [B, H, T, T]
        return self.gate[None, :, None, None] * r

    # ------------------------------------------------------------------ #
    def _descriptor(self, rho, overlap, valid):
        K = rho.shape[-1]
        frac = overlap.float() / self.R
        parts = []
        for k in range(K):
            r = rho[..., k]
            if self.mode == "magnitude":
                parts += [r.abs(), frac[..., k], valid[..., k].float()]
            else:  # full / zero_delay
                parts += [r.real, r.imag, r.abs(), frac[..., k], valid[..., k].float()]
        return torch.stack(parts, dim=-1)

    def _real_mlp(self, z):
        B, T, R = z.shape
        zr = torch.view_as_real(z)                     # [B, T, R, 2]
        zi = zr[:, :, None, :, :]                      # [B, T, 1, R, 2]
        zj = zr[:, None, :, :, :]                      # [B, 1, T, R, 2]
        feat = torch.cat((zi.expand(B, T, T, R, 2),
                          zj.expand(B, T, T, R, 2)), dim=-2)  # [B, T, T, 2R, 2]
        return self.mlp(feat.reshape(B, T, T, 4 * R))

    def _shuffle(self, r):
        """Deterministic per-sample re-pairing of the j index (experiment A5).

        Keeps the original QK, positions and V intact; only the relation pairing
        is permuted, so the compute budget is unchanged while correct i<->j
        correspondence is destroyed (section 7).
        """
        B, T, T2, H = r.shape
        key = (B, T)
        if self._perm_key != key:
            gen = torch.Generator().manual_seed(self.shuffle_seed)  # randperm is CPU-only
            perm = torch.stack([torch.randperm(T, generator=gen) for _ in range(B)]).to(r.device)
            self._perm = perm
            self._perm_key = key
        idx = self._perm[:, None, :, None].expand(B, T, T, H)
        return r.gather(2, idx)

    # ------------------------------------------------------------------ #
    def param_count(self):
        return sum(p.numel() for p in self.parameters())
