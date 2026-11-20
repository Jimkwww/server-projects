"""SSM_L round-1 pretraining models.

Backbones (reuse the old project's building blocks unchanged; init is bit-identical to the
old G / Q_corr classifiers, verified by tests/test_encoder_init.py):

- G: ordinary Global Transformer. PatchEmbedding -> 6x Attention Block -> LayerNorm. Uses the
  historical absolute sinusoidal token positions. patch=8, dim=176, depth=6, heads=4.
- Q: original real Patch projection + 8-channel learnable complex recursive filter
  (ComplexRecurBranch, 8 complex states -> 16 real interleaved corr descriptor) + 16->176
  filtered-correlation fusion (zero-init Linear) + 6-layer S1 (ComplexSSM). The filter's 32
  real scalar parameters (raw_alpha[8] + theta[8] + b[8 complex]) participate in gradients.

Both encoders expose the pre-pooling [B, 128, 176] tokens. Pretraining adds a per-token
Linear(177 -> 16) reconstruction head (16 = 2*8 real IQ samples in [I0,Q0,I1,Q1,...] order),
shared capacity between MAE and denoising; the extra scalar slot carries the known sigma for
denoising and a constant zero for MAE.
"""
import math
import torch
from torch import nn

from ssm_exp.model import (PatchEmbedding, Block, ComplexRecurBranch, sinusoidal_positions,
                           Classifier, build_round5_model)


def stable_state_corr_pack(q, patch, eps=1e-8):
    """Per-channel patch-internal normalized adjacent complex correlation, identical to the old
    project's state_corr_pack for all non-degenerate inputs, but NaN-stable for fully-zero
    (MAE-masked) patches.

    The old denominator `sqrt(a*b) + eps` has gradient 0.5*b/sqrt(a*b) -> 0/0 when a=b=0 (a
    fully masked patch), which NaNs the aux-branch backward. Clamping `a*b` at eps^2 before the
    sqrt keeps the denominator EXACTLY `sqrt(a*b)+eps` whenever a*b > eps^2 (i.e. every real
    signal) while making the gradient finite at zero.
    """
    b, length, s = q.shape
    qp = q.reshape(b, length // patch, patch, s)
    cur, prev = qp[:, :, 1:, :], qp[:, :, :-1, :]
    num = (cur * prev.conj()).sum(2)
    ab = cur.abs().square().sum(2) * prev.abs().square().sum(2)
    den = ab.clamp_min(eps * eps).sqrt() + eps
    r = num / den
    return torch.stack((r.real, r.imag), -1).reshape(b, length // patch, 2 * s)


def mask_patches(x, mask, patch):
    """Zero out the masked patches of x [B,2,L]. mask [B,N] bool, True = masked.

    Zero-fill (not position removal) so the ordered recurrence keeps the exact time span;
    masked patches still occupy their positions and receive visible context through the
    recurrence's own decay (no implicit time re-warping)."""
    b, c, l = x.shape
    n = l // patch
    xp = x.view(b, c, n, patch)
    keep = (~mask).to(x.dtype).unsqueeze(1).unsqueeze(-1)  # [B,1,N,1]
    return (xp * keep).view(b, c, l)


class Encoder(nn.Module):
    """G or Q encoder. Exposes pre-pooling tokens; NO classification head (added downstream).

    The parameter names mirror the old Classifier's (front/blocks/norm/aux_branch/aux_fusion/
    aux_mean/aux_std) so `build_encoder` can load the old model's state_dict keys verbatim and
    the two are bit-identical at init."""

    def __init__(self, cfg, backbone, seed=None):
        super().__init__()
        if backbone not in ('G', 'Q'):
            raise ValueError("backbone must be 'G' or 'Q'")
        self.backbone = backbone
        self.cfg = dict(cfg)
        dim, patch = cfg['dim'], cfg['patch']
        self.patch = patch
        self.front = PatchEmbedding(patch, dim)
        if backbone == 'Q':
            self.aux_branch = ComplexRecurBranch(seed=(seed + 70002) if seed is not None else None)
            self.aux_readout = 'corr'
            self.aux_fusion = nn.Linear(16, dim, bias=False)
            nn.init.zeros_(self.aux_fusion.weight)
        else:
            self.aux_branch = None
            self.aux_readout = None
            self.aux_fusion = None
        variant = 'G' if backbone == 'G' else 'Q_corr'
        self.blocks = nn.ModuleList([Block({**cfg, 'variant': variant}) for _ in range(cfg['depth'])])
        self.norm = nn.LayerNorm(dim)
        self.pos_mode = 'sinusoidal' if backbone == 'G' else 'none'
        self.register_buffer('aux_mean', torch.zeros(16))
        self.register_buffer('aux_std', torch.ones(16))

    def aux_descriptor(self, x):
        """Raw 16-dim filtered-correlation descriptor before standardization (no detach, so
        gradient flows to the ComplexRecurBranch parameters)."""
        q = self.aux_branch(x)
        return stable_state_corr_pack(q, self.patch)

    def forward_tokens(self, x, mask=None):
        """x [B,2,L] max_abs-normalized -> tokens [B,N,dim] (after final LayerNorm, before
        pooling). If mask [B,N] is given (MAE), the masked patches of x are zeroed first, so
        both the front projection and the Q recurrence/correlation see only masked content."""
        if mask is not None:
            x = mask_patches(x, mask, self.patch)
        u = self.front(x)
        if self.pos_mode == 'sinusoidal':
            u = u + sinusoidal_positions(u.shape[1], u.shape[2], u.device, u.dtype)
        if self.aux_branch is not None:
            f = (self.aux_descriptor(x) - self.aux_mean) / self.aux_std
            u = u + self.aux_fusion(f)
        p = u.new_zeros(u.shape[0], u.shape[1], 3)
        for block in self.blocks:
            u = block(u, p, None)
        return self.norm(u)

    def features(self, x, mask=None):
        """Mean-pooled tokens [B,dim] (the exact pre-head representation the old Classifier
        feeds to its linear head)."""
        return self.forward_tokens(x, mask).mean(1)


class ReconstructHead(nn.Module):
    """Per-token Linear(dim+1 -> 2*patch) reconstruction. The extra scalar slot is appended to
    every token position: denoising broadcasts the sample's known sigma; MAE feeds zero."""

    def __init__(self, dim, patch):
        super().__init__()
        self.dim, self.patch = dim, patch
        self.proj = nn.Linear(dim + 1, 2 * patch)

    def forward(self, tokens, extra):
        # tokens [B,N,dim], extra [B,N] (scalar per position, already broadcast by caller).
        z = torch.cat((tokens, extra.unsqueeze(-1)), -1)  # [B,N,dim+1]
        out = self.proj(z)                                # [B,N,2*patch]
        b, n, _ = out.shape
        # 2*patch reals in [I0,Q0,I1,Q1,...] order (slot 2p+q = channel q at within-patch p)
        # -> [B,N,patch,2] -> [B,2,N,patch] -> [B,2,L], so channel q lands at time 8n+p.
        return out.reshape(b, n, self.patch, 2).permute(0, 3, 1, 2).reshape(b, 2, n * self.patch)


class PretrainModel(nn.Module):
    """encoder + reconstruction head for MAE / denoising pretraining."""

    def __init__(self, encoder, patch):
        super().__init__()
        self.encoder = encoder
        self.patch = patch
        self.decoder = ReconstructHead(encoder.cfg['dim'], patch)

    def forward(self, x, mask=None, extra=None):
        tokens = self.encoder.forward_tokens(x, mask=mask)
        b, n = tokens.shape[0], tokens.shape[1]
        extra = torch.zeros(b, n, device=x.device, dtype=x.dtype) if extra is None else extra
        return self.decoder(tokens, extra)


class DownstreamModel(nn.Module):
    """Frozen encoder + mean pool + linear classification head (probe / fine-tune / scratch)."""

    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.cfg['dim'], num_classes)

    def forward(self, x, mask=None):
        return self.head(self.encoder.features(x, mask))


def build_encoder(cfg, backbone, seed=None):
    """Build an Encoder and load the old project's G / Q_corr init verbatim.

    For Q this reproduces build_round5_model (S1-shared front/blocks/norm, per-seed aux branch);
    for G it reproduces Classifier({'variant':'G'}). The classification head and the unused
    physical-feature buffers of the old Classifier are dropped (re-added downstream / never used).
    """
    variant = 'G' if backbone == 'G' else 'Q_corr'
    if variant == 'Q_corr':
        ref = build_round5_model({**cfg, 'variant': 'Q_corr'}, seed)
    else:
        ref = Classifier({**cfg, 'variant': 'G'})
    enc = Encoder(cfg, backbone, seed)
    sd = {k: v for k, v in ref.state_dict().items()
          if not k.startswith('head.') and not k.startswith('phys_')}
    enc.load_state_dict(sd, strict=True)
    return enc
