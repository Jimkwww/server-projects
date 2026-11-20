"""Round-4 encoder builders (no-CFO downstream + Q learning-phase study).

Adds, on top of the round-1/round-2 builders (which are NOT modified):

- 'G_corr': plain IQ PatchEmbedding + raw-IQ local complex-correlation fusion
  (physical_features with the energy slot zeroed -> only r.real, r.imag) + Global
  attention. This reproduces the historical Classifier({'variant':'G_corr'}) exactly,
  but exposes the pre-pooling token interface the pretrain/downstream code expects.
  phys_mean/phys_std (3-dim: log-energy, r.real, r.imag) are fit on the fixed rng-0
  8192-sample train subset, exactly like the old project's fit_physics.

- 'G_Qlearn': reused from build_encoder_r2 (Global + learnable 8-channel Q filter branch).

- fixed-Q control: freeze_aux(enc) freezes ONLY the ComplexRecurBranch parameters
  (raw_alpha/theta/b). The aux_fusion layer, the trunk and (downstream) the head keep
  training. This is the round-4 definition of "fixed Q" (frozen filter, not frozen fusion,
  not frozen encoder).

MAE masking is applied to the raw input BEFORE any auxiliary descriptor is computed, for
every aux-bearing backbone (G_corr included), so the descriptor can never leak the target.
"""
import torch
from torch import nn

from ssm_exp.model import (PatchEmbedding, Block, sinusoidal_positions, physical_features,
                           max_abs_norm, Classifier)
from pretrain.model import mask_patches, build_encoder
from round2_overnight.models import build_encoder_r2, Encoder2

R4_BACKBONES = ('G', 'Q', 'S1plain', 'G_Qlearn', 'G_corr')


class GcorrEncoder(nn.Module):
    """G_corr = plain PatchEmbedding + raw-IQ local complex-correlation fusion + Global
    attention. Pre-pooling token interface (forward_tokens/features)."""

    def __init__(self, cfg, seed=None):
        super().__init__()
        self.cfg = dict(cfg)
        dim, patch = cfg['dim'], cfg['patch']
        self.patch = patch
        self.backbone = 'G_corr'
        self.front = PatchEmbedding(patch, dim)
        # fusion of the standardized physical feature (energy slot zeroed in forward).
        # NOTE: the historical Classifier({'variant':'G_corr'}) uses the DEFAULT Linear init
        # for this fusion (NOT zero-init), so build_encoder_r4 loads the reference state_dict
        # verbatim to stay bit-identical. The default init here is only a placeholder.
        self.fusion = nn.Linear(3, dim, bias=False)
        self.blocks = nn.ModuleList([Block({**cfg, 'variant': 'G_corr'})
                                     for _ in range(cfg['depth'])])
        self.norm = nn.LayerNorm(dim)
        self.pos_mode = 'sinusoidal'
        self.aux_branch = None
        self.aux_fusion = None
        self.aux_readout = None
        self.register_buffer('phys_mean', torch.zeros(3))
        self.register_buffer('phys_std', torch.ones(3))
        self.register_buffer('aux_mean', torch.zeros(16))
        self.register_buffer('aux_std', torch.ones(16))

    def forward_tokens(self, x, mask=None):
        if mask is not None:
            x = mask_patches(x, mask, self.patch)
        u = self.front(x)
        u = u + sinusoidal_positions(u.shape[1], u.shape[2], u.device, u.dtype)
        p = (physical_features(x, self.patch) - self.phys_mean) / self.phys_std
        p = p.clone()
        p[..., 0] = 0.0  # energy slot zeroed (correlation only), matching historical G_corr
        u = u + self.fusion(p)
        pz = u.new_zeros(u.shape[0], u.shape[1], 3)
        for block in self.blocks:
            u = block(u, pz, None)
        return self.norm(u)

    def features(self, x, mask=None):
        return self.forward_tokens(x, mask).mean(1)


def build_encoder_r4(cfg, backbone, seed=None):
    if backbone in ('G', 'Q'):
        return build_encoder(cfg, backbone, seed)
    if backbone in ('S1plain', 'G_Qlearn'):
        return build_encoder_r2(cfg, backbone, seed)
    if backbone == 'G_corr':
        # Bit-identical to the historical Classifier({'variant':'G_corr'}) (the head and the
        # default phys_* buffers are dropped; phys_mean/phys_std are fit later by fit_phys_stats).
        ref = Classifier({**cfg, 'variant': 'G_corr'})
        enc = GcorrEncoder(cfg, seed)
        sd = {k: v for k, v in ref.state_dict().items() if not k.startswith('head.')}
        enc.load_state_dict(sd, strict=True)
        return enc
    raise ValueError('unknown backbone ' + repr(backbone))


def freeze_aux(enc):
    """Freeze the Q filter branch (raw_alpha/theta/b) only. Returns the encoder."""
    if getattr(enc, 'aux_branch', None) is not None:
        for p in enc.aux_branch.parameters():
            p.requires_grad_(False)
    return enc


def unfreeze_aux(enc):
    if getattr(enc, 'aux_branch', None) is not None:
        for p in enc.aux_branch.parameters():
            p.requires_grad_(True)
    return enc


def aux_param_hash(enc):
    """SHA256 of the aux_branch parameter bytes (used to prove 'fixed Q' really stayed fixed)."""
    import hashlib
    h = hashlib.sha256()
    if getattr(enc, 'aux_branch', None) is None:
        return None
    for name, p in enc.aux_branch.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def has_aux(backbone):
    return backbone in ('Q', 'G_Qlearn')
