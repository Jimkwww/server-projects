"""Round-2 backbone extensions.

Adds the two missing cells of the D 2x2 ablation matrix while keeping the round-1 G/Q
encoders bit-identical (verified by test_bitidentity.py):

- 'S1plain': plain IQ Patch + S1 (ComplexSSM, variant 'S1'), NO Q branch.  This is exactly
  the historical S1 backbone (Q minus the aux correlation branch).
- 'G_Qlearn': plain IQ Patch + learnable 8-channel Q branch (ComplexRecurBranch seed+70002,
  learnable raw_alpha/theta/b) + Global attention (variant 'G_Q_corr', sinusoidal positions).
  The Q branch is initialized from the same-seed Q-MAE init state and the Global trunk from
  the same-seed G-MAE init state (both use seed+70002 for the aux branch and the seeded global
  RNG for the G trunk, matching the round-1 shared-init cache exactly).

The encoder interface matches round-1 pretrain.model.Encoder: pre-pooling tokens via
forward_tokens/features, aux_descriptor + aux_mean/aux_std buffers for the Q branch, and the
same state_dict key names, so downstream probe/finetune code is unchanged.
"""
import torch
from torch import nn

from ssm_exp.model import (PatchEmbedding, Block, ComplexRecurBranch, sinusoidal_positions,
                           Classifier, build_round5_model, build_global_qcorr_model)
from pretrain.model import stable_state_corr_pack, mask_patches, build_encoder

BACKBONES = ('G', 'Q', 'S1plain', 'G_Qlearn')
_VARIANT = {'G': 'G', 'Q': 'Q_corr', 'S1plain': 'S1', 'G_Qlearn': 'G_Q_corr'}
_POS_MODE = {'G': 'sinusoidal', 'Q': 'none', 'S1plain': 'none', 'G_Qlearn': 'sinusoidal'}
_HAS_AUX = {'G': False, 'Q': True, 'S1plain': False, 'G_Qlearn': True}


class Encoder2(nn.Module):
    """4-backbone encoder exposing pre-pooling [B,N,dim] tokens (same interface as round-1
    pretrain.model.Encoder)."""

    def __init__(self, cfg, backbone, seed=None):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError("backbone must be one of " + repr(BACKBONES))
        self.backbone = backbone
        self.cfg = dict(cfg)
        dim, patch = cfg['dim'], cfg['patch']
        self.patch = patch
        self.front = PatchEmbedding(patch, dim)
        self.aux_branch = None
        self.aux_fusion = None
        self.aux_readout = None
        if _HAS_AUX[backbone]:
            self.aux_branch = ComplexRecurBranch(seed=(seed + 70002) if seed is not None else None)
            self.aux_readout = 'corr'
            self.aux_fusion = nn.Linear(16, dim, bias=False)
            nn.init.zeros_(self.aux_fusion.weight)
        self.blocks = nn.ModuleList([Block({**cfg, 'variant': _VARIANT[backbone]})
                                     for _ in range(cfg['depth'])])
        self.norm = nn.LayerNorm(dim)
        self.pos_mode = _POS_MODE[backbone]
        self.register_buffer('aux_mean', torch.zeros(16))
        self.register_buffer('aux_std', torch.ones(16))

    def aux_descriptor(self, x):
        q = self.aux_branch(x)
        return stable_state_corr_pack(q, self.patch)

    def forward_tokens(self, x, mask=None):
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
        return self.forward_tokens(x, mask).mean(1)


def build_encoder_r2(cfg, backbone, seed=None):
    """Build a 4-backbone encoder. 'G'/'Q' delegate to the round-1 builder (bit-identical);
    'S1plain'/'G_Qlearn' are initialized from the historical Classifier variants."""
    if backbone in ('G', 'Q'):
        return build_encoder(cfg, backbone, seed)
    if backbone == 'G_Qlearn':
        ref = build_global_qcorr_model({**cfg, 'variant': 'G_Q_corr'}, seed)
    else:  # S1plain
        ref = Classifier({**cfg, 'variant': 'S1'})
    enc = Encoder2(cfg, backbone, seed)
    sd = {k: v for k, v in ref.state_dict().items()
          if not k.startswith('head.') and not k.startswith('phys_')}
    enc.load_state_dict(sd, strict=True)
    return enc
