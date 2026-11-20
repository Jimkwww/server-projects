"""Round6 encoder builders: G / Q / Q_bi.

- G, Q: delegate to pretrain.model.build_encoder, byte-identical to round5 G-MAE / Q-MAE.
- Q_bi: the SAME Q front (PatchEmbedding + ComplexRecurBranch seed+70002 + zero-init
  16->dim aux_fusion) + a BIDIRECTIONAL S1 trunk: 6 x Block(variant='A64') =
  BidirectionalSSM with state 64 PER DIRECTION (forward 64 + backward 64). The Q front stays
  unidirectional; only the SSM trunk's scan direction + state allocation change.

  Q_bi init: front / aux_branch / aux_fusion / norm / aux stats reuse Q's round5 shared-init
  (byte-identical); the bidirectional blocks (A64) are freshly random-initialized by the
  seeded global RNG at build time (NOT reused from Q's unidirectional ComplexSSM state=128,
  whose keys/shapes differ).

Note on parameter counts (recorded in model_info, not assumed): A64 has two independent
_Direction blocks (each B/C/selector/decay), so its selector first layer Linear(dim+3,hidden)
is duplicated relative to Q's single selector. B/C matrices total the same (64+64 vs 128
state); the selector input projection is the extra cost. Actual counts are written at build.
"""
import torch
from torch import nn

from ssm_exp.model import PatchEmbedding, Block, ComplexRecurBranch
from pretrain.model import (stable_state_corr_pack, mask_patches, build_encoder,
                            ReconstructHead)

R6_BACKBONES = ('G', 'Q', 'Q_bi')


class QbiEncoder(nn.Module):
    """Q front + bidirectional A64 SSM trunk. Pre-pooling token interface (forward_tokens /
    features), same state_dict key names as the round1/round2 encoders where the modules are
    shared (front/aux_branch/aux_fusion/norm/aux_mean/aux_std)."""

    def __init__(self, cfg, seed=None):
        super().__init__()
        self.backbone = 'Q_bi'
        self.cfg = dict(cfg)
        dim, patch = cfg['dim'], cfg['patch']
        self.patch = patch
        self.front = PatchEmbedding(patch, dim)
        self.aux_branch = ComplexRecurBranch(seed=(seed + 70002) if seed is not None else None)
        self.aux_readout = 'corr'
        self.aux_fusion = nn.Linear(16, dim, bias=False)
        nn.init.zeros_(self.aux_fusion.weight)
        self.blocks = nn.ModuleList([Block({**cfg, 'variant': 'A64'}) for _ in range(cfg['depth'])])
        self.norm = nn.LayerNorm(dim)
        self.pos_mode = 'none'
        self.register_buffer('aux_mean', torch.zeros(16))
        self.register_buffer('aux_std', torch.ones(16))

    def aux_descriptor(self, x):
        q = self.aux_branch(x)
        return stable_state_corr_pack(q, self.patch)

    def forward_tokens(self, x, mask=None):
        # MAE mask is applied to raw IQ BEFORE the Q branch descriptor, so neither scan
        # direction can leak masked IQ through the aux branch.
        if mask is not None:
            x = mask_patches(x, mask, self.patch)
        u = self.front(x)
        if self.aux_branch is not None:
            f = (self.aux_descriptor(x) - self.aux_mean) / self.aux_std
            u = u + self.aux_fusion(f)
        p = u.new_zeros(u.shape[0], u.shape[1], 3)
        for block in self.blocks:
            u = block(u, p, None)
        return self.norm(u)

    def features(self, x, mask=None):
        return self.forward_tokens(x, mask).mean(1)


def build_encoder_r6(cfg, backbone, seed=None):
    if backbone in ('G', 'Q'):
        return build_encoder(cfg, backbone, seed)
    if backbone == 'Q_bi':
        return QbiEncoder(cfg, seed)
    raise ValueError('unknown backbone ' + repr(backbone))


class JointPretrainModel(nn.Module):
    """Shared encoder + two independent reconstruction heads (MAE + denoise).

    L_joint = 0.5 * L_MAE + 0.5 * L_DN. The two heads are independent (separate ReconstructHead
    parameters); only the encoder is shared. No dynamic weighting / gradient-conflict
    correction (round6 task book section 5)."""

    def __init__(self, encoder, patch):
        super().__init__()
        self.encoder = encoder
        self.patch = patch
        self.mae_decoder = ReconstructHead(encoder.cfg['dim'], patch)
        self.dn_decoder = ReconstructHead(encoder.cfg['dim'], patch)

    def forward_mae(self, x, mask):
        tokens = self.encoder.forward_tokens(x, mask=mask)
        b, n = tokens.shape[0], tokens.shape[1]
        extra = torch.zeros(b, n, device=x.device, dtype=x.dtype)
        return self.mae_decoder(tokens, extra)

    def forward_dn(self, x, sigma):
        tokens = self.encoder.forward_tokens(x, mask=None)
        b, n = tokens.shape[0], tokens.shape[1]
        extra = sigma[:, None].expand(b, n)
        return self.dn_decoder(tokens, extra)
