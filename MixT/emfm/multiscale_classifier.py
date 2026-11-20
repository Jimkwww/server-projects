"""Multi-scale classification head for the hierarchical (Swin) encoder.

Uses forward hooks to collect the final token features of every encoder stage,
mean-pools each stage independently (weighted by per-stage valid masks), then
concatenates the pooled vectors into the classification head.

This is a NEW file; it does not modify the frozen emfm/models.py. The encoder
itself is the unmodified IQEncoder, so pretrained weights load unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .models import IQEncoder


class MultiScaleClassifier(nn.Module):
    def __init__(self, model_config, classes=24):
        super().__init__()
        self.encoder = IQEncoder(**model_config)
        self.patch_size = model_config['patch_size']
        self.dims = list(model_config['dims'])
        # Per-stage LayerNorm balances the different scales/dims before concat.
        self.stage_norms = nn.ModuleList([nn.LayerNorm(d) for d in self.dims])
        self.head = nn.Linear(sum(self.dims), classes)
        # Hooks on each stage's last block capture that stage's token features.
        self._stage_outs = {}
        self._hooks = []
        for s in range(len(self.dims)):
            blk = self.encoder.stages[s][-1]
            self._hooks.append(blk.register_forward_hook(self._make_hook(s)))

    def _make_hook(self, s):
        def hook(module, inp, out):
            self._stage_outs[s] = out
        return hook

    def _patch_valids(self, valid_mask):
        # Mirror encoder patch-ization + merge valid logic (models.py).
        B, L = valid_mask.shape
        valid = valid_mask
        pad = (-L) % self.patch_size
        if pad:
            valid = F.pad(valid, (0, pad), value=False)
        v = valid.reshape(B, -1, self.patch_size).any(-1)
        valids = [v]
        for _ in range(len(self.dims) - 1):
            v = valids[-1]
            if v.shape[1] % 2:
                v = F.pad(v, (0, 1), value=False)
            valids.append(v[:, 0::2] | v[:, 1::2])
        return valids

    def forward(self, iq, valid_mask):
        self._stage_outs.clear()
        self.encoder(iq, valid_mask)  # fills self._stage_outs via hooks
        valids = self._patch_valids(valid_mask)
        pools = []
        for s in range(len(self.dims)):
            feat = self._stage_outs[s]          # [B, n_s, dim_s]; invalid tokens zeroed
            v = valids[s]                        # [B, n_s]
            num = (feat * v[..., None].to(feat.dtype)).sum(1)
            den = v.sum(1, keepdim=True).clamp_min(1).to(feat.dtype)
            pools.append(self.stage_norms[s](num / den))
        return self.head(torch.cat(pools, dim=-1))
