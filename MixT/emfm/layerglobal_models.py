"""Layered-Global encoder: Swin's 3-stage hierarchy + patch merging, but GLOBAL
attention at every stage instead of windowed / shifted-window attention.

This is experiment A of the structure-isolation plan (local agent step 2):
  dims=[64,128,256], depths=[2,2,2], heads=[2,4,8], patch_size=8,
  token flow 128->64->32, single-scale head. The ONLY change vs the Swin config
  is attention scope: every Block is built with window=0 (full attention) and
  shift=0 (no shifting).

NEW file — the frozen emfm/models.py is not modified. Block/Merge/positions are
imported from it. Because window=0, Block.attend falls through to self.attn,
so parameter count is IDENTICAL to the Swin encoder (head dim 256), making the
comparison with sup_swin_50_s2024 (0.3407) clean.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .models import Block, Merge, positions


class LayerGlobalEncoder(nn.Module):
    def __init__(self, patch_size=8, dims=(64, 128, 256), depths=(2, 2, 2),
                 heads=(2, 4, 8), kind="layerglobal", window=0, shifted=False,
                 ratio=4, dropout=0.):
        super().__init__()
        # kind/window/shifted are accepted for config/metadata compatibility but
        # are intentionally ignored: attention is ALWAYS global in this model.
        if not (len(dims) == len(depths) == len(heads)):
            raise ValueError("Invalid architecture")
        if patch_size < 1 or any(d < 1 for d in depths):
            raise ValueError("Invalid patch/depth")
        self.patch_size = patch_size
        self.embed = nn.Linear(2 * patch_size, dims[0])
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        for s, (dim, depth, nh) in enumerate(zip(dims, depths, heads)):
            # window=0, shift=0 -> Block.attend returns self.attn(x, valid).
            self.stages.append(nn.ModuleList([
                Block(dim, nh, ratio, 0, 0, dropout) for _ in range(depth)]))
            if s + 1 < len(dims):
                self.merges.append(Merge(dim, dims[s + 1]))
        self.norm = nn.LayerNorm(dims[-1])
        self.output_dim = dims[-1]

    def forward(self, iq, valid):
        # Mirrors IQEncoder.forward (models.py) minus the mask-token path.
        if iq.ndim != 3 or iq.shape[-1] != 2 or valid.shape != iq.shape[:2]:
            raise ValueError("Expected IQ[B,L,2], valid[B,L]")
        if not valid.any(dim=1).all():
            raise ValueError("Every sample needs at least one valid point")
        iq = iq.masked_fill(~valid[..., None], 0)
        pad = (-iq.shape[1]) % self.patch_size
        iq = F.pad(iq, (0, 0, 0, pad))
        valid = F.pad(valid, (0, pad), value=False)
        b = iq.shape[0]
        x = self.embed(iq.reshape(b, -1, 2 * self.patch_size))
        valid = valid.reshape(b, -1, self.patch_size).any(-1)
        x = (x + positions(x.shape[1], x.shape[2], x.device, x.dtype)).masked_fill(~valid[..., None], 0)
        for s, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, valid)
            if s < len(self.merges):
                x, valid = self.merges[s](x, valid)
        return self.norm(x).masked_fill(~valid[..., None], 0), valid


class LayerGlobalClassifier(nn.Module):
    def __init__(self, model_config, classes=24):
        super().__init__()
        mc = dict(model_config)
        mc.pop("kind", None)
        self.encoder = LayerGlobalEncoder(**mc)
        self.head = nn.Linear(self.encoder.output_dim, classes)

    def forward(self, iq, valid_mask):
        x, mask = self.encoder(iq, valid_mask)
        pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)
