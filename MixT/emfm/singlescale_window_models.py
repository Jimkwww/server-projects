"""Single-scale Window encoder: Global's single stage (dim=176, depth=6, heads=4,
128 tokens, no merge), but with window=8 shifted-window attention at every block.

This is experiment B of the structure-isolation plan (local agent step 3): the
mirror of layerglobal. The ONLY change vs the Global config is attention scope:
window=8 + shift instead of full attention. Window adds no parameters, so the
count is IDENTICAL to Global (2,251,592), making the comparison with
sup_global_50_s2024 (0.3869) clean.

NEW file — the frozen emfm/models.py is not modified. Block/positions are
imported from it.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .models import Block, positions


class SingleScaleWindowEncoder(nn.Module):
    def __init__(self, patch_size=8, dims=(176,), depths=(6,), heads=(4,),
                 kind="singlescale_window", window=8, shifted=True, ratio=4, dropout=0.):
        super().__init__()
        # kind is accepted for config/metadata compatibility.
        if not (len(dims) == len(depths) == len(heads) == 1):
            raise ValueError("Single-scale window encoder uses exactly one stage")
        if patch_size < 1 or window < 1 or depths[0] < 1:
            raise ValueError("Invalid patch/window/depth")
        self.patch_size = patch_size
        self.embed = nn.Linear(2 * patch_size, dims[0])
        dim, depth, nh = dims[0], depths[0], heads[0]
        # window=window, shift=window//2 on odd blocks — same rule as Swin's
        # stages in models.py, but applied to the single global-resolution stage.
        self.stages = nn.ModuleList([nn.ModuleList([
            Block(dim, nh, ratio, window, window // 2 if shifted and i % 2 else 0, dropout)
            for i in range(depth)])])
        self.merges = nn.ModuleList()  # empty: no merge, 128 tokens throughout
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


class SingleScaleWindowClassifier(nn.Module):
    def __init__(self, model_config, classes=24):
        super().__init__()
        mc = dict(model_config)
        mc.pop("kind", None)
        self.encoder = SingleScaleWindowEncoder(**mc)
        self.head = nn.Linear(self.encoder.output_dim, classes)

    def forward(self, iq, valid_mask):
        x, mask = self.encoder(iq, valid_mask)
        pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)
