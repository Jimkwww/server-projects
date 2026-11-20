"""Pre-LN global Transformer and hierarchical shifted-window 1D encoder.

Both use the SAME sinusoidal absolute token positions for the first controlled
comparison. This is a Swin-inspired 1D model, not an exact port of 2D Swin.
Shifted windows use explicit left/right padding (no cyclic wraparound).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def positions(n, d, device, dtype):
    t = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000) / d))
    p = torch.zeros(n, d, device=device)
    p[:, 0::2] = torch.sin(t * freq)
    p[:, 1::2] = torch.cos(t * freq[:d // 2])
    return p.to(dtype)


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must divide evenly into heads")
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, valid):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        # Empty padding-only windows get one dummy key; outputs are zeroed below.
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(dim=1)
        scores = (q @ k.transpose(-2, -1)) * ((d // self.heads) ** -.5)
        scores = scores.masked_fill(~safe[:, None, None, :], float('-inf'))
        weights = F.dropout(scores.float().softmax(-1).to(x.dtype), self.dropout, self.training)
        out = (weights @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out).masked_fill(~valid[..., None], 0)


class Block(nn.Module):
    def __init__(self, dim, heads, ratio=4, window=0, shift=0, dropout=0.):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ratio), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * ratio, dim), nn.Dropout(dropout))

    def attend(self, x, valid):
        if not self.window:
            return self.attn(x, valid)
        b, n, d = x.shape
        # Keep the window grid independent of other samples' padded lengths.
        left = self.shift
        right = (-(n + left)) % self.window
        z = F.pad(x, (0, 0, left, right))
        mask = F.pad(valid, (left, right), value=False)
        z = self.attn(z.reshape(-1, self.window, d), mask.reshape(-1, self.window))
        return z.reshape(b, -1, d)[:, left:left + n]

    def forward(self, x, valid):
        x = x + self.attend(self.norm1(x), valid)
        x = x + self.ffn(self.norm2(x))
        return x.masked_fill(~valid[..., None], 0)


class Merge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.proj = nn.Linear(2 * dim, out_dim, bias=False)

    def forward(self, x, valid):
        x = x.masked_fill(~valid[..., None], 0)
        if x.shape[1] % 2:
            x = F.pad(x, (0, 0, 0, 1))
            valid = F.pad(valid, (0, 1), value=False)
        x = torch.cat((x[:, 0::2], x[:, 1::2]), dim=-1)
        valid = valid[:, 0::2] | valid[:, 1::2]
        return self.proj(self.norm(x)).masked_fill(~valid[..., None], 0), valid


class IQEncoder(nn.Module):
    def __init__(self, kind, patch_size=8, dims=(64, 128, 256), depths=(2, 2, 2),
                 heads=(2, 4, 8), window=8, shifted=True, ratio=4, dropout=0.):
        super().__init__()
        if kind not in ('global', 'swin') or not (len(dims) == len(depths) == len(heads)):
            raise ValueError("Invalid architecture")
        if kind == 'global' and len(dims) != 1:
            raise ValueError("Global baseline uses one resolution")
        if patch_size < 1 or window < 1 or any(d < 1 for d in depths):
            raise ValueError("Invalid patch/window/depth")
        self.patch_size = patch_size
        self.embed = nn.Linear(2 * patch_size, dims[0])
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        for s, (dim, depth, nh) in enumerate(zip(dims, depths, heads)):
            self.stages.append(nn.ModuleList([
                Block(dim, nh, ratio, 0 if kind == 'global' else window,
                      window // 2 if shifted and i % 2 else 0, dropout)
                for i in range(depth)]))
            if s + 1 < len(dims):
                self.merges.append(Merge(dim, dims[s + 1]))
        self.norm = nn.LayerNorm(dims[-1])
        self.output_dim = dims[-1]

    def forward(self, iq, valid, patch_mask=None, mask_token=None):
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
        if patch_mask is not None:
            if patch_mask.shape != valid.shape or mask_token is None:
                raise ValueError("Mask must match input patch grid and supply mask_token")
            if (patch_mask & ~valid).any():
                raise ValueError("Cannot mask padding-only patches")
            # Replacement before any attention/merging; no raw-signal skip path.
            x = torch.where(patch_mask[..., None], mask_token.to(x.dtype), x)
        x = (x + positions(x.shape[1], x.shape[2], x.device, x.dtype)).masked_fill(~valid[..., None], 0)
        for s, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, valid)
            if s < len(self.merges):
                x, valid = self.merges[s](x, valid)
        return self.norm(x).masked_fill(~valid[..., None], 0), valid


class Classifier(nn.Module):
    def __init__(self, model_config, classes=24):
        super().__init__()
        self.encoder = IQEncoder(**model_config)
        self.head = nn.Linear(self.encoder.output_dim, classes)

    def forward(self, iq, valid_mask):
        x, mask = self.encoder(iq, valid_mask)
        pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)
