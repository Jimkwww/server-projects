"""Pre-LN global Transformer backbone.

``positions`` / ``Attention`` / ``Block`` are verbatim copies of the frozen
``SwinE/pretrain_v1/emfm/models.py`` (the Global baseline). The only addition is
``GlobalBackbone``, which applies sinusoidal absolute positions + ``depth``
global blocks + a final LayerNorm on top of the tokenizer's embeddings — i.e.
everything the baseline ``IQEncoder`` does AFTER its patch-embedding linear
layer. The patch embedding itself is replaced by the tokenizer's projection, so
the backbone (positions, attention, norm, head) is byte-for-byte the baseline
architecture.
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


class MetadataPosEmbed(nn.Module):
    """Learned band embedding + sinusoidal time embedding from ``time_center``.

    Phase-2 ablation: replaces the baseline flattened-sequence sinusoidal
    positions with the token's physical location. The band id is a learned
    embedding; the time part reuses the same sinusoidal formula as
    :func:`positions`, but over the continuous ``time_center`` (sample units)
    instead of the flat integer index. This decouples position from sequence
    order, so e.g. a band-major WPT/DWT flatten is no longer mis-positioned.
    """

    def __init__(self, dim, num_bands):
        super().__init__()
        self.dim = dim
        self.band = nn.Embedding(num_bands, dim)

    def forward(self, time_center, band_id):
        t = time_center.to(torch.float32)[:, None]  # [T, 1]
        d = self.dim
        freq = torch.exp(torch.arange(0, d, 2, device=t.device).float() * (-math.log(10000) / d))
        p = torch.zeros(t.shape[0], d, device=t.device)
        p[:, 0::2] = torch.sin(t * freq)
        p[:, 1::2] = torch.cos(t * freq[:d // 2])
        return p.to(self.band.weight.dtype) + self.band(band_id)


class GlobalBackbone(nn.Module):
    """Single-resolution global attention: positions + depth blocks + LayerNorm.

    Matches the Global baseline ``IQEncoder`` (kind='global', dims=[176],
    depths=[6], heads=[4]) minus the embed linear layer and minus the
    window/merge machinery (single stage, no merge).

    ``pos`` (optional [T, D]) overrides the default sinusoidal absolute
    positions, for the metadata position-encoding ablation.
    """

    def __init__(self, dim=176, depth=6, heads=4, ratio=4, dropout=0.):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must divide evenly into heads")
        self.dim, self.depth, self.heads = dim, depth, heads
        self.blocks = nn.ModuleList([
            Block(dim, heads, ratio, 0, 0, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.output_dim = dim

    def forward(self, embeddings, valid, pos=None):
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.dim or valid.shape != embeddings.shape[:2]:
            raise ValueError("Expected embeddings[B,T,D], valid[B,T]")
        if not valid.any(dim=1).all():
            raise ValueError("Every sample needs at least one valid token")
        if pos is None:
            pos = positions(embeddings.shape[1], embeddings.shape[2], embeddings.device, embeddings.dtype)
        elif pos.ndim != 2 or pos.shape != (embeddings.shape[1], embeddings.shape[2]):
            raise ValueError("Expected pos[T,D]")
        x = embeddings.masked_fill(~valid[..., None], 0)
        x = (x + pos[None]).masked_fill(~valid[..., None], 0)
        for block in self.blocks:
            x = block(x, valid)
        return self.norm(x).masked_fill(~valid[..., None], 0)
