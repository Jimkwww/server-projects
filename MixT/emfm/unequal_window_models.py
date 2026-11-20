"""Unequal-length contiguous windows with cross-layer cyclic rotation.

The Global backbone (single stage, dim=176, depth=6, heads=4, 128 tokens, no
merge) with the ONLY change being attention scope: tokens are partitioned into
contiguous non-overlapping windows of UNEQUAL length; attention is computed
within each window only, all heads sharing the same partition. Across layers the
window-length list is either fixed (rotation="fixed", experiment U-F) or rotated
right by one position per layer (rotation="cyclic", experiment U-C).

Token count, resolution, patch embedding, sinusoidal absolute positions, QKV /
output projections, LayerNorm placement, MLP, residuals, mean pooling and the
classification head are identical to Global. No token merging: every layer's
input/output is [B, 128, dim].

Window lists sum to N=128 (the token count for L=1024, patch=8). The encoder
asserts the runtime token count equals sum(window_list); unequal windows are not
rescaled to arbitrary lengths.

Two attention impls are provided and must agree exactly on identical weights:
  - "dense_mask": full N x N scores masked by the layer's allowed matrix
    (reference; used for correctness and phase-1 accuracy).
  - "per_window": Q/K/V sliced per contiguous window, independent attention,
    concatenated back in original order.

NEW file — the frozen emfm/models.py is not modified. Only `positions` is
imported from it.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .models import positions


def rotate_right(w, r):
    """Right-rotate a list by r positions: W[-r:] + W[:-r] (r=0 -> unchanged)."""
    r = r % len(w)
    return list(w[-r:] + w[:-r]) if r else list(w)


def window_intervals(w, n):
    """Half-open [start, end) intervals partitioning [0, n) for window lengths w."""
    starts = []
    s = 0
    for wi in w:
        starts.append(s)
        s += wi
    if s != n:
        raise ValueError(f"window lengths sum to {s}, expected {n}")
    return [(starts[i], starts[i] + w[i]) for i in range(len(w))]


def allowed_matrix(w, n):
    """A[i, j] = 1 iff token i and j fall in the same window (diagonal always 1)."""
    a = torch.zeros(n, n, dtype=torch.bool)
    for s, e in window_intervals(w, n):
        a[s:e, s:e] = True
    return a


class UnequalWindowAttention(nn.Module):
    """Attention restricted to per-layer unequal windows.

    Same parameters as the Global `Attention` (qkv, proj); the only difference is
    the attention scope, passed per call as `allowed` ([n, n] bool) and
    `intervals` (list of (start, end)).
    """

    def __init__(self, dim, heads, dropout=0., impl="dense_mask"):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must divide evenly into heads")
        if impl not in ("dense_mask", "per_window"):
            raise ValueError("impl must be dense_mask or per_window")
        self.heads, self.dropout, self.impl = heads, dropout, impl
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = (dim // heads) ** -0.5

    @staticmethod
    def _safe(valid):
        # Per-sample guarantee of >=1 attended key (mirrors models.Attention).
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(dim=1)
        return safe

    def forward(self, x, valid, allowed, intervals):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        safe = self._safe(valid)
        if self.impl == "dense_mask":
            scores = (q @ k.transpose(-2, -1)) * self.scale
            mask = safe[:, None, None, :] & allowed[None, None, :, :]
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.dropout(scores.float().softmax(-1).to(x.dtype),
                                self.dropout, self.training)
            out = (weights @ v).transpose(1, 2).reshape(b, n, d)
        else:  # per_window
            outs = []
            for s, e in intervals:
                qw, kw, vw = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e]
                sw = safe[:, s:e]
                scores = (qw @ kw.transpose(-2, -1)) * self.scale
                scores = scores.masked_fill(~sw[:, None, None, :], float("-inf"))
                w = F.dropout(scores.float().softmax(-1).to(x.dtype),
                              self.dropout, self.training)
                outs.append(w @ vw)  # [b, heads, w, dk]
            out = torch.cat(outs, dim=2).transpose(1, 2).reshape(b, n, d)
        return self.proj(out).masked_fill(~valid[..., None], 0)


class UnequalWindowBlock(nn.Module):
    def __init__(self, dim, heads, ratio=4, dropout=0., impl="dense_mask",
                 allowed=None, intervals=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = UnequalWindowAttention(dim, heads, dropout, impl)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ratio), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * ratio, dim),
                                 nn.Dropout(dropout))
        # Non-persistent: derivable from config, but registered so it follows
        # device and mixed-precision casts like any other buffer.
        self.register_buffer("allowed", allowed, persistent=False)
        self.intervals = intervals  # list[(int, int)]; plain attr, not a tensor

    def forward(self, x, valid):
        x = x + self.attn(self.norm1(x), valid, self.allowed, self.intervals)
        x = x + self.ffn(self.norm2(x))
        return x.masked_fill(~valid[..., None], 0)


class UnequalWindowEncoder(nn.Module):
    def __init__(self, patch_size=8, dims=(176,), depths=(6,), heads=(4,),
                 kind="unequal_window", window_list=(4, 8, 16, 24, 32, 44),
                 rotation="cyclic", window_impl="dense_mask", ratio=4, dropout=0.):
        super().__init__()
        # kind/window/shifted are accepted for config/metadata compatibility.
        if not (len(dims) == len(depths) == len(heads) == 1):
            raise ValueError("UnequalWindowEncoder uses exactly one stage")
        if rotation not in ("cyclic", "fixed"):
            raise ValueError("rotation must be cyclic or fixed")
        if any(w < 1 for w in window_list):
            raise ValueError("window lengths must be positive")
        self.patch_size = patch_size
        self.rotation = rotation
        self.window_list = tuple(window_list)
        self.n_tokens = sum(window_list)
        self.embed = nn.Linear(2 * patch_size, dims[0])
        dim, depth, nh = dims[0], depths[0], heads[0]
        k = len(window_list)
        # Mirror Global's structure: a single stage (ModuleList) of `depth`
        # blocks, so parameter names match IQEncoder(kind='global') exactly and
        # a Global checkpoint loads with strict=True when window_list=[N].
        blocks = []
        for l in range(depth):
            r = l % k if rotation == "cyclic" else 0
            wl = rotate_right(window_list, r)
            intervals = window_intervals(wl, self.n_tokens)
            allowed = allowed_matrix(wl, self.n_tokens)
            blocks.append(UnequalWindowBlock(dim, nh, ratio, dropout, window_impl,
                                             allowed, intervals))
        self.stages = nn.ModuleList([nn.ModuleList(blocks)])
        self.merges = nn.ModuleList()  # empty: no merge, N tokens throughout
        self.norm = nn.LayerNorm(dims[-1])
        self.output_dim = dims[-1]

    def forward(self, iq, valid):
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
        if x.shape[1] != self.n_tokens:
            raise ValueError(
                f"token count {x.shape[1]} != window partition total {self.n_tokens}")
        x = (x + positions(x.shape[1], x.shape[2], x.device, x.dtype)) \
            .masked_fill(~valid[..., None], 0)
        for s, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, valid)
            if s < len(self.merges):
                x, valid = self.merges[s](x, valid)
        return self.norm(x).masked_fill(~valid[..., None], 0), valid


class UnequalWindowClassifier(nn.Module):
    def __init__(self, model_config, classes=24):
        super().__init__()
        mc = dict(model_config)
        for key in ("kind", "window", "shifted"):
            mc.pop(key, None)
        self.encoder = UnequalWindowEncoder(**mc)
        self.head = nn.Linear(self.encoder.output_dim, classes)

    def forward(self, iq, valid_mask):
        x, mask = self.encoder(iq, valid_mask)
        pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)
