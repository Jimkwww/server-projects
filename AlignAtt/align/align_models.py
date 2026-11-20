"""Relation-augmented global encoder (experiment matrix A0-A6).

Mirrors ``emfm/models.py`` IQEncoder with ``kind='global'`` (single resolution,
global attention) but adds an optional relation bias to attention logits. The
relation descriptor is computed once per forward from the raw patch IQ and is
injected into a configurable set of layers (default: first layer only).

The ONLY structural change vs the frozen baseline is the additive bias term; with
``relation.mode='none'`` (or gate == 0) the computation path reduces to the
baseline (verified by ``test_correlation.py``).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .correlation import iq_to_complex
from .relation_bias import RelationBias


def positions(n, d, device, dtype):
    t = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000) / d))
    p = torch.zeros(n, d, device=device)
    p[:, 0::2] = torch.sin(t * freq)
    p[:, 1::2] = torch.cos(t * freq[:d // 2])
    return p.to(dtype)


class BiasAttention(nn.Module):
    """Global attention with an optional additive per-head bias on the logits."""

    def __init__(self, dim, heads, dropout=0.):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must divide evenly into heads")
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, valid, bias=None):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(dim=1)
        scores = (q @ k.transpose(-2, -1)) * ((d // self.heads) ** -.5)
        scores = scores.float()
        if bias is not None:
            scores = scores + bias
        scores = scores.masked_fill(~safe[:, None, None, :], float('-inf'))
        weights = F.dropout(scores.softmax(-1).to(x.dtype), self.dropout, self.training)
        out = (weights @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out).masked_fill(~valid[..., None], 0)


class RelationOnlyAttention(BiasAttention):
    """Diagnostic A6: drop the QK content term, keep only the relation bias."""

    def forward(self, x, valid, bias=None):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(dim=1)
        scores = bias if bias is not None else torch.zeros_like(q @ k.transpose(-2, -1))
        scores = scores.masked_fill(~safe[:, None, None, :], float('-inf'))
        weights = F.dropout(scores.softmax(-1).to(x.dtype), self.dropout, self.training)
        out = (weights @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out).masked_fill(~valid[..., None], 0)


class BiasBlock(nn.Module):
    def __init__(self, dim, heads, ratio=4, dropout=0., relation_only=False):
        super().__init__()
        attn = RelationOnlyAttention if relation_only else BiasAttention
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = attn(dim, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ratio), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * ratio, dim), nn.Dropout(dropout))

    def forward(self, x, valid, bias=None):
        x = x + self.attn(self.norm1(x), valid, bias)
        x = x + self.ffn(self.norm2(x))
        return x.masked_fill(~valid[..., None], 0)


class RelationGlobalEncoder(nn.Module):
    """Single-resolution global encoder (the fixed Global baseline) + relation bias.

    ``model_config`` is the baseline ``global`` config (patch_size/dims/depths/
    heads/window/shifted/ratio/dropout) extended with an optional ``relation``
    dict. ``kind`` is accepted for config compatibility and ignored.
    """

    def __init__(self, model_config):
        super().__init__()
        mc = dict(model_config)
        mc.pop("kind", None)
        mc.pop("window", None)
        mc.pop("shifted", None)
        relation = mc.pop("relation", None) or {}
        if not (len(mc["dims"]) == len(mc["depths"]) == len(mc["heads"]) == 1):
            raise ValueError("RelationGlobalEncoder targets the single-resolution Global baseline")
        self.patch_size = mc["patch_size"]
        dim = mc["dims"][0]
        depth = mc["depths"][0]
        heads = mc["heads"][0]
        ratio = mc.get("ratio", 4)
        dropout = mc.get("dropout", 0.)
        self.embed = nn.Linear(2 * self.patch_size, dim)
        mode = relation.get("mode", "none")
        self.relation_R = int(relation.get("R", self.patch_size))
        if self.relation_R < 1:
            raise ValueError("relation.R must be >= 1")
        self.inject = relation.get("inject", "first")
        if self.inject not in ("first", "first_half", "all"):
            raise ValueError(f"Unknown relation inject: {self.inject}")
        self.relation = (RelationBias(
            heads, mode, R=self.relation_R,
            tau_radius=relation.get("tau_radius", 2),
            mlp_hidden=relation.get("mlp_hidden", 16),
            eps=relation.get("eps", 1e-6),
            gate_init=relation.get("gate_init", 0.0),
            shuffle=relation.get("shuffle", False),
            shuffle_seed=relation.get("shuffle_seed", 0))
            if mode != "none" else None)
        use_qk = relation.get("use_qk", True)
        self.stages = nn.ModuleList([nn.ModuleList([
            BiasBlock(dim, heads, ratio, dropout, relation_only=not use_qk)
            for _ in range(depth)])])
        self.norm = nn.LayerNorm(dim)
        self.output_dim = dim
        self._inject_idx = self._resolve_inject(depth)

    def _resolve_inject(self, depth):
        if self.inject == "first":
            return {0}
        if self.inject == "first_half":
            return set(range((depth + 1) // 2))
        return set(range(depth))

    def forward(self, iq, valid):
        if iq.ndim != 3 or iq.shape[-1] != 2 or valid.shape != iq.shape[:2]:
            raise ValueError("Expected IQ[B,L,2], valid[B,L]")
        if not valid.any(dim=1).all():
            raise ValueError("Every sample needs at least one valid point")
        iq = iq.masked_fill(~valid[..., None], 0)
        pad = (-iq.shape[1]) % self.patch_size
        iq = F.pad(iq, (0, 0, 0, pad))
        valid = F.pad(valid, (0, pad), value=False)     # sample-level, padded length
        b = iq.shape[0]

        # Relation path: raw-IQ context segments (computed once per forward).
        bias = None
        if self.relation is not None:
            z, sample_valid = self._context(iq, valid, self.relation_R)
            token_valid = valid.reshape(b, -1, self.patch_size).any(-1)
            bias = self.relation(z, token_valid, sample_valid)   # [B, H, T, T] or None

        x = self.embed(iq.reshape(b, -1, 2 * self.patch_size))
        valid = valid.reshape(b, -1, self.patch_size).any(-1)
        x = (x + positions(x.shape[1], x.shape[2], x.device, x.dtype)).masked_fill(~valid[..., None], 0)

        for blocks in self.stages:
            for i, block in enumerate(blocks):
                x = block(x, valid, bias if i in self._inject_idx else None)
        return self.norm(x).masked_fill(~valid[..., None], 0), valid

    def _context(self, iq, valid, R):
        """Extract R-point context segments centred on each patch, plus validity.

        ``iq`` is the padded ``[B, L, 2]`` input, ``valid`` the sample-level
        ``[B, L]`` mask (False = boundary/padding). Returns complex ``[B, T, R]``
        segments with invalid samples zeroed, and the ``[B, T, R]`` sample mask.
        With ``R == patch_size`` this is exactly the patch grid.
        """
        B, L, _ = iq.shape
        P = self.patch_size
        T = L // P
        half = R // 2
        starts = torch.arange(T, device=iq.device) * P + P // 2 - half
        idx = starts[:, None] + torch.arange(R, device=iq.device)[None, :]
        in_range = (idx >= 0) & (idx < L)
        idx_c = idx.clamp(0, L - 1)
        z = iq_to_complex(iq[:, idx_c])                 # [B, T, R] complex
        sv = valid[:, idx_c] & in_range[None]           # [B, T, R] bool
        return z.masked_fill(~sv, 0), sv


class AlignClassifier(nn.Module):
    def __init__(self, model_config, classes=24):
        super().__init__()
        self.encoder = RelationGlobalEncoder(model_config)
        self.head = nn.Linear(self.encoder.output_dim, classes)

    def forward(self, iq, valid_mask):
        x, mask = self.encoder(iq, valid_mask)
        pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)

    def relation_param_count(self):
        return self.encoder.relation.param_count() if self.encoder.relation else 0
