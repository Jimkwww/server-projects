"""Glue: tokenizer -> Global backbone -> mean-pool classification head.

Keeps the backbone and head identical to the Global baseline; only the
tokenizer (input representation) is swapped. The config drives tokenizer
selection via ``cfg["tokenizer"]``.
"""
import torch
from torch import nn

from .backbone import GlobalBackbone, MetadataPosEmbed
from .tokenizers import build_tokenizer


class TokenClassifier(nn.Module):
    def __init__(self, config, classes=24):
        super().__init__()
        self.tokenizer = build_tokenizer(config["tokenizer"])
        bb = config.get("backbone", {})
        self.backbone = GlobalBackbone(
            dim=bb.get("dim", 176), depth=bb.get("depth", 6),
            heads=bb.get("heads", 4), ratio=bb.get("ratio", 4),
            dropout=bb.get("dropout", 0.0))
        self.pos_encoding = bb.get("pos_encoding", "sinusoidal")
        if self.pos_encoding == "metadata":
            self.pos_embed = MetadataPosEmbed(
                dim=self.backbone.dim, num_bands=bb.get("num_bands", 1))
        elif self.pos_encoding != "sinusoidal":
            raise ValueError(f"Unknown pos_encoding {self.pos_encoding!r}")
        self.head = nn.Linear(self.backbone.output_dim, classes)
        self.classes = classes

    def forward(self, iq, valid_mask):
        tb = self.tokenizer(iq, valid_mask)
        pos = None
        if self.pos_encoding == "metadata":
            pos = self.pos_embed(tb.time_center, tb.band_id)
        x = self.backbone(tb.embeddings, tb.valid_mask, pos=pos)
        pooled = x.sum(1) / tb.valid_mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)
