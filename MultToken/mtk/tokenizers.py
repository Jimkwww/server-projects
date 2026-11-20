"""Multiscale / multi-representation tokenizers for raw IQ signals.

Every tokenizer maps raw IQ ``[B, L, 2]`` + valid mask ``[B, L]`` to a
:class:`TokenBatch`. Design rules (see ``01_multiscale_tokenization_experiment.md``):

* The backbone, classification head and training protocol are shared and
  unchanged; only the front-end (input representation) differs.
* Embedding dim ``D`` and token count ``T`` are kept equal across methods
  (D=176, T=128 for N=1024) so the comparison is single-factor.
* Fixed front-ends (T1 DCT, T3 WPT, T4 DWT, T5 STFT) are non-learnable and run
  in numpy on the CPU; the projection to ``D`` is their only learnable part.
* The learnable front-end (T2 conv) runs in torch so it is differentiable.
* Complex coefficients are always kept as (re, im) pairs; nothing is reduced to
  magnitude/power, and nothing keeps only positive frequencies.
* ``time_center`` / ``time_support`` / ``scale_id`` / ``band_id`` are recorded
  metadata; the Phase-1 backbone still uses the baseline sinusoidal absolute
  positions over the flattened token sequence. The "time-center + scale/band
  metadata" position encoding is a Phase-2 ablation, not applied here.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    import pywt
except ImportError:  # pragma: no cover - import guard for environments w/o pywt
    pywt = None

try:
    from scipy.fft import dct
    from scipy.signal import ShortTimeFFT, windows
except ImportError:  # pragma: no cover
    dct = ShortTimeFFT = windows = None


@dataclass
class TokenBatch:
    embeddings: torch.Tensor            # [B, T, D]
    valid_mask: torch.Tensor            # [B, T] bool
    time_center: torch.Tensor           # [T] float, sample-space center (metadata)
    time_support: torch.Tensor          # [T] float, effective support in samples (metadata)
    scale_id: torch.Tensor              # [T] long, scale / branch id (metadata)
    band_id: torch.Tensor               # [T] long, band id (metadata)
    metadata: dict = field(default_factory=dict)


def _device_like(t: torch.Tensor):
    return t.device


def _grid_centers(T, stride, support, device, offset=0.0):
    return (torch.arange(T, device=device, dtype=torch.float32) * stride + offset + support / 2)


def _ones_long(T, device, value=0):
    return torch.full((T,), value, dtype=torch.long, device=device)


def _full_like(t: torch.Tensor, value):
    return torch.full((t.shape[1],), value, dtype=torch.float32, device=t.device)


class _Proj(nn.Module):
    """Shared learnable projection: token coefficients (in_dim) -> D."""

    def __init__(self, in_dim, dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)


# ---------------------------------------------------------------------------
# T0 — raw I/Q fixed patch, the primary baseline
# ---------------------------------------------------------------------------
class T0PatchTokenizer(nn.Module):
    """Flatten P consecutive I/Q samples (2P values) and project to D.

    Identical to the baseline patch embedding: ``nn.Linear(2P, D)`` over
    non-overlapping patches. Should reproduce the Global baseline exactly.
    """

    def __init__(self, patch_size=8, dim=176):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.proj = nn.Linear(2 * patch_size, dim)

    def forward(self, iq, valid):
        B, L, _ = iq.shape
        P = self.patch_size
        iq = iq.masked_fill(~valid[..., None], 0)
        pad = (-L) % P
        iq = F.pad(iq, (0, 0, 0, pad))
        valid = F.pad(valid, (0, pad), value=False)
        T = iq.shape[1] // P
        x = self.proj(iq.reshape(B, T, 2 * P))
        v = valid.reshape(B, T, P).any(-1)
        dev = x.device
        return TokenBatch(
            embeddings=x, valid_mask=v,
            time_center=_grid_centers(T, P, P, dev),
            time_support=_full_like(x, P),
            scale_id=_ones_long(T, dev, 0), band_id=_ones_long(T, dev, 0),
            metadata={"name": "t0", "patch_size": P, "token_count": T,
                      "projection_in_dim": 2 * P, "dim": self.dim})


# ---------------------------------------------------------------------------
# T1 — fixed in-block orthogonal transform (control group)
# ---------------------------------------------------------------------------
class T1OrthoTokenizer(nn.Module):
    """Apply a fixed orthogonal transform (DCT-II) inside each block, then the
    same Linear(2P -> D) as T0. Token count and projection are unchanged.

    Because A is orthogonal/invertible, ``W @ A`` spans the same function family
    as the raw linear projection ``W``; this group exists to check that any
    apparent gain is NOT simply a change of basis (see doc section 5 T1).
    """

    def __init__(self, patch_size=8, dim=176, transform="dct"):
        super().__init__()
        if transform != "dct":
            raise ValueError("T1 currently supports transform='dct' only")
        self.patch_size = patch_size
        self.dim = dim
        self.proj = nn.Linear(2 * patch_size, dim)

    @staticmethod
    def analyze(iq_np, patch_size):
        # iq_np: [B, L, 2] float32; returns [B, T, 2P] coefficients (same basis count)
        B, L, _ = iq_np.shape
        P = patch_size
        x = iq_np[:, :(L // P) * P].reshape(B, -1, P, 2)
        y = dct(x, type=2, axis=2, norm="ortho")  # orthonormal DCT-II along time
        return y.reshape(B, -1, 2 * P).astype(np.float32)

    def forward(self, iq, valid):
        B, L, _ = iq.shape
        P = self.patch_size
        dev = iq.device
        pad = (-L) % P
        iq = iq.masked_fill(~valid[..., None], 0)
        valid = F.pad(valid, (0, pad), value=False)
        coef = torch.from_numpy(self.analyze(iq.detach().cpu().numpy(), P)).to(dev)
        x = self.proj(coef)
        v = valid.reshape(B, -1, P).any(-1)
        return TokenBatch(
            embeddings=x, valid_mask=v,
            time_center=_grid_centers(x.shape[1], P, P, dev),
            time_support=_full_like(x, P),
            scale_id=_ones_long(x.shape[1], dev, 0), band_id=_ones_long(x.shape[1], dev, 0),
            metadata={"name": "t1", "transform": "dct2_orthonormal", "patch_size": P,
                      "token_count": x.shape[1], "projection_in_dim": 2 * P, "dim": self.dim})


# ---------------------------------------------------------------------------
# T2 — multiscale time filtering (simple strong control)
# ---------------------------------------------------------------------------
class T2MultiscaleConv(nn.Module):
    """Three learnable 1-D conv branches with kernel lengths ~P, 2P, 4P, all at
    stride P on the same time grid. Branch outputs are concatenated (total
    channels == 2P) then mapped to D, keeping token count and projection width
    equal to T0. Conv weights are the only learnable front-end here; the
    receptive-field lengths are reported in the metadata.
    """

    def __init__(self, patch_size=8, dim=176, kernels=(8, 16, 32), channels=(6, 5, 5),
                 freeze_conv=False):
        super().__init__()
        if sum(channels) != 2 * patch_size:
            raise ValueError("T2 total branch channels must equal 2P to match T0 projection width")
        if len(kernels) != len(channels):
            raise ValueError("kernels and channels must align")
        self.patch_size = patch_size
        self.dim = dim
        self.kernels = tuple(kernels)
        self.channels = tuple(channels)
        self.freeze_conv = freeze_conv
        self.convs = nn.ModuleList()
        for k, c in zip(kernels, channels):
            pad = (k - patch_size) // 2  # keeps output length == L / stride
            self.convs.append(nn.Conv1d(2, c, k, stride=patch_size, padding=pad, bias=False))
        if freeze_conv:
            # Control: fixed random multiscale filters, only the projection learns.
            for conv in self.convs:
                conv.requires_grad_(False)
        self.proj = nn.Linear(sum(channels), dim)

    def forward(self, iq, valid):
        B, L, _ = iq.shape
        P = self.patch_size
        dev = iq.device
        iq = iq.masked_fill(~valid[..., None], 0)
        pad = (-L) % P
        iq = F.pad(iq, (0, 0, 0, pad))
        valid = F.pad(valid, (0, pad), value=False)
        z = iq.transpose(1, 2)  # [B, 2, L]
        outs = [conv(z) for conv in self.convs]  # each [B, c, T]
        T = outs[0].shape[-1]
        cat = torch.cat(outs, dim=1).transpose(1, 2)  # [B, T, 2P]
        x = self.proj(cat)
        v = valid.reshape(B, -1, P).any(-1)
        # Effective support of a token = max receptive field across branches.
        support = float(max(self.kernels))
        return TokenBatch(
            embeddings=x, valid_mask=v,
            time_center=_grid_centers(T, P, P, dev),
            time_support=torch.full((T,), support, device=dev),
            scale_id=_ones_long(T, dev, 0), band_id=_ones_long(T, dev, 0),
            metadata={"name": "t2", "kernels": list(self.kernels),
                      "channels": list(self.channels), "stride": P,
                      "per_branch_support": list(self.kernels),
                      "token_count": T, "projection_in_dim": sum(self.channels),
                      "dim": self.dim, "learnable_frontend": True,
                      "freeze_conv": self.freeze_conv})


# ---------------------------------------------------------------------------
# B — control: single large-kernel conv + 16-dim bottleneck (vs T2's 3 branches)
# ---------------------------------------------------------------------------
class BSingleConv(nn.Module):
    """Control B: one learnable 1-D conv (2 -> 16, kernel=32) at stride P on the
    same time grid as T2, then the same ``Linear(16 -> D)`` projection. Keeps
    T2's rank<=16 bottleneck but drops the short/medium kernel time-support
    restriction (all 16 channel kernels now span the full 32 taps). No
    activation / norm / pooling / residual anywhere in the front-end.

    For L=1024, P=8: conv padding=12 -> output length 128, so embeddings are
    [B, 128, D] exactly like T0/T2.
    """

    def __init__(self, patch_size=8, dim=176, kernel_size=32, bottleneck=16):
        super().__init__()
        if (kernel_size - patch_size) % 2:
            raise ValueError("kernel_size - patch_size must be even for center alignment")
        self.patch_size = patch_size
        self.dim = dim
        self.kernel_size = kernel_size
        self.bottleneck = bottleneck
        self.conv = nn.Conv1d(2, bottleneck, kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size) // 2, bias=False)
        self.proj = nn.Linear(bottleneck, dim)

    def forward(self, iq, valid):
        B, L, _ = iq.shape
        P = self.patch_size
        dev = iq.device
        iq = iq.masked_fill(~valid[..., None], 0)
        pad = (-L) % P
        iq = F.pad(iq, (0, 0, 0, pad))
        valid = F.pad(valid, (0, pad), value=False)
        z = iq.transpose(1, 2)            # [B, 2, L]
        y = self.conv(z)                  # [B, 16, T]
        y = y.transpose(1, 2)             # [B, T, 16]
        x = self.proj(y)                  # [B, T, D]
        T = x.shape[1]
        v = valid.reshape(B, -1, P).any(-1)
        support = float(self.kernel_size)
        return TokenBatch(
            embeddings=x, valid_mask=v,
            time_center=_grid_centers(T, P, P, dev),
            time_support=torch.full((T,), support, device=dev),
            scale_id=_ones_long(T, dev, 0), band_id=_ones_long(T, dev, 0),
            metadata={"name": "b", "kernel_size": self.kernel_size,
                      "bottleneck": self.bottleneck, "stride": P,
                      "token_count": T, "projection_in_dim": self.bottleneck,
                      "dim": self.dim, "learnable_frontend": True})


# ---------------------------------------------------------------------------
# Shared fixed-analysis helpers (numpy)
# ---------------------------------------------------------------------------
def _wpt_leaves(x, wavelet, level, mode="periodization"):
    """Full binary wavelet-packet tree to ``level``; returns 2^level leaves in
    natural (dyadic / low-then-high) order."""
    leaves = [x]
    for _ in range(level):
        nxt = []
        for node in leaves:
            cA, cD = pywt.dwt(node, wavelet, mode=mode)
            nxt += [cA, cD]
        leaves = nxt
    return leaves


def _freq_order(level):
    """Gray-code permutation mapping natural (dyadic) leaf index -> frequency
    order index. ``leaf_freq[j]`` is the natural index of the j-th frequency
    bin. Recorded for metadata only (token sequence uses natural order)."""
    n = 1 << level
    gray = [i ^ (i >> 1) for i in range(n)]
    return gray


class T3WPTTokenizer(nn.Module):
    """Fixed wavelet-packet transform to depth J (2^J leaves), then per-leaf
    coefficient patches of P grouped into tokens. I and Q share the same tree,
    boundary mode and coefficient order; each token is a (re, im) pair of leaf
    coefficients and is projected to D with a single shared Linear(2P, D).
    """

    def __init__(self, wavelet="haar", level=2, patch_size=8, dim=176, mode="periodization"):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.patch_size = patch_size
        self.dim = dim
        self.mode = mode
        self.proj = nn.Linear(2 * patch_size, dim)

    def analyze(self, iq_np):
        # returns (coef [B, T, 2P], meta dict)
        B, L, _ = iq_np.shape
        P = self.patch_size
        leaves_per = 1 << self.level
        tokens_i = []
        tokens_q = []
        leaf_len = L // leaves_per
        if L % leaves_per or leaf_len % P:
            raise ValueError(f"L={L} not divisible by leaves*P for WPT depth {self.level}")
        for b in range(B):
            li = _wpt_leaves(iq_np[b, :, 0], self.wavelet, self.level, self.mode)
            lq = _wpt_leaves(iq_np[b, :, 1], self.wavelet, self.level, self.mode)
            for leaf in li:
                tokens_i.append(leaf.reshape(-1, P))
            for leaf in lq:
                tokens_q.append(leaf.reshape(-1, P))
        # tokens_i: [B * leaves_per, leaf_len/P, P]
        ti = np.stack(tokens_i).reshape(B, leaves_per, -1, P)
        tq = np.stack(tokens_q).reshape(B, leaves_per, -1, P)
        # band-major, then time: [B, leaves_per, T_leaf, 2P]
        coef = np.concatenate([ti, tq], axis=-1).reshape(B, leaves_per * (leaf_len // P), 2 * P)
        meta = {"name": "t3", "wavelet": self.wavelet, "level": self.level,
                "mode": self.mode, "leaves": leaves_per,
                "tokens_per_leaf": leaf_len // P, "natural_order": list(range(leaves_per)),
                "freq_order_natural_indices": _freq_order(self.level),
                "coeff_spacing_samples": 1 << self.level,
                "token_count": leaves_per * (leaf_len // P)}
        return coef.astype(np.float32), meta

    def forward(self, iq, valid):
        dev = iq.device
        P = self.patch_size
        iq = iq.masked_fill(~valid[..., None], 0)
        coef, meta = self.analyze(iq.detach().cpu().numpy())
        coef = torch.from_numpy(coef).to(dev)
        x = self.proj(coef)
        T = x.shape[1]
        # WPT coefficients are global (each mixes many samples); with fixed
        # length N there is no per-token validity structure. All tokens valid.
        v = torch.ones((valid.shape[0], T), dtype=torch.bool, device=dev)
        spacing = 1 << self.level
        t_leaf = meta["tokens_per_leaf"]
        # center/support per leaf: offset = leaf_id * spacing (approx, filter support extra)
        centers = []
        bands = []
        for leaf in range(meta["leaves"]):
            c = (np.arange(t_leaf) + 0.5) * P * spacing
            centers.append(c)
            bands.append(np.full(t_leaf, leaf))
        meta.update({"projection_in_dim": 2 * self.patch_size, "dim": self.dim})
        return TokenBatch(
            embeddings=x, valid_mask=v,
            time_center=torch.tensor(np.concatenate(centers), device=dev, dtype=torch.float32),
            time_support=torch.full((T,), float(P * spacing), device=dev),
            scale_id=_ones_long(T, dev, self.level),
            band_id=torch.tensor(np.concatenate(bands), device=dev, dtype=torch.long),
            metadata=meta)


class T4DWTokenizer(nn.Module):
    """Multilevel DWT keeping every component ``a_J, d_J, ..., d_1`` (critical
    sampling). Each component's coefficients are patched into P-groups and all
    tokens share one Linear(2P, D). Complements T3: coarse-fine scale split
    rather than a uniform full tree.
    """

    def __init__(self, wavelet="db4", level=3, patch_size=8, dim=176, mode="periodization"):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.patch_size = patch_size
        self.dim = dim
        self.mode = mode
        self.proj = nn.Linear(2 * patch_size, dim)

    def analyze(self, iq_np):
        B, L, _ = iq_np.shape
        P = self.patch_size
        N = L
        # components in wavedec order [a_J, d_J, d_{J-1}, ..., d_1]; spacing = 2^scale
        scales = [self.level] * 2 + list(range(self.level - 1, 0, -1))
        lengths = [N >> s for s in scales]  # periodization: component at scale s has N/2^s coeffs
        tokens_per = [n // P for n in lengths]
        if any(n % P for n in lengths):
            raise ValueError(f"component length not divisible by P={P}: {lengths}")
        # per-token metadata (identical across the batch; fixed-length data)
        scale_ids, band_ids, centers, supports = [], [], [], []
        for k, (s, n) in enumerate(zip(scales, lengths)):
            for tt in range(n // P):
                scale_ids.append(s)
                band_ids.append(k)
                centers.append((tt + 0.5) * P * (1 << s))
                supports.append(P * (1 << s))
        # data coefficients [B, T, 2P] (I part then Q part), component-major order
        patch_i, patch_q = [], []
        for b in range(B):
            ci = pywt.wavedec(iq_np[b, :, 0], self.wavelet, mode=self.mode, level=self.level)
            cq = pywt.wavedec(iq_np[b, :, 1], self.wavelet, mode=self.mode, level=self.level)
            patch_i.extend([c.reshape(-1, P) for c in ci])
            patch_q.extend([c.reshape(-1, P) for c in cq])
        ti = np.concatenate(patch_i, axis=0)  # [B*T, P]
        tq = np.concatenate(patch_q, axis=0)
        T = sum(tokens_per)
        coef = np.concatenate([ti, tq], axis=-1).reshape(B, T, 2 * P)
        meta = {"name": "t4", "wavelet": self.wavelet, "level": self.level,
                "mode": self.mode, "component_order": ["a%d" % self.level] +
                ["d%d" % (self.level - i) for i in range(self.level)],
                "component_scales": scales, "tokens_per_component": tokens_per,
                "token_count": T}
        return coef.astype(np.float32), meta, (scale_ids, band_ids, centers, supports)

    def forward(self, iq, valid):
        dev = iq.device
        iq = iq.masked_fill(~valid[..., None], 0)
        coef, meta, aux = self.analyze(iq.detach().cpu().numpy())
        scale_ids, band_ids, centers, supports = aux
        coef = torch.from_numpy(coef).to(dev)
        x = self.proj(coef)
        T = x.shape[1]
        v = torch.ones((valid.shape[0], T), dtype=torch.bool, device=dev)
        meta.update({"projection_in_dim": 2 * self.patch_size, "dim": self.dim})
        return TokenBatch(
            embeddings=x, valid_mask=v,
            time_center=torch.tensor(centers, device=dev, dtype=torch.float32),
            time_support=torch.tensor(supports, device=dev, dtype=torch.float32),
            scale_id=torch.tensor(scale_ids, device=dev, dtype=torch.long),
            band_id=torch.tensor(band_ids, device=dev, dtype=torch.long),
            metadata=meta)


class T5STFTTokenizer(nn.Module):
    """Complex two-sided STFT. Screening uses a single rectangular window with
    critical sampling (hop == nperseg) so the transform is exactly invertible
    and the cell count is N (== baseline token budget). Time-frequency blocks of
    (freq_block x time_block) complex cells are flattened into tokens and
    projected to D. ``hann + hop/2`` (overlapping, COLA) is a Phase-2 variant.
    """

    def __init__(self, nperseg=32, hop=None, freq_block=4, time_block=2, dim=176):
        super().__init__()
        self.nperseg = nperseg
        self.hop = hop if hop is not None else nperseg
        self.freq_block = freq_block
        self.time_block = time_block
        self.dim = dim
        self.proj = nn.Linear(2 * freq_block * time_block, dim)

    def analyze(self, iq_np):
        B, L, _ = iq_np.shape
        # Critical-sampled STFT (boxcar window, hop == nperseg) reduces to a
        # block DFT; computed with numpy FFT for exact invertibility and an exact
        # cell count == N (so the token budget matches the baseline).
        if self.hop != self.nperseg:
            raise ValueError("Screening T5 uses critical sampling (hop == nperseg); "
                             "overlapping hann+hop/2 is a Phase-2 variant")
        z = iq_np[..., 0] + 1j * iq_np[..., 1]  # [B, L]
        nblk = L // self.nperseg
        z = z[:, :nblk * self.nperseg].reshape(B, nblk, self.nperseg)  # [B, Tf, nperseg]
        spec = np.fft.fft(z, axis=-1)  # [B, Tf, F] (freq in numpy order)
        spec = spec.transpose(0, 2, 1)  # [B, F, Tf]
        F, Tf = self.nperseg, nblk
        fb, tb = self.freq_block, self.time_block
        if F % fb or Tf % tb:
            raise ValueError(f"STFT grid {F}x{Tf} not divisible by block {fb}x{tb}")
        grid = spec.reshape(B, F // fb, fb, Tf // tb, tb).transpose(0, 1, 3, 2, 4)
        grid = grid.reshape(B, (F // fb) * (Tf // tb), fb * tb)  # [B, T, cells]
        coef = np.stack([grid.real, grid.imag], axis=-1).reshape(B, -1, 2 * fb * tb)
        T = coef.shape[1]
        freq_groups = F // fb
        time_groups = Tf // tb
        centers = ((np.arange(T) // freq_groups) * tb + tb / 2) * self.hop
        supports = np.full(T, self.nperseg * tb, dtype=np.float32)
        meta = {"name": "t5", "window": "boxcar", "nperseg": self.nperseg,
                "hop": self.hop, "fft_mode": "twosided (numpy.fft)", "freq_block": fb,
                "time_block": tb, "freq_bins": F, "time_frames": Tf,
                "freq_groups": freq_groups, "time_groups": time_groups,
                "cells_per_token": fb * tb, "token_count": T,
                "freq_axis_order": "0..F/2-1 positive, then negative (numpy.fft order)"}
        return coef.astype(np.float32), meta, (centers.astype(np.float32), supports)

    def forward(self, iq, valid):
        dev = iq.device
        iq = iq.masked_fill(~valid[..., None], 0)
        coef, meta, aux = self.analyze(iq.detach().cpu().numpy())
        centers, supports = aux
        coef = torch.from_numpy(coef).to(dev)
        x = self.proj(coef)
        T = x.shape[1]
        v = torch.ones((valid.shape[0], T), dtype=torch.bool, device=dev)
        meta.update({"projection_in_dim": 2 * self.freq_block * self.time_block,
                     "dim": self.dim})
        return TokenBatch(
            embeddings=x, valid_mask=v,
            time_center=torch.tensor(centers, device=dev),
            time_support=torch.tensor(supports, device=dev),
            scale_id=_ones_long(T, dev, 0),
            band_id=torch.tensor(np.repeat(np.arange(meta["freq_groups"]), meta["time_groups"]),
                                 device=dev, dtype=torch.long),
            metadata=meta)


_TOKENIZERS = {
    "t0": T0PatchTokenizer,
    "t1": T1OrthoTokenizer,
    "t2": T2MultiscaleConv,
    "b": BSingleConv,
    "t3": T3WPTTokenizer,
    "t4": T4DWTokenizer,
    "t5": T5STFTTokenizer,
}


def build_tokenizer(cfg: dict) -> nn.Module:
    name = cfg["name"]
    if name not in _TOKENIZERS:
        raise ValueError(f"Unknown tokenizer {name!r}; choose from {list(_TOKENIZERS)}")
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return _TOKENIZERS[name](**kwargs)
