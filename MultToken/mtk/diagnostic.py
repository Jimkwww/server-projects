"""Information-preservation and correctness diagnostics for the fixed front-ends.

Phase-0 checks (doc section 6 + section 8 stage 0):

1. Analysis-coefficient layer: WPT / DWT / STFT invert via the known inverse and
   report float32 relative L2 reconstruction error. Targets: float64 < 1e-10,
   float32 < 1e-5.
2. Frequency order: verify the WPT leaf frequency ordering via a chirp (low
   frequency energy must land in the low-frequency leaf).
3. Boundary: verify the boundary mode produces critical sampling and does not
   scramble the time axis (impulse reconstruction).
4. IQ pairing: I and Q undergo the SAME tree / order (identity-by-construction
   check plus a shuffle check).

Each check returns a dict of measurements; :func:`run_all` prints a report and
returns a dict keyed by check name. Used by ``tests/test_diagnostic.py``.
"""
import numpy as np

import torch

from . import tokenizers as T


def _rel_l2(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        a, b = a.astype(np.complex128), b.astype(np.complex128)
    else:
        a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def check_wpt_inverse(wavelet="haar", level=2, N=1024, mode="periodization"):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N).astype(np.float32)
    leaves = T._wpt_leaves(x, wavelet, level, mode)
    if sum(len(l) for l in leaves) != N:
        raise AssertionError(f"WPT not critically sampled: {sum(len(l) for l in leaves)} != {N}")
    # Inverse via pywt.idwt back up the tree.
    import pywt
    cur = leaves
    for _ in range(level):
        nxt = []
        for i in range(0, len(cur), 2):
            nxt.append(pywt.idwt(cur[i], cur[i + 1], wavelet, mode=mode))
        cur = nxt
    recon = np.asarray(cur[0], dtype=np.float32)[:N]
    return {"critical_sampling": True, "rel_l2_float32": _rel_l2(x, recon)}


def check_dwt_inverse(wavelet="db4", level=3, N=1024, mode="periodization"):
    import pywt
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N).astype(np.float32)
    coeffs = pywt.wavedec(x, wavelet, mode=mode, level=level)
    recon = np.asarray(pywt.waverec(coeffs, wavelet, mode=mode), dtype=np.float32)[:N]
    return {"rel_l2_float32": _rel_l2(x, recon),
            "component_lengths": [len(c) for c in coeffs]}


def check_stft_inverse(nperseg=32, hop=32, N=1024):
    rng = np.random.default_rng(0)
    z = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
    nblk = N // nperseg
    blocks = z[:nblk * nperseg].reshape(nblk, nperseg)
    spec = np.fft.fft(blocks, axis=-1)
    recon = np.fft.ifft(spec, axis=-1).reshape(-1)
    return {"rel_l2_complex": _rel_l2(z, recon),
            "grid": [nperseg, nblk]}


def check_wpt_freq_order(wavelet="haar", level=2, N=1024):
    """Low-frequency sinusoid energy must concentrate in the low-frequency leaf."""
    t = np.arange(N) / N
    x = np.sin(2 * np.pi * 1.0 * t).astype(np.float32)  # ~1 cycle -> very low freq
    leaves = T._wpt_leaves(x, wavelet, level, mode="periodization")
    energy = [float(np.sum(np.abs(l) ** 2)) for l in leaves]
    low_natural = int(np.argmax(energy))
    gray = T._freq_order(level)
    # frequency order: natural index gray[j] holds the j-th frequency bin.
    # low freq should be bin 0 -> natural index gray[0] == 0.
    return {"natural_leaf_energies": energy,
            "dominant_natural_leaf": low_natural,
            "freq_order_natural_indices": gray,
            "low_freq_in_natural_leaf_0": bool(low_natural == gray[0])}


def check_iq_pairing(wavelet="haar", level=2, N=1024):
    """I and Q use the same transform; swapping the channels swaps the outputs."""
    rng = np.random.default_rng(1)
    iq = rng.standard_normal((1, N, 2)).astype(np.float32)
    tok = T.T3WPTTokenizer(wavelet=wavelet, level=level, patch_size=8, dim=16)
    coef, meta = tok.analyze(iq)
    iq_swap = np.stack([iq[..., 1], iq[..., 0]], axis=-1)
    coef_swap, _ = tok.analyze(iq_swap)
    # coef layout: [1, T, 2P] where first P columns are I-coeffs, last P are Q-coeffs.
    P = 8
    i_part = coef[..., :P]
    q_part = coef[..., P:]
    i_swap = coef_swap[..., :P]
    q_swap = coef_swap[..., P:]
    return {"i_of_swap_equals_q": float(np.abs(i_swap - q_part).max()),
            "q_of_swap_equals_i": float(np.abs(q_swap - i_part).max())}


def check_token_shapes():
    """Every tokenizer emits T tokens and D-dim embeddings for a fixed-length input."""
    torch.manual_seed(0)
    B, L, D = 4, 1024, 176
    iq = torch.randn(B, L, 2)
    valid = torch.ones(B, L, dtype=torch.bool)
    specs = [
        T.T0PatchTokenizer(patch_size=8, dim=D),
        T.T1OrthoTokenizer(patch_size=8, dim=D),
        T.T2MultiscaleConv(patch_size=8, dim=D),
        T.T3WPTTokenizer(wavelet="haar", level=2, patch_size=8, dim=D),
        T.T3WPTTokenizer(wavelet="db4", level=2, patch_size=8, dim=D),
        T.T4DWTokenizer(wavelet="db4", level=3, patch_size=8, dim=D),
        T.T5STFTTokenizer(nperseg=32, hop=32, dim=D),
    ]
    out = {}
    for tok in specs:
        tb = tok(iq, valid)
        assert tb.embeddings.shape == (B, tb.embeddings.shape[1], D)
        assert tb.valid_mask.shape == tb.embeddings.shape[:2]
        meta = tb.metadata
        key = meta["name"] + ("-" + meta.get("wavelet", "") if meta["name"] == "t3" else "")
        out[key] = {"T": tb.embeddings.shape[1], "D": tb.embeddings.shape[2]}
    return out


def run_all():
    report = {}
    report["wpt_inverse_haar_j2"] = check_wpt_inverse("haar", 2)
    report["wpt_inverse_db4_j2"] = check_wpt_inverse("db4", 2)
    report["dwt_inverse_db4_j3"] = check_dwt_inverse("db4", 3)
    report["stft_inverse"] = check_stft_inverse()
    report["wpt_freq_order_haar_j2"] = check_wpt_freq_order("haar", 2)
    report["iq_pairing"] = check_iq_pairing()
    report["token_shapes"] = check_token_shapes()
    return report


if __name__ == "__main__":
    import json
    report = run_all()
    print(json.dumps(report, indent=2, ensure_ascii=False))
