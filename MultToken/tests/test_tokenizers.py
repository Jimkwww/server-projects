"""Unit tests for tokenizer shapes, T/D invariance, and T0==baseline embedding."""
import torch
from mtk.tokenizers import (
    build_tokenizer, T0PatchTokenizer, T1OrthoTokenizer, T2MultiscaleConv,
    T3WPTTokenizer, T4DWTokenizer, T5STFTTokenizer,
)


def _make(batch=4, length=1024):
    torch.manual_seed(0)
    return torch.randn(batch, length, 2), torch.ones(batch, length, dtype=torch.bool)


def test_t0_matches_baseline_embed():
    # T0 projection must be the exact baseline patch embedding math: flatten
    # 2P values per patch then Linear(2P -> D).
    iq, valid = _make(batch=2)
    tok = T0PatchTokenizer(patch_size=8, dim=176)
    tb = tok(iq, valid)
    # Manual baseline embedding for comparison.
    x = iq.reshape(2, -1, 16)
    expected = tok.proj(x)
    assert torch.allclose(tb.embeddings, expected, atol=1e-6)
    assert tb.embeddings.shape == (2, 128, 176)
    assert tb.valid_mask.all()


def test_all_tokenizers_same_T_and_D():
    iq, valid = _make(batch=2)
    specs = {
        "t0": dict(patch_size=8, dim=176),
        "t1": dict(patch_size=8, dim=176),
        "t2": dict(patch_size=8, dim=176),
        "t3_haar": dict(wavelet="haar", level=2, patch_size=8, dim=176),
        "t3_db4": dict(wavelet="db4", level=2, patch_size=8, dim=176),
        "t4": dict(wavelet="db4", level=3, patch_size=8, dim=176),
        "t5": dict(nperseg=32, hop=32, dim=176),
    }
    for name, kw in specs.items():
        cfg = {"name": name.split("_")[0], **kw}
        tok = build_tokenizer(cfg)
        tb = tok(iq, valid)
        assert tb.embeddings.shape[1] == 128, f"{name}: T={tb.embeddings.shape[1]}"
        assert tb.embeddings.shape[2] == 176, f"{name}: D={tb.embeddings.shape[2]}"
        assert tb.valid_mask.shape == (2, 128)


def test_t3_leaf_budget_and_iq_order():
    iq, valid = _make(batch=1)
    tok = T3WPTTokenizer(wavelet="haar", level=2, patch_size=8, dim=16)
    tb = tok(iq, valid)
    assert tb.embeddings.shape == (1, 128, 16)
    assert list(tb.scale_id) == [2] * 128
    # band ids: 4 leaves x 32 tokens each, leaf-major
    assert list(tb.band_id[:32]) == [0] * 32
    assert list(tb.band_id[32:64]) == [1] * 32
    assert tb.band_id.max() == 3


def test_t5_grid():
    iq, valid = _make(batch=1)
    tok = T5STFTTokenizer(nperseg=32, hop=32, dim=16)
    tb = tok(iq, valid)
    assert tb.embeddings.shape == (1, 128, 16)
    assert tok.proj.in_features == 16


def test_projection_param_counts_reported():
    # Fair-budget check: T0/T1/T3/T4/T5 projection is Linear(16->176);
    # T2 total branch channels == 16 so its projection is also 16->176.
    for tok in [T0PatchTokenizer(), T1OrthoTokenizer(), T2MultiscaleConv(),
                T3WPTTokenizer(), T4DWTokenizer(), T5STFTTokenizer()]:
        assert tok.proj.in_features == 16, f"{type(tok).__name__} proj in_features != 16"
