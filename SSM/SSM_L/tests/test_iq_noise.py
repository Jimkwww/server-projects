"""Acceptance: injected-noise statistics (task book section 6, check 2)."""
import math
import pytest
import torch

from pretrain.iq_noise import corrupt_iq


def _batch(n=4096, t=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, t, generator=g)
    # max_abs-normalize so power is bounded and non-trivial
    m = x.abs().reshape(n, -1).max(dim=1, keepdim=True).values
    x = x / m.clamp_min(1e-8).unsqueeze(-1)
    return x


def test_shape_and_levels():
    x = _batch()
    gen = torch.Generator().manual_seed(1)
    y, sigma, level = corrupt_iq(x, gen)
    assert y.shape == x.shape
    assert sigma.shape == (x.shape[0],)
    assert level.shape == (x.shape[0],)
    assert set(level.tolist()) <= {0.0, 5.0, 10.0, 15.0}
    assert torch.isfinite(y).all()


def test_noise_zero_mean_equal_var_uncorrelated():
    x = _batch(n=16384, t=128)
    gen = torch.Generator().manual_seed(2)
    y, sigma, level = corrupt_iq(x, gen)
    noise = y - x                       # [B,2,T] = sigma * epsilon
    mi = noise[:, 0].mean()
    mq = noise[:, 1].mean()
    assert abs(mi) < 0.05 and abs(mq) < 0.05, (mi, mq)
    vi = noise[:, 0].var()
    vq = noise[:, 1].var()
    # I/Q variance equal (noise is independent Gaussian per channel)
    assert abs(vi - vq) / max(vi, vq) < 0.1, (vi, vq)
    # I/Q correlation near zero
    c = (noise[:, 0] * noise[:, 1]).mean() / (noise[:, 0].std() * noise[:, 1].std() + 1e-9)
    assert abs(c) < 0.05, c


@pytest.mark.parametrize("level", [0, 5, 10, 15])
def test_power_ratio(level):
    # force a single level via a fixed generator draw on a large batch, then verify the ratio.
    x = _batch(n=65536, t=64)
    gen = torch.Generator().manual_seed(3)
    # draw many times and keep only samples assigned the target level
    y, sigma, lvl = corrupt_iq(x, gen)
    sel = lvl == float(level)
    if sel.sum() < 1000:
        # re-draw to get enough samples at this level
        while sel.sum() < 1000:
            y2, s2, l2 = corrupt_iq(x, gen)
            y[~sel] = y2[~sel]; lvl[~sel] = l2[~sel]
            sel = lvl == float(level)
    xs, ys = x[sel], y[sel]
    p_x = xs.square().sum(dim=(1, 2))
    p_n = (ys - xs).square().sum(dim=(1, 2))
    snr_db = 10 * torch.log10((p_x / p_n).clamp_min(1e-12)).mean().item()
    assert abs(snr_db - level) <= 0.3, (level, snr_db)


def test_zero_input_no_nan():
    x = torch.zeros(8, 2, 64)
    gen = torch.Generator().manual_seed(4)
    y, sigma, _ = corrupt_iq(x, gen)
    assert torch.isfinite(y).all()
    assert (y == 0).all()
    assert (sigma == 0).all()


def test_tiny_magnitude_finite():
    x = torch.full((8, 2, 64), 1e-30)
    gen = torch.Generator().manual_seed(5)
    y, sigma, _ = corrupt_iq(x, gen)
    assert torch.isfinite(y).all()
    assert torch.isfinite(sigma).all()


def test_seed_reproducibility():
    x = _batch(n=64, t=128)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    y1, s1, l1 = corrupt_iq(x, g1)
    y2, s2, l2 = corrupt_iq(x, g2)
    assert torch.equal(y1, y2) and torch.equal(s1, s2) and torch.equal(l1, l2)
