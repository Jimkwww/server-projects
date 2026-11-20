"""IQ corruption for the SSM_L round-1 denoising pretraining task.

The corruption protocol (from the round-1 task book, section 4, denoising):

- The original observation x is joint max_abs-normalized ONCE (per-sample joint max
  |component| over I and Q), i.e. the same normalization the historical Global uses.
- Noise y is added to the normalized x; neither y nor the target is re-normalized per peak.
- Injected level per sample is drawn uniformly from {0, 5, 10, 15} dB, defined as the ratio
  of the original observation total power to the *expected* injected-noise power (NOT the
  dataset's true SNR). Re-sampled independent I/Q Gaussian noise on every access.
- Zero-power samples: sigma = 0 for a zero-power sample (finite output, no NaN).
"""
import math
import torch


def corrupt_iq(x: torch.Tensor, generator: torch.Generator):
    """Return noisy observation, per-real-channel sigma, and injected level."""
    if x.ndim != 3 or x.shape[1] != 2 or not x.is_floating_point():
        raise ValueError("Expected floating IQ with shape [B, 2, T]")
    levels = x.new_tensor([0., 5., 10., 15.])
    indices = torch.randint(len(levels), (x.shape[0],), device=x.device,
                            generator=generator)
    level_db = levels[indices]
    power = x.square().sum(dim=1).mean(dim=-1)
    sigma = (power / 2).sqrt() * torch.exp(-math.log(10) * level_db / 20)
    epsilon = torch.randn(x.shape, device=x.device, dtype=x.dtype,
                          generator=generator)
    return x + sigma[:, None, None] * epsilon, sigma, level_db
