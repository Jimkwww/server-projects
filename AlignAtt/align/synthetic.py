"""Synthetic complex baseband waveforms for the phase-1 relation diagnostic.

Kept deliberately separate from the real RML2018A samples (section 8 phase 1 of
the plan): everything here is generated on the fly from fixed seeds and is never
drawn from the dataset.
"""
import numpy as np

_CONST = {
    "bpsk": np.array([-1.0, 1.0], dtype=np.float64),
    "qpsk": np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=np.complex128) / np.sqrt(2),
}


def _qam(levels):
    grid = levels[:, None] + 1j * levels[None, :]
    return (grid / np.sqrt(np.mean(np.abs(grid) ** 2))).ravel()


_CONST["qam16"] = _qam(np.array([-3.0, -1.0, 1.0, 3.0]))
_CONST["qam64"] = _qam(np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]))


def modulations():
    return list(_CONST.keys())


def gen_waveform(mod, n_samples, rng, sps=4):
    """``n_samples`` IQ samples of ``mod`` with rectangular pulse shaping."""
    const = _CONST[mod]
    n_sym = (n_samples + sps - 1) // sps
    syms = const[rng.integers(0, len(const), n_sym)]
    return np.repeat(syms, sps)[:n_samples].astype(np.complex64)


def add_noise(x, snr_db, rng):
    """Add complex Gaussian noise at ``snr_db`` (None/inf = noiseless)."""
    if snr_db is None or np.isinf(snr_db):
        return x
    p = float(np.mean(np.abs(x) ** 2))
    sigma = np.sqrt(p / (10 ** (snr_db / 10)))
    n = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) / np.sqrt(2)
    return (x + sigma * n).astype(np.complex64)


def controlled_pair(mod, P, d, rng, snr_db=None, phi=None):
    """Same waveform, integer delay ``tau`` and optional rotation ``phi``.

    Returns ``(zi, zj, tau_true)``: two P-point segments of one waveform with a
    known integer delay, plus independent observation noise on each.
    """
    tau = int(rng.integers(0, d + 1))
    phi = phi if phi is not None else float(rng.uniform(0, 2 * np.pi))
    base = gen_waveform(mod, P + d, rng)
    zi = add_noise(base[0:P], snr_db, rng)
    zj = add_noise(base[tau:tau + P] * np.exp(1j * phi), snr_db, rng)
    return zi, zj, tau


def same_class_pair(mod, P, rng, snr_db=None):
    """Two independent symbol draws of the SAME modulation (no shared waveform)."""
    return (add_noise(gen_waveform(mod, P, rng), snr_db, rng),
            add_noise(gen_waveform(mod, P, rng), snr_db, rng), None)


def diff_class_pair(mod_i, mod_j, P, rng, snr_db=None):
    return (add_noise(gen_waveform(mod_i, P, rng), snr_db, rng),
            add_noise(gen_waveform(mod_j, P, rng), snr_db, rng), None)


def noise_pair(P, rng):
    def g():
        return (rng.standard_normal(P) + 1j * rng.standard_normal(P)).astype(np.complex64) / np.sqrt(2)
    return g(), g(), None
