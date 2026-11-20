"""Phase 1: relation diagnostic on synthetic pairs (section 8, no training).

Builds the four pairing classes and reports whether the complex-correlation
descriptor separates true matches from false ones:

* controlled transform : same waveform, integer delay + rotation + noise
* same-class independent symbols : independent draws of one modulation
* different class            : two different modulations
* independent noise          : pure complex Gaussian

For each category we report the distribution of max|rho| over lags, and for the
controlled case the delay error (argmax lag vs true lag) and the sharpness of the
soft distribution over lags. A small SNR sweep shows how the true/false gap
behaves as noise grows.

Output: stdout summary table + JSON written to ``--out``.
"""
import argparse
import json
import numpy as np
import torch

from .correlation import rho_all
from .synthetic import (controlled_pair, same_class_pair, diff_class_pair,
                        noise_pair, modulations)


def run_category(zpairs, d, device):
    """zpairs: (N, 2, P) complex64. Returns per-pair max|rho|, argmax lag, entropy."""
    z = torch.as_tensor(zpairs, dtype=torch.complex64, device=device)
    rho, _, valid = rho_all(z, d)                    # [N, 2, 2, K]
    r = rho[:, 0, 1, :]                              # [N, K] off-diagonal pair
    a = r.abs().square()
    a = a.masked_fill(~valid[:, 0, 1, :], 0.0)
    max_abs = r.abs().max(-1).values.cpu().numpy()
    argmax = a.argmax(-1).cpu().numpy()              # lag index k, tau = k - d
    p = torch.softmax(a / 0.1, dim=-1)               # temperature 0.1 (section 6.2)
    p = torch.where(p > 0, p, torch.ones_like(p))
    entropy = -(p * p.log()).sum(-1).cpu().numpy() / np.log(a.shape[-1])
    return max_abs, argmax, entropy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--snr", default="", help="comma-separated dB; 'inf'/'noiseless' = clean")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/diagnose.json")
    a = ap.parse_args()
    def _parse(s):
        s = s.strip()
        if s in ("", "inf", "noiseless", "none"):
            return None
        return float(s)
    snr_list = [_parse(s) for s in a.snr.split(",")] if a.snr.strip() else [None]
    rng = np.random.default_rng(a.seed)
    device = torch.device(a.device)
    mods = modulations()

    print(f"# Phase-1 relation diagnostic  (pairs={a.pairs}, P={a.P}, d={a.d})")
    summary = {}
    for snr in snr_list:
        label = "noiseless" if snr is None else f"SNR {snr:g}dB"
        print(f"\n## {label}")
        rows = {}
        # controlled transform (mixed delays 0..d)
        zc = np.empty((a.pairs, 2, a.P), dtype=np.complex64)
        taus = []
        for i in range(a.pairs):
            zc[i, 0], zc[i, 1], tau = controlled_pair("qpsk", a.P, a.d, rng, snr)
            taus.append(tau)
        ma, am, ent = run_category(zc, a.d, device)
        err = (am - a.d) - np.array(taus)             # tau = k - d
        rows["controlled"] = dict(max_abs_mean=float(ma.mean()), max_abs_median=float(np.median(ma)),
                                  delay_acc=float((err == 0).mean()), delay_mae=float(np.abs(err).mean()),
                                  entropy_mean=float(ent.mean()))
        # same class independent symbols
        zs = np.empty((a.pairs, 2, a.P), dtype=np.complex64)
        for i in range(a.pairs):
            zs[i, 0], zs[i, 1], _ = same_class_pair("qpsk", a.P, rng, snr)
        ma, _, ent = run_category(zs, a.d, device)
        rows["same_class"] = dict(max_abs_mean=float(ma.mean()), max_abs_median=float(np.median(ma)),
                                  entropy_mean=float(ent.mean()))
        # different class
        zd = np.empty((a.pairs, 2, a.P), dtype=np.complex64)
        for i in range(a.pairs):
            zd[i, 0], zd[i, 1], _ = diff_class_pair("qpsk", "qam16", a.P, rng, snr)
        ma, _, ent = run_category(zd, a.d, device)
        rows["diff_class"] = dict(max_abs_mean=float(ma.mean()), max_abs_median=float(np.median(ma)),
                                  entropy_mean=float(ent.mean()))
        # independent noise
        zn = np.empty((a.pairs, 2, a.P), dtype=np.complex64)
        for i in range(a.pairs):
            zn[i, 0], zn[i, 1], _ = noise_pair(a.P, rng)
        ma, _, ent = run_category(zn, a.d, device)
        rows["noise"] = dict(max_abs_mean=float(ma.mean()), max_abs_median=float(np.median(ma)),
                             entropy_mean=float(ent.mean()))

        summary[label] = rows
        print(f"{'category':<16}{'max|rho| mean':>15}{'median':>10}{'delay acc':>12}{'delay MAE':>12}{'entropy':>10}")
        for name, m in rows.items():
            acc = f"{m.get('delay_acc', 0):.3f}" if "delay_acc" in m else "-"
            mae = f"{m.get('delay_mae', 0):.2f}" if "delay_mae" in m else "-"
            print(f"{name:<16}{m['max_abs_mean']:>15.4f}{m['max_abs_median']:>10.4f}"
                  f"{acc:>12}{mae:>12}{m['entropy_mean']:>10.4f}")

    with open(a.out, "w") as f:
        json.dump(dict(args=vars(a), results=summary), f, indent=2)
    print(f"\nSaved: {a.out}")


if __name__ == "__main__":
    main()
