"""Revision-1 Section 5: SSM state statistics on a fixed, reproducible subset (NOT a single
model). Reports per-layer rho quantiles, near-1 fraction, write-gate distribution, state RMS,
instantaneous tau quantiles + log-distribution (with the clipping rule documented), and the
retention product R(t,l,k)=prod rho(t-j,k) computed in the log domain.

Models: all 18 S1plain models (A: random/MAE/denoise x 3 seeds; B: scratch/MAE/denoise x 3
seeds, RML; scratch wired from the old SSM project). Fixed subset = rng(0)-chosen 512 RML val
samples (index sha256 recorded).

Outputs: state_statistics.json + figures in revision1/reports/figs/.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_r1 import load_s1plain_downstream, R1, R4, R2, SEEDS, task_cfg
from memory_diagnosis import forward_s1, load_a_encoder
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.engine import make_loader
from ssm_exp.model import max_abs_norm

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBSET_N = 512
RHO_CLIP_LO = 1e-9
RHO_CLIP_HI = 1.0 - 1e-9
T_LIST = (1, 4, 16, 32, 64, 128)


def collect_states(enc, x, device):
    """Return per-layer rho [B,N,S], g [B,N,S], h_mag [B,N,S] on a fixed batch."""
    enc.eval()
    x = max_abs_norm(x.to(device))
    per_layer = []
    with torch.no_grad():
        u = enc.front(x)
        p = u.new_zeros(u.shape[0], u.shape[1], 3)
        for block in enc.blocks:
            ssm = block.mix
            xn = block.norm1(u)
            from memory_diagnosis import _ssm_terms
            v, a, g, rho = _ssm_terms(ssm, xn, p)
            h = _scan_full(a, g * v)
            per_layer.append((rho.detach().cpu(), g.detach().cpu(), h.abs().detach().cpu()))
            y = ssm.gain * F.linear(h, ssm.C).real + ssm.direct * xn
            u = u + block.drop(y)
            u = u + block.drop(block.ffn(block.norm2(u)))
    return per_layer


def _scan_full(a, b):
    from ssm_exp.model import parallel_scan
    return parallel_scan(a, b)


def layer_stats(rho, g, h_mag):
    rho = rho.numpy().ravel()
    g = g.numpy().ravel()
    h_mag = h_mag.numpy().ravel()
    # tau computed in float64 with an explicit clip so rho==1.0 (float32 rounding) does not
    # produce log(1)=0 -> division by zero. Clipping rule: rho in [1e-9, 1-1e-9]; the upper
    # clip bounds tau at -1/log(1-1e-9) ~ 1e9 tokens (a floor for the 'infinite' channels).
    rho64 = rho.astype(np.float64)
    tau = -1.0 / np.log(np.clip(rho64, RHO_CLIP_LO, RHO_CLIP_HI))
    logtau = np.log10(np.clip(tau, 1e-3, 1e9))
    tau_hist, tau_edges = np.histogram(logtau, bins=60, range=(-3, 9))
    return {
        'rho_q': {q: float(np.percentile(rho, q)) for q in (1, 5, 25, 50, 75, 90, 99, 99.9)},
        'rho_near1_frac': {'>0.9': float((rho > 0.9).mean()),
                           '>0.99': float((rho > 0.99).mean()),
                           '>0.999': float((rho > 0.999).mean()),
                           '>0.9999': float((rho > 0.9999).mean())},
        'gate': {'mean': float(g.mean()), 'q10': float(np.percentile(g, 10)),
                 'q50': float(np.percentile(g, 50)), 'q90': float(np.percentile(g, 90)),
                 'satur_frac': float(((g < 0.01) | (g > 0.99)).mean())},
        'state_rms': float(np.sqrt((h_mag ** 2).mean())),
        'tau_q': {q: float(np.percentile(tau, q)) for q in (1, 5, 25, 50, 75, 90, 99)},
        'tau_log_mean': float(logtau.mean()), 'tau_log_std': float(logtau.std()),
        'tau_log_hist': {'counts': tau_hist.tolist(), 'edges': tau_edges.tolist()},
    }


def retention_curve(rho):
    """log R(t) = mean over (batch, n>=t, state) of sum_{j<t} log rho(n-j). Returns {t: R}."""
    # rho: [B,N,S]
    logrho = torch.log(rho.clamp_min(1e-12))
    logcum = logrho.cumsum(dim=1)  # [B,N,S]
    B, N, S = rho.shape
    out = {}
    for t in T_LIST:
        if t >= N:
            out[t] = None
            continue
        # mean over n in [t, N), channels, batch
        seg = logcum[:, t:] - logcum[:, :-t]
        logR = seg.mean().item()
        out[t] = float(np.exp(logR))
    return out


def collect_model(enc, x, device):
    """Per-layer stats + retention for one model on the fixed subset."""
    per_layer = collect_states(enc, x, device)
    out = []
    for li, (rho, g, hm) in enumerate(per_layer):
        st = layer_stats(rho, g, hm)
        st['layer'] = li
        st['retention_R'] = retention_curve(rho)
        out.append(st)
    return out


def model_paths():
    """(name, kind, loader) for all 18 S1plain RML models."""
    models = []
    cfg = task_cfg('rml')
    for init in ('random', 'mae', 'denoise'):
        for s in SEEDS:
            models.append((f'A_{init}_seed{s}', 'A', init, s))
    for init in ('scratch', 'mae', 'denoise'):
        for s in SEEDS:
            models.append((f'B_{init}_seed{s}', 'B', init, s))
    return models, cfg


def load_model(kind, init, seed, cfg, device):
    if kind == 'A':
        return load_a_encoder(init, seed, cfg, device)
    # B group (finetuned downstream): return the encoder only
    if init == 'scratch':
        ckpt = Path('/home/user_wzk/wzk/projects/SSM/SSM/runs/clean') / f'S1_seed{seed}' / 'best.pt'
        return load_s1plain_downstream(ckpt, device, old_classifier=True)[0].encoder
    ckpt = R4 / 'runs' / 'rml' / init / f'S1plain_seed{seed}' / 'best.pt'
    return load_s1plain_downstream(ckpt, device, old_classifier=False)[0].encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')

    m = read_manifest(str(ROOT / 'configs' / 'data.json'))
    val = IQDataset(m, 'val')
    idx = np.random.default_rng(0).choice(len(val), SUBSET_N, replace=False)
    idx.sort()
    x = torch.stack([val[i][0] for i in idx])  # [512, 2, 1024]
    idx_sha = hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()

    models, cfg = model_paths()
    per_model = {}
    for name, kind, init, seed in models:
        enc = load_model(kind, init, seed, cfg, device)
        per_model[name] = collect_model(enc, x, device)
        print(f'[stats] {name}', flush=True)

    (R1 / 'reports').mkdir(parents=True, exist_ok=True)
    out = {'subset_n': SUBSET_N, 'val_idx_sha256': idx_sha, 'rho_clip': [RHO_CLIP_LO, RHO_CLIP_HI],
           'T_list': list(T_LIST), 'models': per_model}
    (R1 / 'reports' / 'state_statistics.json').write_text(json.dumps(out, indent=2))
    _make_figures(out)
    print('[done] state_statistics.json + figures')


def _aggregate(per_model, key_fn):
    """mean + std across models of key_fn(layer_dict)."""
    per_layer = {}
    for name, layers in per_model.items():
        for li, L in enumerate(layers):
            per_layer.setdefault(li, []).append(key_fn(L))
    agg = []
    for li in sorted(per_layer):
        vals = per_layer[li]
        agg.append({'layer': li, 'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)) if len(vals) > 1 else 0.0,
                    'n_models': len(vals)})
    return agg


def _make_figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs = R1 / 'reports' / 'figs'
    figs.mkdir(parents=True, exist_ok=True)
    per_model = out['models']

    # gather per-layer rho/tau/g concatenated across models (downsampled for plotting)
    def gather(field):
        d = {}
        for name, layers in per_model.items():
            for li, L in enumerate(layers):
                d.setdefault(li, []).append(L[field])
        return d

    # fig 1: rho quantiles per layer (mean over models)
    rho_d = {}
    qs = [1, 5, 25, 50, 75, 90, 99, 99.9]
    for name, layers in per_model.items():
        for li, L in enumerate(layers):
            rho_d.setdefault(li, []).append([L['rho_q'][q] for q in qs])
    fig, ax = plt.subplots(figsize=(8, 4))
    for li in sorted(rho_d):
        xs = np.array(rho_d[li])
        ax.plot(qs, xs.mean(0), marker='o', label=f'L{li}')
    ax.set_xlabel('rho quantile (%)'); ax.set_ylabel('rho'); ax.set_ylim(0, 1)
    ax.legend(ncol=2, fontsize=7); ax.set_title('per-layer rho quantiles (mean over 18 models)')
    fig.tight_layout(); fig.savefig(figs / 'state_stats_rho_quantiles.png', dpi=110)
    plt.close(fig)

    # fig 2: retention R(t) per layer (log y)
    all_logR = {}
    for name, layers in per_model.items():
        for li, L in enumerate(layers):
            rr = L['retention_R']
            all_logR.setdefault(li, []).append([np.log(rr[t]) for t in T_LIST if rr[t] is not None])
    fig, ax = plt.subplots(figsize=(8, 4))
    for li in sorted(all_logR):
        arr = np.array(all_logR[li])
        m = arr.mean(0); s = arr.std(0)
        Ts_ok = [t for t in T_LIST if t < 128][:len(m)]
        ax.plot(Ts_ok, m, marker='o', label=f'L{li}')
        ax.fill_between(Ts_ok, m - s, m + s, alpha=0.15)
    ax.set_xlabel('t (tokens back)'); ax.set_ylabel('log R(t)')
    ax.legend(ncol=2, fontsize=7)
    ax.set_title('retention product log R(t) per layer (mean±std over 18 models)')
    fig.tight_layout(); fig.savefig(figs / 'state_stats_retention.png', dpi=110)
    plt.close(fig)

    # fig 3: tau log10 histogram aggregated across models (layers 0 and 5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, li in zip(axes, (0, 5)):
        counts = None; edges = None
        for name, layers in per_model.items():
            if len(layers) > li:
                h = layers[li]['tau_log_hist']
                if counts is None:
                    counts = np.array(h['counts'], dtype=np.float64)
                    edges = np.array(h['edges'])
                else:
                    counts += np.array(h['counts'], dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, counts, width=edges[1] - edges[0], alpha=0.7)
        ax.set_xlabel('log10(tau) [tokens]'); ax.set_ylabel('count')
        ax.set_title(f'layer {li} tau distribution (summed over 18 models)')
    fig.tight_layout(); fig.savefig(figs / 'state_stats_tau_hist.png', dpi=110)
    plt.close(fig)
    print('[figs] state_stats_rho_quantiles.png state_stats_retention.png state_stats_tau_hist.png')


if __name__ == '__main__':
    main()
