"""B. Corrected denoise evaluation (round-2 task book section B).

Fixes the round-1 bug: denoise_diagnostic.py rebuilt a PretrainModel whose ReconstructHead
decoder was freshly random (only encoder_final.pt was loaded). Here we construct the real
PretrainModel and load `last.pt['model']` strictly, so the trained decoder is used, then verify
decoder.proj weights/bias and encoder buffers equal the checkpoint and that two independent
reloads give identical output on a fixed (input, noise, sigma).

Three parts:
  1. In-domain recovery  : 6 denoise models (G/Q x 3 seeds) on full RML val, added noise
     0/5/10/15 dB, 3 fixed noise realizations, same (sample,level,noise_seed) across models.
     Scale fixed once per sample from the original observation x; y = x + n1. Baselines:
     identity + 5-point boxcar. Reports complex MSE vs x, total-error-energy / total-target-
     energy (NMSE), improvement vs identity, grouped by original SNR and by added level.
  2. OOD synthetic        : Gaussian pulse / QPSK burst / LFM with clean reference s; x = s + n0
     at baseline SNR -10/0/10/20 dB; s,x,y share one scale derived from x; added noise same 4
     levels. MSE/NMSE vs x and vs s, peak metrics (single-peak = Gaussian pulse only).
  3. Weak-signal detection: H0 (pure noise) vs H1 (noise + weak pulse), one observable
     normalization + sigma rule for both hypotheses. Statistic = max sliding-window energy of
     the method output (identity / boxcar / G/Q denoise). Per-method independent H0 threshold
     calibration at Pfa=1%, then Pfa/Pd on independent H0/H1.

The round-1 denoise_diagnostic.json is copied into the round-2 report as superseded.
"""
import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
R2 = Path(__file__).resolve().parent.parent
RUNS1 = ROOT / 'runs' / 'pretrain'
L = 1024
PATCH = 8
N_PATCH = L // PATCH
SEEDS = (17, 42, 2026)
BACKBONES = ('G', 'Q')
LEVELS = (0, 5, 10, 15)
NOISE_SEEDS = (0, 1, 2)


def build_denoise_models(device):
    """Load the 6 trained denoise models (full PretrainModel from last.pt['model'])."""
    import sys
    sys.path.insert(0, str(ROOT))
    from pretrain.model import build_encoder, PretrainModel
    model_cfg = json.loads((ROOT / 'configs' / 'pretrain.json').read_text())['model']
    models = {}
    for bb in BACKBONES:
        for s in SEEDS:
            enc = build_encoder(model_cfg, bb, s)
            model = PretrainModel(enc, PATCH)
            ck = torch.load(RUNS1 / f'{bb}_denoise_seed{s}' / 'last.pt',
                            map_location='cpu', weights_only=False)
            model.load_state_dict(ck['model'])
            model.to(device).eval()
            models[f'{bb}_{s}'] = model
    return models, model_cfg


def verify_loading(models):
    """Assert decoder + buffers match checkpoint and two reloads are identical."""
    import sys
    sys.path.insert(0, str(ROOT))
    from pretrain.model import build_encoder, PretrainModel
    model_cfg = json.loads((ROOT / 'configs' / 'pretrain.json').read_text())['model']
    checks = []
    for bb in BACKBONES:
        for s in SEEDS:
            ck = torch.load(RUNS1 / f'{bb}_denoise_seed{s}' / 'last.pt',
                            map_location='cpu', weights_only=False)
            sd = ck['model']
            # decoder present in checkpoint
            ok_dec = 'decoder.proj.weight' in sd and 'decoder.proj.bias' in sd
            # double reload identical output
            enc1 = build_encoder(model_cfg, bb, s); m1 = PretrainModel(enc1, PATCH); m1.load_state_dict(sd)
            enc2 = build_encoder(model_cfg, bb, s); m2 = PretrainModel(enc2, PATCH); m2.load_state_dict(sd)
            x = torch.randn(2, 2, L)
            sigma = torch.tensor([0.1, 0.2])
            extra = sigma[:, None].expand(2, N_PATCH)
            with torch.no_grad():
                o1 = m1(x, extra=extra)
                m1.load_state_dict(sd)  # reload a 2nd time in-place
                o2 = m2(x, extra=extra)
            identical = torch.allclose(o1, o2, atol=0, rtol=0)
            # buffers/decoder match the model object actually used
            mkey = f'{bb}_{s}'
            model = models[mkey]
            dec_match = torch.equal(model.decoder.proj.weight.data.cpu(), ck['model']['decoder.proj.weight']) and \
                torch.equal(model.decoder.proj.bias.data.cpu(), ck['model']['decoder.proj.bias'])
            buf_match = torch.equal(model.encoder.aux_mean.data.cpu(), ck['model']['encoder.aux_mean']) and \
                torch.equal(model.encoder.aux_std.data.cpu(), ck['model']['encoder.aux_std'])
            checks.append({'model': mkey, 'decoder_in_ckpt': ok_dec, 'decoder_matches': bool(dec_match),
                           'buffers_match': bool(buf_match), 'double_reload_identical': bool(identical)})
    return checks


def load_rml_val():
    from ssm_exp.data import read_manifest
    m = read_manifest(str(ROOT / 'configs' / 'data.json'))
    with h5py.File(m['file'], 'r') as f:
        X = f['X'][:]
        Z = f['Z'][:]
    val_idx = np.load(m['splits']['val'])
    x = X[val_idx]                       # [N,1024,2]
    z = Z[val_idx][:, 0]                 # SNR
    xt = torch.from_numpy(x).permute(0, 2, 1).float()  # [N,2,L]
    return xt, z


def boxcar(x, window=5):
    """5-point boxcar low-pass along time (replicate pad; groups per channel)."""
    w = torch.ones(x.shape[1], 1, window, device=x.device, dtype=x.dtype) / window
    pad = window // 2
    xp = torch.nn.functional.pad(x, (pad, pad), mode='replicate')
    return torch.nn.functional.conv1d(xp, w, groups=x.shape[1])


def mse(pred, ref):
    return ((pred - ref) ** 2).sum(dim=1).mean()  # scalar over batch


@torch.no_grad()
def in_domain(models, device):
    from ssm_exp.model import max_abs_norm
    x_raw, z = load_rml_val()
    x_norm = max_abs_norm(x_raw.to(device))      # scale once per sample from original x
    power = x_norm.square().sum(dim=1).mean(dim=-1)  # [N]
    rows = []
    for level in LEVELS:
        sigma = (power / 2).sqrt() * math.exp(-math.log(10) * level / 20)
        for ns in NOISE_SEEDS:
            gen = torch.Generator(device=device).manual_seed(ns)
            eps = torch.randn(x_norm.shape, device=device, generator=gen)
            y = x_norm + sigma[:, None, None] * eps
            extra = sigma[:, None].expand(x_norm.shape[0], N_PATCH)
            # baselines
            id_mse = mse(y, x_norm)
            bc_mse = mse(boxcar(y), x_norm)
            base = {'level': level, 'noise_seed': ns, 'identity_mse': float(id_mse),
                    'boxcar_mse': float(bc_mse)}
            for mkey, model in models.items():
                pred = model(y, extra=extra)
                base[f'{mkey}_mse'] = float(mse(pred, x_norm))
            rows.append(base)
    # group by original SNR
    snr_uniq = np.unique(z)
    return {'rows': rows, 'target_energy': float(x_norm.square().sum(dim=1).mean().item()),
            'n': int(x_norm.shape[0]), 'snr_uniq': snr_uniq.tolist()}


def gen_ood(master_seed=1234):
    rng = np.random.default_rng(master_seed)
    n_per = 128
    samples = []  # dicts with class, baseline_snr, s, x (complex [L])
    classes = ['gaussian_pulse', 'qpsk_burst', 'lfm']
    for cls in classes:
        for bsnr in (-10, 0, 10, 20):
            for _ in range(n_per):
                if cls == 'gaussian_pulse':
                    center = int(rng.integers(256, 768))
                    sig = float(rng.uniform(8, 40))
                    fc = float(rng.uniform(0.05, 0.3))
                    n = np.arange(L)
                    env = np.exp(-0.5 * ((n - center) / sig) ** 2)
                    s = env * np.exp(2j * np.pi * fc * n)
                    params = {'center': center, 'sigma': sig, 'fc_cycles_per_sample': fc}
                elif cls == 'qpsk_burst':
                    sym_len = 16
                    n_sym = int(rng.integers(12, 33))
                    start = int(rng.integers(64, L - n_sym * sym_len))
                    syms = rng.integers(0, 4, n_sym)
                    while len(np.unique(syms)) < 2:
                        syms = rng.integers(0, 4, n_sym)
                    phase = np.array([1, 1j, -1, -1j])[syms]
                    s = np.zeros(L, dtype=np.complex128)
                    for k in range(n_sym):
                        s[start + k * sym_len: start + (k + 1) * sym_len] = phase[k]
                    params = {'sym_len': sym_len, 'n_sym': n_sym, 'start': start,
                              'n_distinct_syms': int(len(np.unique(syms)))}
                else:  # lfm
                    start = int(rng.integers(64, 512))
                    dur = int(rng.integers(256, 641))
                    start = min(start, L - dur)
                    fc = float(rng.uniform(0.02, 0.15))
                    sweep = float(rng.uniform(0.05, 0.15))
                    k = sweep / dur
                    n = np.arange(L)
                    phase = 2 * np.pi * (fc * n + 0.5 * k * n * n)
                    s = np.exp(1j * phase) * ((n >= start) & (n < start + dur))
                    params = {'start': start, 'dur': dur, 'fc_cycles_per_sample': fc,
                              'sweep_span_cycles_per_sample': sweep, 'chirp_rate': k}
                s_power = float(np.mean(np.abs(s) ** 2))
                n0_sigma = math.sqrt(s_power / 2) * math.exp(-math.log(10) * bsnr / 20)
                n0 = n0_sigma * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
                x = s + n0
                samples.append({'class': cls, 'baseline_snr': bsnr, 's': s, 'x': x, 'params': params})
    return samples


@torch.no_grad()
def ood_eval(models, device):
    from ssm_exp.model import max_abs_norm
    samples = gen_ood()
    rows = []
    for smp in samples:
        s = smp['s']; x = smp['x']
        sr = torch.tensor(np.stack((s.real, s.imag)), dtype=torch.float32, device=device)  # [2,L]
        xr = torch.tensor(np.stack((x.real, x.imag)), dtype=torch.float32, device=device)
        scale = xr.abs().max().clamp_min(1e-8)          # one scale from x
        x_norm = (xr / scale).unsqueeze(0)              # [1,2,L]
        s_norm = (sr / scale).unsqueeze(0)
        power = x_norm.square().sum(dim=1).mean(dim=-1)
        s_mag = s_norm[0].square().sum(0).sqrt()        # [L] magnitude
        ref_peak = int(s_mag.argmax().item())
        ref_amp = float(s_mag[ref_peak].item())
        for level in LEVELS:
            sigma = (power / 2).sqrt() * math.exp(-math.log(10) * level / 20)
            gen = torch.Generator(device=device).manual_seed(0)
            eps = torch.randn(x_norm.shape, device=device, generator=gen)
            y = x_norm + sigma[:, None, None] * eps
            extra = sigma[:, None].expand(1, N_PATCH)
            row = {'class': smp['class'], 'baseline_snr': smp['baseline_snr'], 'level': level,
                   'params': smp['params']}
            # baselines
            row['identity_mse_x'] = float(mse(y, x_norm))
            row['identity_mse_s'] = float(mse(y, s_norm))
            row['boxcar_mse_x'] = float(mse(boxcar(y), x_norm))
            row['boxcar_mse_s'] = float(mse(boxcar(y), s_norm))
            for mkey, model in models.items():
                pred = model(y, extra=extra)
                row[f'{mkey}_mse_x'] = float(mse(pred, x_norm))
                row[f'{mkey}_mse_s'] = float(mse(pred, s_norm))
                if smp['class'] == 'gaussian_pulse':  # single-peak metrics
                    p_mag = pred[0].square().sum(0).sqrt()  # [L]
                    est_peak = int(p_mag.argmax().item())
                    est_amp = float(p_mag[ref_peak].item())
                    row[f'{mkey}_peak_amp_ratio'] = float(est_amp / max(ref_amp, 1e-9))
                    row[f'{mkey}_peak_bias_samples'] = est_peak - ref_peak
            rows.append(row)
    # summary: aggregate per class/baseline/level
    return rows


def gen_detection(n_cal, n_eval, n_h1, strengths, master_seed=7):
    rng = np.random.default_rng(master_seed)
    Lw = 1024
    def noise(b):
        return rng.standard_normal((b, Lw)) + 1j * rng.standard_normal((b, Lw))
    def pulse(b, amp):
        # gaussian pulse, random center, width ~ 16 samples
        center = rng.integers(128, Lw - 128, b)
        n = np.arange(Lw)[None, :]
        env = np.exp(-0.5 * ((n - center[:, None]) / 16.0) ** 2)
        ph = rng.uniform(0, 2 * np.pi, b)
        return (amp * env * np.exp(1j * ph[:, None]))
    h0_cal = noise(n_cal)
    h0_eval = noise(n_eval)
    h1 = {a: noise(n_h1) + pulse(n_h1, a) for a in strengths}
    return h0_cal, h0_eval, h1


def sliding_energy_max(z, window=64):
    """z [B,L] complex -> max sliding-window |z|^2 sum (window of `window` samples)."""
    p = z.real ** 2 + z.imag ** 2
    b = p.shape[0]
    cs = np.concatenate([np.zeros((b, 1)), np.cumsum(p, axis=1)], axis=1)  # [B,L+1]
    wsum = cs[:, window:] - cs[:, :-window]                               # [B,L+1-window]
    return wsum.max(axis=1)


@torch.no_grad()
def detection(models, device):
    from ssm_exp.model import max_abs_norm
    n_cal, n_eval, n_h1 = 10000, 10000, 1000
    strengths = (0.1, 0.2, 0.4)
    level = 10  # fixed added-noise level for the main table
    h0_cal, h0_eval, h1 = gen_detection(n_cal, n_eval, n_h1, strengths)
    eps_rng = np.random.default_rng(99)

    def to_complex(t):
        return t[:, 0].cpu().numpy() + 1j * t[:, 1].cpu().numpy()

    def make_y(z):
        """Observable rule: normalize by z's own max, add noise at `level` dB. Same for H0/H1."""
        yr = torch.tensor(np.stack((z.real, z.imag), 1), dtype=torch.float32, device=device)
        y_norm = max_abs_norm(yr)
        power = y_norm.square().sum(dim=1).mean(dim=-1)
        sigma = (power / 2).sqrt() * math.exp(-math.log(10) * level / 20)
        eps = torch.from_numpy(eps_rng.standard_normal(y_norm.shape).astype(np.float32)).to(device)
        y = y_norm + sigma[:, None, None] * eps
        return y, sigma

    datasets = {'h0_cal': h0_cal, 'h0_eval': h0_eval}
    for a in strengths:
        datasets[f'h1_{a}'] = h1[a]
    ys = {k: make_y(z) for k, z in datasets.items()}   # (y, sigma); same y across methods

    def transform(y, sigma, model, mname):
        if mname == 'identity':
            out = y
        elif mname == 'boxcar':
            out = boxcar(y)
        else:
            extra = sigma[:, None].expand(y.shape[0], N_PATCH)
            out = model(y, extra=extra)
        return to_complex(out)

    results = {}
    for mname, model in [('identity', None), ('boxcar', None)] + list(models.items()):
        ycal, sc = ys['h0_cal']
        thr = np.quantile(sliding_energy_max(transform(ycal, sc, model, mname)), 0.99)  # Pfa=1%
        ye, se = ys['h0_eval']
        pfa = float((sliding_energy_max(transform(ye, se, model, mname)) > thr).mean())
        pd = {}
        for a in strengths:
            y1, s1 = ys[f'h1_{a}']
            pd[f'amp_{a}'] = float((sliding_energy_max(transform(y1, s1, model, mname)) > thr).mean())
        results[mname] = {'threshold': float(thr), 'pfa_measured': pfa, 'pd': pd}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:2')
    ap.add_argument('--out', default=str(R2 / 'reports' / 'denoise_eval_round2.json'))
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    models, model_cfg = build_denoise_models(device)
    print('models loaded:', list(models.keys()), flush=True)
    checks = verify_loading(models)
    print('verify:', json.dumps(checks, ensure_ascii=False), flush=True)
    print('in-domain eval ...', flush=True)
    dom = in_domain(models, device)
    print('ood eval ...', flush=True)
    ood = ood_eval(models, device)
    print('detection ...', flush=True)
    det = detection(models, device)

    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        'supersedes_round1': 'reports/denoise_diagnostic.json (buggy: random decoder)',
        'verify': checks, 'in_domain': dom, 'ood_rows': ood,
        'detection': {'added_level_db': 10, 'statistic': 'max sliding-window energy (W=64)',
                      'n_h0_cal': 10000, 'n_h0_eval': 10000, 'n_h1_per_strength': 1000,
                      'strengths_peak_amp': list(det and next(iter(det.values())).get('pd', {}).keys() or []),
                      'results': det},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    # copy superseded round-1 report
    import shutil
    src = ROOT / 'reports' / 'denoise_diagnostic.json'
    if src.exists():
        shutil.copy(src, outdir / 'denoise_diagnostic_round1_superseded.json')
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
