"""SSM_L round-1 denoising diagnostic (task book section 7).

Builds a small fixed batch of controlled waveforms with a clean reference s, adds a known
baseline noise to form the original observation x, then adds artificial noise (the pretraining
perturbation) to form y. Reports squared reconstruction error against x and against s for:
input y, identity mapping, a simple boxcar low-pass filter (bandwidth stated below), and the
trained denoising model. Also records pulse-peak attenuation and position shift.

Diagnostic samples are synthetic and independent of the RML2018A training set.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from pretrain.model import build_encoder, PretrainModel
from pretrain.iq_noise import corrupt_iq
from ssm_exp.model import max_abs_norm

ROOT = Path(__file__).resolve().parent.parent
L = 1024
PATCH = 8


def make_waveforms():
    """8 synthetic complex waveforms (as [B,2,L] real I/Q) with a clean reference s."""
    t = torch.arange(L, dtype=torch.float64) / L
    xs = []
    for k in range(8):
        if k % 3 == 0:
            # Gaussian pulse at a fixed position (k-dependent)
            mu = 0.2 + 0.05 * k
            sig = 0.008
            env = torch.exp(-0.5 * ((t - mu) / sig) ** 2)
            s = env * torch.exp(2j * math.pi * 0.25 * t)
        elif k % 3 == 1:
            # QPSK burst in [0.15, 0.35]
            syms = torch.tensor([1, -1, 1j, -1j, 1, 1j, -1, -1j], dtype=torch.cfloat)
            idx = ((t * 64).long().clamp(0, 7))
            s = syms[idx] * ((t > 0.15) & (t < 0.35)).double()
        else:
            # linear chirp burst
            phase = 2 * math.pi * (0.05 * t + 0.3 * t ** 2)
            s = torch.exp(1j * phase) * ((t > 0.1) & (t < 0.5)).double()
        xs.append(s)
    s = torch.stack(xs)  # [B,L] complex
    sr = torch.stack((s.real, s.imag), 1).float()  # [B,2,L]
    return sr, s


def boxcar(x, window):
    """Simple boxcar low-pass filter along time (bandwidth ~= 1/(2*window) normalized)."""
    w = torch.ones(x.shape[1], 1, window, device=x.device, dtype=x.dtype) / window
    pad = window // 2
    xp = torch.nn.functional.pad(x, (pad, pad), mode='replicate')
    return torch.nn.functional.conv1d(xp, w, groups=x.shape[1])


def mse(pred, ref):
    return ((pred - ref) ** 2).sum(dim=1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--level-db', type=float, default=10.0)
    ap.add_argument('--snr-db', type=float, default=20.0, help='baseline noise SNR (s -> x)')
    ap.add_argument('--window', type=int, default=5)
    args = ap.parse_args()
    device = torch.device(args.device)

    sr, s = make_waveforms()
    s = s.to(device)
    sr = sr.to(device)

    # baseline noise: x = s + noise at args.snr_db relative to s's power (per real channel)
    torch.manual_seed(0)
    s_power = sr.square().sum(1).mean(-1)
    base_sigma = (s_power / 2).sqrt() * math.exp(-math.log(10) * args.snr_db / 20)
    x = sr + base_sigma[:, None, None] * torch.randn_like(sr)

    # Normalize the original observation x ONCE (joint max_abs), matching pretraining. The
    # artificial noise is then added to the normalized x; y is NOT re-normalized (task book 4).
    xn = max_abs_norm(x)
    sn = max_abs_norm(sr)  # clean reference on the same scale, for the "vs s" comparison

    power = xn.square().sum(1).mean(-1)
    sigma = (power / 2).sqrt() * math.exp(-math.log(10) * args.level_db / 20)
    gen = torch.Generator(device=device).manual_seed(1)
    eps = torch.randn(x.shape, device=device, generator=gen)
    y = xn + sigma[:, None, None] * eps

    outputs = {'input_y': y, 'boxcar': boxcar(y, args.window)}
    results = {}

    for backbone in ('G', 'Q'):
        cfg = json.loads((ROOT / 'configs' / 'pretrain.json').read_text())['model']
        enc = build_encoder(cfg, backbone, 17).to(device)
        ck = torch.load(ROOT / 'runs' / 'pretrain' / f'{backbone}_denoise_seed17' / 'encoder_final.pt',
                        map_location='cpu', weights_only=False)
        enc.load_state_dict(ck['encoder'])
        model = PretrainModel(enc, PATCH).to(device).eval()
        with torch.no_grad():
            n_patches = L // PATCH
            extra = sigma[:, None].expand(x.shape[0], n_patches)
            pred = model(y, extra=extra)
        outputs[f'model_{backbone}'] = pred

    # MSE against the original observation x and against the clean reference s
    for name, out in outputs.items():
        results[name] = {'mse_x': mse(out, xn), 'mse_s': mse(out, sn)}
    results['identity'] = {'mse_x': mse(y, xn), 'mse_s': mse(y, sn)}  # output = input y

    # pulse peak attenuation + position shift on the pulse-like samples (0, 3, 6)
    peak = {}
    pulse_idx = [0, 3, 6]
    for name, out in outputs.items():
        att, shift = [], []
        out_c = torch.complex(out[:, 0], out[:, 1])
        for i in pulse_idx:
            ref_peak = s[i].abs().argmax().item()
            est_peak = out_c[i].abs().argmax().item()
            att.append((out_c[i].abs()[ref_peak] / max(s[i].abs()[ref_peak].item(), 1e-9)).item())
            shift.append(est_peak - ref_peak)
        peak[name] = {'peak_attenuation_ratio': float(sum(att) / len(att)),
                      'mean_position_shift_samples': float(sum(shift) / len(shift))}

    outdir = ROOT / 'reports'
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {'level_db': args.level_db, 'baseline_snr_db': args.snr_db,
               'window': args.window, 'bandwidth_note': f'boxcar window={args.window}, '
               f'cutoff ~1/(2*window) normalized cycles/sample',
               'mse': results, 'peak': peak}
    (outdir / 'denoise_diagnostic.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    torch.save({'s': s.cpu(), 'x': x.cpu(), 'y': y.cpu(),
                'outputs': {k: v.cpu() for k, v in outputs.items()}},
               outdir / 'denoise_diagnostic.pt')
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
