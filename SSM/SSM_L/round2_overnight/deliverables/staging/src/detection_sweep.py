"""Supplementary detection amplitude sweep (extends run_b's Pfa=1% detection table).

run_b's main table uses weak pulse peak amplitudes 0.1/0.2/0.4 (below the receiver-noise max),
where no method beats the Pfa floor. This sweep adds stronger amplitudes to locate where the
max-sliding-window-energy detector (applied to identity / boxcar / G/Q-denoise outputs) starts
to separate H1 from H0, and records the same observable normalization+added-noise rule.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
R2 = Path(__file__).resolve().parent.parent
L = 1024
N_PATCH = L // 8


@torch.no_grad()
def main():
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(R2 / 'scripts'))
    from run_b import build_denoise_models, gen_detection, sliding_energy_max, boxcar
    from ssm_exp.model import max_abs_norm

    device = torch.device('cuda:0')
    models, _ = build_denoise_models(device)
    strengths = (0.4, 0.8, 1.6, 3.2)
    level = 10
    n_cal, n_eval, n_h1 = 10000, 10000, 1000
    h0_cal, h0_eval, h1 = gen_detection(n_cal, n_eval, n_h1, strengths)
    eps_rng = np.random.default_rng(99)

    def to_complex(t):
        return t[:, 0].cpu().numpy() + 1j * t[:, 1].cpu().numpy()

    def make_y(z):
        yr = torch.tensor(np.stack((z.real, z.imag), 1), dtype=torch.float32, device=device)
        y_norm = max_abs_norm(yr)
        power = y_norm.square().sum(dim=1).mean(dim=-1)
        sigma = (power / 2).sqrt() * math.exp(-math.log(10) * level / 20)
        eps = torch.from_numpy(eps_rng.standard_normal(y_norm.shape).astype(np.float32)).to(device)
        return y_norm + sigma[:, None, None] * eps, sigma

    datasets = {'h0_cal': h0_cal, 'h0_eval': h0_eval}
    for a in strengths:
        datasets[f'h1_{a}'] = h1[a]
    ys = {k: make_y(z) for k, z in datasets.items()}

    def transform(y, sigma, model, mname, chunk=2000):
        outs = []
        for i in range(0, y.shape[0], chunk):
            yc, sc = y[i:i + chunk], sigma[i:i + chunk]
            if mname == 'identity':
                out = yc
            elif mname == 'boxcar':
                out = boxcar(yc)
            else:
                extra = sc[:, None].expand(yc.shape[0], N_PATCH)
                out = model(yc, extra=extra)
            outs.append(to_complex(out))
        return np.concatenate(outs, axis=0)

    out = {}
    for mname, model in [('identity', None), ('boxcar', None)] + list(models.items()):
        ycal, sc = ys['h0_cal']
        thr = np.quantile(sliding_energy_max(transform(ycal, sc, model, mname)), 0.99)
        ye, se = ys['h0_eval']
        pfa = float((sliding_energy_max(transform(ye, se, model, mname)) > thr).mean())
        pd = {}
        for a in strengths:
            y1, s1 = ys[f'h1_{a}']
            pd[str(a)] = float((sliding_energy_max(transform(y1, s1, model, mname)) > thr).mean())
        out[mname] = {'threshold': float(thr), 'pfa': pfa, 'pd': pd}

    p = R2 / 'reports' / 'detection_sweep_round2.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({'added_level_db': level, 'strengths_peak_amp': list(strengths),
                             'n_h0_cal': n_cal, 'n_h0_eval': n_eval, 'n_h1': n_h1,
                             'results': out}, indent=2))
    for k, v in out.items():
        print(k, 'pfa', round(v['pfa'], 4), 'pd', {a: round(p, 3) for a, p in v['pd'].items()},
              flush=True)
    print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
