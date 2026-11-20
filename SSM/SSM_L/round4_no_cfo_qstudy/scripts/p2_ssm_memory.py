"""P2: SSM effective-memory diagnosis on S1plain (no Q branch, so no recurrence recursion).

For each S1plain downstream model (random/MAE/denoise init x seeds, as produced by P0):

A. State statistics: per SSM layer, record the diagonal decay magnitude |a|=rho, the write
   gate g, and the state magnitude |h| on the val split; derive the effective timescale
   tau = -1/log(rho) (tokens to decay to 1/e).
B. Segment reset: re-run the forward pass with the recurrence state forced to zero every K
   tokens (K in {1,4,16,64}; a correct state cutoff, not a detach) and report val OA vs K.
   A model that depends on long-range state will lose accuracy at small K; a model that only
   needs short context will not.

Run after the S1plain downstream runs exist:
  python round4_no_cfo_qstudy/scripts/p2_ssm_memory.py --task rml --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ssm_exp.data import read_manifest, audit, IQDataset
from ssm_exp.engine import make_loader
from ssm_exp.model import max_abs_norm, parallel_scan, serial_scan, ComplexSSM
from ssm_exp.utils import seed_all, save_json, grouped_metrics
from pretrain.model import DownstreamModel
from round4_no_cfo_qstudy.models_r4 import build_encoder_r4

ROOT = Path(__file__).resolve().parent.parent.parent
R4 = ROOT / 'round4_no_cfo_qstudy'


def segment_scan(a, b, K):
    """Inclusive affine recurrence with the state zeroed at every K-th token (n % K == 0)."""
    B, N, S = a.shape
    h = torch.zeros_like(b[:, 0])
    out = []
    for n in range(N):
        if n % K == 0:
            h = torch.zeros_like(h)
        h = a[:, n] * h + b[:, n]
        out.append(h)
    return torch.stack(out, 1)


def _ssm_terms(ssm, u, p, capture=False):
    v = F.linear(u.to(ssm.B.dtype), ssm.B)
    base_delta = ssm.raw_delta.view(1, 1, -1)
    aux = torch.zeros_like(p)  # S1 uses zero physical slots
    d_raw, g_raw = ssm.selector(torch.cat((u, aux), -1)).chunk(2, -1)
    delta = F.softplus(base_delta + d_raw)
    g = torch.sigmoid(g_raw)
    rho = torch.exp(-F.softplus(ssm.raw_alpha) * delta)
    a = torch.polar(rho.expand_as(v.real), ssm.theta.expand_as(v.real))
    if capture:
        return v, a, g, rho
    return v, a, g, rho


def forward_s1(enc, x, K=None, capture=None):
    """Replicate Encoder2.forward_tokens for S1plain. K=None -> normal parallel scan;
    K>=1 -> segment reset every K tokens. capture (list) collects per-layer rho/g/h."""
    u = enc.front(x)
    p = u.new_zeros(u.shape[0], u.shape[1], 3)
    for block in enc.blocks:
        ssm = block.mix
        xn = block.norm1(u)
        v, a, g, rho = _ssm_terms(ssm, xn, p)
        h = parallel_scan(a, g * v) if K is None else segment_scan(a, g * v, K)
        if capture is not None:
            capture.append(dict(rho=rho.detach(), g=g.detach(), h=h.detach()))
        y = ssm.gain * F.linear(h, ssm.C).real + ssm.direct * xn
        u = u + block.drop(y)
        u = u + block.drop(block.ffn(block.norm2(u)))
    return enc.norm(u)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck['config']['model']
    seed = ck['config']['train']['seed']
    classes = cfg['classes']
    enc = build_encoder_r4(cfg, 'S1plain', seed).to(device)
    model = DownstreamModel(enc, classes).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, cfg, classes


def state_stats(model, loader, device, max_batches=64):
    model.eval()
    per_layer = []
    n_batch = 0
    for x, y, z, i in loader:
        if n_batch >= max_batches:
            break
        x = max_abs_norm(x.to(device))
        with torch.no_grad():
            cap = []
            forward_s1(model.encoder, x, K=None, capture=cap)
        for li, c in enumerate(cap):
            if len(per_layer) <= li:
                per_layer.append(dict(rho=[], g=[], h_mag=[]))
            rho = c['rho'].reshape(-1).cpu().numpy()
            g = c['g'].reshape(-1).cpu().numpy()
            h_mag = c['h'].abs().reshape(-1).cpu().numpy()
            per_layer[li]['rho'].append(rho)
            per_layer[li]['g'].append(g)
            per_layer[li]['h_mag'].append(h_mag)
        n_batch += 1
    out = []
    for li, acc in enumerate(per_layer):
        rho = np.concatenate(acc['rho'])
        g = np.concatenate(acc['g'])
        h_mag = np.concatenate(acc['h_mag'])
        rho_pos = rho[rho > 0]
        tau = -1.0 / np.log(np.clip(rho_pos, 1e-6, 1 - 1e-6))
        out.append(dict(
            layer=li,
            rho_mean=float(rho.mean()), rho_p50=float(np.percentile(rho, 50)),
            rho_p90=float(np.percentile(rho, 90)), rho_frac_gt_0_999=float((rho > 0.999).mean()),
            tau_mean_tokens=float(tau.mean()), tau_p50=float(np.percentile(tau, 50)),
            tau_p90=float(np.percentile(tau, 90)),
            gate_mean=float(g.mean()), gate_frac_sat=float(((g < 0.01) | (g > 0.99)).mean()),
            state_rms=float(np.sqrt((h_mag ** 2).mean())), state_max=float(h_mag.max())))
    return out


def eval_segment(model, loader, device, K):
    model.eval()
    ys, ps = [], []
    for x, y, z, i in loader:
        x = max_abs_norm(x.to(device))
        with torch.no_grad():
            toks = forward_s1(model.encoder, x, K=K)
            logits = model.head(toks.mean(1))
        ys.extend(y.tolist())
        ps.extend(logits.argmax(-1).cpu().tolist())
    ys = np.asarray(ys); ps = np.asarray(ps)
    return grouped_metrics(ys, ps, np.zeros_like(ys), model.encoder.cfg['classes'])['overall']['oa']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['rml', 'radar'], default='rml')
    p.add_argument('--init', default=None, help='comma list subset of scratch,mae,denoise')
    p.add_argument('--seeds', default='17,42,2026')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    device = torch.device(a.device)
    inits = a.init.split(',') if a.init else ['scratch', 'mae', 'denoise']
    seeds = [int(s) for s in a.seeds.split(',')]

    if a.task == 'rml':
        manifest = ROOT / 'configs/data.json'
    else:
        manifest = ROOT / 'round2_overnight/configs/radchar.json'
    m = read_manifest(str(manifest))
    val = IQDataset(m, 'val')
    cfg_for_load = json.loads((R4 / 'configs' / f'downstream_nocfo_{a.task}.json').read_text())
    vl = make_loader(val, cfg_for_load['train']['batch_size'], a.workers)

    results = {'task': a.task, 'inits': inits, 'seeds': seeds}
    for init in inits:
        for seed in seeds:
            # RML scratch S1plain is reused from the old SSM project (not in round4 runs).
            ckpt = R4 / 'runs' / a.task / init / f'S1plain_seed{seed}' / 'best.pt'
            if not ckpt.exists():
                results[f'{init}_{seed}'] = {'missing': str(ckpt)}
                print(f'[skip] {ckpt} missing', flush=True)
                continue
            model, cfg, classes = load_model(ckpt, device)
            ss = state_stats(model, vl, device)
            seg = {f'K_{K}': eval_segment(model, vl, device, K) for K in (1, 4, 16, 64)}
            seg['K_none'] = eval_segment(model, vl, device, None)
            results[f'{init}_{seed}'] = {'state_stats': ss, 'segment_oa': seg}
            print(json.dumps({f'{init}_{seed}': seg}), flush=True)

    out = Path(a.out) if a.out else R4 / 'reports' / f'p2_ssm_memory_{a.task}.json'
    save_json(out, results)
    print(f'[done] {out}', flush=True)


if __name__ == '__main__':
    main()
