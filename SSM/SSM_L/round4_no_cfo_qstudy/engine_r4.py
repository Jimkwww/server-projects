"""Round-4 engine helpers: G_corr physical-feature stat fitting + shared-init for pretrain.

Reuses round2_overnight.engine.fit_aux_stats for the Q-family aux descriptors (bit-identical
protocol). Adds fit_phys_stats for G_corr (same fixed rng-0 8192-sample subset, max_abs_norm
input, identical to the old project's fit_physics).
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from ssm_exp.data import IQDataset
from ssm_exp.engine import make_loader
from ssm_exp.model import max_abs_norm, physical_features
from ssm_exp.utils import seed_all
from round2_overnight.engine import fit_aux_stats
from .models_r4 import build_encoder_r4, freeze_aux


@torch.no_grad()
def fit_phys_stats(encoder, ds, cfg, device):
    """Fit the 3-dim physical-feature standardization (log-energy, r.real, r.imag) on the
    fixed rng-0 8192-sample train subset, exactly like the old project's fit_physics."""
    size = min(len(ds), cfg['train']['physical_fit_samples'])
    positions = np.random.default_rng(0).choice(len(ds), size, replace=False).tolist()
    loader = make_loader(Subset(ds, positions), cfg['train']['batch_size'], 0)
    total = torch.zeros(3, dtype=torch.float64, device=device)
    square = total.clone()
    n = 0
    for x, _, _, _ in loader:
        x = x.to(device)
        p = physical_features(max_abs_norm(x), cfg['model']['patch']).reshape(-1, 3).double()
        total += p.sum(0)
        square += p.square().sum(0)
        n += len(p)
    mean = total / n
    std = (square / n - mean.square()).clamp_min(1e-6).sqrt()
    encoder.phys_mean.copy_(mean.float())
    encoder.phys_std.copy_(std.float())
    return {'samples': size, 'tokens': n, 'mean': mean.tolist(), 'std': std.tolist(),
            'ids_sha256': hashlib.sha256(ds.ids[positions].tobytes()).hexdigest()}


def build_shared_init_r4(cfg, backbone, seed, ds, device, cache_dir, freeze_q=False):
    """Pretrain shared-init for round-4 backbones. For Q/G_Qlearn fits aux stats; for G_corr
    fits phys stats. Optional freeze_q freezes the Q filter before the (only) aux-stat fit."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = 'fixedq' if freeze_q else 'learnq'
        cache = cache_dir / f'{backbone}_seed{seed}_{tag}.pt'
        if cache.exists():
            ck = torch.load(cache, map_location='cpu', weights_only=False)
            seed_all(seed)
            enc = build_encoder_r4(cfg['model'], backbone, seed).to(device)
            if freeze_q:
                freeze_aux(enc)
            enc.load_state_dict(ck['encoder'])
            return enc, ck.get('aux'), ck.get('phys')
    seed_all(seed)
    enc = build_encoder_r4(cfg['model'], backbone, seed).to(device)
    if freeze_q:
        freeze_aux(enc)
    aux = None
    phys = None
    if backbone in ('Q', 'G_Qlearn'):
        aux = fit_aux_stats(enc, ds, cfg, device)
    elif backbone == 'G_corr':
        phys = fit_phys_stats(enc, ds, cfg, device)
    if cache_dir is not None:
        torch.save({'encoder': enc.state_dict(), 'aux': aux, 'phys': phys}, cache)
    return enc, aux, phys
