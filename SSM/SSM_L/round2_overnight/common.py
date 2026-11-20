"""Shared round-2 helpers: encoder loading, random-init checkpoints, feature caching, metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ssm_exp.data import IQDataset, read_manifest, audit
from ssm_exp.engine import make_loader
from ssm_exp.model import max_abs_norm
from ssm_exp.utils import seed_all, grouped_metrics

from .models import build_encoder_r2, BACKBONES

ROOT = Path(__file__).resolve().parent.parent
R2 = Path(__file__).resolve().parent


def cfg_model(path):
    return json.loads((ROOT / 'configs' / path).read_text())['model']


def load_encoder(model_cfg, backbone, seed, ckpt_path, device):
    pt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    seed_all(seed)
    enc = build_encoder_r2(model_cfg, backbone, seed).to(device)
    enc.load_state_dict(pt['encoder'])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def make_random_checkpoint(model_cfg, backbone, seed, out_path, device='cpu',
                           shared_init_dir=None):
    """Save a random-init encoder in encoder_final.pt format (with fit aux stats for Q-family)."""
    seed_all(seed)
    enc = build_encoder_r2(model_cfg, backbone, seed).to(device)
    aux = None
    if backbone in ('Q', 'G_Qlearn'):
        # Reuse round-1 shared-init aux stats when the backbone matches, else fit fresh.
        if shared_init_dir is not None:
            src = Path(shared_init_dir) / f'{backbone}_seed{seed}.pt'
            if src.exists():
                aux = torch.load(src, map_location='cpu', weights_only=False)['aux']
            elif backbone == 'G_Qlearn':
                # G_Qlearn aux stats == same-seed Q aux stats (identical aux_branch init).
                qsrc = Path(shared_init_dir) / f'Q_seed{seed}.pt'
                if qsrc.exists():
                    aux = torch.load(qsrc, map_location='cpu', weights_only=False)['aux']
        if aux is None:
            from .engine import fit_aux_stats
            m = read_manifest(str(ROOT / 'configs' / 'data.json'))
            ds = IQDataset(m, 'train')
            aux = fit_aux_stats(enc, ds, {'train': {'batch_size': 128,
                                                     'aux_fit_samples': 8192}}, device)
        enc.aux_mean.copy_(torch.tensor(aux['mean'], dtype=torch.float32))
        enc.aux_std.copy_(torch.tensor(aux['std'], dtype=torch.float32))
    torch.save({'encoder': enc.state_dict(), 'config': {'model': model_cfg},
                'provenance': {'random_init': True, 'backbone': backbone, 'seed': seed},
                'aux': aux}, out_path)
    return enc


def features_for_encoder(enc, ds, cfg_batch, device):
    enc.eval()
    feats, ys, zs, ids = [], [], [], []
    loader = make_loader(ds, cfg_batch, 0)
    for x, y, z, i in loader:
        x = max_abs_norm(x.to(device))
        feats.append(enc.features(x).cpu())
        ys.append(y)
        zs.append(z)
        ids.append(i)
    return (torch.cat(feats), torch.cat(ys), torch.cat(zs), torch.cat(ids))


def sd_hash(state_dict):
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def metrics_from_preds(y, pred, snr, classes):
    return grouped_metrics(y.tolist(), pred.tolist(), snr.tolist(), classes)
