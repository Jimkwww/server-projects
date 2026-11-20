"""SSM_L round-1 pretraining engine (MAE + denoising).

Per (backbone, seed) the encoder init and (for Q) the frozen aux-correlation statistics are
built once and cached under runs/shared_init, so the MAE and denoising runs of the same
backbone+seed start from byte-identical encoder weights and shared statistics. Data order,
noise stream and mask stream are all separate generators with per-seed fixed seeds, so the
four groups of a seed see the same (sample order, noise, mask) pairs.

The main result uses the epoch-50 weights (task book section 5); no best-checkpoint selection
is applied to pretraining. Checkpoints store the full model/optimizer/scheduler/RNG state so an
interrupted run resumes exactly.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ssm_exp.data import IQDataset, read_manifest, audit
from ssm_exp.engine import make_loader
from ssm_exp.model import max_abs_norm
from ssm_exp.utils import seed_all, save_json, save_checkpoint, environment, rng_state, restore_rng

from .iq_noise import corrupt_iq
from pretrain.model import PretrainModel, mask_patches
from .models import build_encoder_r2, BACKBONES


def load_config(path):
    c = json.loads(Path(path).read_text())
    if not {'model', 'train', 'protocol'} <= c.keys():
        raise ValueError('incomplete config')
    if c['train']['precision'] != 'fp32':
        raise ValueError('v1 supports fp32 only')
    if c['train']['scheduler'] not in ('cosine', 'constant'):
        raise ValueError('invalid scheduler')
    return c


def resume_key(c):
    """Config identity used to refuse resuming a checkpoint with mismatched hyperparameters.
    `epochs` (a run-length bound, also toggled by the smoke --epochs flag) is excluded so an
    interrupted run can be resumed/extended to a different total without a spurious mismatch."""
    ck = {k: v for k, v in c.items() if k != 'train'}
    ck['train'] = {k: v for k, v in c['train'].items() if k != 'epochs'}
    return ck


def sample_mask(b, n, ratio, generator, device):
    """Sample a per-sample binary patch mask [B,N] (True = masked) with exactly round(n*ratio)
    masked patches each, keeping at least one visible and one target patch."""
    k = int(round(n * ratio))
    k = max(1, min(n - 1, k))
    r = torch.rand(b, n, device=device, generator=generator)
    mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    mask.scatter_(1, r.topk(k, dim=1).indices, True)
    return mask


@torch.no_grad()
def fit_aux_stats(encoder, ds, cfg, device):
    """Fit the frozen 16-dim filtered-correlation standardization on a fixed 8192-sample train
    subset (raw IQ -> max_abs_norm -> aux descriptor), using the initial aux-branch parameters.
    Shared by MAE and denoising (identical init -> identical stats)."""
    size = min(len(ds), cfg['train']['aux_fit_samples'])
    positions = np.random.default_rng(0).choice(len(ds), size, replace=False).tolist()
    loader = make_loader(Subset(ds, positions), cfg['train']['batch_size'], 0)
    total = torch.zeros(16, dtype=torch.float64, device=device)
    square = total.clone()
    n = 0
    for x, _, _, _ in loader:
        x = x.to(device)
        f = encoder.aux_descriptor(max_abs_norm(x)).reshape(-1, 16).double()
        total += f.sum(0)
        square += f.square().sum(0)
        n += len(f)
    mean = total / n
    std = (square / n - mean.square()).clamp_min(1e-6).sqrt()
    encoder.aux_mean.copy_(mean.float())
    encoder.aux_std.copy_(std.float())
    return {'samples': size, 'tokens': n, 'mean': mean.tolist(), 'std': std.tolist(),
            'ids_sha256': hashlib.sha256(ds.ids[positions].tobytes()).hexdigest()}


def build_shared_init(cfg, backbone, seed, ds, device, cache_dir):
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f'{backbone}_seed{seed}.pt'
        if cache.exists():
            ck = torch.load(cache, map_location='cpu', weights_only=False)
            seed_all(seed)
            enc = build_encoder_r2(cfg['model'], backbone, seed).to(device)
            enc.load_state_dict(ck['encoder'])
            return enc, ck['aux']
    seed_all(seed)
    enc = build_encoder_r2(cfg['model'], backbone, seed).to(device)
    aux = fit_aux_stats(enc, ds, cfg, device) if backbone in ('Q', 'G_Qlearn') else None
    if cache_dir is not None:
        torch.save({'encoder': enc.state_dict(), 'aux': aux}, cache)
    return enc, aux


def denoise_loss(pred, x):
    """mean over (batch, time) positions of (Ihat-I)^2 + (Qhat-Q)^2."""
    return ((pred - x) ** 2).sum(dim=1).mean()


def mae_loss(pred, x, mask, patch):
    """Same per-time-point complex squared error, but only over masked (target) positions;
    denominator = actual number of target time positions."""
    err = ((pred - x) ** 2).sum(dim=1)                 # [B,T] = (Ihat-I)^2+(Qhat-Q)^2
    mt = mask.repeat_interleave(patch, dim=1)          # [B,T] bool
    return (err * mt).sum() / mt.sum().clamp_min(1)


def code_hash():
    h = hashlib.sha256()
    root = Path(__file__).parent.parent
    for d in (Path(__file__).parent, root / 'pretrain', root / 'ssm_exp'):
        for p in sorted(d.glob('*.py')):
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@torch.no_grad()
def val_recon_loss(model, loader, task, cfg, device, noise_gen, mask_gen):
    model.eval()
    total = 0.0
    count = 0
    for x, _, _, _ in loader:
        x = max_abs_norm(x.to(device))
        if task == 'denoise':
            y, sigma, _ = corrupt_iq(x, noise_gen)
            extra = sigma[:, None].expand(x.shape[0], x.shape[2] // cfg['model']['patch'])
            pred = model(y, extra=extra)
            total += denoise_loss(pred, x).item() * x.shape[0]
        else:
            mask = sample_mask(x.shape[0], x.shape[2] // cfg['model']['patch'],
                               cfg['train']['mask_ratio'], mask_gen, x.device)
            pred = model(x, mask=mask)
            total += mae_loss(pred, x, mask, cfg['model']['patch']).item() * x.shape[0]
        count += x.shape[0]
    return total / count


def train(args):
    c = load_config(args.config)
    if args.backbone not in BACKBONES:
        raise ValueError('backbone must be one of ' + repr(BACKBONES))
    if args.task not in ('mae', 'denoise'):
        raise ValueError('task must be mae or denoise')
    if args.seed is not None:
        c['train']['seed'] = args.seed
    if args.epochs is not None:
        if not args.smoke:
            raise ValueError('--epochs override only allowed with --smoke')
        c['train']['epochs'] = args.epochs
    if not args.smoke and not c['protocol'].get('verified'):
        raise ValueError('protocol.verified must be true before full training')
    out = Path(args.out)
    if (out / 'last.pt').exists() and not args.resume:
        raise FileExistsError('use --resume or a new run directory')
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('nonempty output directory')
    out.mkdir(parents=True, exist_ok=True)

    seed = c['train']['seed']
    device = torch.device(args.device)
    seed_all(seed)
    m = read_manifest(args.manifest)
    data_audit = audit(m)
    if m['classes'] != c['model']['classes']:
        raise ValueError('class count mismatch')
    if m['normalization'] != 'none':
        raise ValueError('pretraining requires a raw (normalization=none) manifest')
    ds = IQDataset(m, 'train')
    val = IQDataset(m, 'val')
    if ds[0][0].shape[-1] != c['model']['input_length']:
        raise ValueError('input length mismatch')
    if args.smoke:
        ds.ids = ds.ids[:min(256, len(ds))]
        val.ids = val.ids[:min(256, len(val))]

    # Separate RNG streams: data order (seed), noise (seed+1e5), mask (seed+2e5), val (seed+3e5).
    gen = torch.Generator().manual_seed(seed)
    noise_gen = torch.Generator(device=device).manual_seed(seed + 100000)
    mask_gen = torch.Generator(device=device).manual_seed(seed + 200000)
    val_noise_gen = torch.Generator(device=device).manual_seed(seed + 300000)
    val_mask_gen = torch.Generator(device=device).manual_seed(seed + 400000)

    dl = make_loader(ds, c['train']['batch_size'], args.workers, True, gen)
    vl = make_loader(val, c['train']['batch_size'], args.workers)

    enc, aux = build_shared_init(c, args.backbone, seed, ds, device,
                                 None if args.smoke else Path(args.runs_root) / 'shared_init')
    model = PretrainModel(enc, c['model']['patch']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['train']['lr'],
                            weight_decay=c['train']['weight_decay'],
                            betas=tuple(c['train']['betas']), eps=c['train']['adam_eps'])
    total_steps = c['train']['epochs'] * len(dl)
    warmup_steps = max(1, int(total_steps * c['train']['warmup_fraction']))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if c['train']['scheduler'] == 'constant':
            return 1.
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup_steps) / max(1, total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    provenance = {'data': data_audit, 'code_sha256': code_hash(),
                  'environment': environment(), 'smoke': args.smoke,
                  'task': args.task, 'backbone': args.backbone}

    start = 0
    if args.resume:
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        if resume_key(ck['config']) != resume_key(c) or ck['provenance'] != provenance:
            raise ValueError('resume config/data/code/environment differs; create a new run')
        if args.backbone in ('Q', 'G_Qlearn'):
            if 'aux' not in ck or ck['aux'] is None:
                raise ValueError('resume: checkpoint lacks aux stats; refusing to silently refit')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start = ck['epoch'] + 1
        restore_rng(ck['rng'], gen)
        noise_gen.set_state(ck['noise_rng'])
        mask_gen.set_state(ck['mask_rng'])
        val_noise_gen.set_state(ck['val_noise_rng'])
        val_mask_gen.set_state(ck['val_mask_rng'])
        aux = ck['aux']
        # In interrupted runs discard uncommitted history lines beyond the last checkpoint.
        history = out / 'history.jsonl'
        if history.exists():
            lines = [json.loads(x) for x in history.read_text().splitlines() if x.strip()]
            history.write_text(''.join(json.dumps(x) + '\n' for x in lines if x['epoch'] < start))
    else:
        save_json(out / 'config.json', c)
        save_json(out / 'provenance.json', provenance)
        save_json(out / 'model_info.json', {
            'backbone': args.backbone, 'task': args.task, 'aux': aux,
            'trainable_real_scalar_parameters': sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()),
        })

    patch = c['model']['patch']
    n_patches = c['model']['input_length'] // patch
    for epoch in range(start, c['train']['epochs']):
        model.train()
        total = 0.0
        n = 0
        t0 = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for batch, (x, _, _, _) in enumerate(dl):
            x = max_abs_norm(x.to(device))
            if args.task == 'denoise':
                y, sigma, _ = corrupt_iq(x, noise_gen)
                extra = sigma[:, None].expand(x.shape[0], n_patches)
                pred = model(y, extra=extra)
                loss = denoise_loss(pred, x)
            else:
                mask = sample_mask(x.shape[0], n_patches, c['train']['mask_ratio'], mask_gen, x.device)
                pred = model(x, mask=mask)
                loss = mae_loss(pred, x, mask, patch)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['train']['grad_clip'], error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            total += loss.item() * x.shape[0]
            n += x.shape[0]
        train_seconds = time.perf_counter() - t0
        vloss = val_recon_loss(model, vl, args.task, c, device, val_noise_gen, val_mask_gen)
        row = {'epoch': epoch, 'lr': opt.param_groups[0]['lr'], 'train_loss': total / n,
               'val_loss': vloss, 'train_seconds': train_seconds,
               'peak_training_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
               'step': scheduler.last_epoch + 1}
        ck = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
              'scheduler': scheduler.state_dict(), 'config': c, 'provenance': provenance,
              'epoch': epoch, 'rng': rng_state(gen), 'noise_rng': noise_gen.get_state(),
              'mask_rng': mask_gen.get_state(), 'val_noise_rng': val_noise_gen.get_state(),
              'val_mask_rng': val_mask_gen.get_state(), 'aux': aux}
        save_checkpoint(out / 'last.pt', ck)
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps({'epoch': epoch, 'train_loss': total / n, 'val_loss': vloss}), flush=True)

    # Final weights = epoch-50 (last epoch). Extract a standalone encoder checkpoint for downstream.
    ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    enc_sd = {k.replace('encoder.', ''): v for k, v in ck['model'].items() if k.startswith('encoder.')}
    torch.save({'encoder': enc_sd, 'config': c, 'provenance': provenance, 'aux': ck['aux']},
               out / 'encoder_final.pt')
    save_json(out / 'DONE.json', {'epochs': c['train']['epochs'], 'final_epoch': ck['epoch'],
                                  'task': args.task, 'backbone': args.backbone, 'smoke': args.smoke})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--backbone', choices=BACKBONES, required=True)
    p.add_argument('--task', choices=['mae', 'denoise'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--runs-root', required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--epochs', type=int)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
