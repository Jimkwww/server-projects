"""Section 6: step-budgeted MAE pretraining on F1/F2 corpora.

Same protocol as round-2 engine (RNG streams, optimizer, cosine warmup, grad clip, MAE loss,
per-corpus shared-init so Q-family aux stats are fit on the right corpus), but with a fixed
STEP budget instead of an epoch budget:

    F0 = 7800 steps = 50 epochs x ceil(19968/128=156)
    F1/F2 = 7800 steps = 998400 window exposures (batch 128), i.e. ~10 passes over 100k.

Final weights = last step (NOT val-selected). Writes encoder_final.pt at the end.

Usage:
  python pretrain_scale.py --config configs/pretrain.json --manifest configs/manifest_f1.json \
      --backbone G --task mae --out round3/runs/pretrain_scale/f1/G_seed17 \
      --runs-root round3/runs/pretrain_scale/f1 --max-steps 7800 --seed 17 --device cuda:0
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ssm_exp.data import IQDataset, read_manifest, audit
from ssm_exp.engine import make_loader
from ssm_exp.model import max_abs_norm
from ssm_exp.utils import seed_all, save_json, save_checkpoint, environment, rng_state, restore_rng
from pretrain.model import PretrainModel
from round2_overnight.engine import (load_config, resume_key, sample_mask, build_shared_init,
                                     mae_loss, val_recon_loss)
from round2_overnight.models import BACKBONES


def train_scale(args):
    c = load_config(args.config)
    if args.backbone not in BACKBONES:
        raise ValueError('backbone must be one of ' + repr(BACKBONES))
    if args.task != 'mae':
        raise ValueError('scale training is MAE-only')
    if args.seed is not None:
        c['train']['seed'] = args.seed
    if not args.smoke and not c['protocol'].get('verified'):
        raise ValueError('protocol.verified must be true before full training')
    out = Path(args.out)
    if (out / 'encoder_final.pt').exists() and not args.resume:
        print(f'[skip] {out.name} already final', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)

    seed = c['train']['seed']
    device = torch.device(args.device)
    seed_all(seed)
    m = read_manifest(args.manifest)
    if m['classes'] != c['model']['classes']:
        raise ValueError('class count mismatch')
    if m['normalization'] != 'none':
        raise ValueError('pretraining requires a raw (normalization=none) manifest')
    ds = IQDataset(m, 'train')
    val = IQDataset(m, 'val')
    if ds[0][0].shape[-1] != c['model']['input_length']:
        raise ValueError('input length mismatch')

    max_steps = args.max_steps
    steps_per_pass = math.ceil(len(ds) / c['train']['batch_size'])

    # RNG streams (same as round-2): data order, noise(1e5), mask(2e5), val(3e5), val mask(4e5).
    gen = torch.Generator().manual_seed(seed)
    mask_gen = torch.Generator(device=device).manual_seed(seed + 200000)
    val_noise_gen = torch.Generator(device=device).manual_seed(seed + 300000)
    val_mask_gen = torch.Generator(device=device).manual_seed(seed + 400000)

    dl = make_loader(ds, c['train']['batch_size'], args.workers, True, gen)
    vl = make_loader(val, c['train']['batch_size'], args.workers)

    runs_root = Path(args.runs_root)
    enc, aux = build_shared_init(c, args.backbone, seed, ds, device,
                                 None if args.smoke else runs_root / 'shared_init')
    model = PretrainModel(enc, c['model']['patch']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['train']['lr'],
                            weight_decay=c['train']['weight_decay'],
                            betas=tuple(c['train']['betas']), eps=c['train']['adam_eps'])
    total_steps = max_steps
    warmup_steps = max(1, int(total_steps * c['train']['warmup_fraction']))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup_steps) / max(1, total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    provenance = {'data': audit(m), 'environment': environment(), 'smoke': args.smoke,
                  'task': args.task, 'backbone': args.backbone, 'corpus': args.corpus,
                  'max_steps': max_steps, 'steps_per_pass': steps_per_pass,
                  'manifest': str(Path(args.manifest).resolve())}

    start_step = 0
    start_pass = 0
    if args.resume and (out / 'last.pt').exists():
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_step = ck['step']
        start_pass = ck['pass']
        restore_rng(ck['rng'], gen)
        mask_gen.set_state(ck['mask_rng'])
        val_noise_gen.set_state(ck['val_noise_rng'])
        val_mask_gen.set_state(ck['val_mask_rng'])
        aux = ck['aux']
        print(f'[resume] step={start_step} pass={start_pass}', flush=True)
    else:
        save_json(out / 'config.json', c)
        save_json(out / 'provenance.json', provenance)
        save_json(out / 'model_info.json', {
            'backbone': args.backbone, 'task': args.task, 'aux': aux,
            'trainable_real_scalar_parameters':
                sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())})

    patch = c['model']['patch']
    n_patches = c['model']['input_length'] // patch

    step = start_step
    pass_ = start_pass
    t0 = time.perf_counter()
    while step < max_steps:
        model.train()
        total = 0.0
        n = 0
        for x, _, _, _ in dl:
            x = max_abs_norm(x.to(device))
            mask = sample_mask(x.shape[0], n_patches, c['train']['mask_ratio'], mask_gen, x.device)
            pred = model(x, mask=mask)
            loss = mae_loss(pred, x, mask, patch)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['train']['grad_clip'],
                                           error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            total += loss.item() * x.shape[0]
            n += x.shape[0]
            step += 1
            if step >= max_steps:
                break
        pass_ += 1
        vloss = val_recon_loss(model, vl, args.task, c, device, val_noise_gen, val_mask_gen)
        row = {'pass': pass_, 'lr': opt.param_groups[0]['lr'], 'train_loss': total / n,
               'val_loss': vloss, 'train_seconds': time.perf_counter() - t0, 'step': step}
        ck = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
              'scheduler': scheduler.state_dict(), 'config': c, 'provenance': provenance,
              'step': step, 'pass': pass_, 'rng': rng_state(gen), 'mask_rng': mask_gen.get_state(),
              'val_noise_rng': val_noise_gen.get_state(), 'val_mask_rng': val_mask_gen.get_state(),
              'aux': aux}
        save_checkpoint(out / 'last.pt', ck)
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps({'pass': pass_, 'step': step, 'train_loss': total / n,
                          'val_loss': vloss, 'elapsed': time.perf_counter() - t0}), flush=True)

    # Final weights = last step. Extract encoder for downstream.
    ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    enc_sd = {k.replace('encoder.', ''): v for k, v in ck['model'].items() if k.startswith('encoder.')}
    torch.save({'encoder': enc_sd, 'config': c, 'provenance': provenance, 'aux': ck['aux']},
               out / 'encoder_final.pt')
    save_json(out / 'DONE.json', {'max_steps': max_steps, 'final_step': ck['step'],
                                  'task': args.task, 'backbone': args.backbone, 'smoke': args.smoke})
    print(f'[done] {out.name} step={ck["step"]}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--backbone', choices=BACKBONES, required=True)
    p.add_argument('--task', choices=['mae'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--runs-root', required=True)
    p.add_argument('--corpus', required=True)
    p.add_argument('--max-steps', type=int, required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    train_scale(args)


if __name__ == '__main__':
    main()
