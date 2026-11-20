"""Round5 (fixed-epoch data-scaling): epoch-budgeted MAE pretraining on F1/F2 corpora.

Same protocol as round-2/round-3 engine (RNG streams, AdamW, cosine+5% warmup, grad clip,
MAE loss, per-corpus shared-init so Q-family aux stats are fit on the right corpus), but with a
fixed EPOCH budget (50 full epochs) instead of a fixed STEP budget:

    F1/F2 = 50 epochs = 50 x ceil(100000/128=782) = 39,100 optimizer steps
    F0    = 50 epochs = 50 x ceil(19968/128=156) = 7,800 optimizer steps (historical, reused)

Intermediate checkpoints are saved at the 7,800th optimizer step and at the end of epochs
20/30/40/50. The 50-epoch checkpoint is the main evaluation object. Train from random init
(the round-3 shared_init cache), NOT resumed from any trained weights.

Usage:
  python pretrain_epoch.py --config configs/pretrain.json --manifest <f1/f2 manifest> \
      --backbone G --out round5_epoch_scale/runs/pretrain/f1/G_seed17 \
      --shared-init-dir round5_epoch_scale/runs/pretrain/f1/shared_init \
      --corpus f1 --epochs 50 --seed 17 --device cuda:0
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ssm_exp.data import IQDataset, read_manifest, audit
from ssm_exp.engine import make_loader
from ssm_exp.model import max_abs_norm
from ssm_exp.utils import seed_all, save_json, save_checkpoint, environment, rng_state, restore_rng
from pretrain.model import PretrainModel
from round2_overnight.engine import (load_config, sample_mask, build_shared_init,
                                     mae_loss, val_recon_loss)
from round2_overnight.models import BACKBONES


def sd_hash(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(d[k].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _extract_encoder_sd(model):
    return {k.replace('encoder.', ''): v
            for k, v in model.state_dict().items() if k.startswith('encoder.')}


def train_epoch(args):
    c = load_config(args.config)
    if args.backbone not in BACKBONES:
        raise ValueError('backbone must be one of ' + repr(BACKBONES))
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

    epochs = args.epochs
    steps_per_pass = math.ceil(len(ds) / c['train']['batch_size'])
    total_steps = epochs * steps_per_pass

    # RNG streams (same as round-2/3): data order(seed), noise(1e5), mask(2e5), val(3e5), val mask(4e5).
    gen = torch.Generator().manual_seed(seed)
    mask_gen = torch.Generator(device=device).manual_seed(seed + 200000)
    val_noise_gen = torch.Generator(device=device).manual_seed(seed + 300000)
    val_mask_gen = torch.Generator(device=device).manual_seed(seed + 400000)

    dl = make_loader(ds, c['train']['batch_size'], args.workers, True, gen)
    vl = make_loader(val, c['train']['batch_size'], args.workers)

    shared_init_dir = None if args.smoke else Path(args.shared_init_dir)
    enc, aux = build_shared_init(c, args.backbone, seed, ds, device, shared_init_dir)
    init_hash = sd_hash(enc.state_dict())
    model = PretrainModel(enc, c['model']['patch']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['train']['lr'],
                            weight_decay=c['train']['weight_decay'],
                            betas=tuple(c['train']['betas']), eps=c['train']['adam_eps'])
    warmup_steps = max(1, int(total_steps * c['train']['warmup_fraction']))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup_steps) / max(1, total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    provenance = {'data': audit(m), 'environment': environment(), 'smoke': args.smoke,
                  'task': 'mae', 'backbone': args.backbone, 'corpus': args.corpus,
                  'epochs': epochs, 'total_steps': total_steps, 'steps_per_pass': steps_per_pass,
                  'shared_init_dir': str(Path(args.shared_init_dir).resolve()),
                  'shared_init_encoder_sha256': init_hash,
                  'manifest': str(Path(args.manifest).resolve()),
                  'budget': 'fixed-epoch (50 full epochs)'}

    # intermediate checkpoint plan
    ckpt_steps = {7800: 'ckpt_step7800.pt'}
    ckpt_epochs = {20: 'ckpt_epoch20.pt', 30: 'ckpt_epoch30.pt',
                   40: 'ckpt_epoch40.pt', 50: 'ckpt_epoch50.pt'}

    start_step = 0
    start_pass = 0
    n_total = 0
    if args.resume and (out / 'last.pt').exists():
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_step = ck['step']
        start_pass = ck['pass']
        n_total = ck.get('cum_samples', start_step * c['train']['batch_size'])
        restore_rng(ck['rng'], gen)
        mask_gen.set_state(ck['mask_rng'])
        val_noise_gen.set_state(ck['val_noise_rng'])
        val_mask_gen.set_state(ck['val_mask_rng'])
        aux = ck['aux']
        print(f'[resume] step={start_step} pass={start_pass} cum_samples={n_total}', flush=True)
    else:
        save_json(out / 'config.json', c)
        save_json(out / 'provenance.json', provenance)
        save_json(out / 'model_info.json', {
            'backbone': args.backbone, 'task': 'mae', 'aux': aux,
            'shared_init_encoder_sha256': init_hash,
            'trainable_real_scalar_parameters':
                sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())})

    patch = c['model']['patch']
    n_patches = c['model']['input_length'] // patch

    def save_encoder_ckpt(name, step, pass_):
        torch.save({'encoder': _extract_encoder_sd(model), 'config': c, 'provenance': provenance,
                    'aux': aux, 'step': step, 'pass': pass_}, out / name)

    step = start_step
    pass_ = start_pass
    t0 = time.perf_counter()
    peak_mem = 0
    while pass_ < epochs:
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
            n_total += x.shape[0]
            step += 1
            if step in ckpt_steps and not (out / ckpt_steps[step]).exists():
                save_encoder_ckpt(ckpt_steps[step], step, pass_)
        pass_ += 1
        vloss = val_recon_loss(model, vl, 'mae', c, device, val_noise_gen, val_mask_gen)
        peak_mem = max(peak_mem, int(torch.cuda.max_memory_allocated(device)))
        row = {'pass': pass_, 'epoch': pass_, 'lr': opt.param_groups[0]['lr'],
               'train_loss': total / n, 'val_loss': vloss,
               'cum_samples': n_total, 'step': step,
               'train_seconds': time.perf_counter() - t0, 'peak_mem_bytes': peak_mem}
        ck = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
              'scheduler': scheduler.state_dict(), 'config': c, 'provenance': provenance,
              'step': step, 'pass': pass_, 'cum_samples': n_total,
              'rng': rng_state(gen), 'mask_rng': mask_gen.get_state(),
              'val_noise_rng': val_noise_gen.get_state(), 'val_mask_rng': val_mask_gen.get_state(),
              'aux': aux}
        save_checkpoint(out / 'last.pt', ck)
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        if pass_ in ckpt_epochs:
            save_encoder_ckpt(ckpt_epochs[pass_], step, pass_)
        print(json.dumps({'pass': pass_, 'step': step, 'train_loss': total / n,
                          'val_loss': vloss, 'cum_samples': n_total,
                          'peak_mem_mb': peak_mem / 1e6,
                          'elapsed': time.perf_counter() - t0}), flush=True)

    # Final weights = end of last epoch. Extract encoder for downstream.
    ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    torch.save({'encoder': _extract_encoder_sd(model), 'config': c, 'provenance': provenance,
                'aux': ck['aux'], 'step': ck['step'], 'pass': ck['pass']},
               out / 'encoder_final.pt')
    save_json(out / 'DONE.json', {'epochs': epochs, 'total_steps': total_steps,
                                  'final_step': ck['step'], 'final_pass': ck['pass'],
                                  'cum_samples': ck['cum_samples'],
                                  'peak_mem_bytes': peak_mem,
                                  'task': 'mae', 'backbone': args.backbone, 'smoke': args.smoke})
    print(f'[done] {out.name} step={ck["step"]} pass={ck["pass"]}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--backbone', choices=BACKBONES, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--shared-init-dir', required=True)
    p.add_argument('--corpus', required=True)
    p.add_argument('--epochs', type=int, required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    train_epoch(args)


if __name__ == '__main__':
    main()
