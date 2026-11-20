"""Round6 pretraining: objective comparison (MAE / denoise / joint) + bidirectional SSM.

Fixed-epoch budget, identical to round5 (50 full epochs = 39,100 optimizer steps on F1):
AdamW(lr 3e-4, wd 0.01, betas (0.9,0.999), eps 1e-8), cosine + 5% warmup, grad_clip 1.0,
fp32, no CFO, patch=8, batch 128 drop_last=False. Train from random init (round5 shared-init
cache for G/Q, Q_bi derives its front/aux/norm from Q's cache); NOT resumed from trained weights.

Tasks:
  mae      : zero-fill 50% masked patches, per-token reconstruction (round5 protocol).
  denoise  : corrupt with synthetic Gaussian noise at 0/5/10/15 dB (additional-noise SNR);
             target = original x; sigma enters ONLY the denoise head (extra scalar).
  joint    : shared encoder + independent MAE and denoise heads. Each batch is split into ~2
             equal subsets (one task per sample per pass, re-randomized each epoch via the
             split RNG); both subsets forward, gradients accumulate, ONE optimizer update per
             original batch. L_joint = 0.5*L_MAE + 0.5*L_DN.

RNG streams (round2 convention, extended): data order(seed), noise(seed+100000),
mask(seed+200000), val noise(seed+300000), val mask(seed+400000), joint split(seed+500000).
G/Q reuse round5's shared-init so data order + mask + encoder init are byte-identical to
round5 G-MAE/Q-MAE.

Backbones: G, Q, Q_bi (see scripts/models_r6.py).

Usage:
  python pretrain_r6.py --config configs/pretrain.json --manifest <f1 manifest> \
      --backbone G --task denoise --out round6_objective_bidirectional/runs/pretrain/f1/G_denoise_seed17 \
      --shared-init-dir round6_objective_bidirectional/runs/pretrain/f1/shared_init \
      --round5-shared-init-dir round5_epoch_scale/runs/pretrain/f1/shared_init \
      --corpus f1 --epochs 50 --seed 17 --device cuda:1
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
from round2_overnight.engine import (load_config, sample_mask, mae_loss, denoise_loss,
                                     val_recon_loss)
from round2_overnight.iq_noise import corrupt_iq
from round6_objective_bidirectional.scripts.models_r6 import (build_encoder_r6,
                                                              JointPretrainModel, R6_BACKBONES)


def sd_hash(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(d[k].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _extract_encoder_sd(model):
    return {k.replace('encoder.', ''): v
            for k, v in model.state_dict().items() if k.startswith('encoder.')}


def _real_scalar_params(module):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in module.parameters())


def build_shared_init_r6(cfg, backbone, seed, ds, device, r6_cache_dir, round5_dir):
    """Build/load the encoder init + aux stats for round6.

    - G/Q: load round5's cached {G,Q}_seed{seed}.pt verbatim (byte-identical to round5 MAE).
    - Q_bi: front/aux_branch/aux_fusion/norm/aux stats come from round5's Q_seed{seed}.pt;
      the A64 blocks stay at their fresh random init (built under seed_all(seed)).
      Cached under r6_cache_dir/Q_bi_seed{seed}.pt for reproducibility.
    """
    if backbone in ('G', 'Q'):
        src = Path(round5_dir) / f'{backbone}_seed{seed}.pt'
        ck = torch.load(src, map_location='cpu', weights_only=False)
        seed_all(seed)
        enc = build_encoder_r6(cfg['model'], backbone, seed).to(device)
        enc.load_state_dict(ck['encoder'])
        return enc, ck['aux'], str(src)

    # Q_bi
    cache = Path(r6_cache_dir) / f'Q_bi_seed{seed}.pt'
    q_src = Path(round5_dir) / f'Q_seed{seed}.pt'
    if cache.exists():
        ck = torch.load(cache, map_location='cpu', weights_only=False)
        seed_all(seed)
        enc = build_encoder_r6(cfg['model'], 'Q_bi', seed).to(device)
        enc.load_state_dict(ck['encoder'])
        return enc, ck['aux'], f'{q_src} (front/aux/norm) + fresh A64 blocks'

    qck = torch.load(q_src, map_location='cpu', weights_only=False)
    seed_all(seed)
    enc = build_encoder_r6(cfg['model'], 'Q_bi', seed).to(device)
    # Copy only the Q modules that Q_bi shares; leave the A64 blocks at fresh random init.
    shared = {k: v for k, v in qck['encoder'].items()
              if k.startswith(('front.', 'aux_branch.', 'aux_fusion.', 'norm.',
                               'aux_mean', 'aux_std'))}
    missing, unexpected = enc.load_state_dict(shared, strict=False)
    if unexpected:
        raise RuntimeError(f'Q_bi init unexpected keys: {unexpected}')
    # everything missing must be inside the bidirectional blocks (not shared with Q)
    assert all(k.startswith('blocks.') for k in missing), f'unexpected missing keys: {missing}'
    Path(r6_cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save({'encoder': enc.state_dict(), 'aux': qck['aux']}, cache)
    return enc, qck['aux'], f'{q_src} (front/aux/norm) + fresh A64 blocks'


@torch.no_grad()
def joint_val_losses(model, loader, cfg, device, val_noise_gen, val_mask_gen):
    """Joint validation: evaluate the MAE head and the denoise head separately over the FULL
    validation set (not split), plus the weighted total. Both heads use the shared encoder."""
    model.eval()
    patch = cfg['model']['patch']
    n_patches = cfg['model']['input_length'] // patch
    mae_total = dn_total = 0.0
    count = 0
    for x, _, _, _ in loader:
        x = max_abs_norm(x.to(device))
        mask = sample_mask(x.shape[0], n_patches, cfg['train']['mask_ratio'], val_mask_gen, x.device)
        pred_mae = model.forward_mae(x, mask)
        mae_total += mae_loss(pred_mae, x, mask, patch).item() * x.shape[0]
        y, sigma, _ = corrupt_iq(x, val_noise_gen)
        pred_dn = model.forward_dn(y, sigma)
        dn_total += denoise_loss(pred_dn, x).item() * x.shape[0]
        count += x.shape[0]
    mae_val = mae_total / count
    dn_val = dn_total / count
    return mae_val, dn_val, 0.5 * mae_val + 0.5 * dn_val


def train_epoch(args):
    c = load_config(args.config)
    if args.backbone not in R6_BACKBONES:
        raise ValueError('backbone must be one of ' + repr(R6_BACKBONES))
    if args.task not in ('mae', 'denoise', 'joint'):
        raise ValueError('task must be mae/denoise/joint')
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
    if args.smoke:
        ds.ids = ds.ids[:min(256, len(ds))]
        val.ids = val.ids[:min(256, len(val))]

    epochs = args.epochs
    steps_per_pass = math.ceil(len(ds) / c['train']['batch_size'])
    total_steps = epochs * steps_per_pass

    # RNG streams (round2 convention + joint split stream).
    gen = torch.Generator().manual_seed(seed)
    noise_gen = torch.Generator(device=device).manual_seed(seed + 100000)
    mask_gen = torch.Generator(device=device).manual_seed(seed + 200000)
    val_noise_gen = torch.Generator(device=device).manual_seed(seed + 300000)
    val_mask_gen = torch.Generator(device=device).manual_seed(seed + 400000)
    split_gen = torch.Generator(device=device).manual_seed(seed + 500000)

    dl = make_loader(ds, c['train']['batch_size'], args.workers, True, gen)
    vl = make_loader(val, c['train']['batch_size'], args.workers)

    r6_shared = Path(args.shared_init_dir)
    enc, aux, init_src = build_shared_init_r6(c, args.backbone, seed, ds, device,
                                              r6_shared, args.round5_shared_init_dir)
    init_hash = sd_hash(enc.state_dict())
    patch = c['model']['patch']
    if args.task == 'joint':
        model = JointPretrainModel(enc, patch).to(device)
    else:
        model = PretrainModel(enc, patch).to(device)

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
                  'task': args.task, 'backbone': args.backbone, 'corpus': args.corpus,
                  'epochs': epochs, 'total_steps': total_steps, 'steps_per_pass': steps_per_pass,
                  'shared_init_source': init_src,
                  'shared_init_encoder_sha256': init_hash,
                  'manifest': str(Path(args.manifest).resolve()),
                  'budget': 'fixed-epoch (50 full epochs)',
                  'rng_streams': {'data': seed, 'noise': seed + 100000, 'mask': seed + 200000,
                                  'val_noise': seed + 300000, 'val_mask': seed + 400000,
                                  'joint_split': seed + 500000}}

    # intermediate checkpoint plan (round6: epoch 10/20/30/40/50)
    ckpt_epochs = {10: 'ckpt_epoch10.pt', 20: 'ckpt_epoch20.pt', 30: 'ckpt_epoch30.pt',
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
        noise_gen.set_state(ck['noise_rng'])
        mask_gen.set_state(ck['mask_rng'])
        val_noise_gen.set_state(ck['val_noise_rng'])
        val_mask_gen.set_state(ck['val_mask_rng'])
        split_gen.set_state(ck['split_rng'])
        aux = ck['aux']
        print(f'[resume] step={start_step} pass={start_pass} cum_samples={n_total}', flush=True)
    else:
        save_json(out / 'config.json', c)
        save_json(out / 'provenance.json', provenance)
        save_json(out / 'model_info.json', {
            'backbone': args.backbone, 'task': args.task, 'aux': aux,
            'shared_init_source': init_src,
            'shared_init_encoder_sha256': init_hash,
            'encoder_real_scalar_parameters': _real_scalar_params(enc),
            'total_real_scalar_parameters': _real_scalar_params(model),
        })

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
        n_mae = n_dn = 0
        mae_total = dn_total = 0.0
        for x, _, _, _ in dl:
            x = max_abs_norm(x.to(device))
            B = x.shape[0]
            if args.task == 'mae':
                mask = sample_mask(B, n_patches, c['train']['mask_ratio'], mask_gen, x.device)
                pred = model(x, mask=mask)
                loss = mae_loss(pred, x, mask, patch)
            elif args.task == 'denoise':
                y, sigma, _ = corrupt_iq(x, noise_gen)
                extra = sigma[:, None].expand(B, n_patches)
                pred = model(y, extra=extra)
                loss = denoise_loss(pred, x)
            else:  # joint
                half = B // 2
                perm = torch.randperm(B, generator=split_gen, device=x.device)
                mae_idx, dn_idx = perm[:half], perm[half:]
                x_mae, x_dn = x[mae_idx], x[dn_idx]
                mask = sample_mask(len(x_mae), n_patches, c['train']['mask_ratio'], mask_gen, x.device)
                pred_mae = model.forward_mae(x_mae, mask)
                l_mae = mae_loss(pred_mae, x_mae, mask, patch)
                y, sigma, _ = corrupt_iq(x_dn, noise_gen)
                pred_dn = model.forward_dn(y, sigma)
                l_dn = denoise_loss(pred_dn, x_dn)
                loss = 0.5 * l_mae + 0.5 * l_dn
                mae_total += l_mae.item() * len(x_mae)
                dn_total += l_dn.item() * len(x_dn)
                n_mae += len(x_mae)
                n_dn += len(x_dn)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['train']['grad_clip'],
                                           error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            total += loss.item() * B
            n += B
            n_total += B
            step += 1
        pass_ += 1
        if args.task == 'joint':
            vmae, vdn, vtotal = joint_val_losses(model, vl, c, device, val_noise_gen, val_mask_gen)
            tloss = total / n
        else:
            vloss = val_recon_loss(model, vl, args.task, c, device, val_noise_gen, val_mask_gen)
            tloss = total / n
        peak_mem = max(peak_mem, int(torch.cuda.max_memory_allocated(device)))
        row = {'pass': pass_, 'epoch': pass_, 'lr': opt.param_groups[0]['lr'],
               'train_loss': tloss, 'cum_samples': n_total, 'step': step,
               'train_seconds': time.perf_counter() - t0, 'peak_mem_bytes': peak_mem}
        if args.task == 'joint':
            row.update({'train_loss_mae': mae_total / max(1, n_mae),
                        'train_loss_dn': dn_total / max(1, n_dn),
                        'val_loss_mae': vmae, 'val_loss_dn': vdn, 'val_loss': vtotal,
                        'samples_mae': n_mae, 'samples_dn': n_dn})
        else:
            row['val_loss'] = vloss
        ck = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
              'scheduler': scheduler.state_dict(), 'config': c, 'provenance': provenance,
              'step': step, 'pass': pass_, 'cum_samples': n_total,
              'rng': rng_state(gen), 'noise_rng': noise_gen.get_state(),
              'mask_rng': mask_gen.get_state(), 'val_noise_rng': val_noise_gen.get_state(),
              'val_mask_rng': val_mask_gen.get_state(), 'split_rng': split_gen.get_state(),
              'aux': aux}
        save_checkpoint(out / 'last.pt', ck)
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        if pass_ in ckpt_epochs:
            save_encoder_ckpt(ckpt_epochs[pass_], step, pass_)
        print(json.dumps({'pass': pass_, 'step': step, 'train_loss': tloss,
                          'val_loss': row['val_loss'], 'cum_samples': n_total,
                          'peak_mem_mb': peak_mem / 1e6,
                          'elapsed': time.perf_counter() - t0}), flush=True)

    ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    torch.save({'encoder': _extract_encoder_sd(model), 'config': c, 'provenance': provenance,
                'aux': ck['aux'], 'step': ck['step'], 'pass': ck['pass']},
               out / 'encoder_final.pt')
    save_json(out / 'DONE.json', {'epochs': epochs, 'total_steps': total_steps,
                                  'final_step': ck['step'], 'final_pass': ck['pass'],
                                  'cum_samples': ck['cum_samples'],
                                  'peak_mem_bytes': peak_mem,
                                  'task': args.task, 'backbone': args.backbone, 'smoke': args.smoke})
    print(f'[done] {out.name} step={ck["step"]} pass={ck["pass"]}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--backbone', choices=R6_BACKBONES, required=True)
    p.add_argument('--task', choices=['mae', 'denoise', 'joint'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--shared-init-dir', required=True)
    p.add_argument('--round5-shared-init-dir', required=True)
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
