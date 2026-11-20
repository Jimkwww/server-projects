"""SSM_L round-1 downstream training: frozen linear probe, full fine-tune, and supervised
scratch baselines. All three read raw IQ (max_abs-normalized in-loop) and share the same
clean-val-OA checkpoint selection (earliest tie)."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ssm_exp.data import IQDataset, read_manifest, audit
from ssm_exp.engine import make_loader
from ssm_exp.model import max_abs_norm, perturb
from ssm_exp.utils import seed_all, save_json, save_checkpoint, environment, rng_state, restore_rng, grouped_metrics

from .model import build_encoder, DownstreamModel
from .engine import fit_aux_stats, resume_key


def load_config(path):
    c = json.loads(Path(path).read_text())
    if not {'model', 'train', 'protocol'} <= c.keys():
        raise ValueError('incomplete config')
    if c['train']['precision'] != 'fp32':
        raise ValueError('v1 supports fp32 only')
    return c


@torch.no_grad()
def evaluate(model, loader, device, frequency=0., phase=0., save_predictions=None):
    model.eval()
    ys, ps, zs, ids = [], [], [], []
    total_loss = 0.0
    count = 0
    for x, y, z, i in loader:
        x = max_abs_norm(x.to(device))
        if frequency != 0 or phase != 0:
            x = perturb(x, frequency, phase)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite evaluation logits')
        total_loss += F.cross_entropy(logits, y.to(device), reduction='sum').item()
        count += len(y)
        ys.extend(y.cpu().tolist())
        ps.extend(logits.argmax(-1).cpu().tolist())
        zs.extend(z.tolist())
        ids.extend(i.tolist())
    if not count:
        raise ValueError('empty evaluation set')
    result = grouped_metrics(ys, ps, zs, model.encoder.cfg['classes'])
    result['loss'] = total_loss / count
    if save_predictions:
        np.savez_compressed(save_predictions, y=np.array(ys), pred=np.array(ps),
                            snr=np.array(zs), ids=np.array(ids))
    return result


@torch.no_grad()
def precompute_features(encoder, ds, cfg, device):
    encoder.eval()
    feats, ys, zs, ids = [], [], [], []
    loader = make_loader(ds, cfg['train']['batch_size'], 0)
    for x, y, z, i in loader:
        x = max_abs_norm(x.to(device))
        feats.append(encoder.features(x).cpu())
        ys.append(y)
        zs.append(z)
        ids.append(i)
    return (torch.cat(feats), torch.cat(ys), torch.cat(zs), torch.cat(ids))


def _fresh_head(dim, classes, seed):
    seed_all(seed)
    return nn.Linear(dim, classes)


def _sd_hash(state_dict):
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_probe(args):
    c = load_config(args.config)
    c['train']['seed'] = args.seed
    if args.epochs is not None and args.smoke:
        c['train']['epochs'] = args.epochs
    out = Path(args.out)
    if (out / 'last.pt').exists() and not args.resume:
        raise FileExistsError('use --resume or a new run directory')
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    m = read_manifest(args.manifest)
    data_audit = audit(m)
    if m['normalization'] != 'none':
        raise ValueError('downstream requires a raw (normalization=none) manifest')

    # Load the frozen pretrained encoder (epoch-50).
    pt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    seed = c['train']['seed']
    encoder = build_encoder(c['model'], args.backbone, seed).to(device)
    encoder.load_state_dict(pt['encoder'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder_hash_before = _sd_hash(encoder.state_dict())

    ds = IQDataset(m, 'train')
    val = IQDataset(m, 'val')
    if args.smoke:
        ds.ids = ds.ids[:min(256, len(ds))]
        val.ids = val.ids[:min(256, len(val))]

    # Precompute frozen mean-pooled features once (encoder is frozen; dropout=0).
    ftr, ytr, ztr, itr = precompute_features(encoder, ds, c, device)
    fvl, yvl, zvl, ivl = precompute_features(encoder, val, c, device)
    train_ds = torch.utils.data.TensorDataset(ftr, ytr)
    val_ds = torch.utils.data.TensorDataset(fvl, yvl)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=c['train']['batch_size'], shuffle=True,
                                           generator=torch.Generator().manual_seed(seed))
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=c['train']['batch_size'])

    head = _fresh_head(c['model']['dim'], c['model']['classes'], seed).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=c['train']['lr'],
                            weight_decay=c['train']['weight_decay'],
                            betas=tuple(c['train']['betas']), eps=c['train']['adam_eps'])
    total_steps = c['train']['epochs'] * len(train_dl)
    warmup_steps = max(1, int(total_steps * c['train']['warmup_fraction']))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return .5 * (1 + math.cos(math.pi * min(1., step / max(1, total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    provenance = {'data': data_audit, 'environment': environment(), 'smoke': args.smoke,
                  'pretrained': str(Path(args.pretrained).resolve()),
                  'pretrained_sha256': hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest()}

    def eval_probe():
        head.eval()
        correct = 0
        n = 0
        for f, y in val_dl:
            logits = head(f.to(device))
            correct += (logits.argmax(-1).cpu() == y).sum().item()
            n += len(y)
        return correct / n

    start = 0
    best = -1.0
    best_epoch = -1
    if args.resume:
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        head.load_state_dict(ck['head'])
        opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start = ck['epoch'] + 1
        best, best_epoch = ck['best'], ck['best_epoch']
    else:
        save_json(out / 'config.json', c)
        save_json(out / 'provenance.json', provenance)

    for epoch in range(start, c['train']['epochs']):
        head.train()
        total = 0.0
        n = 0
        for f, y in train_dl:
            logits = head(f.to(device))
            loss = F.cross_entropy(logits, y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
            total += loss.item() * len(y)
            n += len(y)
        val_oa = eval_probe()
        improved = val_oa > best
        if improved:
            best, best_epoch = val_oa, epoch
        row = {'epoch': epoch, 'lr': opt.param_groups[0]['lr'], 'train_loss': total / n,
               'val_oa': val_oa, 'best': best, 'best_epoch': best_epoch}
        save_checkpoint(out / 'last.pt', {'head': head.state_dict(), 'optimizer': opt.state_dict(),
                                          'scheduler': scheduler.state_dict(), 'epoch': epoch,
                                          'best': best, 'best_epoch': best_epoch, 'config': c,
                                          'provenance': provenance})
        if improved:
            save_checkpoint(out / 'best.pt', {'head': head.state_dict(), 'epoch': epoch,
                                              'best': best, 'best_epoch': best_epoch})
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps({'epoch': epoch, 'train_loss': total / n, 'val_oa': val_oa,
                          'best_epoch': best_epoch}), flush=True)

    # Final clean-val predictions from the best head on precomputed features.
    ck = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    head.load_state_dict(ck['head'])
    head.eval()
    preds = head(fvl.to(device)).argmax(-1).cpu().tolist()
    result = grouped_metrics(yvl.tolist(), preds, zvl.tolist(), c['model']['classes'])
    np.savez_compressed(out / 'val_predictions.npz', y=yvl.numpy(), pred=np.array(preds),
                        snr=zvl.numpy(), ids=ivl.numpy())
    encoder_hash_after = _sd_hash(encoder.state_dict())
    save_json(out / 'best_val.json', {'best_epoch': ck['epoch'], 'selection': 'val OA, earliest tie',
                                      'metrics': result, 'smoke': args.smoke,
                                      'encoder_frozen': encoder_hash_before == encoder_hash_after,
                                      'encoder_hash_before': encoder_hash_before,
                                      'encoder_hash_after': encoder_hash_after})
    save_json(out / 'DONE.json', {'epochs': c['train']['epochs'], 'best_epoch': ck['epoch'],
                                  'final_epoch': c['train']['epochs'] - 1, 'smoke': args.smoke})


def train_supervised(args, mode):
    """mode in {'finetune','scratch'}: full-model supervised training."""
    c = load_config(args.config)
    c['train']['seed'] = args.seed
    if args.epochs is not None and args.smoke:
        c['train']['epochs'] = args.epochs
    out = Path(args.out)
    if (out / 'last.pt').exists() and not args.resume:
        raise FileExistsError('use --resume or a new run directory')
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seed_all(c['train']['seed'])
    m = read_manifest(args.manifest)
    data_audit = audit(m)
    if m['normalization'] != 'none':
        raise ValueError('downstream requires a raw (normalization=none) manifest')

    encoder = build_encoder(c['model'], args.backbone, c['train']['seed']).to(device)
    if mode == 'finetune':
        pt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        encoder.load_state_dict(pt['encoder'])
    model = DownstreamModel(encoder, c['model']['classes']).to(device)

    ds = IQDataset(m, 'train')
    val = IQDataset(m, 'val')
    if args.smoke:
        ds.ids = ds.ids[:min(256, len(ds))]
        val.ids = val.ids[:min(256, len(val))]
    if mode == 'scratch' and args.backbone == 'Q':
        # Scratch Q must fit the frozen aux-correlation statistics from train IQ exactly as the
        # old Q_corr did (same fixed rng-0 subset, same init -> identical to the pretrain stats).
        fit_aux_stats(encoder, ds, c, device)
    gen = torch.Generator().manual_seed(c['train']['seed'])
    aug_gen = torch.Generator(device=device).manual_seed(c['train']['seed'] + 100000)
    dl = make_loader(ds, c['train']['batch_size'], args.workers, True, gen)
    vl = make_loader(val, c['train']['batch_size'], args.workers)

    opt = torch.optim.AdamW(model.parameters(), lr=c['train']['lr'],
                            weight_decay=c['train']['weight_decay'],
                            betas=tuple(c['train']['betas']), eps=c['train']['adam_eps'])
    total_steps = c['train']['epochs'] * len(dl)
    warmup_steps = max(1, int(total_steps * c['train']['warmup_fraction']))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup_steps) / max(1, total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    provenance = {'data': data_audit, 'environment': environment(), 'smoke': args.smoke, 'mode': mode}
    if mode == 'finetune':
        provenance['pretrained'] = str(Path(args.pretrained).resolve())
        provenance['pretrained_sha256'] = hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest()

    start = 0
    best = -1.0
    best_epoch = -1
    if args.resume:
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        if resume_key(ck['config']) != resume_key(c):
            raise ValueError('resume config differs; create a new run')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start = ck['epoch'] + 1
        best, best_epoch = ck['best'], ck['best_epoch']
        restore_rng(ck['rng'], gen)
        aug_gen.set_state(ck['augment_rng'])
    else:
        save_json(out / 'config.json', c)
        save_json(out / 'provenance.json', provenance)

    for epoch in range(start, c['train']['epochs']):
        model.train()
        total = 0.0
        n = 0
        t0 = time.perf_counter()
        for batch, (x, y, _, _) in enumerate(dl):
            x = x.to(device)
            extent = c['train']['augment_cfo_max']
            if c['train'].get('augment_on_raw', False):
                if extent > 0:
                    x = perturb(x, (2 * torch.rand(len(x), device=device, generator=aug_gen) - 1) * extent)
                x = max_abs_norm(x)
            elif extent > 0:
                x = max_abs_norm(x)
                x = perturb(x, (2 * torch.rand(len(x), device=device, generator=aug_gen) - 1) * extent)
            else:
                x = max_abs_norm(x)
            logits = model(x)
            loss = F.cross_entropy(logits, y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if c['train']['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), c['train']['grad_clip'], error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            total += loss.item() * len(y)
            n += len(y)
        train_seconds = time.perf_counter() - t0
        scores = evaluate(model, vl, device)
        improved = scores['overall']['oa'] > best
        if improved:
            best, best_epoch = scores['overall']['oa'], epoch
        row = {'epoch': epoch, 'lr': opt.param_groups[0]['lr'], 'train_loss': total / n,
               'val': scores, 'train_seconds': train_seconds,
               'step': scheduler.last_epoch + 1}
        ck = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
              'scheduler': scheduler.state_dict(), 'config': c, 'provenance': provenance,
              'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
              'rng': rng_state(gen), 'augment_rng': aug_gen.get_state()}
        save_checkpoint(out / 'last.pt', ck)
        if improved:
            save_checkpoint(out / 'best.pt', ck)
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps({'epoch': epoch, 'train_loss': total / n, 'val_oa': scores['overall']['oa'],
                          'val_f1': scores['overall']['macro_f1'], 'best_epoch': best_epoch}), flush=True)

    ck = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    result = evaluate(model, vl, device, save_predictions=out / 'val_predictions.npz')
    save_json(out / 'best_val.json', {'best_epoch': ck['epoch'], 'selection': 'val OA, earliest tie',
                                      'metrics': result, 'smoke': args.smoke})
    save_json(out / 'DONE.json', {'epochs': c['train']['epochs'], 'best_epoch': ck['epoch'],
                                  'final_epoch': c['train']['epochs'] - 1, 'smoke': args.smoke})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['probe', 'finetune', 'scratch'], required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--backbone', choices=['G', 'Q'], required=True)
    p.add_argument('--pretrained', default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--epochs', type=int)
    a = p.parse_args()
    if a.mode == 'probe':
        if not a.pretrained:
            p.error('--pretrained required for probe')
        train_probe(a)
    elif a.mode == 'finetune':
        if not a.pretrained:
            p.error('--pretrained required for finetune')
        train_supervised(a, 'finetune')
    else:
        train_supervised(a, 'scratch')


if __name__ == '__main__':
    main()
