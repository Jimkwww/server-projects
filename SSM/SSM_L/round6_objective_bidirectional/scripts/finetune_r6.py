"""Round6 full fine-tune (no CFO) for G/Q/Q_bi encoders.

Protocol identical to round4 no-CFO downstream (round5 finetune reused it): 50 epochs,
AdamW(lr 3e-4, wd 0.01), cosine + 5% warmup, grad_clip 1.0, fp32, batch 128, no CFO, head
re-initialized from scratch, clean-val-OA earliest-tie checkpoint selection. RML reports val;
RadChar selects on val(=dev) and evaluates test(=eval) once at the end via --eval-split test.

Differs from downstream_r4.py only in the encoder builder (build_encoder_r6 -> G/Q/Q_bi).
Q_bi keeps its bidirectional A64 trunk in both fine-tune and inference. The Q aux branch is
NOT frozen (round6 has no freeze-aux setting).

Usage:
  python finetune_r6.py --config configs/downstream_nocfo_rml.json --manifest <rml manifest> \
      --backbone Q_bi --pretrained <encoder_final.pt> --out <out_dir> --seed 17 --device cuda:1
  # radar: add --eval-split test
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
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ssm_exp.data import IQDataset, read_manifest, audit
from ssm_exp.engine import make_loader
from ssm_exp.model import max_abs_norm
from ssm_exp.utils import (seed_all, save_json, save_checkpoint, environment, rng_state,
                           restore_rng, grouped_metrics)
from pretrain.model import DownstreamModel
from round2_overnight.engine import resume_key
from round6_objective_bidirectional.scripts.models_r6 import build_encoder_r6, R6_BACKBONES


def load_config(path):
    c = json.loads(Path(path).read_text())
    if not {'model', 'train', 'protocol'} <= c.keys():
        raise ValueError('incomplete config')
    if c['train']['precision'] != 'fp32':
        raise ValueError('v1 supports fp32 only')
    if c['train']['augment_cfo_max'] != 0.0:
        raise ValueError('round6 downstream requires augment_cfo_max=0.0 (no CFO)')
    return c


@torch.no_grad()
def evaluate(model, loader, device, save_predictions=None):
    model.eval()
    ys, ps, zs, ids = [], [], [], []
    total_loss = 0.0
    count = 0
    for x, y, z, i in loader:
        x = max_abs_norm(x.to(device))
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


def train(args):
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
    if m['classes'] != c['model']['classes']:
        raise ValueError('manifest/config class count mismatch')

    encoder = build_encoder_r6(c['model'], args.backbone, c['train']['seed']).to(device)
    pt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    encoder.load_state_dict(pt['encoder'])
    model = DownstreamModel(encoder, c['model']['classes']).to(device)

    ds = IQDataset(m, 'train')
    val = IQDataset(m, 'val')
    if args.smoke:
        ds.ids = ds.ids[:min(256, len(ds))]
        val.ids = val.ids[:min(256, len(val))]

    gen = torch.Generator().manual_seed(c['train']['seed'])
    dl = make_loader(ds, c['train']['batch_size'], args.workers, True, gen)
    vl = make_loader(val, c['train']['batch_size'], args.workers)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=c['train']['lr'], weight_decay=c['train']['weight_decay'],
                            betas=tuple(c['train']['betas']), eps=c['train']['adam_eps'])
    total_steps = c['train']['epochs'] * len(dl)
    warmup_steps = max(1, int(total_steps * c['train']['warmup_fraction']))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup_steps) / max(1, total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    provenance = {'data': data_audit, 'environment': environment(), 'smoke': args.smoke,
                  'mode': 'finetune',
                  'pretrained': str(Path(args.pretrained).resolve()),
                  'pretrained_sha256': hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest()}

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
    else:
        save_json(out / 'config.json', c)
        save_json(out / 'provenance.json', provenance)

    for epoch in range(start, c['train']['epochs']):
        model.train()
        total = 0.0
        n = 0
        t0 = time.perf_counter()
        for x, y, _, _ in dl:
            x = max_abs_norm(x.to(device))
            logits = model(x)
            loss = F.cross_entropy(logits, y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if c['train']['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), c['train']['grad_clip'],
                                               error_if_nonfinite=True)
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
               'val': scores, 'train_seconds': train_seconds, 'step': scheduler.last_epoch + 1}
        ck = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
              'scheduler': scheduler.state_dict(), 'config': c, 'provenance': provenance,
              'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
              'rng': rng_state(gen)}
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
    extra = {'best_epoch': ck['epoch'], 'selection': 'val OA, earliest tie', 'metrics': result,
             'smoke': args.smoke}
    if args.eval_split:
        ev = IQDataset(m, args.eval_split)
        evl = make_loader(ev, c['train']['batch_size'], args.workers)
        eres = evaluate(model, evl, device, save_predictions=out / 'eval_predictions.npz')
        extra[f'eval_{args.eval_split}'] = eres
    save_json(out / 'best_val.json', extra)
    save_json(out / 'DONE.json', {'epochs': c['train']['epochs'], 'best_epoch': ck['epoch'],
                                  'final_epoch': c['train']['epochs'] - 1, 'smoke': args.smoke})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--backbone', choices=R6_BACKBONES, required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--epochs', type=int)
    p.add_argument('--eval-split', default=None)
    a = p.parse_args()
    train(a)


if __name__ == '__main__':
    main()
