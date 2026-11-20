"""SERVER: single-device masked IQ pretraining with independent RNG and fixed val masks."""
import argparse
import json
import math
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import sha256
from .masked_model import MaskedIQModel, make_patch_mask, masked_error
from .pretrain_data import PretrainIQ, load_bundle, collate_pretrain
from .train import seed_all, worker_seed, autocast, dump, save_checkpoint


@torch.no_grad()
def validate(model, batches, device, precision, ratio, mask_seed):
    model.eval()
    total, count, zero = 0., 0, 0.
    per_snr = {}
    for b in batches:
        mask = make_patch_mask(b['valid_mask'], model.encoder.patch_size, ratio,
                               sample_ids=b['sample_id'], seed=mask_seed).to(device)
        x, valid = b['iq'].to(device), b['valid_mask'].to(device)
        with autocast(device, precision):
            pred = model(x, valid, mask)
            error, components, baseline = masked_error(pred, x, valid, mask, model.encoder.patch_size)
        if not torch.isfinite(error):
            raise FloatingPointError('Nonfinite validation error')
        total += float(error)
        count += int(components)
        zero += float(baseline)
        for snr in b['snr'].unique():
            select = (b['snr'] == snr).to(device)
            e, c, z = masked_error(pred[select], x[select], valid[select], mask[select], model.encoder.patch_size)
            stat = per_snr.setdefault(str(float(snr)), [0., 0, 0.])
            stat[0] += float(e); stat[1] += int(c); stat[2] += float(z)
    return {'masked_mse': total / count, 'zero_prediction_mse': zero / count,
            'mse_divided_by_zero_baseline': total / zero if zero else None,
            'masked_iq_components': count,
            'per_snr': {k: {'mse': v[0]/v[1], 'zero_mse': v[2]/v[1]} for k,v in per_snr.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--h5', required=True)
    p.add_argument('--bundle', required=True, help='Generated dataset.json')
    p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=['smoke','dev'], default='smoke')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--cache', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', help='Own trusted checkpoint; epoch boundaries only')
    # Enables a controlled resume equivalence check without changing the schedule.
    p.add_argument('--stop-after-epochs', type=int, help='Stop cleanly after N epochs, preserving configured schedule')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    t = cfg['training']; epochs = t['epochs'][a.stage]
    if a.workers < 0 or (a.cache and a.workers != 0):
        p.error('Cached selected subset uses workers=0; --no-cache supports workers>0')
    if epochs < 1 or t['batch_size'] < 1 or not 0 < cfg['objective']['mask_ratio'] < 1:
        p.error('Invalid epochs/batch/mask settings')
    if a.stop_after_epochs is not None and a.stop_after_epochs < 1:
        p.error('--stop-after-epochs must be positive')
    if cfg['objective'].get('name', 'masked_iq_mse') != 'masked_iq_mse':
        p.error('This implementation supports masked_iq_mse only')
    if cfg['data']['normalization'] != 'max_abs':
        p.error('First pretraining bundle uses max_abs only')
    bundle, rows = load_bundle(a.bundle)
    if bundle['classification_manifest_sha256'] != cfg['data']['classification_manifest_sha256']:
        raise ValueError('Downstream exclusion manifest mismatch')
    device = torch.device(a.device)
    if device.type not in ('cuda','cpu'):
        p.error('Use server CPU or CUDA')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    precision = t['precision']
    if precision not in ('fp32','bf16'):
        p.error('Supported precision: fp32/bf16')
    if precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        p.error('bf16 unavailable')
    out = Path(a.output)
    if a.resume:
        if Path(a.resume).resolve().parent != out.resolve():
            p.error('Resume in the existing run directory')
    out.mkdir(parents=True, exist_ok=bool(a.resume))
    seed_all(a.seed)
    model = MaskedIQModel(cfg['model']).to(device)
    metadata = {'config': cfg, 'stage': a.stage, 'seed': a.seed, 'device': str(device),
                'workers': a.workers, 'cache': a.cache,
                'bundle_sha256': sha256(a.bundle),
                'classification_manifest_sha256': bundle['classification_manifest_sha256'],
                'h5_path': str(Path(a.h5).resolve()),
                'python': platform.python_version(), 'torch': torch.__version__,
                'parameters': sum(p.numel() for p in model.parameters()),
                'encoder_parameters': sum(p.numel() for p in model.encoder.parameters()),
                'reconstruction_parameters': sum(p.numel() for p in model.reconstruction.parameters()),
                'mask_token_parameters': model.mask_token.numel(),
                'source_hashes': {str(f.relative_to(Path(__file__).parent)): sha256(f)
                                  for f in sorted(Path(__file__).parent.rglob('*.py'))}}
    print('Reading and checking selected IQ only; no Y/class labels enter training.', flush=True)
    train_ds = PretrainIQ(a.h5, rows['train_'+a.stage], a.cache)
    val_ds = PretrainIQ(a.h5, rows['val_'+a.stage], a.cache)
    metadata.update(train_samples=len(train_ds), val_samples=len(val_ds))
    gen = torch.Generator().manual_seed(a.seed)
    val_gen = torch.Generator().manual_seed(a.seed+1)
    masks = torch.Generator().manual_seed(a.seed+2)
    kwargs = {'multiprocessing_context':'spawn'} if a.workers else {}
    def dl(ds, shuffle, rng):
        return DataLoader(ds, batch_size=t['batch_size'], shuffle=shuffle, num_workers=a.workers,
                          collate_fn=collate_pretrain, worker_init_fn=worker_seed, generator=rng,
                          pin_memory=device.type=='cuda', **kwargs)
    train_dl, val_dl = dl(train_ds, True, gen), dl(val_ds, False, val_gen)
    opt = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    total_steps = epochs * len(train_dl)
    warmup = max(1, int(total_steps * t['warmup_fraction']))
    def schedule(step):
        if step < warmup:
            return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total_steps-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    start, best, step = 0, float('inf'), 0
    if a.resume:
        state = torch.load(a.resume, map_location='cpu', weights_only=False)
        if state.get('format') != 'emfm_masked_iq_v1' or state['metadata'] != metadata:
            raise ValueError('Resume metadata mismatch: keep code/config/data/device unchanged')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start, best, step = state['epoch']+1, state['best'], state['step']
        gen.set_state(state['loader_rng']); val_gen.set_state(state['val_rng']); masks.set_state(state['mask_rng'])
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        # Remove any log line written after the last durable epoch checkpoint.
        log = out/'metrics.jsonl'
        if log.exists():
            kept = [r for r in log.read_text().splitlines() if json.loads(r)['epoch'] < start]
            log.write_text(''.join(r+'\n' for r in kept))
    dump(out/'metadata.json', metadata)
    ratio, mask_seed = cfg['objective']['mask_ratio'], cfg['objective']['validation_mask_seed']
    if not a.resume:
        initial = validate(model, val_dl, device, precision, ratio, mask_seed)
        dump(out/'initial_validation.json', initial)
        print('Initial validation: '+json.dumps(initial), flush=True)
    for epoch in range(start, epochs):
        if a.stop_after_epochs is not None and epoch >= a.stop_after_epochs:
            break
        model.train()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        begin = time.perf_counter()
        error_sum, components_sum, samples = 0., 0, 0
        for b in train_dl:
            mask = make_patch_mask(b['valid_mask'], model.encoder.patch_size, ratio, generator=masks).to(device)
            x, valid = b['iq'].to(device), b['valid_mask'].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device, precision):
                pred = model(x, valid, mask)
                error, components, _ = masked_error(pred, x, valid, mask, model.encoder.patch_size)
                loss = error / components
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite pretraining loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'], error_if_nonfinite=True)
            opt.step(); scheduler.step(); step += 1
            error_sum += float(error.detach()); components_sum += int(components); samples += len(x)
            if step % 100 == 0:
                print(json.dumps({'epoch':epoch,'step':step,'masked_mse':float(loss.detach())}), flush=True)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds = time.perf_counter()-begin
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        val = validate(model, val_dl, device, precision, ratio, mask_seed)
        improved = val['masked_mse'] < best
        best = min(best, val['masked_mse'])
        record = {'epoch':epoch, 'step':step, 'train_masked_mse':error_sum/components_sum,
                  'train_samples_s':samples/seconds,'train_seconds':seconds,'peak_train_allocated_bytes':peak,
                  'validation':val,'lr_next':scheduler.get_last_lr()[0]}
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        state = {'format':'emfm_masked_iq_v1', 'metadata':metadata, 'model':model.state_dict(),
                 'encoder':model.encoder.state_dict(), 'optimizer':opt.state_dict(), 'scheduler':scheduler.state_dict(),
                 'epoch':epoch,'step':step,'best':best, 'loader_rng':gen.get_state(),'val_rng':val_gen.get_state(),
                 'mask_rng':masks.get_state(),'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
                 'python_rng':random.getstate(), 'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        if improved:
            save_checkpoint(out/'best.pt',state)
            dump(out/'best_metrics.json', {'epoch':epoch,**val})
        save_checkpoint(out/'last.pt',state)
        print(json.dumps(record),flush=True)
    train_ds.close(); val_ds.close()


if __name__ == '__main__':
    main()
