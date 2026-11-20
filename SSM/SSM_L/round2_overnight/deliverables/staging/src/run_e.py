"""E. Independent radar transfer (round-2 task book section E).

RadChar-Tiny (5 radar waveform types, 512 complex IQ) preprocessed by prep_radchar.py into the
RML manifest format. Frozen round-1 encoders G/Q x {random, MAE, denoise} x 3 seeds, head
rebuilt for 5 classes. No SNR/class/source fed to the encoder (SNR only for reporting).

For each encoder:
  - cache mean-pooled radar train/dev/eval features once (resume-safe),
  - full-label probe: AdamW head (30 epochs, cosine, lr 1e-3) selected on dev, eval ONCE on eval,
  - 10-per-class x 5 label seeds: fixed-standardization L2 logistic regression, eval on eval.
Reports record-level OA / macro-F1 / per-class recall.
"""
import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
R2 = Path(__file__).resolve().parent.parent
RUNS1 = ROOT / 'runs' / 'pretrain'
RADAR_FEAT = R2 / 'runs' / 'radar_features'
SEEDS = (17, 42, 2026)
BACKBONES = ('G', 'Q')
PRETRAINS = ('random', 'mae', 'denoise')
CLASSES = 5


def ckpt_for(bb, pt, seed):
    if pt == 'random':
        return R2 / 'runs' / 'random_ckpt' / f'{bb}_seed{seed}.pt'
    return RUNS1 / f'{bb}_{pt}_seed{seed}' / 'encoder_final.pt'


def cache_radar_features(bb, pt, seed, device, force=False):
    import sys
    sys.path.insert(0, str(ROOT))
    from ssm_exp.data import IQDataset, read_manifest
    from round2_overnight.common import load_encoder, features_for_encoder

    out = RADAR_FEAT / f'{bb}_{pt}_seed{seed}.pt'
    if out.exists() and not force:
        return out
    ck = ckpt_for(bb, pt, seed)
    if pt == 'random' and not ck.exists():
        from round2_overnight.common import make_random_checkpoint
        model_cfg = json.loads((ROOT / 'configs' / 'pretrain.json').read_text())['model']
        make_random_checkpoint(model_cfg, bb, seed, str(ck), device='cpu',
                               shared_init_dir=str(ROOT / 'runs' / 'pretrain' / 'shared_init'))
    model_cfg = json.loads((ROOT / 'configs' / 'pretrain.json').read_text())['model']
    enc = load_encoder(model_cfg, bb, seed, str(ck), device)
    m = read_manifest(str(R2 / 'configs' / 'radchar.json'))
    payload = {}
    for split in ('train', 'val', 'test'):
        ds = IQDataset(m, split)
        feats, ys, zs, ids = features_for_encoder(enc, ds, 128, device)
        payload[split] = {'feats': feats, 'y': ys, 'snr': zs, 'idx': ids}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out


def adamw_probe(feat, seed, epochs=30, device=None):
    device = device or torch.device('cpu')
    Xtr, ytr = feat['train']['feats'].to(device), feat['train']['y'].to(device)
    Xdv, ydv = feat['val']['feats'].to(device), feat['val']['y'].to(device)
    Xev, yev = feat['test']['feats'].to(device), feat['test']['y'].to(device)
    dim = Xtr.shape[1]
    torch.manual_seed(seed)
    head = nn.Linear(dim, CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.0,
                            betas=(0.9, 0.999), eps=1e-8)
    tr_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, ytr),
                                        batch_size=128, shuffle=True,
                                        generator=torch.Generator().manual_seed(seed))
    total_steps = epochs * len(tr_dl)
    warmup = max(1, int(total_steps * 0.05))
    def lr(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., step / max(1, total_steps - warmup))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr)
    best = -1.0
    best_state = None
    step = 0
    for epoch in range(epochs):
        head.train()
        for f, y in tr_dl:
            logits = head(f)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        head.eval()
        with torch.no_grad():
            dv_oa = (head(Xdv).argmax(-1) == ydv).float().mean().item()
        if dv_oa > best:
            best = dv_oa
            best_state = deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    with torch.no_grad():
        ev_pred = head(Xev).argmax(-1)
        ev_oa = (ev_pred == yev).float().mean().item()
    return best, ev_oa, ev_pred.cpu()


def lr_lowlabel(feat, label_seeds=(0, 1, 2, 3, 4), budget=10):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    Xtr, ytr = feat['train']['feats'].numpy(), feat['train']['y'].numpy()
    Xev, yev = feat['test']['feats'].numpy(), feat['test']['y'].numpy()
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xev_s = scaler.transform(Xev)
    rows = []
    for ls in label_seeds:
        rng = np.random.default_rng(ls)
        idx = []
        for c in range(CLASSES):
            ci = np.where(ytr == c)[0]
            idx.append(rng.choice(ci, size=min(budget, len(ci)), replace=False))
        idx = np.concatenate(idx)
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        clf.fit(Xtr_s[idx], ytr[idx])
        pred = clf.predict(Xev_s)
        rows.append({'label_seed': ls, 'oa': float(accuracy_score(yev, pred)),
                     'macro_f1': float(f1_score(yev, pred, average='macro')),
                     'recall': recall_score(yev, pred, average=None).tolist()})
    return rows


def metrics_for(ev_y, ev_pred, ev_snr):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    ev_y = ev_y.numpy(); ev_pred = ev_pred.numpy(); ev_snr = ev_snr.numpy()
    return {
        'oa': float(accuracy_score(ev_y, ev_pred)),
        'macro_f1': float(f1_score(ev_y, ev_pred, average='macro')),
        'recall': recall_score(ev_y, ev_pred, average=None).tolist(),
        'n': int(len(ev_y)),
        'oa_low_snr': float(accuracy_score(ev_y[ev_snr <= 0], ev_pred[ev_snr <= 0])),
        'oa_high_snr': float(accuracy_score(ev_y[ev_snr > 0], ev_pred[ev_snr > 0])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default=str(R2 / 'reports' / 'radar_transfer_round2.json'))
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--force-cache', action='store_true')
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    specs = [(bb, pt, s) for bb in BACKBONES for pt in PRETRAINS for s in SEEDS]
    results = {}
    for i, (bb, pt, s) in enumerate(specs):
        key = f'{bb}_{pt}_seed{s}'
        f = cache_radar_features(bb, pt, s, device, force=args.force_cache)
        print(f'[{i+1}/{len(specs)}] cached {key}', flush=True)
        if args.cache_only:
            continue
        feat = torch.load(f, map_location='cpu', weights_only=False)
        dev_best, ev_oa, ev_pred = adamw_probe(feat, s, device=device)
        ev_metrics = metrics_for(feat['test']['y'], ev_pred, feat['test']['snr'])
        pred_dir = R2 / 'runs' / 'radar_eval_preds'
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_dir / f'{key}.npz', y=feat['test']['y'].numpy(),
                            pred=ev_pred.numpy(), snr=feat['test']['snr'].numpy(),
                            idx=feat['test']['idx'].numpy())
        low = lr_lowlabel(feat)
        results[key] = {'full_label': {'dev_best_oa': dev_best, 'eval': ev_metrics},
                        'low_label_10_per_class': low}
        print(f'  {key}: dev {dev_best:.4f} eval {ev_oa:.4f}', flush=True)

    if not args.cache_only:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'results': results}, indent=2))
        print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
