"""Section 4: radar transfer with the UNIFIED readout (StandardScaler + L2 LR, C=1, lbfgs).

Extends round-2 run_e (G/Q x {random,mae,denoise} x 3 seeds, AdamW probe) to G_Qlearn/S1plain,
and switches the primary readout to the unified LR protocol used in section 3 (radar-train
scaler fit). Round-2 AdamW remains as a separate reference; we also compute AdamW for the 18 new
encoders so the reference covers all 4 structures.

For each of 36 encoders: cache radar train/dev/eval features (resume-safe), unified LR full-label
eval (OA / macro-F1 / per-class recall / low-high SNR), save eval record IDs/labels/preds/SNR +
encoder weight hash. Per-seed paired deltas computed at the end.

Note (labeling): RadChar-Tiny has no official split; prep_radchar.py created a stratified
record-level random split (60/20/20). This is therefore an intra-dataset record-level split
transfer evaluation, NOT a cross-simulation (held-out generator params) generalization test.

Usage:
  python section4_radar.py --cache-only --backbones G_Qlearn S1plain --device cuda:0
  python section4_radar.py --readout-only
"""
import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
R2 = ROOT / 'round2_overnight'
R3 = ROOT / 'round3_scale_transfer'
RADAR_FEAT2 = R2 / 'runs' / 'radar_features'
RADAR_FEAT3 = R3 / 'runs' / 'radar_features'
SEEDS = (17, 42, 2026)
BACKBONES = ('G', 'Q', 'G_Qlearn', 'S1plain')
PRETRAINS = ('random', 'mae', 'denoise')
CLASSES = 5


def model_cfg():
    return json.loads((ROOT / 'configs' / 'pretrain.json').read_text())['model']


def ckpt_for(bb, pt, seed):
    if pt == 'random':
        return R2 / 'runs' / 'random_ckpt' / f'{bb}_seed{seed}.pt'
    if bb in ('G', 'Q'):
        return ROOT / 'runs' / 'pretrain' / f'{bb}_{pt}_seed{seed}' / 'encoder_final.pt'
    return R2 / 'runs' / 'pretrain' / f'{bb}_{pt}_seed{seed}' / 'encoder_final.pt'


def feat_path(bb, pt, seed):
    if bb in ('G', 'Q'):
        return RADAR_FEAT2 / f'{bb}_{pt}_seed{seed}.pt'
    return RADAR_FEAT3 / f'{bb}_{pt}_seed{seed}.pt'


def cache_radar(bb, pt, seed, device):
    from ssm_exp.data import IQDataset, read_manifest
    from round2_overnight.common import load_encoder, features_for_encoder

    out = feat_path(bb, pt, seed)
    if out.exists():
        return out, 'skip'
    ck = ckpt_for(bb, pt, seed)
    if not ck.exists():
        return out, f'missing-ckpt:{ck.name}'
    enc = load_encoder(model_cfg(), bb, seed, str(ck), device)
    m = read_manifest(str(R2 / 'configs' / 'radchar.json'))
    payload = {}
    for split in ('train', 'val', 'test'):
        ds = IQDataset(m, split)
        feats, ys, zs, ids = features_for_encoder(enc, ds, 128, device)
        payload[split] = {'feats': feats, 'y': ys, 'snr': zs, 'idx': ids}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out, 'ok'


def encoder_sd_hash(bb, pt, seed):
    ck = ckpt_for(bb, pt, seed)
    ptf = torch.load(ck, map_location='cpu', weights_only=False)
    h = hashlib.sha256()
    for k in sorted(ptf['encoder']):
        h.update(k.encode())
        h.update(ptf['encoder'][k].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def lr_full(feat):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    Xtr = feat['train']['feats'].numpy().astype(np.float32)
    ytr = feat['train']['y'].numpy().astype(np.int64)
    Xev = feat['test']['feats'].numpy().astype(np.float32)
    yev = feat['test']['y'].numpy().astype(np.int64)
    zev = feat['test']['snr'].numpy()
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xev))
    lo, hi = zev <= 0, zev > 0
    return {
        'oa': float(accuracy_score(yev, pred)),
        'macro_f1': float(f1_score(yev, pred, average='macro')),
        'recall': recall_score(yev, pred, average=None).tolist(),
        'n': int(len(yev)),
        'oa_low_snr': float(accuracy_score(yev[lo], pred[lo])) if lo.any() else None,
        'oa_high_snr': float(accuracy_score(yev[hi], pred[hi])) if hi.any() else None,
    }, pred


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
    for _ in range(epochs):
        head.train()
        for f, y in tr_dl:
            logits = head(f)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
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
    return best, ev_oa, ev_pred.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--readout-only', action='store_true')
    ap.add_argument('--backbones', nargs='+', default=list(BACKBONES))
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--skip-adamw', action='store_true')
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.readout_only:
        pred_dir = R3 / 'runs' / 'radar_eval_preds'
        pred_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for bb in BACKBONES:
            for pt in PRETRAINS:
                for s in SEEDS:
                    key = f'{bb}_{pt}_seed{s}'
                    f = feat_path(bb, pt, s)
                    if not f.exists():
                        results[key] = {'missing': True}
                        print(f'{key} MISSING', flush=True)
                        continue
                    feat = torch.load(f, map_location='cpu', weights_only=False)
                    lr_m, lr_pred = lr_full(feat)
                    wh = encoder_sd_hash(bb, pt, s)
                    np.savez_compressed(pred_dir / f'{key}_lr.npz',
                                        y=feat['test']['y'].numpy(), pred=lr_pred,
                                        snr=feat['test']['snr'].numpy(),
                                        idx=feat['test']['idx'].numpy())
                    row = {'full_label_lr': lr_m, 'encoder_weight_sha256': wh}
                    if not args.skip_adamw:
                        dev_best, ev_oa, ev_pred = adamw_probe(feat, s, device=device)
                        np.savez_compressed(pred_dir / f'{key}_adamw.npz',
                                            y=feat['test']['y'].numpy(), pred=ev_pred,
                                            snr=feat['test']['snr'].numpy(),
                                            idx=feat['test']['idx'].numpy())
                        row['adamw_reference'] = {'dev_best_oa': float(dev_best),
                                                  'eval_oa': float(ev_oa)}
                    results[key] = row
                    print(f'{key}: LR OA {lr_m["oa"]:.4f} f1 {lr_m["macro_f1"]:.4f} '
                          f'low {lr_m["oa_low_snr"]} high {lr_m["oa_high_snr"]}', flush=True)
        out = R3 / 'reports' / 'section4_radar_transfer.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            'protocol': 'StandardScaler(radar-train) + L2 LR(C=1, lbfgs); AdamW kept as reference',
            'labeling': '同数据集内记录级随机划分的迁移评价（RadChar 无官方划分，60/20/20 分层）',
            'results': results}, indent=2))
        print('wrote', out, flush=True)
        return

    specs = [(bb, pt, s) for bb in args.backbones for pt in PRETRAINS for s in args.seeds]
    for i, (bb, pt, s) in enumerate(specs):
        f, status = cache_radar(bb, pt, s, device)
        print(f'[{i+1}/{len(specs)}] {bb}_{pt}_seed{s}: {status}', flush=True)


if __name__ == '__main__':
    main()
