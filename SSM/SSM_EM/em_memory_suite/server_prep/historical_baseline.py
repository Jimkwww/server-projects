"""Historical baseline: SSM_L G / Q encoders (round5 F1-MAE checkpoints) evaluated on this
package's data + simple linear head, so the comparison is on the SAME task head and SAME data
splits as the new reference models.

Protocol difference vs this package (recorded, not silently papered over):
  - Historical encoders take [B,2,L] with max_abs_norm normalization (joint max abs component);
    this package uses raw [B,L,2] without per-sample normalization. We adapt input layout +
    normalization here and record it.
  - Historical dim=176/depth=6 (vs new dim=64/depth=3); parameter counts are NOT matched
    (reported separately). This is the historical configuration, not a parameter-matched control.
  - Historical G uses sinusoidal absolute positions; Q uses none. Both handle L=512 and L=1024.

Modes: probe (freeze encoder, train linear head 50ep) and finetune (train all, 50ep), matching
this package's downstream protocol (AdamW lr 3e-4, cosine). Reports test OA/macro-F1 per seed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
SSML = Path('/home/user_wzk/wzk/projects/SSM/SSM_L')
sys.path.insert(0, str(SSML))
sys.path.insert(0, str(ROOT))

from ssm_exp.model import max_abs_norm
from pretrain.model import Encoder as HistEncoder, build_encoder
from emmem.data import IQDataset
from emmem.metrics import classification

CFG = {'patch': 8, 'dim': 176, 'depth': 6, 'heads': 4, 'state': 128, 'selector_hidden': 64,
       'classes': 24, 'input_length': 1024, 'ffn_ratio': 4, 'dropout': 0.0, 'position': 'none',
       'direction': 'forward', 'scan': 'parallel', 'ssm_gain': 1.0}

CKPT = {
    'G': SSML / 'round5_epoch_scale/runs/pretrain/f1/G_seed{seed}/encoder_final.pt',
    'Q': SSML / 'round5_epoch_scale/runs/pretrain/f1/Q_seed{seed}/encoder_final.pt',
}


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_hist(backbone, seed, device):
    ck = torch.load(str(CKPT[backbone]).format(seed=seed), map_location='cpu', weights_only=False)
    enc = HistEncoder(CFG, backbone, seed)
    enc.load_state_dict(ck['encoder'], strict=True)
    return enc.to(device)


@torch.no_grad()
def extract(enc, x, device):
    x = max_abs_norm(x.permute(0, 2, 1).to(device))  # [B,L,2] -> [B,2,L] normalized
    return enc.forward_tokens(x).mean(1)  # [B,dim]


def run_one(backbone, seed, manifest, task, mode, epochs, device, out_dir):
    seed_all(seed)
    enc = build_hist(backbone, seed, device)
    meta = json.loads(Path(manifest).read_text())
    spec = meta['tasks'][task]
    head = nn.Linear(CFG['dim'], spec['outputs']).to(device)
    if mode == 'probe':
        for p in enc.parameters():
            p.requires_grad_(False)

    def feat_loader(split):
        ds = IQDataset(manifest, split, task)
        return DataLoader(ds, batch_size=128, shuffle=(split == 'train'), num_workers=4, pin_memory=True)

    if mode == 'probe':
        # Precompute frozen features once, then train the linear head on cached features.
        def cache(split):
            xs, ys = [], []
            for x, y, _ in feat_loader(split):
                xs.append(extract(enc, x, device).cpu())
                ys.append(y.numpy())
            return torch.cat(xs), np.concatenate(ys)
        Xtr, Ytr = cache('train')
        Xva, Yva = cache('val')
        Xte, Yte = cache('test')
        Xtr, Xva, Xte = Xtr.to(device), Xva.to(device), Xte.to(device)
        Ytr_t = torch.from_numpy(Ytr).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        n_tr = len(Xtr)
        best = -1.0
        for epoch in range(epochs):
            head.train()
            perm = torch.randperm(n_tr, device=device)
            for i in range(0, n_tr, 128):
                idx = perm[i:i + 128]
                logits = head(Xtr[idx])
                loss = torch.nn.functional.cross_entropy(logits, Ytr_t[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
            head.eval()
            v = classification(Yva, head(Xva).argmax(-1).cpu().numpy(), spec['outputs'])
            if v['macro_f1'] > best:
                best = v['macro_f1']
        head.eval()
        result = classification(Yte, head(Xte).argmax(-1).cpu().numpy(), spec['outputs'])
    else:
        params = list(enc.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loader = feat_loader('train')
        best = -1.0
        for epoch in range(epochs):
            enc.train()
            for x, y, _ in loader:
                x = max_abs_norm(x.permute(0, 2, 1).to(device))
                y = y.to(device)
                logits = head(enc.forward_tokens(x).mean(1))
                loss = torch.nn.functional.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            sched.step()
            enc.eval()
            ys, ps = [], []
            with torch.no_grad():
                for x, y, _ in feat_loader('val'):
                    ps.append(head(extract(enc, x, device)).argmax(-1).cpu().numpy())
                    ys.append(y.numpy())
            v = classification(np.concatenate(ys), np.concatenate(ps), spec['outputs'])
            if v['macro_f1'] > best:
                best = v['macro_f1']
        ys, ps = [], []
        with torch.no_grad():
            for x, y, _ in feat_loader('test'):
                ps.append(head(extract(enc, x, device)).argmax(-1).cpu().numpy())
                ys.append(y.numpy())
        result = classification(np.concatenate(ys), np.concatenate(ps), spec['outputs'])
    result['backbone'] = backbone
    result['seed'] = seed
    result['task'] = task
    result['mode'] = mode
    result['params'] = sum(p.numel() * (2 if p.is_complex() else 1) for p in enc.parameters())
    out = Path(out_dir) / f'{backbone}_s{seed}_{task}_{mode}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--backbone', choices=['G', 'Q'], required=True)
    ap.add_argument('--seeds', default='17,42,2026')
    ap.add_argument('--modes', default='probe,finetune')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    modes = a.modes.split(',')
    for s in seeds:
        for m in modes:
            r = run_one(a.backbone, s, a.manifest, a.task, m, a.epochs, a.device, a.out)
            print(json.dumps({k: r[k] for k in ['backbone', 'seed', 'task', 'mode', 'oa', 'macro_f1']}))


if __name__ == '__main__':
    main()
