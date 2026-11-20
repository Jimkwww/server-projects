"""Section 0 check: reproduce round-1 frozen-probe val OA from the saved weights.

Loads round-1 G_mae_seed17 encoder_final.pt + probe best.pt head, recomputes the frozen
mean-pooled val features and val OA, and compares against the round-1 recorded best_val.json.
This confirms the round-2 loading pipeline is bit-faithful before any round-2 results are built.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.model import max_abs_norm
from ssm_exp.utils import seed_all
from pretrain.model import build_encoder

ROOT = Path(__file__).resolve().parent.parent.parent
DEVICE = 'cuda'


def main():
    cfg = json.loads((ROOT / 'configs' / 'probe.json').read_text())
    mc = cfg['model']
    seed = 17
    pt = torch.load(ROOT / 'runs' / 'pretrain' / 'G_mae_seed17' / 'encoder_final.pt', map_location='cpu', weights_only=False)
    head_ck = torch.load(ROOT / 'runs' / 'probe' / 'G_mae_seed17' / 'best.pt', map_location='cpu', weights_only=False)
    recorded = json.loads((ROOT / 'runs' / 'probe' / 'G_mae_seed17' / 'best_val.json').read_text())
    rec_oa = recorded['metrics']['overall']['oa']

    seed_all(seed)
    enc = build_encoder(mc, 'G', seed).to(DEVICE)
    enc.load_state_dict(pt['encoder'])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    from torch import nn
    head = nn.Linear(mc['dim'], mc['classes']).to(DEVICE)
    head.load_state_dict(head_ck['head'])
    head.eval()

    m = read_manifest(str(ROOT / 'configs' / 'data.json'))
    val = IQDataset(m, 'val')
    ys, ps = [], []
    bs = 256
    for start in range(0, len(val), bs):
        xs, yl = [], []
        for j in range(start, min(start + bs, len(val))):
            x, y, _, _ = val[j]
            xs.append(x)
            yl.append(y)
        xb = max_abs_norm(torch.stack(xs).to(DEVICE))
        with torch.no_grad():
            logits = head(enc.features(xb))
        ps.extend(logits.argmax(-1).cpu().tolist())
        ys.extend(yl)
    oa = np.mean(np.array(ys) == np.array(ps))
    print(f'recomputed val OA = {oa:.6f}')
    print(f'recorded  val OA = {rec_oa:.6f}')
    print(f'abs diff        = {abs(oa - rec_oa):.2e}')
    ok = abs(oa - rec_oa) < 1e-6
    print('RESULT:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
