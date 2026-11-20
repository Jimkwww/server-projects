"""Round6 encoder inference latency (protocol section 9).

Measures mean-pooled encoder forward (features) wall-clock on the target device for batch 1 and
batch 128 (RML 1024-length, RadChar 512-length), warmup + CUDA sync. Uses the epoch50 encoder.

Writes round6_objective_bidirectional/reports/latency.json + latency.csv.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ssm_exp.utils import seed_all
from ssm_exp.model import max_abs_norm
from round6_objective_bidirectional.scripts.models_r6 import build_encoder_r6

R6 = ROOT / 'round6_objective_bidirectional'
R5 = ROOT / 'round5_epoch_scale'
SEEDS = (17, 42, 2026)
GROUPS = [('G', 'MAE'), ('G', 'denoise'), ('G', 'joint'), ('Q', 'MAE'),
          ('Q', 'denoise'), ('Q', 'joint'), ('Q_bi', 'MAE'),
          ('Q_bi', 'denoise'), ('Q_bi', 'joint')]
REPEATS = 50
WARMUP = 10


def ckpt_path(bb, obj, seed):
    if obj == 'MAE' and bb in ('G', 'Q'):
        return R5 / 'runs' / 'pretrain' / 'f1' / f'{bb}_seed{seed}' / 'encoder_final.pt'
    return R6 / 'runs' / 'pretrain' / 'f1' / f'{bb}_{obj.lower()}_seed{seed}' / 'encoder_final.pt'


def measure(enc, x, device):
    x = max_abs_norm(x.to(device))
    for _ in range(WARMUP):
        enc.features(x)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        enc.features(x)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / REPEATS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:1')
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    rows = []
    for bb, obj in GROUPS:
        rec = {'group': f'{bb}-{obj}', 'backbone': bb, 'objective': obj}
        for s in SEEDS:
            p = ckpt_path(bb, obj, s)
            if not p.exists():
                continue
            ck = torch.load(p, map_location='cpu', weights_only=False)
            cfg = ck['config']['model']
            seed = ck['config']['train']['seed']
            seed_all(seed)
            enc = build_encoder_r6(cfg, bb, seed).to(device)
            enc.load_state_dict(ck['encoder'])
            enc.eval()
            for length, tag in ((1024, 'rml'), (512, 'radar')):
                for batch in (1, 128):
                    x = torch.randn(batch, 2, length)
                    ms = measure(enc, x, device) * 1000.0
                    rec[f'{tag}_b{batch}_ms'] = ms
        rows.append(rec)
    (R6 / 'reports').mkdir(parents=True, exist_ok=True)
    (R6 / 'reports' / 'latency.json').write_text(json.dumps(rows, indent=2))
    if rows:
        with open(R6 / 'reports' / 'latency.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f'[latency] {len(rows)} groups measured')
    for r in rows:
        print('  ', r['group'], 'rml_b1=%.2fms rml_b128=%.2fms radar_b1=%.2fms radar_b128=%.2fms'
              % (r.get('rml_b1_ms', float('nan')), r.get('rml_b128_ms', float('nan')),
                 r.get('radar_b1_ms', float('nan')), r.get('radar_b128_ms', float('nan'))))


if __name__ == '__main__':
    main()
