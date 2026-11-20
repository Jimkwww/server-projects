"""Generate random-init encoder checkpoints for all 4 backbones x 3 seeds.

G/Q read the round-1 shared-init cache (frozen aux stats already fit); G_Qlearn/S1plain are
built fresh via build_encoder_r2 (G_Qlearn reuses the same-seed Q aux stats, which are
identical because both aux branches use seed+70002). Output is written in encoder_final.pt
format under round2_overnight/runs/random_ckpt/ so the round-2 downstream probe can consume it.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.utils import seed_all
from round2_overnight.common import make_random_checkpoint
from round2_overnight.models import BACKBONES

ROOT = Path(__file__).resolve().parent.parent.parent
R2 = Path(__file__).resolve().parent.parent
SEEDS = (17, 42, 2026)
SHARED_INIT = ROOT / 'runs' / 'pretrain' / 'shared_init'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', choices=BACKBONES, action='append', default=None)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    backbones = args.backbone or BACKBONES
    model_cfg = json.loads((ROOT / 'configs' / 'pretrain.json').read_text())['model']
    out_dir = R2 / 'runs' / 'random_ckpt'
    out_dir.mkdir(parents=True, exist_ok=True)
    for bb in backbones:
        for s in SEEDS:
            out = out_dir / f'{bb}_seed{s}.pt'
            if out.exists():
                print(f'[skip] {out.name}')
                continue
            make_random_checkpoint(model_cfg, bb, s, str(out), device=args.device,
                                   shared_init_dir=str(SHARED_INIT))
            print(f'[ok] {out.name}')


if __name__ == '__main__':
    main()
