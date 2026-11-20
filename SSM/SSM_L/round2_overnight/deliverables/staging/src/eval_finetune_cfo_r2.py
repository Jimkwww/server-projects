"""D. round-2 finetune CFO eval (+-0.005, round-1 protocol).

Loads each round-2 finetune best.pt (clean-val selected), evaluates on RML val at clean /
CFO +0.005 / CFO -0.005 cycles/sample, writes reports/finetune_cfo_round2.json.
Covers G_Qlearn / S1plain x {mae, denoise} x 3 seeds (12 runs).
"""
import argparse
import json
from pathlib import Path

import torch

from pretrain.model import DownstreamModel
from pretrain.downstream import evaluate
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.engine import make_loader
from round2_overnight.models import build_encoder_r2

ROOT = Path(__file__).resolve().parent.parent.parent
R2 = Path(__file__).resolve().parent.parent
BACKBONES = ('G_Qlearn', 'S1plain')
TASKS = ('mae', 'denoise')
SEEDS = (17, 42, 2026)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    m = read_manifest(str(ROOT / 'configs' / 'data.json'))
    vl = make_loader(IQDataset(m, 'val'), 128, args.workers)

    rows = {}
    for bb in BACKBONES:
        for task in TASKS:
            for seed in SEEDS:
                run = R2 / 'runs' / 'finetune' / f'{bb}_{task}_seed{seed}'
                if not (run / 'best.pt').exists():
                    rows[f'{bb}_{task}_seed{seed}'] = {'missing': True}
                    print(f'{bb}_{task}_seed{seed} MISSING', flush=True)
                    continue
                ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
                cfg = ck['config']
                encoder = build_encoder_r2(cfg['model'], bb, cfg['train']['seed']).to(device)
                model = DownstreamModel(encoder, cfg['model']['classes']).to(device)
                model.load_state_dict(ck['model'])
                model.eval()
                out = {}
                for name, freq in [('clean', 0.0), ('cfo_plus', 0.005), ('cfo_minus', -0.005)]:
                    r = evaluate(model, vl, device, frequency=freq)
                    out[name] = {'oa': r['overall']['oa'], 'macro_f1': r['overall']['macro_f1']}
                rows[f'{bb}_{task}_seed{seed}'] = {'best_epoch': ck['epoch'], 'metrics': out}
                print(f'{bb}_{task}_seed{seed} clean={out["clean"]["oa"]:.4f} '
                      f'+0.005={out["cfo_plus"]["oa"]:.4f} -0.005={out["cfo_minus"]["oa"]:.4f}',
                      flush=True)

    outdir = R2 / 'reports'
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'finetune_cfo_round2.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print('wrote', outdir / 'finetune_cfo_round2.json', flush=True)


if __name__ == '__main__':
    main()
