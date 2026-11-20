"""SSM_L round-1 secondary metric: finetune clean-selected weights under CFO +-0.005.

Loads each finetune best.pt (selected on clean val OA), evaluates on the val split at
clean / CFO +0.005 / CFO -0.005 cycles per sample (positive and negative reported
separately), and writes reports/finetune_cfo.json + reports/finetune_cfo.md.
"""
import argparse
import json
from pathlib import Path

import torch

from pretrain.model import build_encoder, DownstreamModel
from pretrain.downstream import evaluate
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.engine import make_loader

ROOT = Path(__file__).resolve().parent.parent
GROUPS = ['G_mae', 'G_denoise', 'Q_mae', 'Q_denoise']
SEEDS = [17, 42, 2026]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    device = torch.device(args.device)
    m = read_manifest(str(ROOT / 'configs' / 'data.json'))
    val = IQDataset(m, 'val')
    vl = make_loader(val, 128, args.workers)

    rows = {}
    for group in GROUPS:
        for seed in SEEDS:
            run = ROOT / 'runs' / 'finetune' / f'{group}_seed{seed}'
            ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
            cfg = ck['config']
            backbone = 'G' if group.startswith('G') else 'Q'
            encoder = build_encoder(cfg['model'], backbone, cfg['train']['seed']).to(device)
            model = DownstreamModel(encoder, cfg['model']['classes']).to(device)
            model.load_state_dict(ck['model'])
            model.eval()
            out = {}
            for name, freq in [('clean', 0.0), ('cfo_plus', 0.005), ('cfo_minus', -0.005)]:
                r = evaluate(model, vl, device, frequency=freq)
                out[name] = {'oa': r['overall']['oa'], 'macro_f1': r['overall']['macro_f1'],
                             'low_oa': r['low']['oa'] if r.get('low') else None,
                             'high_oa': r['high']['oa'] if r.get('high') else None}
            rows[f'{group}_seed{seed}'] = {'best_epoch': ck['epoch'], 'metrics': out}
            print(f'{group}_seed{seed} best_epoch={ck["epoch"]} '
                  f'clean={out["clean"]["oa"]:.4f} +0.005={out["cfo_plus"]["oa"]:.4f} '
                  f'-0.005={out["cfo_minus"]["oa"]:.4f}', flush=True)

    outdir = ROOT / 'reports'
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'finetune_cfo.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    md = ['# finetune clean-selected weights under CFO ±0.005 (val OA %)', '']
    md.append('| run | clean | +0.005 | −0.005 | best_epoch |')
    md.append('|---|---|---|---|---|')
    for k, v in rows.items():
        md.append(f"| {k} | {100*v['metrics']['clean']['oa']:.2f} | "
                  f"{100*v['metrics']['cfo_plus']['oa']:.2f} | "
                  f"{100*v['metrics']['cfo_minus']['oa']:.2f} | {v['best_epoch']} |")
    (outdir / 'finetune_cfo.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {outdir}/finetune_cfo.json', flush=True)


if __name__ == '__main__':
    main()
