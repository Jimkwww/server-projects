"""Section 2: re-evaluate G_Qlearn-scratch + G_Qlearn-{mae,denoise} finetunes from actual
best.pt checkpoints at clean / CFO +-0.005, with low/high-SNR splits and per-sample predictions.

Covers 9 runs: scratch x3 (round3), finetune x6 (round2). Also reports mean+-std aggregation
and the deltas vs G-scratch (0.4418, round-1 baseline from reports/results.json).

Outputs:
  reports/section2_finetune_cfo.json   (per-run + aggregate metrics)
  reports/section2_predictions/        (per-sample npz per run)
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pretrain.model import DownstreamModel
from pretrain.downstream import evaluate
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.engine import make_loader
from ssm_exp.model import max_abs_norm, perturb
from round2_overnight.models import build_encoder_r2

R3 = ROOT / 'round3_scale_transfer'
R2 = ROOT / 'round2_overnight'
SEEDS = (17, 42, 2026)
G_SCRATCH_OA = 0.4418  # round-1 reports/results.json mean over seeds


def load_model(run, device):
    ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    cfg = ck['config']
    encoder = build_encoder_r2(cfg['model'], 'G_Qlearn', cfg['train']['seed']).to(device)
    model = DownstreamModel(encoder, cfg['model']['classes']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, cfg, ck['epoch']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    m = read_manifest(str(ROOT / 'configs' / 'data.json'))
    vl = make_loader(IQDataset(m, 'val'), 128, args.workers)

    pred_dir = R3 / 'reports' / 'section2_predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for seed in SEEDS:
        runs[f'scratch_seed{seed}'] = R3 / 'runs' / 'scratch' / f'G_Qlearn_scratch_seed{seed}'
    for task in ('mae', 'denoise'):
        for seed in SEEDS:
            runs[f'{task}_seed{seed}'] = R2 / 'runs' / 'finetune' / f'G_Qlearn_{task}_seed{seed}'

    rows = {}
    for name, run in runs.items():
        if not (run / 'best.pt').exists():
            rows[name] = {'missing': True}
            print(f'{name} MISSING', flush=True)
            continue
        model, cfg, best_epoch = load_model(run, device)
        out = {}
        for label, freq in [('clean', 0.0), ('cfo_plus', 0.005), ('cfo_minus', -0.005)]:
            r = evaluate(model, vl, device, frequency=freq)
            out[label] = {
                'oa': r['overall']['oa'],
                'macro_f1': r['overall']['macro_f1'],
                'low_oa': r['low']['oa'] if r['low'] else None,
                'high_oa': r['high']['oa'] if r['high'] else None,
                'low_f1': r['low']['macro_f1'] if r['low'] else None,
                'high_f1': r['high']['macro_f1'] if r['high'] else None,
            }
        # per-sample predictions: re-run with save for clean + both CFO in one pass
        preds = {}
        for label, freq in [('clean', 0.0), ('cfo_plus', 0.005), ('cfo_minus', -0.005)]:
            ys, ps, zs, ids = [], [], [], []
            model.eval()
            for x, y, z, i in vl:
                x = x.to(device)
                x = max_abs_norm(x)
                if freq != 0:
                    x = perturb(x, freq)
                logits = model(x)
                ys.extend(y.cpu().tolist())
                ps.extend(logits.argmax(-1).cpu().tolist())
                zs.extend(z.tolist())
                ids.extend(i.tolist())
            preds[label] = {'y': np.array(ys), 'pred': np.array(ps),
                            'snr': np.array(zs), 'ids': np.array(ids)}
        np.savez_compressed(pred_dir / f'{name}.npz',
                            y_clean=preds['clean']['y'], pred_clean=preds['clean']['pred'],
                            snr=preds['clean']['snr'], ids=preds['clean']['ids'],
                            y_cfo_plus=preds['cfo_plus']['y'], pred_cfo_plus=preds['cfo_plus']['pred'],
                            y_cfo_minus=preds['cfo_minus']['y'], pred_cfo_minus=preds['cfo_minus']['pred'])
        rows[name] = {'best_epoch': best_epoch, 'metrics': out}
        print(f"{name} (ep{best_epoch}) clean={out['clean']['oa']:.4f} "
              f"+0.005={out['cfo_plus']['oa']:.4f} -0.005={out['cfo_minus']['oa']:.4f} "
              f"low={out['clean']['low_oa']:.4f} high={out['clean']['high_oa']:.4f}", flush=True)

    # aggregation per group
    def agg(group, key):
        vals = [rows[g]['metrics'][key]['oa'] for g in group if not rows.get(g, {}).get('missing')]
        return {'mean': float(statistics.mean(vals)),
                'std': float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                'per_seed': vals}

    groups = {
        'scratch': [f'scratch_seed{s}' for s in SEEDS],
        'mae': [f'mae_seed{s}' for s in SEEDS],
        'denoise': [f'denoise_seed{s}' for s in SEEDS],
    }
    agg_rows = {}
    for gname, group in groups.items():
        agg_rows[gname] = {k: agg(group, k) for k in ('clean', 'cfo_plus', 'cfo_minus')}

    # deltas vs G-scratch baseline (clean val OA)
    deltas = {
        'mae_finetune_minus_scratch': agg_rows['mae']['clean']['mean'] - agg_rows['scratch']['clean']['mean'],
        'denoise_finetune_minus_scratch': agg_rows['denoise']['clean']['mean'] - agg_rows['scratch']['clean']['mean'],
        'G_Qlearn_scratch_minus_G_scratch': agg_rows['scratch']['clean']['mean'] - G_SCRATCH_OA,
        'mae_pretrain_gain_over_G_Qlearn_scratch': agg_rows['mae']['clean']['mean'] - agg_rows['scratch']['clean']['mean'],
        'denoise_pretrain_gain_over_G_Qlearn_scratch': agg_rows['denoise']['clean']['mean'] - agg_rows['scratch']['clean']['mean'],
        'G_scratch_baseline': G_SCRATCH_OA,
    }

    result = {'per_run': rows, 'aggregate': agg_rows, 'deltas': deltas}
    out_path = R3 / 'reports' / 'section2_finetune_cfo.json'
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print('wrote', out_path, flush=True)


if __name__ == '__main__':
    main()
