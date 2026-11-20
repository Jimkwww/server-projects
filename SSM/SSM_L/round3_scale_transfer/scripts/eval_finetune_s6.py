"""Section 6 downstream part 3: re-evaluate the 18 F1/F2 finetunes at clean / CFO +-0.005 with
low/high-SNR splits + per-sample predictions. Aggregates mean+-std and reports deltas vs the
F0 (round-1/round-2) MAE finetune baselines.
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
R3 = ROOT / 'round3_scale_transfer'
R2 = ROOT / 'round2_overnight'
from pretrain.model import DownstreamModel
from pretrain.downstream import evaluate
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.engine import make_loader
from round2_overnight.models import build_encoder_r2

MODELS = ('G', 'G_Qlearn', 'Q')
CORPORA = ('f1', 'f2')
SEEDS = (17, 42, 2026)
# F0 MAE finetune clean-val OA baselines (round-1/round-2, section-2 protocol)
F0_MAE_FINETUNE = {'G': 0.4834, 'G_Qlearn': 0.5347, 'Q': 0.4862}


def load_model(run, backbone, seed, device):
    ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    cfg = ck['config']
    encoder = build_encoder_r2(cfg['model'], backbone, seed).to(device)
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

    pred_dir = R3 / 'reports' / 'section6_predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = {}
    for corp in CORPORA:
        for model in MODELS:
            for seed in SEEDS:
                name = f'{corp}_{model}_seed{seed}'
                run = R3 / 'runs' / 'finetune_s6' / name
                if not (run / 'best.pt').exists():
                    rows[name] = {'missing': True}
                    print(f'{name} MISSING', flush=True)
                    continue
                model_, cfg, best_epoch = load_model(run, model, seed, device)
                out = {}
                preds = {}
                for label, freq in [('clean', 0.0), ('cfo_plus', 0.005), ('cfo_minus', -0.005)]:
                    r = evaluate(model_, vl, device, frequency=freq)
                    out[label] = {'oa': r['overall']['oa'], 'macro_f1': r['overall']['macro_f1'],
                                  'low_oa': r['low']['oa'] if r['low'] else None,
                                  'high_oa': r['high']['oa'] if r['high'] else None}
                # per-sample predictions for the three conditions
                from ssm_exp.model import max_abs_norm, perturb
                for label, freq in [('clean', 0.0), ('cfo_plus', 0.005), ('cfo_minus', -0.005)]:
                    ys, ps, zs, ids = [], [], [], []
                    model_.eval()
                    for x, y, z, i in vl:
                        x = max_abs_norm(x.to(device))
                        if freq != 0:
                            x = perturb(x, freq)
                        logits = model_(x)
                        ys.extend(y.cpu().tolist()); ps.extend(logits.argmax(-1).cpu().tolist())
                        zs.extend(z.tolist()); ids.extend(i.tolist())
                    preds[label] = (np.array(ys), np.array(ps), np.array(zs), np.array(ids))
                np.savez_compressed(pred_dir / f'{name}.npz',
                                    y_clean=preds['clean'][0], pred_clean=preds['clean'][1],
                                    snr=preds['clean'][2], ids=preds['clean'][3],
                                    y_plus=preds['cfo_plus'][0], pred_plus=preds['cfo_plus'][1],
                                    y_minus=preds['cfo_minus'][0], pred_minus=preds['cfo_minus'][1])
                rows[name] = {'best_epoch': best_epoch, 'metrics': out}
                print(f"{name} (ep{best_epoch}) clean={out['clean']['oa']:.4f} "
                      f"+0.005={out['cfo_plus']['oa']:.4f} -0.005={out['cfo_minus']['oa']:.4f}",
                      flush=True)

    # aggregate
    def agg(keys, cond):
        vals = [rows[k]['metrics'][cond]['oa'] for k in keys if not rows.get(k, {}).get('missing')]
        return {'mean': float(statistics.mean(vals)),
                'std': float(statistics.stdev(vals)) if len(vals) > 1 else 0.0}

    agg_rows = {}
    for corp in CORPORA:
        for model in MODELS:
            keys = [f'{corp}_{model}_seed{s}' for s in SEEDS]
            agg_rows[f'{corp}_{model}'] = {c: agg(keys, c) for c in ('clean', 'cfo_plus', 'cfo_minus')}

    deltas = {}
    for corp in CORPORA:
        for model in MODELS:
            deltas[f'{corp}_{model}_minus_F0_mae'] = \
                agg_rows[f'{corp}_{model}']['clean']['mean'] - F0_MAE_FINETUNE[model]

    out = R3 / 'reports' / 'section6_finetune_cfo.json'
    out.write_text(json.dumps({'per_run': rows, 'aggregate': agg_rows,
                               'deltas': deltas, 'F0_mae_finetune_baseline': F0_MAE_FINETUNE},
                              indent=2))
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
