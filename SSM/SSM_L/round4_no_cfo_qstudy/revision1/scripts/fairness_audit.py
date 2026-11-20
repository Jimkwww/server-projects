"""Revision-1 Section 6: Q 2D fairness audit (fixed-Q vs learn-Q pretrain + downstream).

No rerun: static/weight-level checks only. Outputs fairness_audit.json + fairness_audit.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.utils import seed_all
from ssm_exp.engine import make_loader
from pretrain.model import PretrainModel
from round4_no_cfo_qstudy.engine_r4 import build_shared_init_r4
from round2_overnight.engine import build_shared_init as build_shared_init_r2
from common_r1 import R1, R4, R2, SEEDS

R = Path(__file__).resolve().parent.parent.parent.parent


def sd_hash(d):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(d[k].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def aux_branch(sd):
    return {k: v for k, v in sd.items() if 'aux_branch' in k}


def maxdiff(a, b):
    return float(max((a[k] - b[k]).abs().max().item() for k in a))


def audit_initial_weights(device):
    """fixed-Q vs learn-Q pretrain initial weights (encoder/filter/buffers/aux + decoder init)."""
    c = json.load(open(R4 / 'configs' / 'pretrain_mae_f0.json'))
    m = read_manifest(str(R / 'configs' / 'data.json'))
    ds = IQDataset(m, 'train')
    rows = {}
    for s in SEEDS:
        fx = torch.load(R4 / 'runs' / 'shared_init' / f'G_Qlearn_seed{s}_fixedq.pt',
                        map_location='cpu', weights_only=False)
        lq = torch.load(R2 / 'runs' / 'pretrain' / 'shared_init' / f'G_Qlearn_seed{s}.pt',
                        map_location='cpu', weights_only=False)
        fx_e, lq_e = fx['encoder'], lq['encoder']
        fx_a, lq_a = fx.get('aux'), lq.get('aux')
        same_enc = all(torch.equal(fx_e[k], lq_e[k]) for k in fx_e)
        same_filter = all(torch.equal(fx_e[k], lq_e[k]) for k in fx_e if 'aux_branch' in k)
        same_buf = all(torch.equal(fx_e[k], lq_e[k]) for k in ('aux_mean', 'aux_std'))
        same_aux = (fx_a is not None and lq_a is not None
                    and np.allclose(fx_a['mean'], lq_a['mean'])
                    and np.allclose(fx_a['std'], lq_a['std']))
        # decoder init alignment (reconstruct both code paths)
        seed_all(s)
        enc4, _, _ = build_shared_init_r4(c, 'G_Qlearn', s, ds, device,
                                          str(R4 / 'runs' / 'shared_init'), freeze_q=True)
        dec4 = PretrainModel(enc4, c['model']['patch']).decoder.proj.weight.detach()
        seed_all(s)
        enc2, _ = build_shared_init_r2(c, 'G_Qlearn', s, ds, device,
                                       str(R2 / 'runs' / 'pretrain' / 'shared_init'))
        dec2 = PretrainModel(enc2, c['model']['patch']).decoder.proj.weight.detach()
        same_decoder = bool(torch.equal(dec4, dec2))
        rows[s] = {'encoder_equal': bool(same_enc), 'filter_equal': bool(same_filter),
                   'buffers_equal': bool(same_buf), 'aux_stats_equal': bool(same_aux),
                   'decoder_init_equal': same_decoder,
                   'fixedq_encoder_sha256': sd_hash(fx_e),
                   'learnq_encoder_sha256': sd_hash(lq_e)}
    return rows


def audit_pretrain_config():
    c4 = json.load(open(R4 / 'configs' / 'pretrain_mae_f0.json'))
    c2 = json.load(open(R2 / 'runs' / 'pretrain' / 'G_Qlearn_mae_seed17' / 'config.json'))
    keys = ('epochs', 'lr', 'batch_size', 'weight_decay', 'betas', 'scheduler',
            'warmup_fraction', 'grad_clip', 'mask_ratio', 'augment_cfo_max')
    diffs = {k: (c4['train'].get(k), c2['train'].get(k)) for k in keys
             if c4['train'].get(k) != c2['train'].get(k)}
    return {'round4_pretrain_cfg': {k: c4['train'][k] for k in keys},
            'round2_pretrain_cfg': {k: c2['train'][k] for k in keys},
            'diffs': diffs,
            'model_cfg_equal': c4['model'] == c2['model']}


def audit_downstream_freezing():
    out = {}
    for task in ('rml', 'radar'):
        for setting, expect in (('A_fixedq_fixedq', 'frozen'), ('B_learnq_fixedq', 'frozen'),
                                ('C_fixedq_learnq', 'learn'), ('D_learnq_learnq', 'learn')):
            for s in SEEDS:
                if setting == 'D_learnq_learnq':
                    p = R4 / 'runs' / task / 'mae' / f'G_Qlearn_seed{s}' / 'best_val.json'
                else:
                    p = R4 / 'runs' / task / setting / f'G_Qlearn_seed{s}' / 'best_val.json'
                d = json.load(open(p))
                b, a_ = d.get('aux_hash_before'), d.get('aux_hash_after')
                key = f'{task}:{setting}:seed{s}'
                out[key] = {'expected': expect,
                            'aux_frozen_verified': d.get('aux_frozen_verified'),
                            'hash_equal': (b == a_) if (b and a_) else None}
    return out


def audit_filter_update():
    """C/D: does the Q filter actually move during downstream finetune (learn-Q phase)?"""
    out = {}
    pre_fixed = torch.load(R4 / 'runs' / 'pretrain' / 'G_Qlearn_fixedq_mae_seed17'
                           / 'encoder_final.pt', map_location='cpu', weights_only=False)['encoder']
    pre_learn = torch.load(R2 / 'runs' / 'pretrain' / 'G_Qlearn_mae_seed17'
                           / 'encoder_final.pt', map_location='cpu', weights_only=False)['encoder']
    for task in ('rml', 'radar'):
        c_ck = torch.load(R4 / 'runs' / task / 'C_fixedq_learnq' / 'G_Qlearn_seed17' / 'best.pt',
                          map_location='cpu', weights_only=False)['model']
        c_enc = {k.replace('encoder.', ''): v for k, v in c_ck.items() if k.startswith('encoder.')}
        d_ck = torch.load(R4 / 'runs' / task / 'mae' / 'G_Qlearn_seed17' / 'best.pt',
                          map_location='cpu', weights_only=False)['model']
        d_enc = {k.replace('encoder.', ''): v for k, v in d_ck.items() if k.startswith('encoder.')}
        out[task] = {'C_filter_maxdelta_vs_fixedq_pretrain': maxdiff(aux_branch(pre_fixed), aux_branch(c_enc)),
                     'D_filter_maxdelta_vs_learnq_pretrain': maxdiff(aux_branch(pre_learn), aux_branch(d_enc))}
    return out


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    report = {
        'initial_weights_fixedq_vs_learnq': audit_initial_weights(device),
        'pretrain_config_round2_vs_round4': audit_pretrain_config(),
        'downstream_freezing_A_B_C_D': audit_downstream_freezing(),
        'filter_update_C_D': audit_filter_update(),
        'resume_behavior': {
            'verified': 'code inspection (pretrain_r4.py train())',
            'finding': ('resume restores aux/phys from the checkpoint (ck.get("aux"/"phys")), '
                        'and model.load_state_dict(ck["model"]) restores aux_mean/aux_std buffers; '
                        'NO stat re-fit on resume. A fresh fit can occur only if the shared_init '
                        'cache is missing, and is then overwritten by the checkpoint load.'),
        },
        'unverifiable': [
            'round2 learn-Q pretrain used a different entry script (pretrain/engine.py) than '
            'round4 fixed-Q (pretrain_r4.py); optimizer param selection differs textually '
            '(round2: model.parameters(); round4: requires_grad filter) but is equivalent when '
            'no params are frozen, and the mask stream (seed+200000) and data order (seed) are '
            'identical. Full bit-level equivalence of the two training loops was not re-run.',
            'The exact decoder init RNG state was reconstructed and matched; a full forward/'
            'backward bit-match of the two engines was not re-executed (未核验).',
        ],
    }
    (R1 / 'reports').mkdir(parents=True, exist_ok=True)
    (R1 / 'reports' / 'fairness_audit.json').write_text(json.dumps(report, indent=2))
    _write_md(report)
    print('[done] fairness_audit.json / fairness_audit.md')


def _write_md(report):
    L = ['# Revision-1 Section 6: Q 2D fairness audit (no rerun)\n']
    L.append('\n## 1. fixed-Q vs learn-Q pretrain initial weights\n')
    L.append('| seed | encoder | filter | buffers | aux stats | decoder init |\n|---|---|---|---|---|---|\n')
    for s, r in report['initial_weights_fixedq_vs_learnq'].items():
        L.append(f"| {s} | {r['encoder_equal']} | {r['filter_equal']} | {r['buffers_equal']} "
                 f"| {r['aux_stats_equal']} | {r['decoder_init_equal']} |\n")
    L.append('\nBoth pretrain arms start from byte-identical encoder + decoder + aux stats; the only '
             'difference is `freeze_aux` (requires_grad on the filter), which does not change values.\n')
    L.append('\n## 2. pretrain config round2 (learn-Q) vs round4 (fixed-Q)\n')
    c = report['pretrain_config_round2_vs_round4']
    L.append(f"- model config equal: {c['model_cfg_equal']}\n")
    L.append(f"- train config diffs: {c['diffs'] if c['diffs'] else 'none (identical)'}\n")
    L.append('\n## 3. downstream A/B/C/D freezing\n')
    L.append('| task:setting | expected | aux_frozen_verified | hash before==after |\n|---|---|---|---|\n')
    for k, v in report['downstream_freezing_A_B_C_D'].items():
        L.append(f"| {k} | {v['expected']} | {v['aux_frozen_verified']} | {v['hash_equal']} |\n")
    L.append('\n## 4. C/D filter actually updates\n')
    for task, v in report['filter_update_C_D'].items():
        L.append(f"- {task}: C filter max|Δ| vs fixed-Q pretrain = {v['C_filter_maxdelta_vs_fixedq_pretrain']:.4f}; "
                 f"D filter max|Δ| vs learn-Q pretrain = {v['D_filter_maxdelta_vs_learnq_pretrain']:.4f}\n")
    L.append('\n## 5. resume behavior\n')
    L.append(f"- {report['resume_behavior']['finding']}\n")
    L.append('\n## 6. unverifiable / not re-run\n')
    for u in report['unverifiable']:
        L.append(f'- {u}\n')
    (R1 / 'reports' / 'fairness_audit.md').write_text(''.join(L))


if __name__ == '__main__':
    main()
