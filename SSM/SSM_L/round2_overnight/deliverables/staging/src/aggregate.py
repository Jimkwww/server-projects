"""Round-2 aggregation: split hashes, checkpoint SHA256, key metric tables, curves.

Reads all round-2 outputs (A/B/C/D/E) and writes:
  reports/aggregate_round2.json   — nested summary of every section
  reports/split_hashes.json       — SHA256 of split index files
  reports/checkpoint_sha256.json  — SHA256 of the new checkpoints
  reports/curves/*.png            — training curves (pretrain val_loss, probe/finetune val_oa)
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
R2 = Path(__file__).resolve().parent.parent
RUNS = R2 / 'runs'
REP = R2 / 'reports'
SEEDS = (17, 42, 2026)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def split_hashes():
    out = {}
    for name in ('cls_dev_train', 'cls_dev_val', 'cls_dev_test'):
        p = ROOT / 'data' / f'{name}.npy'
        out[name] = {'file': str(p), 'sha256': sha256_file(p), 'n': int(len(np.load(p)))}
    for name in ('radchar_train', 'radchar_dev', 'radchar_eval'):
        p = R2 / 'data' / f'{name}.npy'
        out[name] = {'file': str(p), 'sha256': sha256_file(p), 'n': int(len(np.load(p)))}
    out['radchar_h5'] = {'file': '/data1/user_wzk/datasets/Radchar/radchar_tiny_rml.h5',
                         'sha256': sha256_file('/data1/user_wzk/datasets/Radchar/radchar_tiny_rml.h5')}
    return out


def ckpt_sha256():
    out = {}
    # new round-2 pretrains (encoder_final + denoise last)
    for bb in ('G_Qlearn', 'S1plain'):
        for task in ('mae', 'denoise'):
            for s in SEEDS:
                d = RUNS / 'pretrain' / f'{bb}_{task}_seed{s}'
                for fn in ('encoder_final.pt', 'last.pt'):
                    p = d / fn
                    if p.exists():
                        out[f'{bb}_{task}_seed{s}/{fn}'] = sha256_file(p)
    # random checkpoints
    for bb in ('G', 'Q', 'G_Qlearn', 'S1plain'):
        for s in SEEDS:
            p = RUNS / 'random_ckpt' / f'{bb}_seed{s}.pt'
            if p.exists():
                out[f'random_ckpt/{bb}_seed{s}.pt'] = sha256_file(p)
    return out


def best_val(run, mode='probe'):
    p = RUNS / mode / run / 'best_val.json'
    if not p.exists():
        return None
    d = json.loads(Path(p).read_text())
    return d['metrics']['overall']


def probe_oa(bb, task, s):
    d = best_val(f'{bb}_{task}_seed{s}')
    return None if d is None else d['oa']


def section_a():
    rows = {}
    for bb in ('G', 'Q'):
        for s in SEEDS:
            rows[f'{bb}_{s}'] = probe_oa(bb, 'random', s)
    return rows


def section_d():
    # round-1 baselines (from task book) + round-2 new backbones
    r1 = {'G_mae': 0.2113, 'G_denoise': 0.1635, 'Q_mae': 0.2640, 'Q_denoise': 0.2165}
    out = {'round1_probe_mean': r1}
    for bb in ('G_Qlearn', 'S1plain'):
        for task in ('mae', 'denoise'):
            oas = [probe_oa(bb, task, s) for s in SEEDS]
            out[f'{bb}_{task}_probe'] = {'per_seed': oas,
                                         'mean': float(np.mean([o for o in oas if o is not None]))
                                         if any(o is not None for o in oas) else None}
            ft = [best_val(f'{bb}_{task}_seed{s}', mode='finetune') for s in SEEDS]
            ft_oa = [None if d is None else d['oa'] for d in ft]
            ft_f1 = [None if d is None else d['macro_f1'] for d in ft]
            out[f'{bb}_{task}_finetune'] = {'per_seed_oa': ft_oa,
                                            'mean_oa': float(np.mean([o for o in ft_oa if o is not None]))
                                            if any(o is not None for o in ft_oa) else None,
                                            'mean_f1': float(np.mean([o for o in ft_f1 if o is not None]))
                                            if any(o is not None for o in ft_f1) else None}
    return out


def section_c():
    p = R2 / 'reports' / 'lowlabel_probe_round2.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    # summarize: for each (bb, pt), mean over seeds of mean over label seeds, per budget
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for key, rows in d['results'].items():
        # key format: {bb}_{pt}_seed{s}
        parts = key.split('_')
        bb, pt = parts[0], parts[1]
        per_budget = defaultdict(list)
        for r in rows:
            per_budget[r['budget']].append(r['oa'])
        for b, v in per_budget.items():
            agg[f'{bb}_{pt}'][b].append(float(np.mean(v)))
    out = {}
    for k, bd in agg.items():
        out[k] = {str(b): {'mean_oa_over_seeds': float(np.mean(v)), 'seeds': v} for b, v in bd.items()}
    return out


def section_e():
    p = R2 / 'reports' / 'radar_transfer_round2.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    out = {}
    for key, v in d['results'].items():
        out[key] = {'full_eval_oa': v['full_label']['eval']['oa'],
                    'full_dev_oa': v['full_label']['dev_best_oa'],
                    'full_eval_f1': v['full_label']['eval']['macro_f1'],
                    'low10_mean_oa': float(np.mean([r['oa'] for r in v['low_label_10_per_class']]))}
    return out


def curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir = REP / 'curves'
    outdir.mkdir(parents=True, exist_ok=True)
    # pretrain val_loss curves (round-2 new backbones)
    fig, ax = plt.subplots(figsize=(8, 5))
    for bb in ('G_Qlearn', 'S1plain'):
        for task in ('mae', 'denoise'):
            for s in SEEDS:
                h = RUNS / 'pretrain' / f'{bb}_{task}_seed{s}' / 'history.jsonl'
                if not h.exists():
                    continue
                rows = [json.loads(x) for x in h.read_text().splitlines() if x.strip()]
                ax.plot([r['epoch'] for r in rows], [r['val_loss'] for r in rows],
                        label=f'{bb}_{task}_{s}')
    ax.set_xlabel('epoch'); ax.set_ylabel('val recon loss'); ax.set_title('Round-2 pretrain val loss')
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(outdir / 'pretrain_val_loss.png', dpi=110); plt.close(fig)

    # probe/finetune val OA curves (round-2 new backbones)
    for mode in ('probe', 'finetune'):
        fig, ax = plt.subplots(figsize=(8, 5))
        for bb in ('G_Qlearn', 'S1plain'):
            for task in ('mae', 'denoise'):
                for s in SEEDS:
                    h = RUNS / mode / f'{bb}_{task}_seed{s}' / 'history.jsonl'
                    if not h.exists():
                        continue
                    rows = [json.loads(x) for x in h.read_text().splitlines() if x.strip()]
                    ys = [r['val_oa'] if 'val_oa' in r else r['val']['overall']['oa'] for r in rows]
                    ax.plot([r['epoch'] for r in rows], ys, label=f'{bb}_{task}_{s}')
        ax.set_xlabel('epoch'); ax.set_ylabel('val OA')
        ax.set_title(f'Round-2 {mode} val OA'); ax.legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(outdir / f'{mode}_val_oa.png', dpi=110); plt.close(fig)
    return str(outdir)


def main():
    payload = {
        'split_hashes': split_hashes(),
        'checkpoint_sha256': ckpt_sha256(),
        'A_random_probe': section_a(),
        'D_structure_ablation': section_d(),
        'C_lowlabel': section_c(),
        'E_radar_transfer': section_e(),
    }
    REP.mkdir(parents=True, exist_ok=True)
    (REP / 'aggregate_round2.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (REP / 'split_hashes.json').write_text(json.dumps(payload['split_hashes'], indent=2))
    (REP / 'checkpoint_sha256.json').write_text(json.dumps(payload['checkpoint_sha256'], indent=2))
    curves_dir = curves()
    print('wrote aggregate_round2.json, split_hashes.json, checkpoint_sha256.json, curves at',
          curves_dir, flush=True)


if __name__ == '__main__':
    main()
