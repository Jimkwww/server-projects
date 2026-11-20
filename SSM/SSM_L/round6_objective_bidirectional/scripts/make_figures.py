"""Round6 figures: (1) pretrain recon loss curves for the 7 groups; (2) main-table bar chart.

Reads history.jsonl (pretrain) and reports/main_table.csv; writes round6_objective_bidirectional/figs/*.png.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
R6 = ROOT / 'round6_objective_bidirectional'
R5 = ROOT / 'round5_epoch_scale'
FIGS = R6 / 'figs'
SEEDS = (17, 42, 2026)

# 9 groups: (backbone, objective, color, linestyle)
GROUPS = [
    ('G', 'MAE', 'tab:blue', ':'),
    ('G', 'denoise', 'tab:blue', '-'),
    ('G', 'joint', 'tab:blue', '--'),
    ('Q', 'MAE', 'tab:green', ':'),
    ('Q', 'denoise', 'tab:green', '-'),
    ('Q', 'joint', 'tab:green', '--'),
    ('Q_bi', 'MAE', 'tab:purple', '-'),
    ('Q_bi', 'denoise', 'tab:purple', '-'),
    ('Q_bi', 'joint', 'tab:purple', '--'),
]


def hist_path(bb, obj, seed):
    if obj == 'MAE' and bb in ('G', 'Q'):
        return R5 / 'runs' / 'pretrain' / 'f1' / f'{bb}_seed{seed}' / 'history.jsonl'
    return R6 / 'runs' / 'pretrain' / 'f1' / f'{bb}_{obj.lower()}_seed{seed}' / 'history.jsonl'


def group_label(bb, obj):
    return {'MAE': f'{bb}-MAE', 'denoise': f'{bb}-denoise', 'joint': f'{bb}-joint'}[obj] if obj != 'MAE' else f'{bb}-MAE'


def training_curves():
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    for bb, obj, color, ls in GROUPS:
        for si, s in enumerate(SEEDS):
            p = hist_path(bb, obj, s)
            if not p.exists():
                continue
            rows = [json.loads(l) for l in p.read_text().splitlines()]
            steps = [r['step'] for r in rows]
            tr = [r['train_loss'] for r in rows]
            vl = [r['val_loss'] for r in rows]
            lbl = group_label(bb, obj) if si == 0 else None
            ax.plot(steps, tr, color=color, ls=ls, alpha=0.3, lw=0.8, label=lbl)
            ax.plot(steps, vl, color=color, ls=ls, alpha=0.9, lw=1.4,
                    label=f'{lbl} val' if si == 0 else None)
    ax.set_title('Round6 pretrain recon loss (solid=train, dashed=val; joint total = weighted)')
    ax.set_xlabel('optimizer step')
    ax.set_ylabel('recon loss')
    ax.legend(fontsize=7, ncol=2)
    for x in (7820, 15640, 23460, 31280, 39100):
        ax.axvline(x, color='gray', alpha=0.2, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIGS / 'training_curves.png', dpi=110)
    plt.close(fig)
    print('[fig] training_curves.png')


def training_curves_by_objective():
    FIGS.mkdir(parents=True, exist_ok=True)
    objectives = {
        'MAE': [('G', 'tab:blue'), ('Q', 'tab:green'), ('Q_bi', 'tab:purple')],
        'denoise': [('G', 'tab:blue'), ('Q', 'tab:green'), ('Q_bi', 'tab:purple')],
        'joint': [('G', 'tab:blue'), ('Q', 'tab:green'), ('Q_bi', 'tab:purple')],
    }
    for obj, bbs in objectives.items():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for bb, color in bbs:
            tr_all, vl_all, steps = [], [], None
            for s in SEEDS:
                p = hist_path(bb, obj, s)
                if not p.exists():
                    continue
                rows = [json.loads(l) for l in p.read_text().splitlines()]
                steps = [r['step'] for r in rows]
                tr_all.append([r['train_loss'] for r in rows])
                vl_all.append([r['val_loss'] for r in rows])
            if not tr_all:
                continue
            tr = np.mean(tr_all, axis=0)
            tr_s = np.std(tr_all, axis=0)
            vl = np.mean(vl_all, axis=0)
            vl_s = np.std(vl_all, axis=0)
            ax.plot(steps, vl, color=color, lw=1.8, label=f'{bb} val')
            ax.fill_between(steps, vl - vl_s, vl + vl_s, color=color, alpha=0.18)
            ax.plot(steps, tr, color=color, lw=0.9, ls='--', alpha=0.55, label=f'{bb} train')
        ax.set_title(f'Round6 pretrain recon loss — {obj} (mean ± std, 3 seeds)')
        ax.set_xlabel('optimizer step')
        ax.set_ylabel('recon loss')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fn = FIGS / f'training_{obj.lower()}.png'
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        print(f'[fig] {fn.name}')


def main_bar():
    csv_path = R6 / 'reports' / 'main_table.csv'
    if not csv_path.exists():
        print('[fig] main_table.csv not ready; skip main_bar')
        return
    rows = list(csv.DictReader(open(csv_path)))
    labels = [r['group'] for r in rows]
    metrics = [('comm_frozen_mean', 'comm frozen'), ('radar_frozen_mean', 'radar frozen'),
               ('comm_ft_mean', 'comm finetune'), ('radar_ft_mean', 'radar finetune')]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (col, name) in enumerate(metrics):
        vals = [float(r[col]) if r[col] not in ('', 'None') else np.nan for r in rows]
        errs = [float(r[col.replace('_mean', '_std')]) if r[col.replace('_mean', '_std')] not in ('', 'None') else 0
                for r in rows]
        ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=2, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel('OA')
    ax.set_title('Round6 main table (mean ± sample-std, 3 seeds)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / 'main_bar.png', dpi=110)
    plt.close(fig)
    print('[fig] main_bar.png')


if __name__ == '__main__':
    training_curves_by_objective()
    main_bar()
    print('[done] figures')
