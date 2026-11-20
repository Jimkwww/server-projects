"""Round5 figures: (1) training/val recon loss vs epoch; (2) frozen readout OA vs checkpoint.

Reads history.jsonl (pretrain) and readout JSONs; writes round5_epoch_scale/figs/*.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
R5 = ROOT / 'round5_epoch_scale'
REP = R5 / 'reports'
FIGS = R5 / 'figs'
BACKBONES = ('G', 'G_Qlearn', 'Q')
CORPORA = ('f1', 'f2')
SEEDS = (17, 42, 2026)
CKPT_NAMES = ('step7800', 'epoch20', 'epoch30', 'epoch40', 'epoch50')
CKPT_X = {'step7800': 7800, 'epoch20': 15640, 'epoch30': 23460, 'epoch40': 31280, 'epoch50': 39100}
COLORS = {'G': 'tab:blue', 'G_Qlearn': 'tab:orange', 'Q': 'tab:green'}


def training_curves():
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ci, corpus in enumerate(CORPORA):
        ax = axes[ci]
        for bb in BACKBONES:
            for s in SEEDS:
                p = R5 / 'runs' / 'pretrain' / corpus / f'{bb}_seed{s}' / 'history.jsonl'
                if not p.exists():
                    continue
                rows = [json.loads(l) for l in p.read_text().splitlines()]
                steps = [r['step'] for r in rows]
                tr = [r['train_loss'] for r in rows]
                vl = [r['val_loss'] for r in rows]
                ax.plot(steps, tr, color=COLORS[bb], alpha=0.3, lw=0.8,
                        label=bb if s == SEEDS[0] else None)
                ax.plot(steps, vl, color=COLORS[bb], alpha=0.9, lw=1.2,
                        linestyle='--', label=f'{bb} val' if s == SEEDS[0] else None)
        ax.set_title(f'{corpus.upper()} MAE recon loss')
        ax.set_xlabel('optimizer step')
        ax.set_ylabel('recon loss')
        ax.legend(fontsize=7)
        # mark intermediate checkpoints
        for nm, x in CKPT_X.items():
            ax.axvline(x, color='gray', alpha=0.25, lw=0.6)
    fig.suptitle('Round5 pretraining recon loss (solid=train, dashed=val, 3 seeds each)')
    fig.tight_layout()
    fig.savefig(FIGS / 'training_curves.png', dpi=110)
    plt.close(fig)
    print('[fig] training_curves.png')


def readout_trend():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ti, task in enumerate(('rml', 'radar')):
        split = 'val' if task == 'rml' else 'test'
        for ci, corpus in enumerate(CORPORA):
            ax = axes[ci][ti]
            for bb in BACKBONES:
                means, stds = [], []
                for ckpt in CKPT_NAMES:
                    oas = []
                    for s in SEEDS:
                        p = (R5 / 'runs' / 'downstream' / 'readout' / corpus /
                             f'{bb}_seed{s}__{ckpt}__{task}.json')
                        if p.exists():
                            d = json.load(open(p))
                            if split in d:
                                oas.append(d[split]['oa'])
                    if oas:
                        means.append(np.mean(oas)); stds.append(np.std(oas, ddof=1))
                    else:
                        means.append(np.nan); stds.append(np.nan)
                xs = [CKPT_X[c] for c in CKPT_NAMES]
                ax.errorbar(xs, means, yerr=stds, color=COLORS[bb], marker='o', ms=4,
                            capsize=3, label=bb)
            ax.set_title(f'{corpus.upper()} {task} ({split})')
            ax.set_xlabel('optimizer step')
            ax.set_ylabel('OA')
            ax.legend(fontsize=7)
            ax.set_xticks(list(CKPT_X.values()))
            ax.set_xticklabels(CKPT_NAMES, rotation=30, fontsize=7)
    fig.suptitle('Round5 frozen readout OA vs training checkpoint (mean ± sample-std)')
    fig.tight_layout()
    fig.savefig(FIGS / 'readout_trend.png', dpi=110)
    plt.close(fig)
    print('[fig] readout_trend.png')


if __name__ == '__main__':
    training_curves()
    readout_trend()
    print('[done] figures')
