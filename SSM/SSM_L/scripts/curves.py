"""SSM_L round-1 training curves (matplotlib PNGs in reports/)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
GROUPS = ['G_mae', 'G_denoise', 'Q_mae', 'Q_denoise']
SEEDS = [17, 42, 2026]
COLORS = {17: '#1f77b4', 42: '#ff7f0e', 2026: '#2ca02c'}


def read_history(path):
    p = Path(path)
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def plot_pretrain():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for gi, g in enumerate(GROUPS):
        ax = axes[gi // 2][gi % 2]
        for s in SEEDS:
            h = read_history(ROOT / 'runs' / 'pretrain' / f'{g}_seed{s}' / 'history.jsonl')
            if not h:
                continue
            ax.plot([r['epoch'] for r in h], [r['train_loss'] for r in h],
                    color=COLORS[s], label=f'train s{s}', lw=1.2)
            ax.plot([r['epoch'] for r in h], [r['val_loss'] for r in h],
                    color=COLORS[s], ls='--', label=f'val s{s}', lw=1.2)
        ax.set_title(g)
        ax.set_xlabel('epoch'); ax.set_ylabel('recon loss'); ax.legend(fontsize=7)
    fig.suptitle('pretraining reconstruction loss (train solid / val dashed)')
    fig.tight_layout()
    fig.savefig(ROOT / 'reports' / 'pretrain_curves.png', dpi=110)
    plt.close(fig)


def plot_downstream(mode):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for gi, g in enumerate(GROUPS):
        ax = axes[gi // 2][gi % 2]
        for s in SEEDS:
            h = read_history(ROOT / 'runs' / mode / f'{g}_seed{s}' / 'history.jsonl')
            if not h:
                continue
            if mode == 'probe':
                ys = [r['val_oa'] for r in h]
            else:
                ys = [r['val']['overall']['oa'] if isinstance(r.get('val'), dict) else r.get('val_oa', 0) for r in h]
            ax.plot([r['epoch'] for r in h], ys, color=COLORS[s], label=f's{s}', lw=1.2)
        ax.set_title(g); ax.set_xlabel('epoch'); ax.set_ylabel('val OA'); ax.legend(fontsize=7)
    fig.suptitle(f'{mode} validation OA')
    fig.tight_layout()
    fig.savefig(ROOT / 'reports' / f'{mode}_curves.png', dpi=110)
    plt.close(fig)


def main():
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    plot_pretrain()
    plot_downstream('probe')
    plot_downstream('finetune')
    print('wrote reports/*_curves.png')


if __name__ == '__main__':
    main()
