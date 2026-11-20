"""Aggregate training curves (history.jsonl) into PNGs for the report.

Produces reports/curves/{task}_{init}_{backbone}.png (train_loss + val_oa over epochs,
one line per seed). matplotlib Agg backend; run after the orchestrator finishes.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
R4 = ROOT / 'round4_no_cfo_qstudy'
RUNS = R4 / 'runs'
OUT = R4 / 'reports' / 'curves'
SEEDS = (17, 42, 2026)
BBS = ('G', 'Q', 'S1plain', 'G_Qlearn', 'G_corr')
INITS = ('scratch', 'mae', 'denoise')


def read_hist(p):
    rows = []
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def plot(run_dirs, title, out_png):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for s, rd in zip(SEEDS, run_dirs):
        if not (rd / 'history.jsonl').exists():
            continue
        rows = read_hist(rd / 'history.jsonl')
        if not rows:
            continue
        ep = [r['epoch'] for r in rows]
        a1.plot(ep, [r['train_loss'] for r in rows], label=f'seed {s}')
        a2.plot(ep, [r['val']['overall']['oa'] for r in rows], label=f'seed {s}')
    a1.set_xlabel('epoch'); a1.set_ylabel('train loss'); a1.set_title('train loss'); a1.legend()
    a2.set_xlabel('epoch'); a2.set_ylabel('val OA'); a2.set_title('val OA'); a2.legend()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for task in ('rml', 'radar'):
        for init in INITS:
            for bb in BBS:
                dirs = [RUNS / task / init / f'{bb}_seed{s}' for s in SEEDS]
                if not any((d / 'history.jsonl').exists() for d in dirs):
                    continue
                plot(dirs, f'{task} {init} {bb}', OUT / f'{task}_{init}_{bb}.png')
                n += 1
    # P1 settings (G_Qlearn)
    for task in ('rml', 'radar'):
        for st in ('A_fixedq_fixedq', 'B_learnq_fixedq', 'C_fixedq_learnq'):
            dirs = [RUNS / task / st / f'G_Qlearn_seed{s}' for s in SEEDS]
            if not any((d / 'history.jsonl').exists() for d in dirs):
                continue
            plot(dirs, f'{task} {st}', OUT / f'{task}_{st}.png')
            n += 1
    print(f'[plot_curves] wrote {n} curve PNGs to {OUT}')


if __name__ == '__main__':
    main()
