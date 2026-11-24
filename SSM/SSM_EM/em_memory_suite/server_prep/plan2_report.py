"""Phase-2 report artifacts: cross-task/cross-data summary + figures."""
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / 'report'
REP.mkdir(exist_ok=True)

MODELS = ['structured', 'generic_ssm', 'generic_transformer', 'generic_ssm_recon']
TASKS = [('amc', 'AMC'), ('radar', 'radar'), ('techrec', 'TECHRec'), ('hisarmod', 'HisarMod')]


def probe_oa(root, tag, model, prefix):
    oas = []
    for f in glob.glob(f'{root}/{tag}/{model}_s*_{prefix}/test_eval/metrics.json'):
        oas.append(json.loads(open(f).read())['metrics']['oa'])
    return float(np.mean(oas)) if oas else None


def main():
    # 200k probe + finetune
    rows = []
    for m in MODELS:
        for tag, label in TASKS:
            p = probe_oa('results/plan2_full', tag, m, 'probe')
            ft = probe_oa('results/plan2_full', tag, m, 'finetune')
            rows.append({'model': m, 'task': label, 'probe_200k': p, 'ft_200k': ft})
    with open(REP / 'plan2_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'task', 'probe_200k', 'ft_200k'])
        w.writeheader()
        w.writerows(rows)
    print('[report] plan2_summary.csv')

    # figure: probe OA by model across 4 tasks (200k)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = REP / 'plan2_figures'
    figdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (tag, label) in zip(axes, TASKS):
        vals = [probe_oa('results/plan2_full', tag, m, 'probe') for m in MODELS]
        colors = ['#1f77b4' if m == 'structured' else '#999999' for m in MODELS]
        ax.bar(MODELS, vals, color=colors)
        ax.set_title(f'{label} probe (200k)')
        ax.set_xticklabels(MODELS, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('OA')
    fig.tight_layout()
    fig.savefig(figdir / 'plan2_probe_4tasks.png', dpi=150)
    plt.close(fig)
    print('[report] plan2_figures/plan2_probe_4tasks.png')


if __name__ == '__main__':
    main()
