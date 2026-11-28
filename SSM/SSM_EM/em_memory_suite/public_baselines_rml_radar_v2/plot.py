import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / 'public_baselines_rml_radar_v1' / 'runs'
FIGS = ROOT / 'figs'; FIGS.mkdir(exist_ok=True)
MODELS = ['q', 'structured', 'emind', 'wavesfm']
COLORS = {'q': '#1f77b4', 'structured': '#d62728', 'emind': '#2ca02c', 'wavesfm': '#ff7f0e'}
TASKS = {'rml': ('RML (24-class)', '../data/rml2018_clsdev/test_snr.npy', 24),
         'radar': ('RadChar (5-class)', '../data/radchar_tiny/test_snr.npy', 5)}

# ---- 1. learning curves (seed17 val_f1) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (task, (tname, _, _)) in zip(axes, TASKS.items()):
    for m in MODELS:
        r = json.loads((V1/task/m/'result.json').read_text())
        h = r['history']
        ax.plot([x['ep'] for x in h], [x['va_f1'] for x in h], color=COLORS[m], label=m, lw=1.8)
    ax.set_title(f'{tname} val macro-F1 (seed17)'); ax.set_xlabel('epoch'); ax.set_ylabel('val macro-F1')
    ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIGS/'fig_learning_curves.png', dpi=130); plt.close(fig)

# ---- 2. per-SNR curves (OA vs SNR bin) ----
def load_pred(task, model, seed):
    if seed == 17:
        return np.load(V1/task/model/'test_pred.npz')
    return np.load(ROOT/'runs'/task/model/f's{seed}_lr0.0003_ep30'/'test_pred.npz')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (task, (tname, snr_path, nc)) in zip(axes, TASKS.items()):
    snr = np.load(snr_path).reshape(-1)
    if task == 'rml':
        bins = np.arange(-20, 32, 4)
    else:
        bins = np.arange(int(np.floor(snr.min())), int(np.ceil(snr.max()))+1, 2)
    centers = (bins[:-1] + bins[1:]) / 2
    for m in MODELS:
        oas = []
        for seed in (17, 42, 2026):
            d = load_pred(task, m, seed); y, p = d['y'], d['pred']
            oa_bin = []
            for i in range(len(bins)-1):
                mask = (snr >= bins[i]) & (snr < bins[i+1])
                oa_bin.append(float((y[mask]==p[mask]).mean()) if mask.sum()>0 else np.nan)
            oas.append(oa_bin)
        oas = np.array(oas); mean = np.nanmean(oas, axis=0)
        ax.plot(centers, mean, color=COLORS[m], label=m, lw=1.8, marker='o', ms=3)
    ax.set_title(f'{tname} OA vs SNR (3-seed mean)'); ax.set_xlabel('SNR (dB)'); ax.set_ylabel('OA')
    ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIGS/'fig_snr.png', dpi=130); plt.close(fig)

# ---- 3. confusion matrices (rml seed17, 4 models) ----
def load_conf(task, model, seed):
    p = ROOT/'eval'/task/model/f's{seed}.json'
    return json.loads(p.read_text())

CLS_NAMES = {'rml': ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK','16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM','SSB-WC','SSB-SC','DSB-WC','DSB-SC','FM','GMSK','OQPSK'],
             'radar': ['CPT','Barker','PolyBarker','Frank','LFM']}
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for ax, m in zip(axes.ravel(), MODELS):
    cm = np.array(load_conf('rml', m, 17)['confusion'])
    cm_n = cm / cm.sum(axis=1, keepdims=True).clip(1e-9)
    im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'RML {m} (seed17), OA={cm.diagonal().sum()/cm.sum():.3f}')
    ax.set_xticks(range(24)); ax.set_yticks(range(24))
    ax.set_xticklabels(CLS_NAMES['rml'], rotation=90, fontsize=5)
    ax.set_yticklabels(CLS_NAMES['rml'], fontsize=5)
fig.colorbar(im, ax=axes, fraction=0.03); fig.tight_layout()
fig.savefig(FIGS/'fig_confusion_rml.png', dpi=130); plt.close(fig)

# radar confusion
fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
for ax, m in zip(axes.ravel(), MODELS):
    cm = np.array(load_conf('radar', m, 17)['confusion'])
    cm_n = cm / cm.sum(axis=1, keepdims=True).clip(1e-9)
    im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'radar {m} (seed17), OA={cm.diagonal().sum()/cm.sum():.3f}')
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLS_NAMES['radar'], rotation=45, fontsize=7)
    ax.set_yticklabels(CLS_NAMES['radar'], fontsize=7)
fig.colorbar(im, ax=axes, fraction=0.03); fig.tight_layout()
fig.savefig(FIGS/'fig_confusion_radar.png', dpi=130); plt.close(fig)
print('figs written:', sorted(p.name for p in FIGS.glob('*.png')))
