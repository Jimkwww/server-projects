#!/usr/bin/env python
"""Build SSM_results_round4_structure.zip.

Contains: configs, all round4 reports (design/correctness/per-seed/CFO/phase/paired diffs/
per-SNR/class/train stats/benchmark/conclusion), per-run artifacts for the 15 round4 runs and
9 baselines (G/G_corr/S1 x 3 seeds), per-sample predictions (unified order-B CFO+phase grid),
final source, checkpoint path + SHA256 manifest. Weights stay on the server (paths + hashes).
"""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
DELIV = ROOT / 'deliverables'
DELIV.mkdir(parents=True, exist_ok=True)
ZIP = DELIV / 'SSM_results_round4_structure.zip'

RUN_FILES = ['config.json', 'provenance.json', 'model_info.json', 'history.jsonl',
             'best_val.json', 'DONE.json', 'val_predictions.npz']

CHECKPOINTS = []
for v in ('G', 'G_corr', 'S1'):
    CHECKPOINTS += [(f'runs/round3_aug/{v}_seed{s}', 'baseline') for s in (17, 42, 2026)]
for v in ('A64', 'A128', 'B_front', 'C_bi_front', 'D_complex'):
    CHECKPOINTS += [(f'runs/round4_structure/{v}_seed{s}', 'round4') for s in (17, 42, 2026)]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


weight_manifest = []
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    # configs
    for f in sorted((ROOT / 'configs').glob('*.json')):
        z.write(f, f'configs/{f.name}')

    # round4 reports only (design/correctness/data-driven tables/conclusion)
    r4 = ROOT / 'reports' / 'round4_structure'
    for f in sorted(r4.glob('*')):
        if f.is_file():
            z.write(f, f'reports/round4_structure/{f.name}')

    # per-run artifacts for all 24 checkpoints
    for rel, tag in CHECKPOINTS:
        run = ROOT / rel
        if not (run / 'best.pt').exists():
            raise FileNotFoundError(f'missing {run}/best.pt')
        name = f'{tag}_{run.name}'
        for rf in RUN_FILES:
            p = run / rf
            if p.exists():
                z.write(p, f'runs/{name}/{rf}')
        log = run.parent / f'{run.name}.log'
        if log.exists():
            z.write(log, f'runs/{name}/train.log')
        for w in ('best.pt', 'last.pt'):
            if (run / w).exists():
                weight_manifest.append({'path': str(run / w), 'sha256': sha256_file(run / w),
                                        'variant': run.name.rsplit('_seed', 1)[0]})

    # per-sample predictions (unified order-B CFO + phase grid)
    for f in sorted((ROOT / 'predictions_round4_structure').glob('*.npz')):
        z.write(f, f'predictions/{f.name}')

    # final source code + scripts
    for f in sorted((ROOT / 'ssm_exp').glob('*.py')):
        z.write(f, f'source/ssm_exp/{f.name}')
    for f in sorted((ROOT / 'scripts').glob('*.py')):
        z.write(f, f'source/scripts/{f.name}')
    for f in sorted((ROOT / 'scripts').glob('*.sh')):
        z.write(f, f'source/scripts/{f.name}')

    # weight manifest (weights stay on server)
    z.writestr('weight_manifest.json', json.dumps({
        'weights_on_server': weight_manifest,
        'note': 'best.pt/last.pt stay on server; SHA256 recorded. predictions/ hold per-sample '
                '(y/pred/snr/ids) for every condition (clean, CFO grid, extended CFO, phase).',
    }, indent=2))

    readme = (
        'SSM round-4 structure deliverable (bidirectional context + complex input structure)\n'
        '==================================================================================\n'
        'Variants: baselines G/G_corr/S1 (round3_aug) + new A64/A128/B_front/C_bi_front/D_complex\n'
        'Contents:\n'
        '- reports/round4_structure/ : design_impl.md, correctness.md, per_seed_main, cfo_curves,\n'
        '  phase_curves, paired_diffs, per_snr, class_analysis, train_stats, benchmark, conclusion\n'
        '- runs/{baseline,round4}_* : config/provenance/model_info/history/best_val/DONE/val_predictions + train.log\n'
        '- predictions/ : per-sample predictions at clean + CFO (0,+-0.001,+-0.002,+-0.005; ext +-0.01,+-0.02)\n'
        '  + global phase (0,pi/4,pi/2,pi,3pi/2), order B (raw IQ -> perturb -> max_abs norm)\n'
        '- configs/ , source/ : configuration and final source\n'
        '- weight_manifest.json : weight paths + SHA256 (weights left on server)\n'
        '\n'
        'Protocol: raw IQ -> perturb(CFO or phase) -> max_abs normalize -> model (order B).\n'
        'Val split only; test split was NOT used. 3 seeds reported with sample std (ddof=1).\n'
    )
    z.writestr('README.txt', readme)

print(f'wrote {ZIP} ({ZIP.stat().st_size/1e6:.2f} MB)')
print(f'{len(weight_manifest)} weight paths recorded')
