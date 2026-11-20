#!/usr/bin/env python
"""Build SSM_results_round1.zip: configs, reports, per-run JSON/CSV/log/npz,
final source, weight path manifest. Excludes .pt weights and optional
per-perturbation pred_*.npz (kept on server, paths recorded in manifest)."""
import json
import zipfile
from pathlib import Path

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
DELIV = ROOT / 'deliverables'
DELIV.mkdir(parents=True, exist_ok=True)
ZIP = DELIV / 'SSM_results_round1.zip'

# Per-run files to include (small, text/JSON/npz); weights excluded.
RUN_FILES = ['config.json', 'provenance.json', 'model_info.json', 'history.jsonl',
             'best_val.json', 'DONE.json', 'val_predictions.npz', 'benchmark.json']

weight_paths = []
src_files = []

with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    # configs
    for f in sorted((ROOT / 'configs').glob('*.json')):
        z.write(f, f'configs/{f.name}')

    # reports (recursively, whole dir)
    for f in sorted((ROOT / 'reports').rglob('*')):
        if f.is_file():
            z.write(f, f'reports/{f.relative_to(ROOT / "reports")}')

    # per-run artifacts
    runs = ROOT / 'runs' / 'clean'
    for run in sorted(runs.iterdir()):
        if not run.is_dir():
            continue
        name = run.name
        for rf in RUN_FILES:
            p = run / rf
            if p.exists():
                z.write(p, f'runs/{name}/{rf}')
        # eval_val results.json (perturbation metrics) — skip large pred_*.npz
        er = run / 'eval_val' / 'results.json'
        if er.exists():
            z.write(er, f'runs/{name}/eval_val/results.json')
        # record weight absolute paths for the manifest
        for w in ('best.pt', 'last.pt'):
            if (run / w).exists():
                weight_paths.append(str(run / w))

    # final source code (the executable deliverable)
    for f in sorted((ROOT / 'ssm_exp').glob('*.py')):
        z.write(f, f'source/ssm_exp/{f.name}')
    for f in sorted((ROOT / 'scripts').glob('*.py')):
        z.write(f, f'source/scripts/{f.name}')
    for f in sorted((ROOT / 'scripts').glob('*.sh')):
        z.write(f, f'source/scripts/{f.name}')

    # weight path manifest
    manifest = {'weights_on_server': weight_paths,
                'perturbation_predictions_on_server':
                    sorted(str(p) for p in runs.glob('*/eval_val/pred_*.npz'))}
    z.writestr('weight_manifest.json', json.dumps(manifest, indent=2))

print(f'wrote {ZIP} ({ZIP.stat().st_size/1e6:.2f} MB)')
print(f'{len(weight_paths)} weight paths recorded')
