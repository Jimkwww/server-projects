#!/usr/bin/env python
"""Build SSM_results_round2.zip.

Includes: configs, all reports (round1 + round2), per-run text/JSON/npz for
runs/clean and runs/round2, per-sample predictions (clean + CFO=+-0.001/+-0.005
under both orders A/B) ACTUALLY inside the zip, final source, weight manifest.
Excludes .pt weights (paths recorded in weight_manifest.json) and the large
round-1 eval_val/pred_*.npz (superseded by the frozen-protocol predictions).
"""
import json
import zipfile
from pathlib import Path

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
DELIV = ROOT / 'deliverables'
DELIV.mkdir(parents=True, exist_ok=True)
ZIP = DELIV / 'SSM_results_round2.zip'

RUN_FILES = ['config.json', 'provenance.json', 'model_info.json', 'history.jsonl',
             'best_val.json', 'DONE.json', 'val_predictions.npz']

weight_paths = []

with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    # configs
    for f in sorted((ROOT / 'configs').glob('*.json')):
        z.write(f, f'configs/{f.name}')

    # reports (recursively)
    for f in sorted((ROOT / 'reports').rglob('*')):
        if f.is_file():
            z.write(f, f'reports/{f.relative_to(ROOT / "reports")}')

    # per-run artifacts for both round-1 (clean) and round-2 (round2)
    for runs_dir, tag in ((ROOT / 'runs' / 'clean', 'r1'),
                          (ROOT / 'runs' / 'round2', 'r2')):
        for run in sorted(runs_dir.iterdir()):
            if not run.is_dir():
                continue
            name = f'{tag}_{run.name}'
            for rf in RUN_FILES:
                p = run / rf
                if p.exists():
                    z.write(p, f'runs/{name}/{rf}')
            er = run / 'eval_val' / 'results.json'
            if er.exists():
                z.write(er, f'runs/{name}/eval_val/results.json')
            for w in ('best.pt', 'last.pt'):
                if (run / w).exists():
                    weight_paths.append(str(run / w))

    # per-sample predictions (frozen protocol), actually in the zip
    for pred_dir, tag in ((ROOT / 'predictions_round2', 'round1'),
                          (ROOT / 'predictions_round2_new', 'round2')):
        for f in sorted(pred_dir.glob('*.npz')):
            z.write(f, f'predictions/{tag}/{f.name}')

    # final source code
    for f in sorted((ROOT / 'ssm_exp').glob('*.py')):
        z.write(f, f'source/ssm_exp/{f.name}')
    for f in sorted((ROOT / 'scripts').glob('*.py')):
        z.write(f, f'source/scripts/{f.name}')
    for f in sorted((ROOT / 'scripts').glob('*.sh')):
        z.write(f, f'source/scripts/{f.name}')

    # weight path manifest
    manifest = {'weights_on_server': weight_paths,
                'note': 'per-sample predictions are inside predictions/ (clean + CFO=+-0.001/+-0.005, orders A/B)'}
    z.writestr('weight_manifest.json', json.dumps(manifest, indent=2))

print(f'wrote {ZIP} ({ZIP.stat().st_size/1e6:.2f} MB)')
print(f'{len(weight_paths)} weight paths recorded')
