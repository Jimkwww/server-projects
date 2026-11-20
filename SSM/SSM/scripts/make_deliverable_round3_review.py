#!/usr/bin/env python
"""Build SSM_results_round3_review.zip.

Contains: configs, all reports (incl. round3_review), per-run text/JSON/npz for the
24 reviewed checkpoints (runs/clean, runs/round2, runs/round3, runs/round3_aug),
per-sample predictions (unified extended CFO grid order B under predictions_round3_review,
plus the original round-3 predictions_round3 / predictions_round3_aug), final source,
and a weight path manifest. Weights stay on the server (paths recorded).
"""
import json
import zipfile
from pathlib import Path

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
DELIV = ROOT / 'deliverables'
DELIV.mkdir(parents=True, exist_ok=True)
ZIP = DELIV / 'SSM_results_round3_review.zip'

RUN_FILES = ['config.json', 'provenance.json', 'model_info.json', 'history.jsonl',
             'best_val.json', 'DONE.json', 'val_predictions.npz']
# Only the 24 reviewed checkpoints (G/G_corr/S1/S1_corr x 3 seeds x pre/post).
REVIEWED = []
for v in ('G', 'S1'):
    REVIEWED += [(f'runs/clean/{v}_seed{s}', 'r1') for s in (17, 42, 2026)]
REVIEWED += [(f'runs/round2/S1_corr_seed{s}', 'r2') for s in (17, 42, 2026)]
REVIEWED += [(f'runs/round3/G_corr_seed{s}', 'r3') for s in (17, 42, 2026)]
for v in ('G', 'G_corr', 'S1', 'S1_corr'):
    REVIEWED += [(f'runs/round3_aug/{v}_seed{s}', 'r3aug') for s in (17, 42, 2026)]

weight_paths = []

with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    # configs
    for f in sorted((ROOT / 'configs').glob('*.json')):
        z.write(f, f'configs/{f.name}')

    # reports (recursively, includes round1/2/3 + round3_review + server audit)
    for f in sorted((ROOT / 'reports').rglob('*')):
        if f.is_file():
            z.write(f, f'reports/{f.relative_to(ROOT / "reports")}')

    # per-run artifacts for the 24 reviewed checkpoints
    for rel, tag in REVIEWED:
        run = ROOT / rel
        if not (run / 'best.pt').exists():
            raise FileNotFoundError(f'missing {run}/best.pt')
        name = f'{tag}_{run.name}'
        for rf in RUN_FILES:
            p = run / rf
            if p.exists():
                z.write(p, f'runs/{name}/{rf}')
        er = run / 'eval_val' / 'results.json'
        if er.exists():
            z.write(er, f'runs/{name}/eval_val/results.json')
        log = run.parent / f'{run.name}.log'
        if log.exists():
            z.write(log, f'runs/{name}/train.log')
        for w in ('best.pt', 'last.pt'):
            if (run / w).exists():
                weight_paths.append(str(run / w))

    # per-sample predictions: unified extended grid (all 24 runs, order B)
    for f in sorted((ROOT / 'predictions_round3_review').glob('*.npz')):
        z.write(f, f'predictions/round3_review/{f.name}')
    # original round-3 predictions (order A/B at +-0.001/+-0.002/+-0.005)
    for pred_dir, tag in ((ROOT / 'predictions_round3', 'round3'),
                          (ROOT / 'predictions_round3_aug', 'round3_aug')):
        for f in sorted(pred_dir.glob('*.npz')):
            z.write(f, f'predictions/{tag}/{f.name}')

    # final source code
    for f in sorted((ROOT / 'ssm_exp').glob('*.py')):
        z.write(f, f'source/ssm_exp/{f.name}')
    for f in sorted((ROOT / 'scripts').glob('*.py')):
        z.write(f, f'source/scripts/{f.name}')
    for f in sorted((ROOT / 'scripts').glob('*.sh')):
        z.write(f, f'source/scripts/{f.name}')

    # weight path manifest (weights stay on server)
    manifest = {
        'weights_on_server': weight_paths,
        'note': ('per-sample predictions: predictions/round3_review (unified order-B extended '
                 'CFO grid, clean + 22 CFO points, all 24 checkpoints), predictions/round3 and '
                 'predictions/round3_aug (original order A/B at +-0.001/+-0.002/+-0.005).'),
    }
    z.writestr('weight_manifest.json', json.dumps(manifest, indent=2))

    # README
    readme = (
        'SSM round-3 review deliverable\n'
        '==============================\n'
        'Contents:\n'
        '- reports/round3_review/ : per-seed main tables, full CFO curves, class/SNR analysis, conclusion\n'
        '- reports/round3/ : prior round-3 (G_corr control + augmentation) reports\n'
        '- runs/{r1,r2,r3,r3aug}_* : per-run config/provenance/model_info/history/best_val/DONE/val_predictions\n'
        '- predictions/ : per-sample predictions (id/label/pred/snr) at each CFO point\n'
        '- configs/ , source/ : configuration and final source code\n'
        '- weight_manifest.json : absolute paths of weights (left on server)\n'
        '\n'
        'Protocol: raw IQ -> inject CFO -> max_abs normalize -> recompute physical desc -> model (order B).\n'
        'Metrics are on the val split only; the test split was not used.\n'
    )
    z.writestr('README.txt', readme)

print(f'wrote {ZIP} ({ZIP.stat().st_size/1e6:.2f} MB)')
print(f'{len(weight_paths)} weight paths recorded')
