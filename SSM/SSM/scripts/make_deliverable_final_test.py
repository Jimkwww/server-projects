#!/usr/bin/env python
"""Build SSM_results_final_test.zip.

Final frozen test evaluation + three-round archive + errata. Weights stay on the
server (paths recorded in weight_manifest.json). Contains the independent test
evaluation (freeze manifest, test tables, test-vs-val, conclusion, errata),
per-sample test predictions, full reports (round1/2/3/round3_review/final_test),
configs, and final source.
"""
import json
import zipfile
from pathlib import Path

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
DELIV = ROOT / 'deliverables'
DELIV.mkdir(parents=True, exist_ok=True)
ZIP = DELIV / 'SSM_results_final_test.zip'

SEEDS = (17, 42, 2026)
VARIANTS = ('G', 'G_corr', 'S1', 'S1_corr')
weight_paths = []

with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    # reports (all rounds incl. final_test) — the three-round archive + errata
    for f in sorted((ROOT / 'reports').rglob('*')):
        if f.is_file():
            z.write(f, f'reports/{f.relative_to(ROOT / "reports")}')

    # per-sample test predictions (132 npz)
    for f in sorted((ROOT / 'predictions_test').glob('*.npz')):
        z.write(f, f'predictions/test/{f.name}')

    # configs
    for f in sorted((ROOT / 'configs').glob('*.json')):
        z.write(f, f'configs/{f.name}')

    # final source
    for f in sorted((ROOT / 'ssm_exp').glob('*.py')):
        z.write(f, f'source/ssm_exp/{f.name}')
    for f in sorted((ROOT / 'scripts').glob('*.py')):
        z.write(f, f'source/scripts/{f.name}')
    for f in sorted((ROOT / 'scripts').glob('*.sh')):
        z.write(f, f'source/scripts/{f.name}')

    # weight paths (the 12 evaluated augmented checkpoints)
    for v in VARIANTS:
        for s in SEEDS:
            d = ROOT / 'runs' / 'round3_aug' / f'{v}_seed{s}'
            for w in ('best.pt', 'last.pt'):
                if (d / w).exists():
                    weight_paths.append(str(d / w))
    z.writestr('weight_manifest.json', json.dumps({
        'weights_on_server': weight_paths,
        'note': 'best.pt selected on clean val OA (earliest tie); test split never used for selection.',
    }, indent=2))

    z.writestr('README.txt', (
        'SSM final test evaluation (frozen protocol)\n'
        '==========================================\n'
        'Independent test-split evaluation of the 12 augmented checkpoints (G/G_corr/S1/S1_corr x 3 seeds).\n'
        '- reports/final_test/freeze_manifest.json : code/config/checkpoint SHA256 + eval spec\n'
        '- reports/final_test/test_main.md/.json : per-seed + mean +/- sample std, per-SNR, CFO tables\n'
        '- reports/final_test/test_vs_val.md : val vs test ranking comparison\n'
        '- reports/final_test/test_conclusion.md : final conclusion (val-dev vs independent test)\n'
        '- reports/final_test/ERRATA.md : three-round errata index\n'
        '- predictions/test/*.npz : per-sample predictions (id/label/pred/snr) at clean +-0.001/+-0.002/+-0.005/+-0.01/+-0.02\n'
        '- reports/ : full round1/2/3/round3_review reports (three-round archive)\n'
        '- weight_manifest.json : weight paths (left on server)\n'
    ))

print(f'wrote {ZIP} ({ZIP.stat().st_size/1e6:.2f} MB)')
print(f'{len(weight_paths)} weight paths recorded')
