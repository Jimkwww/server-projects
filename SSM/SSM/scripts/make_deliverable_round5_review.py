#!/usr/bin/env python
"""Build SSM_results_round5_review.zip (Round 5 supplementary audit + frozen-branch control).

Independent of the original SSM_results_round5_iq.zip (left untouched). Contains:
  reports/round5_review/*  (resume-fix audit, errata, revised conclusion, implementation
    evidence, branch-dependency diagnostic, Q_corr_frozen eval comparison/summary, ...)
  runs/round5_review/Q_corr_frozen_seed* artifacts (config/provenance/model_info/history/
    best_val/DONE + train.log; weights stay on server, SHA256 recorded)
  predictions_round5_review/  (per-sample predictions for Q_corr_frozen)
  configs/resolved_round5_review.json
  source/  (post-fix ssm_exp/*.py + scripts/*.py + scripts/*.sh)
"""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
DELIV = ROOT / 'deliverables'
DELIV.mkdir(parents=True, exist_ok=True)
ZIP = DELIV / 'SSM_results_round5_review.zip'

RUN_FILES = ['config.json', 'provenance.json', 'model_info.json', 'history.jsonl',
             'best_val.json', 'DONE.json']
FROZEN_RUNS = [f'runs/round5_review/Q_corr_frozen_seed{s}' for s in (17, 42, 2026)]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


weight_manifest = []
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    # review reports (all files, recursively to keep resume_fix/ and tests/ subdirs)
    r5r = ROOT / 'reports' / 'round5_review'
    for f in sorted(r5r.rglob('*')):
        if f.is_file():
            z.write(f, f'reports/round5_review/{f.relative_to(r5r)}')

    # frozen runs artifacts
    for rel in FROZEN_RUNS:
        run = ROOT / rel
        if not (run / 'best.pt').exists():
            raise FileNotFoundError(f'missing {run}/best.pt')
        name = run.name  # Q_corr_frozen_seed<seed>
        for rf in RUN_FILES:
            p = run / rf
            if p.exists():
                z.write(p, f'runs/{name}/{rf}')
        log = run.parent / f'{name}.log'
        if log.exists():
            z.write(log, f'runs/{name}/train.log')
        for w in ('best.pt', 'last.pt'):
            if (run / w).exists():
                weight_manifest.append({'path': str(run / w), 'sha256': sha256_file(run / w),
                                        'variant': 'Q_corr_frozen'})

    # per-sample predictions for Q_corr_frozen
    pred = ROOT / 'predictions_round5_review'
    if pred.exists():
        for f in sorted(pred.glob('*.npz')):
            z.write(f, f'predictions/{f.name}')

    # review config
    z.write(ROOT / 'configs' / 'resolved_round5_review.json',
            'configs/resolved_round5_review.json')

    # post-fix source + scripts
    for f in sorted((ROOT / 'ssm_exp').glob('*.py')):
        z.write(f, f'source/ssm_exp/{f.name}')
    for f in sorted((ROOT / 'scripts').glob('*.py')):
        z.write(f, f'source/scripts/{f.name}')
    for f in sorted((ROOT / 'scripts').glob('*.sh')):
        z.write(f, f'source/scripts/{f.name}')

    z.writestr('weight_manifest.json', json.dumps({
        'weights_on_server': weight_manifest,
        'note': 'Q_corr_frozen best.pt/last.pt stay on server; SHA256 recorded. '
                'predictions/ hold per-sample (y/pred/snr/ids) for clean + CFO grid + global phase.',
    }, indent=2))

    readme = (
        'SSM round-5 REVIEW deliverable (supplementary audit + frozen-branch control)\n'
        '===========================================================================\n'
        'Independent of SSM_results_round5_iq.zip (left untouched).\n'
        'Contents:\n'
        '- reports/round5_review/ : resume_fix audit + diffs, errata (10 corrections),\n'
        '  conclusion_revised.md, evidence.md, smoke_overfit.json, test_log.txt,\n'
        '  pretrain_verify.json, branch_dependency.jsonl, eval_comparison.md/.json,\n'
        '  eval_summary.md, q_corr_frozen_train_stats.json, benchmark.json, tests/\n'
        '- runs/Q_corr_frozen_seed* : config/provenance/model_info/history/best_val/DONE + train.log\n'
        '- predictions/ : per-sample predictions for Q_corr_frozen (clean + CFO grid + phase)\n'
        '- configs/resolved_round5_review.json\n'
        '- source/ : post-fix ssm_exp/*.py + scripts/*.py + scripts/*.sh\n'
        '- weight_manifest.json : Q_corr_frozen weight paths + SHA256 (weights left on server)\n'
        '\n'
        'Q_corr_frozen: identical to Q_corr (same init, same S1 main path, W_f zero-init)\n'
        'except the aux-branch recurrence params (raw_alpha/theta/b) are frozen from init.\n'
        'Protocol: raw IQ -> perturb(CFO/phase) -> max_abs normalize -> model (order B).\n'
        'Val split only; test split was NOT used. 3 seeds reported with sample std (ddof=1).\n'
    )
    z.writestr('README.txt', readme)

print(f'wrote {ZIP} ({ZIP.stat().st_size/1e6:.2f} MB)')
print(f'{len(weight_manifest)} weight paths recorded')
