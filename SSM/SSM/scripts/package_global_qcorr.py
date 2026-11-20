#!/usr/bin/env python
"""Build deliverables/SSM_results_global_qcorr.zip for the G_Q_corr round.

Reproducibility contract (weights stay on server): the zip carries run metadata
(config/history/best_val/provenance/model_info/DONE), the 66 per-sample prediction files
of the 6 NEW runs, all tables, acceptance evidence, the source diff vs the round5_review
baseline, the exact commands, and weight_manifest.json (server paths + SHA256 of
best.pt/last.pt — weights themselves are NOT archived).
"""
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
REPORT = ROOT / 'reports' / 'global_qcorr'
PRED = ROOT / 'predictions_global_qcorr'
OUT_ZIP = ROOT / 'deliverables' / 'SSM_results_global_qcorr.zip'
STAGE = Path('/tmp/SSM_results_global_qcorr')

SEEDS = [17, 42, 2026]
NEW_VARIANTS = ['G_Q_corr', 'G_Q_corr_frozen']
NEW_RUNS = [f'{v}_seed{s}' for v in NEW_VARIANTS for s in SEEDS]
RUNS_BY_VARIANT = {
    'G': 'round3_aug', 'G_corr': 'round3_aug',
    'S1': 'round3_aug', 'S1_corr': 'round3_aug',
    'Q_corr': 'round5_iq', 'Q_corr_frozen': 'round5_review',
    'G_Q_corr': 'global_qcorr', 'G_Q_corr_frozen': 'global_qcorr',
}
ALL_VARIANTS = ['G', 'G_corr', 'G_Q_corr', 'G_Q_corr_frozen',
                'S1', 'S1_corr', 'Q_corr', 'Q_corr_frozen']
CONDS = ['clean', 'cfo0p001', 'cfom0p001', 'cfo0p002', 'cfom0p002',
         'cfo0p005', 'cfom0p005', 'cfo0p01', 'cfom0p01', 'cfo0p02', 'cfom0p02']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)
    elif src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    # ---- README -----------------------------------------------------------------
    (STAGE / 'README.md').write_text(README, encoding='utf-8')

    # ---- tables + report ---------------------------------------------------------
    for f in ['report.md', 'cfo_curves.png', 'aggregate.json', 'acceptance_summary.md']:
        if (REPORT / f).exists():
            cp(REPORT / f, STAGE / f)
    tbl = STAGE / 'tables'
    for f in ['main_table.csv', 'paired_diffs.csv', 'special_diffs.csv',
              'cfo_curves.csv', 'per_snr.csv', 'efficiency.csv']:
        if (REPORT / f).exists():
            cp(REPORT / f, tbl / f)
    # conclusion + errata (written by the report step; include if present)
    for f in ['conclusion.md', 'errata.md']:
        if (REPORT / f).exists():
            cp(REPORT / f, STAGE / f)

    # ---- configs ----------------------------------------------------------------
    cfg_dir = STAGE / 'configs'
    cp(ROOT / 'configs' / 'resolved_global_qcorr.json', cfg_dir / 'resolved_global_qcorr.json')
    cp(ROOT / 'configs' / 'data_raw.json', cfg_dir / 'data_raw.json')
    for name in NEW_RUNS:
        cp(ROOT / 'runs' / 'global_qcorr' / name / 'config.json', cfg_dir / f'{name}_config.json')

    # ---- source diff + new files ------------------------------------------------
    sd = STAGE / 'source_diff'
    # baseline from the round5_review deliverable
    import zipfile as _zf
    with _zf.ZipFile(ROOT / 'deliverables' / 'SSM_results_round5_review.zip') as z:
        for n in ['source/ssm_exp/model.py', 'source/ssm_exp/engine.py']:
            base = sd / 'baseline' / Path(n).name
            base.parent.mkdir(parents=True, exist_ok=True)
            base.write_bytes(z.read(n))
    for name in ['model.py', 'engine.py']:
        import difflib
        old = (sd / 'baseline' / name).read_text().splitlines()
        new = (ROOT / 'ssm_exp' / name).read_text().splitlines()
        (sd / f'{name}.diff').write_text(
            '\n'.join(difflib.unified_diff(old, new, fromfile=f'a/ssm_exp/{name}',
                                           tofile=f'b/ssm_exp/{name}', lineterm='')) + '\n',
            encoding='utf-8')
        cp(ROOT / 'ssm_exp' / name, sd / f'full_{name}')
    new_files = [ROOT / 'scripts' / f for f in
                 ['accept_global_qcorr.py', 'eval_global_qcorr.py', 'bench_global_qcorr.py',
                  'aggregate_global_qcorr.py', 'make_report_global_qcorr.py',
                  'package_global_qcorr.py', 'run_global_qcorr.sh']] + \
                [ROOT / 'tests' / f for f in ['test_global_qcorr.py', 'test_global_qcorr_resume.py']]
    for f in new_files:
        cp(f, sd / 'new_files' / f.relative_to(ROOT))

    # ---- env + commands ----------------------------------------------------------
    prov = json.loads((ROOT / 'runs' / 'global_qcorr' / 'G_Q_corr_seed17' / 'provenance.json').read_text())
    env = prov['environment']
    (STAGE / 'env.md').write_text(
        '```\n' + json.dumps(env, indent=2) + '\n```\n\ncode_sha256: ' + prov['code_sha256'] + '\n',
        encoding='utf-8')
    (STAGE / 'commands.md').write_text(commands_text(), encoding='utf-8')

    # ---- acceptance --------------------------------------------------------------
    acc = STAGE / 'acceptance'
    for s in SEEDS:
        cp(REPORT / 'acceptance' / f'seed{s}.json', acc / f'seed{s}.json')
    cp(REPORT / 'acceptance' / 'check4b_resume.json', acc / 'check4b_resume.json')

    # ---- runs (metadata only) ----------------------------------------------------
    run_root = STAGE / 'runs'
    for name in NEW_RUNS:
        src = ROOT / 'runs' / 'global_qcorr' / name
        dst = run_root / name
        for f in ['config.json', 'history.jsonl', 'best_val.json', 'provenance.json',
                  'model_info.json', 'DONE.json']:
            cp(src / f, dst / f)
        # training stdout log
        cp(ROOT / 'runs' / 'global_qcorr' / f'{name}.log', dst / 'train.log')

    # ---- predictions (66 new) ----------------------------------------------------
    for name in NEW_RUNS:
        tag = f'global_qcorr_{name}'
        for cond in CONDS:
            src = PRED / f'{tag}_{cond}.npz'
            if src.exists():
                cp(src, STAGE / 'predictions' / f'{tag}_{cond}.npz')

    # ---- weight manifest (SHA256 only; weights stay on server) -------------------
    manifest = []
    for name in NEW_RUNS:
        for w in ['best.pt', 'last.pt']:
            p = ROOT / 'runs' / 'global_qcorr' / name / w
            manifest.append({'run': name, 'weight': w, 'server_path': str(p),
                             'sha256': sha256(p), 'bytes': p.stat().st_size})
    (STAGE / 'weight_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    # ---- zip ---------------------------------------------------------------------
    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(STAGE.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(STAGE.parent))
    n_files = sum(1 for p in STAGE.rglob('*') if p.is_file())
    print(json.dumps({'zip': str(OUT_ZIP), 'files': n_files, 'bytes': OUT_ZIP.stat().st_size,
                      'weight_manifest_entries': len(manifest)}, indent=2))


def commands_text():
    run_dirs = ','.join(
        f'runs/{RUNS_BY_VARIANT[v]}/{v}_seed{s}' for v in ALL_VARIANTS for s in SEEDS)
    bench_dirs = ','.join(['runs/round3_aug/G_seed17', 'runs/round3_aug/G_corr_seed17',
                           'runs/global_qcorr/G_Q_corr_seed17',
                           'runs/global_qcorr/G_Q_corr_frozen_seed17'])
    return f'''# G_Q_corr round — reproduction commands

## training (6 runs; 3 GPUs in parallel, each seed runs learnable then frozen)
tmux new-session -d -s gqcorr 'bash scripts/run_global_qcorr.sh'
# equivalently, per run (example, seed 17):
python -m ssm_exp.engine --config configs/resolved_global_qcorr.json \\
  --manifest configs/data_raw.json --out runs/global_qcorr/G_Q_corr_seed17 \\
  --variant G_Q_corr --seed 17 --device cuda:0
python -m ssm_exp.engine --config configs/resolved_global_qcorr.json \\
  --manifest configs/data_raw.json --out runs/global_qcorr/G_Q_corr_frozen_seed17 \\
  --variant G_Q_corr_frozen --seed 17 --device cuda:0

## acceptance (4 checks x 3 seeds + resume pytest)
python scripts/accept_global_qcorr.py
python -m pytest tests/test_global_qcorr.py tests/test_global_qcorr_resume.py -q

## evaluation (24 runs x 11 CFO conditions, order B; saves per-sample predictions)
python scripts/eval_global_qcorr.py --runs {run_dirs} --manifest configs/data_raw.json

## efficiency (G / G_corr / G_Q_corr / G_Q_corr_frozen at L=1024)
python scripts/bench_global_qcorr.py --runs {bench_dirs}

## aggregation + report (recomputes OA/Macro-F1 from predictions)
python scripts/aggregate_global_qcorr.py
python scripts/make_report_global_qcorr.py

## packaging
python scripts/package_global_qcorr.py
'''

README = f'''# SSM Global + 8-channel filtered complex correlation (G_Q_corr round)

Protocol: RML2018A_cls_dev_global_qcorr. Built {datetime.now(timezone.utc).isoformat()}.

## What this round asks
On the Global backbone (raw-IQ PatchEmbedding frontend, absolute sinusoidal positions,
6-layer global attention, mean pool, head), replace the raw-IQ complex correlation of
G_corr with an **8-channel filtered complex correlation** (ComplexRecurBranch over raw
samples; 7 adjacent-pair normalized lag-1 complex autocorrelations per patch -> 8 complex
-> 16 real; zero-init Linear(16,176) fusion, added once per token).

- `G_Q_corr`: the 32 filter params (8 decay + 8 angle + 8 complex input coeffs) are learned.
- `G_Q_corr_frozen`: those 32 filter params are fixed at their shared init.

Both train the fusion/backbone/head. Comparisons: G_Q_corr-G_corr, G_Q_corr_frozen-G_corr
(feature-replacement effect), G_Q_corr-G_Q_corr_frozen (learned vs fixed filter),
cross-backbone G_Q_corr-G_corr vs Q_corr-S1_corr, and total gain over plain G.

## Layout
- `report.md` — main tables; `tables/*.csv` + `aggregate.json` — machine-readable.
- `conclusion.md`, `errata.md` — findings and caveats.
- `acceptance/` — 4-check acceptance evidence per seed + resume pytest result.
- `runs/` — 6 new runs (config/history/best_val/provenance/model_info/DONE + train.log).
- `predictions/` — 66 per-sample prediction files (6 runs x 11 conditions).
- `configs/`, `commands.md`, `env.md`, `source_diff/` — reproduction.
- `weight_manifest.json` — server paths + SHA256 of best.pt/last.pt (weights stay on server).

## Key numbers (params, fp32/complex64)
G=2,251,592; G_corr=2,252,120; G_Q_corr=2,254,440; G_Q_corr_frozen=2,254,440
(frozen trainable=2,254,408). See `report.md` for the OA tables.
'''

if __name__ == '__main__':
    main()
