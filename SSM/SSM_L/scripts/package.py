"""SSM_L round-1 deliverable packaging.

Collects the required artifacts into a staging dir, writes a weights manifest (SHA256 paths on
the server, weights themselves stay put), and zips to deliverables/SSM_L_round1_results.zip.

Also prints the completed / reused / failed run count for the final report.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUPS = ['G_mae', 'G_denoise', 'Q_mae', 'Q_denoise']
SEEDS = [17, 42, 2026]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run_counts():
    completed, failed, reused = 0, 0, 6  # 6 reused baselines
    for kind in ('pretrain', 'probe', 'finetune'):
        for g in GROUPS:
            for s in SEEDS:
                d = ROOT / 'runs' / kind / f'{g}_seed{s}'
                if (d / 'DONE.json').exists():
                    completed += 1
                elif (d / 'last.pt').exists():
                    failed += 1
    return completed, failed, reused


def collect_weights_manifest(staging):
    manifest = {}
    def add(key, path):
        if Path(path).exists():
            manifest[key] = {'path': str(path), 'sha256': sha256(path)}

    for g in GROUPS:
        for s in SEEDS:
            add(f'pretrain/{g}_seed{s}/encoder_final.pt',
                ROOT / 'runs' / 'pretrain' / f'{g}_seed{s}' / 'encoder_final.pt')
            add(f'probe/{g}_seed{s}/best.pt', ROOT / 'runs' / 'probe' / f'{g}_seed{s}' / 'best.pt')
            add(f'finetune/{g}_seed{s}/best.pt', ROOT / 'runs' / 'finetune' / f'{g}_seed{s}' / 'best.pt')
    old = ROOT.parent / 'SSM'
    for name, sub, prefix in [('G_scratch', 'round3_aug', 'G'),
                              ('Q_scratch', 'round5_iq', 'Q_corr')]:
        for s in SEEDS:
            add(f'baseline/{name}_seed{s}/best.pt', old / 'runs' / sub / f'{prefix}_seed{s}' / 'best.pt')
    (staging / 'weights_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=str(ROOT / 'deliverables' / 'SSM_L_round1_results.zip'))
    args = ap.parse_args()

    staging = ROOT / 'deliverables' / 'staging'
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    # data inventory + manifests + configs
    for f in ['data/dataset_inventory.csv', 'source_manifest.json',
              'configs/data.json', 'configs/pretrain.json', 'configs/probe.json',
              'configs/finetune.json']:
        src = ROOT / f
        if src.exists():
            dst = staging / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # source code (new code + reused ssm_exp snapshot)
    for d in ['pretrain', 'ssm_exp', 'scripts', 'tests']:
        src = ROOT / d
        dst = staging / 'src' / d
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

    # reports (acceptance, results, curves, cfo, denoise diagnostic)
    reports = ROOT / 'reports'
    (staging / 'reports').mkdir(parents=True, exist_ok=True)
    for name in ['acceptance.md', 'acceptance.json', 'results.json', 'paired_tables.md',
                 'pretrain_curves.png', 'probe_curves.png', 'finetune_curves.png',
                 'finetune_cfo.json', 'finetune_cfo.md', 'denoise_diagnostic.json',
                 'conclusion.md', 'launch_resume.md']:
        p = reports / name
        if p.exists():
            shutil.copy2(p, staging / 'reports' / name)

    # per-sample predictions for every completed run (sample_id/true_label/pred_label/SNR)
    for kind in ('probe', 'finetune'):
        for g in GROUPS:
            for s in SEEDS:
                p = ROOT / 'runs' / kind / f'{g}_seed{s}' / 'val_predictions.npz'
                if p.exists():
                    dst = staging / 'predictions' / kind / f'{g}_seed{s}.npz'
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
    old = ROOT.parent / 'SSM'
    for name, sub, prefix in [('G_scratch', 'round3_aug', 'G'),
                              ('Q_scratch', 'round5_iq', 'Q_corr')]:
        for s in SEEDS:
            p = old / 'runs' / sub / f'{prefix}_seed{s}' / 'val_predictions.npz'
            if p.exists():
                dst = staging / 'predictions' / 'baseline' / f'{name}_seed{s}.npz'
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)

    weights = collect_weights_manifest(staging)
    completed, failed, reused = run_counts()
    summary = {'completed_runs': completed, 'failed_runs': failed, 'reused_baselines': reused,
               'total': completed + failed + reused, 'weights_manifest': weights}
    (staging / 'run_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    zip_path = Path(args.zip)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(staging.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(staging))
    print(f'completed={completed} failed={failed} reused={reused}')
    print(f'wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)')
    shutil.rmtree(staging)


if __name__ == '__main__':
    main()
