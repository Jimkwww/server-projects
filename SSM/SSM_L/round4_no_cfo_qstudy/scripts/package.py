"""Round-4 deliverable packaging (lean).

Bundles ONLY what the task book asks for: code + configs + reports + per-sample predictions.
Weights stay on the server. Excluded on purpose (already summarized in reports/):
  - logs/*.log, __pycache__/*.pyc
  - *.pt / *.pth weights
  - per-run history.jsonl / DONE.json / best_val.json / provenance.json / config.json
    (aggregated by run_manifest.csv + checkpoint_sha256.csv + results_summary.json)
"""
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent   # SSM_L
R4 = ROOT / 'round4_no_cfo_qstudy'
OUT = R4 / 'round4_package.tar.gz'

# Relative-to-ROOT globs to include.
INCLUDE = [
    # round-4 code (scripts + top-level modules)
    'round4_no_cfo_qstudy/*.py',
    'round4_no_cfo_qstudy/scripts/*.py',
    'round4_no_cfo_qstudy/__init__.py',
    # reference source trees the round-4 code imports
    'ssm_exp/**/*.py',
    'pretrain/**/*.py',
    'round2_overnight/**/*.py',
    # configs + data manifests
    'round4_no_cfo_qstudy/configs/*.json',
    'configs/data.json',
    'round2_overnight/configs/radchar.json',
    # reports (md/csv/json + curve PNGs)
    'round4_no_cfo_qstudy/reports/*.md',
    'round4_no_cfo_qstudy/reports/*.csv',
    'round4_no_cfo_qstudy/reports/*.json',
    'round4_no_cfo_qstudy/reports/curves/*.png',
    # per-sample predictions (no weights, no per-run metadata)
    'round4_no_cfo_qstudy/runs/**/val_predictions.npz',
    'round4_no_cfo_qstudy/runs/**/eval_predictions.npz',
    'round4_no_cfo_qstudy/runs/radar_baselines/*_predictions.npz',
]


def main():
    if OUT.exists():
        OUT.unlink()
    seen = set()
    with tarfile.open(OUT, 'w:gz') as tf:
        for pat in INCLUDE:
            for p in sorted(ROOT.glob(pat)):
                if not p.is_file() or p.name == 'round4_package.tar.gz':
                    continue
                arc = str(p.relative_to(ROOT))
                if arc in seen:
                    continue
                seen.add(arc)
                tf.add(p, arcname=arc)
    sz = OUT.stat().st_size / 1e6
    print(f'[package] wrote {OUT} ({sz:.1f} MB, {len(seen)} files)')


if __name__ == '__main__':
    main()
