"""Package the round-2 overnight deliverables into a single zip.

Collects source diffs, configs, data inventory + split hashes, checkpoint SHA256, curves,
per-sample predictions and reports, then writes
  round2_overnight/deliverables/SSM_L_round2_overnight_results.zip
(weights stay on the server; only SHA256 manifests are included).
"""
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R2 = Path(__file__).resolve().parent.parent
RUNS = R2 / 'runs'
REP = R2 / 'reports'
STAGE = R2 / 'deliverables' / 'staging'
ZIP = R2 / 'deliverables' / 'SSM_L_round2_overnight_results.zip'


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / 'src').mkdir(parents=True)
    (STAGE / 'configs').mkdir(parents=True)
    (STAGE / 'reports').mkdir(parents=True)
    (STAGE / 'curves').mkdir(parents=True)
    (STAGE / 'predictions').mkdir(parents=True)

    # source
    for f in sorted(R2.glob('*.py')):
        shutil.copy(f, STAGE / 'src')
    for f in sorted((R2 / 'scripts').glob('*.py')):
        shutil.copy(f, STAGE / 'src')

    # configs (round-2 + radar)
    for f in sorted((R2 / 'configs').glob('*.json')):
        shutil.copy(f, STAGE / 'configs')

    # reports
    for f in sorted(REP.glob('*')):
        if f.is_file():
            shutil.copy(f, STAGE / 'reports')

    # curves
    for f in sorted((REP / 'curves').glob('*.png')):
        shutil.copy(f, STAGE / 'curves')

    # per-sample predictions: A random probes + D probes/finetunes (RML val) + E radar eval
    pred_root = STAGE / 'predictions'
    for kind in ('probe', 'finetune'):
        for d in sorted((RUNS / kind).glob('*')):
            vp = d / 'val_predictions.npz'
            if vp.exists():
                shutil.copy(vp, pred_root / f'{d.name}_val_preds.npz')
    for f in sorted((RUNS / 'radar_eval_preds').glob('*.npz')):
        shutil.copy(f, pred_root / f.name)

    # launch/resume commands + status
    (STAGE / 'commands.md').write_text(
        '# Round-2 launch / resume commands\n\n'
        '```bash\n'
        'cd /home/user_wzk/wzk/projects/SSM/SSM_L\n'
        'source /home/user_wzk/miniconda3/etc/profile.d/conda.sh && conda activate SFM\n'
        '# D (structure ablation, 3-GPU orchestrator, resume-safe)\n'
        'python round2_overnight/scripts/orchestrate_d.py --phase all\n'
        '# A (random frozen probes)\n'
        'python round2_overnight/scripts/run_a.py\n'
        '# B (corrected denoise eval)\n'
        'python round2_overnight/scripts/run_b.py --device cuda:2\n'
        '# C (low-label frozen probe)\n'
        'python round2_overnight/scripts/run_c.py --device cpu\n'
        '# E (radar transfer; cache on GPU, head/LR on CPU)\n'
        'python round2_overnight/scripts/run_e.py --device cuda:0\n'
        '# CFO eval after D finetunes\n'
        'python round2_overnight/scripts/eval_finetune_cfo_r2.py --device cuda:0\n'
        '# aggregation + curves\n'
        'python round2_overnight/scripts/aggregate.py\n'
        '```\n\n'
        'All long jobs were launched under `tmux` (sessions ssml_r2_d / _a / _b / _c / _e).\n'
        'Checkpoint SHA256 + split hashes are in reports/split_hashes.json and '
        'reports/checkpoint_sha256.json.\n'
    )
    shutil.copy(ROOT / 'latest_status.md', STAGE / 'latest_status.md')
    conc = ROOT / 'round2_overnight' / 'reports' / 'conclusion.md'
    if conc.exists():
        shutil.copy(conc, STAGE / 'conclusion.md')

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(STAGE.rglob('*')):
            if f.is_file():
                z.write(f, f.relative_to(STAGE))
    print('wrote', ZIP)
    print('staging files:', sum(1 for f in STAGE.rglob('*') if f.is_file()))


if __name__ == '__main__':
    main()
