"""Package round5 deliverables into a tarball (scripts + reports + figs + predictions; NO weights).

Excludes runs/pretrain weights, shared_init, features caches. Run after everything is done.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R5 = ROOT / 'round5_epoch_scale'
OUT = ROOT / 'deliverables' / 'round5_epoch_scale_package.tar.gz'

OUT.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    'tar', 'czf', str(OUT),
    '--exclude=round5_epoch_scale/runs/pretrain/*/*.pt',
    '--exclude=round5_epoch_scale/runs/pretrain/*/shared_init',
    '--exclude=round5_epoch_scale/runs/downstream/features',
    '--exclude=round5_epoch_scale/runs/downstream/finetune_*/*/*/best.pt',
    '--exclude=round5_epoch_scale/runs/downstream/finetune_*/*/*/last.pt',
    '--exclude=round5_epoch_scale/logs',
    '--exclude=round5_epoch_scale/scripts/*_jobs.txt',
    'round5_epoch_scale',
]
print('[package]', ' '.join(cmd))
subprocess.run(cmd, cwd=str(ROOT), check=True)
print(f'[package] wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    # list what is included (dry-run)
    import sys
    if '--dry-run' in sys.argv:
        subprocess.run(['tar', 'tzf', str(OUT)], check=False)
    else:
        print('[done] package_r5.py')
