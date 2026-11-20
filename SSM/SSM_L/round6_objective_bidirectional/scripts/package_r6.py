"""Package round6 deliverables into a tarball (scripts + reports + figs + predictions; NO weights).

Excludes pretrain weights, shared_init, features caches, best/last downstream checkpoints, logs,
job lists. Run after everything is done.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R6 = ROOT / 'round6_objective_bidirectional'
OUT = ROOT / 'deliverables' / 'round6_objective_bidirectional_package.tar.gz'

OUT.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    'tar', 'czf', str(OUT),
    '--exclude=round6_objective_bidirectional/runs/pretrain/f1/*/*.pt',
    '--exclude=round6_objective_bidirectional/runs/pretrain/f1/shared_init',
    '--exclude=round6_objective_bidirectional/runs/downstream/features',
    '--exclude=round6_objective_bidirectional/runs/downstream/finetune_*/*/*/best.pt',
    '--exclude=round6_objective_bidirectional/runs/downstream/finetune_*/*/*/last.pt',
    '--exclude=round6_objective_bidirectional/logs',
    '--exclude=round6_objective_bidirectional/scripts/*_jobs.txt',
    'round6_objective_bidirectional',
]
print('[package]', ' '.join(cmd))
subprocess.run(cmd, cwd=str(ROOT), check=True)
print(f'[package] wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    import sys
    if '--dry-run' in sys.argv:
        subprocess.run(['tar', 'tzf', str(OUT)], check=False)
    else:
        print('[done] package_r6.py')
