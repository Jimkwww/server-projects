"""A. Random frozen probes: G/Q random-init full-label probes (6 runs).

Generates random-init encoder checkpoints for G and Q from the round-1 shared-init cache, then
runs the round-2 downstream probe (identical to round-1 probe protocol: AdamW head, 30 epochs,
cosine, clean-val-OA earliest-tie selection). The random features are also cached to disk for
reuse by C.
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R2 = Path(__file__).resolve().parent.parent
RUNS = R2 / 'runs'
LOG = R2 / 'logs'
SEEDS = (17, 42, 2026)
BACKBONES = ('G', 'Q')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    # Generate G/Q random checkpoints (reuses round-1 shared-init cache).
    if not args.dry_run:
        subprocess.run([sys.executable, str(R2 / 'scripts' / 'gen_random_ckpts.py'),
                        '--backbone', 'G', '--backbone', 'Q', '--device', 'cpu'], check=True)

    specs = []
    for bb in BACKBONES:
        for s in SEEDS:
            specs.append(dict(backbone=bb, seed=s))

    def done(s):
        return (RUNS / 'probe' / f"{s['backbone']}_random_seed{s['seed']}" / 'DONE.json').exists()

    todo = [s for s in specs if not done(s)]
    print(f'A: {len(todo)} todo / {len(specs)} total', flush=True)
    if args.dry_run:
        for s in todo:
            print('  ', s)
        return

    LOG.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(3, max(1, len(todo)))) as ex:
        futs = {}
        for s in todo:
            pt = RUNS / 'random_ckpt' / f"{s['backbone']}_seed{s['seed']}.pt"
            out = RUNS / 'probe' / f"{s['backbone']}_random_seed{s['seed']}"
            c = [sys.executable, '-m', 'round2_overnight.downstream', '--mode', 'probe',
                 '--config', str(ROOT / 'configs' / 'probe.json'),
                 '--manifest', str(ROOT / 'configs' / 'data.json'),
                 '--backbone', s['backbone'], '--pretrained', str(pt),
                 '--out', str(out), '--seed', str(s['seed']),
                 '--device', f'cuda:{s["seed"] % 3}', '--workers', '4']
            logf = LOG / f"probe_{s['backbone']}_random_seed{s['seed']}.log"
            futs[ex.submit(lambda c=c, logf=logf: (subprocess.run(c, stdout=open(logf, 'w'),
                     stderr=subprocess.STDOUT).returncode == 0, logf))] = s
        for fut in as_completed(futs):
            ok, logf = fut.result()
            s = futs[fut]
            print(f'[{"ok" if ok else "FAIL"}] {s["backbone"]} random seed{s["seed"]} -> {logf}', flush=True)


if __name__ == '__main__':
    main()
