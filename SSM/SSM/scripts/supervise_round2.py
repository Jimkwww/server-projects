#!/usr/bin/env python
"""Round-2 minimal-ablation training supervisor (seed17 first, then 42/2026).

Launched inside a tmux session so it survives VSCode/SSH disconnects. Each
(variant, seed) runs as a single subprocess on one GPU (no DDP). Seeds are
given as argv (e.g. `supervise_round2.py 17` then `supervise_round2.py 42 2026`).
Jobs within a seed wave run in parallel across the 3 GPU slots. Uses --resume
when last.pt exists, skips completed DONE.json runs.
"""
import subprocess
import sys
import threading
import time
from pathlib import Path
import queue

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
CONFIG = 'configs/resolved_round2.json'
MANIFEST = 'configs/data.json'
OUT = ROOT / 'runs' / 'round2'
SEEDS = [int(s) for s in sys.argv[1:]]
VARIANTS = ['S1_energy', 'S1_corr', 'G_full']
GPUS = ['cuda:0', 'cuda:1', 'cuda:2']
WORKERS = 4

gpu_slots = queue.Queue()
for g in GPUS:
    gpu_slots.put(g)
print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def run_job(seed, variant):
    gpu = gpu_slots.get()
    try:
        rundir = OUT / f'{variant}_seed{seed}'
        if (rundir / 'DONE.json').exists():
            log(f'skip {variant}_seed{seed} (DONE)')
            return
        cmd = [sys.executable, '-m', 'ssm_exp.engine', '--config', CONFIG,
               '--manifest', MANIFEST, '--out', str(rundir), '--variant', variant,
               '--seed', str(seed), '--device', gpu, '--workers', str(WORKERS)]
        resume = (rundir / 'last.pt').exists()
        if resume:
            cmd += ['--resume']
        logfile = OUT / f'{variant}_seed{seed}.log'
        with open(logfile, 'a') as f:
            f.write(f'# cmd: {" ".join(cmd)}\n# started: {time.strftime("%F %T")}\n')
        log(f'launch {variant}_seed{seed} on {gpu} (resume={resume})')
        with open(logfile, 'a') as f:
            r = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        log(f'finish {variant}_seed{seed} exit={r.returncode}')
    finally:
        gpu_slots.put(gpu)


def main():
    for seed in SEEDS:
        threads = [threading.Thread(target=run_job, args=(seed, v)) for v in VARIANTS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        log(f'=== seed {seed} wave complete ===')
    log('ALL ROUND2 RUNS DONE')


if __name__ == '__main__':
    main()
