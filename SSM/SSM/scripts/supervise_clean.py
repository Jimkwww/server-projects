#!/usr/bin/env python
"""Detached campaign supervisor: runs all clean runs across 3 GPUs.

Launched inside a tmux session so it survives VSCode/SSH disconnects. Each
(variant, seed) runs as a single subprocess on one GPU (no DDP). Jobs within a
seed run in parallel across the 3 GPU slots; seeds are processed sequentially
(seed 17 first for health check, then 42/2026). Uses --resume when last.pt
exists, skips completed DONE.json runs.
"""
import subprocess
import sys
import threading
import time
from pathlib import Path
import queue

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
CONFIG = 'configs/resolved.json'
MANIFEST = 'configs/data.json'
OUT = ROOT / 'runs' / 'clean'
SEEDS = [17, 42, 2026]
VARIANTS = ['G', 'S0', 'S1', 'S2', 'S2_concat']
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
    log('ALL CLEAN RUNS DONE')


if __name__ == '__main__':
    main()
