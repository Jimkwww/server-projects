#!/usr/bin/env python
"""Generic detached training supervisor (flat pool over variants x seeds).

Usage: supervise_generic.py --out runs/round3 --config configs/resolved_round2.json \
       --manifest configs/data.json --variants G_corr --seeds 17 42 2026

Runs jobs across 3 GPUs (round-robin via a slot queue); skips DONE.json runs,
resumes from last.pt when present. Launch inside tmux so it survives disconnects.
"""
import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path
import queue

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
GPUS = ['cuda:0', 'cuda:1', 'cuda:2']

gpu_slots = queue.Queue()
for g in GPUS:
    gpu_slots.put(g)
print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def run_job(out, config, manifest, variant, seed, workers):
    gpu = gpu_slots.get()
    try:
        rundir = out / f'{variant}_seed{seed}'
        if (rundir / 'DONE.json').exists():
            log(f'skip {variant}_seed{seed} (DONE)')
            return
        cmd = [sys.executable, '-m', 'ssm_exp.engine', '--config', config,
               '--manifest', manifest, '--out', str(rundir), '--variant', variant,
               '--seed', str(seed), '--device', gpu, '--workers', str(workers)]
        resume = (rundir / 'last.pt').exists()
        if resume:
            cmd += ['--resume']
        out.mkdir(parents=True, exist_ok=True)
        logfile = out / f'{variant}_seed{seed}.log'
        with open(logfile, 'a') as f:
            f.write(f'# cmd: {" ".join(cmd)}\n# started: {time.strftime("%F %T")}\n')
        log(f'launch {variant}_seed{seed} on {gpu} (resume={resume})')
        with open(logfile, 'a') as f:
            r = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        log(f'finish {variant}_seed{seed} exit={r.returncode}')
    finally:
        gpu_slots.put(gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--variants', required=True, help='comma-separated')
    ap.add_argument('--seeds', required=True, nargs='+', type=int)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    out = ROOT / a.out
    variants = [v.strip() for v in a.variants.split(',') if v.strip()]
    jobs = [(v, s) for s in a.seeds for v in variants]  # seed-major order
    threads = []
    for v, s in jobs:
        t = threading.Thread(target=run_job, args=(out, a.config, a.manifest, v, s, a.workers))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    log('ALL DONE')


if __name__ == '__main__':
    main()
