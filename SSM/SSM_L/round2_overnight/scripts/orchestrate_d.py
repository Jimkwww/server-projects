"""Round-2 D orchestrator: 2x2 structure ablation (G_Qlearn / S1plain x MAE / denoise).

Runs in dependency order with 3-GPU parallelism, skip-if-complete and --resume for interrupted
runs. Phases:
  random    -> random-init encoder checkpoints (G_Qlearn/S1plain) for the random probes
  pretrain  -> 12 runs: {G_Qlearn,S1plain} x {mae,denoise} x seed{17,42,2026}
  probe     -> 12 pretrain probes + 6 random probes
  finetune  -> 12 runs

G-MAE / Q-MAE (round-1) provide the other two 2x2 cells (read-only).
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R2 = Path(__file__).resolve().parent.parent
BACKBONES = ('G_Qlearn', 'S1plain')
TASKS = ('mae', 'denoise')
SEEDS = (17, 42, 2026)
N_GPU = 3
RUNS = R2 / 'runs'
LOG = R2 / 'logs'


def out_dir(kind, backbone, task, seed):
    return RUNS / kind / f'{backbone}_{task}_seed{seed}'


def done(spec):
    k = spec['kind']
    if k == 'random':
        return (RUNS / 'random_ckpt' / f"{spec['backbone']}_seed{spec['seed']}.pt").exists()
    if k == 'pretrain':
        o = out_dir(k, spec['backbone'], spec['task'], spec['seed'])
        return (o / 'DONE.json').exists() and (o / 'encoder_final.pt').exists()
    o = out_dir(k, spec['backbone'], spec['task'], spec['seed'])
    return (o / 'DONE.json').exists()


def run_one(spec, gpu):
    LOG.mkdir(parents=True, exist_ok=True)
    if spec['kind'] == 'random':
        c = [sys.executable, str(R2 / 'scripts' / 'gen_random_ckpts.py'),
             '--backbone', spec['backbone'], '--device', 'cpu']
        logf = LOG / f"random_{spec['backbone']}_seed{spec['seed']}.log"
    elif spec['kind'] == 'pretrain':
        o = out_dir('pretrain', spec['backbone'], spec['task'], spec['seed'])
        resume = (o / 'last.pt').exists()
        c = [sys.executable, '-m', 'round2_overnight.engine',
             '--config', str(ROOT / 'configs' / 'pretrain.json'),
             '--manifest', str(ROOT / 'configs' / 'data.json'),
             '--backbone', spec['backbone'], '--task', spec['task'],
             '--out', str(o), '--runs-root', str(RUNS / 'pretrain'),
             '--seed', str(spec['seed']), '--device', f'cuda:{gpu}', '--workers', '4']
        if resume:
            c.append('--resume')
        logf = LOG / f"pretrain_{spec['backbone']}_{spec['task']}_seed{spec['seed']}.log"
    else:
        if spec['task'] == 'random':
            pt = RUNS / 'random_ckpt' / f"{spec['backbone']}_seed{spec['seed']}.pt"
        else:
            pt = RUNS / 'pretrain' / f"{spec['backbone']}_{spec['task']}_seed{spec['seed']}" / 'encoder_final.pt'
        o = out_dir(spec['kind'], spec['backbone'], spec['task'], spec['seed'])
        resume = (o / 'last.pt').exists()
        c = [sys.executable, '-m', 'round2_overnight.downstream', '--mode', spec['kind'],
             '--config', str(ROOT / 'configs' / ('probe.json' if spec['kind'] == 'probe' else 'finetune.json')),
             '--manifest', str(ROOT / 'configs' / 'data.json'),
             '--backbone', spec['backbone'], '--pretrained', str(pt),
             '--out', str(o), '--seed', str(spec['seed']), '--device', f'cuda:{gpu}', '--workers', '4']
        if resume:
            c.append('--resume')
        logf = LOG / f"{spec['kind']}_{spec['backbone']}_{spec['task']}_seed{spec['seed']}.log"
    with open(logf, 'w') as f:
        r = subprocess.run(c, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode == 0, str(logf)


def build_specs():
    specs = []
    for bb in BACKBONES:
        for s in SEEDS:
            specs.append(dict(kind='random', backbone=bb, seed=s))
    for bb in BACKBONES:
        for task in TASKS:
            for s in SEEDS:
                specs.append(dict(kind='pretrain', backbone=bb, task=task, seed=s))
    for mode in ('probe', 'finetune'):
        for bb in BACKBONES:
            for task in TASKS:
                for s in SEEDS:
                    specs.append(dict(kind=mode, backbone=bb, task=task, seed=s))
    for bb in BACKBONES:
        for s in SEEDS:
            specs.append(dict(kind='probe', backbone=bb, task='random', seed=s))
    return specs


def run_phase(specs, phase, dry):
    todo = [s for s in specs if s['kind'] == phase and not done(s)]
    total = len([s for s in specs if s['kind'] == phase])
    print(f'===== PHASE {phase}: {len(todo)} todo / {total} total =====', flush=True)
    if dry:
        for s in todo:
            print('  ', s)
        return
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=N_GPU) as ex:
        futs = {ex.submit(run_one, s, s['seed'] % N_GPU): s for s in todo}
        for fut in as_completed(futs):
            ok, logf = fut.result()
            s = futs[fut]
            print(f'[{"ok" if ok else "FAIL"}] {s["kind"]} {s["backbone"]} '
                  f'{s.get("task","")} seed{s["seed"]} -> {logf}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['random', 'pretrain', 'probe', 'finetune', 'all'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    specs = build_specs()
    phases = ['random', 'pretrain', 'probe', 'finetune'] if args.phase in (None, 'all') else [args.phase]
    for phase in phases:
        run_phase(specs, phase, args.dry_run)


if __name__ == '__main__':
    main()
