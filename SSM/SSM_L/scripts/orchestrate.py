"""SSM_L round-1 run orchestrator.

Runs the full matrix in dependency order with one process per GPU (3-way parallel), resuming
interrupted runs and skipping completed ones (a run is complete when DONE.json exists and, for
pretraining, encoder_final.pt exists). Staged per the task book section 5:

  pretrain_s17  -> probe_s17 + finetune_s17  -> pretrain_rest -> probe_rest + finetune_rest

The seed-17 stages are verified (all four groups train and their epoch-50 weights load) before
the remaining seeds are launched; if verification fails the pipeline stops instead of burning
the rest of the budget.

Usage (inside tmux with the SFM env active):
    python scripts/orchestrate.py --all           # everything, staged + verified
    python scripts/orchestrate.py --phase pretrain_s17
    python scripts/orchestrate.py --dry-run
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKBONES = ('G', 'Q')
TASKS = ('mae', 'denoise')
SEEDS = (17, 42, 2026)
N_GPU = 3

PHASE_ORDER = ['pretrain_s17', 'probe_s17', 'finetune_s17',
               'pretrain_rest', 'probe_rest', 'finetune_rest']


def cmd_pretrain(backbone, task, seed, gpu, resume):
    out = ROOT / 'runs' / 'pretrain' / f'{backbone}_{task}_seed{seed}'
    c = [sys.executable, '-m', 'pretrain.engine',
         '--config', str(ROOT / 'configs' / 'pretrain.json'),
         '--manifest', str(ROOT / 'configs' / 'data.json'),
         '--backbone', backbone, '--task', task,
         '--out', str(out), '--runs-root', str(ROOT / 'runs' / 'pretrain'),
         '--seed', str(seed), '--device', f'cuda:{gpu}', '--workers', '4']
    if resume:
        c.append('--resume')
    return c, out


def cmd_downstream(mode, backbone, task, seed, gpu, resume):
    cfg = 'probe' if mode == 'probe' else 'finetune'
    out = ROOT / 'runs' / mode / f'{backbone}_{task}_seed{seed}'
    pt = ROOT / 'runs' / 'pretrain' / f'{backbone}_{task}_seed{seed}' / 'encoder_final.pt'
    c = [sys.executable, '-m', 'pretrain.downstream', '--mode', mode,
         '--config', str(ROOT / 'configs' / f'{cfg}.json'),
         '--manifest', str(ROOT / 'configs' / 'data.json'),
         '--backbone', backbone, '--pretrained', str(pt),
         '--out', str(out), '--seed', str(seed), '--device', f'cuda:{gpu}', '--workers', '4']
    if resume:
        c.append('--resume')
    return c, out


def build_specs():
    specs = []
    for backbone in BACKBONES:
        for task in TASKS:
            for seed in SEEDS:
                phase = ('pretrain_s17' if seed == 17 else 'pretrain_rest')
                specs.append(dict(phase=phase, kind='pretrain', backbone=backbone,
                                  task=task, seed=seed))
    for mode in ('probe', 'finetune'):
        for backbone in BACKBONES:
            for task in TASKS:
                for seed in SEEDS:
                    phase = (f'{mode}_s17' if seed == 17 else f'{mode}_rest')
                    specs.append(dict(phase=phase, kind=mode, backbone=backbone,
                                      task=task, seed=seed))
    return specs


def run_status(spec):
    if spec['kind'] == 'pretrain':
        out = ROOT / 'runs' / 'pretrain' / f"{spec['backbone']}_{spec['task']}_seed{spec['seed']}"
        done = (out / 'DONE.json').exists() and (out / 'encoder_final.pt').exists()
    else:
        out = ROOT / 'runs' / spec['kind'] / f"{spec['backbone']}_{spec['task']}_seed{spec['seed']}"
        done = (out / 'DONE.json').exists()
    return out, done


def verify_seed17_group(kind):
    """All four seed-17 groups must have a valid run of the right epoch count, no NaN."""
    want_epochs = {'pretrain': 50, 'probe': 30, 'finetune': 50}[kind]
    want_final = want_epochs - 1
    ok = True
    detail = []
    for backbone in BACKBONES:
        for task in TASKS:
            if kind == 'pretrain':
                out = ROOT / 'runs' / 'pretrain' / f'{backbone}_{task}_seed17'
                done = out / 'DONE.json'
                hist = out / 'history.jsonl'
                ok_run = done.exists() and (out / 'encoder_final.pt').exists() and hist.exists()
                if ok_run:
                    d = json.loads(done.read_text())
                    rows = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
                    ok_run = d.get('final_epoch') == want_final and len(rows) == want_epochs
                    ok_run = ok_run and all(not (r['train_loss'] != r['train_loss']) for r in rows)
            else:
                out = ROOT / 'runs' / kind / f'{backbone}_{task}_seed17'
                done = out / 'DONE.json'
                hist = out / 'history.jsonl'
                ok_run = done.exists() and hist.exists()
                if ok_run:
                    d = json.loads(done.read_text())
                    ok_run = d.get('final_epoch') == want_final
            ok = ok and ok_run
            detail.append(f'{backbone}-{task}:{ok_run}')
    return ok, detail


def run_spec(spec, gpu, dry_run):
    if spec['kind'] == 'pretrain':
        cmd_fn = cmd_pretrain
    else:
        cmd_fn = lambda b, t, s, g, r: cmd_downstream(spec['kind'], b, t, s, g, r)
    out, done = run_status(spec)
    if done:
        return spec, 'skipped'
    resume = (out / 'last.pt').exists()
    cmd, outdir = cmd_fn(spec['backbone'], spec['task'], spec['seed'], gpu, resume)
    name = f"{spec['kind']}/{spec['backbone']}_{spec['task']}_seed{spec['seed']}"
    log = ROOT / 'logs' / f'{name}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print('DRY ' + ' '.join(cmd))
        return spec, 'dry'
    with open(log, 'a') as fh:
        fh.write(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} gpu={gpu} cmd: {" ".join(cmd)} ===\n')
        fh.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    return spec, ('ok' if proc.returncode == 0 else 'failed')


def run_phase(specs, dry_run):
    # One worker per GPU; each spec is submitted as a worker slot frees up.
    gpu_slots = list(range(N_GPU))
    results = []
    with ThreadPoolExecutor(max_workers=N_GPU) as ex:
        futures = {}

        def submit_next(spec, gpu):
            f = ex.submit(run_spec, spec, gpu, dry_run)
            futures[f] = (spec, gpu)

        for spec in specs:
            if not gpu_slots:
                # wait for any worker to free up
                done = next(as_completed(futures))
                _, gpu = futures.pop(done)
                results.append(done.result())
                gpu_slots.append(gpu)
            gpu = gpu_slots.pop(0)
            submit_next(spec, gpu)
        for f in as_completed(futures):
            results.append(f.result())
    for spec, status in results:
        print(f'[{status}] {spec["kind"]}/{spec["backbone"]}_{spec["task"]}_seed{spec["seed"]}', flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='run the full staged pipeline')
    ap.add_argument('--phase', choices=PHASE_ORDER)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    specs = build_specs()
    by_phase = {}
    for p in PHASE_ORDER:
        by_phase[p] = [s for s in specs if s['phase'] == p]

    if args.phase:
        phases = [args.phase]
    elif args.all:
        phases = PHASE_ORDER
    else:
        print('choose --all or --phase'); sys.exit(1)

    for phase in phases:
        print(f'\n===== PHASE {phase} =====', flush=True)
        run_phase(by_phase[phase], args.dry_run)
        if args.dry_run:
            continue
        if phase == 'pretrain_s17':
            ok, detail = verify_seed17_group('pretrain')
            print('verify seed-17 pretrain:', detail, flush=True)
            if not ok:
                print('SEED-17 PRETRAIN VERIFICATION FAILED — stopping', flush=True)
                sys.exit(2)
        elif phase == 'probe_s17':
            ok, detail = verify_seed17_group('probe')
            print('verify seed-17 probe:', detail, flush=True)
            if not ok:
                print('SEED-17 PROBE VERIFICATION FAILED — stopping', flush=True)
                sys.exit(2)
        elif phase == 'finetune_s17':
            ok, detail = verify_seed17_group('finetune')
            print('verify seed-17 finetune:', detail, flush=True)
            if not ok:
                print('SEED-17 FINETUNE VERIFICATION FAILED — stopping', flush=True)
                sys.exit(2)


if __name__ == '__main__':
    main()
