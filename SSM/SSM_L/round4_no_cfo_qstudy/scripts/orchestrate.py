"""Round-4 job orchestrator: no-CFO downstream (P0), Q-learning-phase study (P1).

Builds the full GPU job list, runs it on 3 GPUs with a FIFO queue, skips any job whose
DONE.json already exists (resumable), and writes run_manifest.csv + reuse_inventory.csv.

Priority order (FIFO over 3 workers):
  1. radar full fine-tune (P0, highest priority)
  2. radar scratch
  3. RML full fine-tune
  4. RML scratch Q/G_Qlearn  (G/S1plain/G_corr scratch are REUSED from the old SSM project)
  5. P1 pretraining (G_corr MAE, fixed-Q G_Qlearn MAE)
  6. P1 downstream (G_corr probe/finetune/scratch, A/B/C finetunes)

Run from the SSM_L root, inside the SFM conda env, in a detached tmux session:
  source .../conda.sh && conda activate SFM && python round4_no_cfo_qstudy/scripts/orchestrate.py
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent   # SSM_L
R4 = ROOT / 'round4_no_cfo_qstudy'
RUNS = R4 / 'runs'
RML_CFG = R4 / 'configs/downstream_nocfo_rml.json'
RADAR_CFG = R4 / 'configs/downstream_nocfo_radar.json'
PRETRAIN_CFG = R4 / 'configs/pretrain_mae_f0.json'
RML_MANIFEST = ROOT / 'configs/data.json'
RADAR_MANIFEST = ROOT / 'round2_overnight/configs/radchar.json'
SEEDS = (17, 42, 2026)
GPUS = (0, 1, 2)
PY = sys.executable

# Existing (round-1/round-2) pretrain encoder checkpoints, keyed (backbone, task).
EXISTING_PT = {
    ('G', 'mae'): 'runs/pretrain/G_mae_seed{s}/encoder_final.pt',
    ('G', 'denoise'): 'runs/pretrain/G_denoise_seed{s}/encoder_final.pt',
    ('Q', 'mae'): 'runs/pretrain/Q_mae_seed{s}/encoder_final.pt',
    ('Q', 'denoise'): 'runs/pretrain/Q_denoise_seed{s}/encoder_final.pt',
    ('G_Qlearn', 'mae'): 'round2_overnight/runs/pretrain/G_Qlearn_mae_seed{s}/encoder_final.pt',
    ('G_Qlearn', 'denoise'): 'round2_overnight/runs/pretrain/G_Qlearn_denoise_seed{s}/encoder_final.pt',
    ('S1plain', 'mae'): 'round2_overnight/runs/pretrain/S1plain_mae_seed{s}/encoder_final.pt',
    ('S1plain', 'denoise'): 'round2_overnight/runs/pretrain/S1plain_denoise_seed{s}/encoder_final.pt',
}

P0_BACKBONES = ('G', 'Q', 'S1plain', 'G_Qlearn')


def build_jobs():
    jobs = []

    # ---- P0: radar + RML downstream ----
    for task, cfg, manifest in (('radar', RADAR_CFG, RADAR_MANIFEST),
                                ('rml', RML_CFG, RML_MANIFEST)):
        eval_flag = ['--eval-split', 'test'] if task == 'radar' else []
        for bb in P0_BACKBONES:
            for seed in SEEDS:
                # scratch (RML G/S1plain scratch are reused from the old SSM project)
                if not (task == 'rml' and bb in ('G', 'S1plain')):
                    out = RUNS / task / 'scratch' / f'{bb}_seed{seed}'
                    jobs.append(dict(
                        id=f'{task}_scratch_{bb}_{seed}', phase='P0', task=task, backbone=bb,
                        init='scratch', seed=seed, out=str(out), requires=[],
                        cmd=[PY, '-m', 'round4_no_cfo_qstudy.downstream_r4', '--mode', 'scratch',
                             '--config', str(cfg), '--manifest', str(manifest), '--backbone', bb,
                             '--out', str(out), '--seed', str(seed)] + eval_flag))
                # full fine-tune from MAE / denoise
                for pt_task in ('mae', 'denoise'):
                    pt = ROOT / EXISTING_PT[(bb, pt_task)].format(s=seed)
                    out = RUNS / task / pt_task / f'{bb}_seed{seed}'
                    jobs.append(dict(
                        id=f'{task}_{pt_task}_{bb}_{seed}', phase='P0', task=task, backbone=bb,
                        init=pt_task, seed=seed, out=str(out), requires=[],
                        cmd=[PY, '-m', 'round4_no_cfo_qstudy.downstream_r4', '--mode', 'finetune',
                             '--config', str(cfg), '--manifest', str(manifest), '--backbone', bb,
                             '--pretrained', str(pt), '--out', str(out), '--seed', str(seed)]
                            + eval_flag))

    # ---- P1 pretraining ----
    for seed in SEEDS:
        out = RUNS / 'pretrain' / f'G_corr_mae_seed{seed}'
        jobs.append(dict(
            id=f'pretrain_gcorr_mae_{seed}', phase='P1', task='pretrain', backbone='G_corr',
            init='mae', seed=seed, out=str(out), requires=[],
            cmd=[PY, '-m', 'round4_no_cfo_qstudy.pretrain_r4', '--config', str(PRETRAIN_CFG),
                 '--manifest', str(RML_MANIFEST), '--backbone', 'G_corr', '--out', str(out),
                 '--runs-root', str(RUNS), '--seed', str(seed)]))
        out = RUNS / 'pretrain' / f'G_Qlearn_fixedq_mae_seed{seed}'
        jobs.append(dict(
            id=f'pretrain_gqlearn_fixedq_mae_{seed}', phase='P1', task='pretrain',
            backbone='G_Qlearn', init='mae_fixedq', seed=seed, out=str(out), requires=[],
            cmd=[PY, '-m', 'round4_no_cfo_qstudy.pretrain_r4', '--config', str(PRETRAIN_CFG),
                 '--manifest', str(RML_MANIFEST), '--backbone', 'G_Qlearn', '--out', str(out),
                 '--runs-root', str(RUNS), '--seed', str(seed), '--freeze-aux']))

    # ---- P1 downstream: G_corr reference ----
    for task, cfg, manifest in (('radar', RADAR_CFG, RADAR_MANIFEST),
                                ('rml', RML_CFG, RML_MANIFEST)):
        eval_flag = ['--eval-split', 'test'] if task == 'radar' else []
        for seed in SEEDS:
            pt = RUNS / 'pretrain' / f'G_corr_mae_seed{seed}' / 'encoder_final.pt'
            req = [str(RUNS / 'pretrain' / f'G_corr_mae_seed{seed}' / 'DONE.json')]
            # frozen readout (probe)
            out = RUNS / task / 'mae_probe' / f'G_corr_seed{seed}'
            jobs.append(dict(
                id=f'{task}_probe_G_corr_{seed}', phase='P1', task=task, backbone='G_corr',
                init='mae_probe', seed=seed, out=str(out), requires=req,
                cmd=[PY, '-m', 'round4_no_cfo_qstudy.downstream_r4', '--mode', 'probe',
                     '--config', str(cfg), '--manifest', str(manifest), '--backbone', 'G_corr',
                     '--pretrained', str(pt), '--out', str(out), '--seed', str(seed)] + eval_flag))
            # MAE fine-tune
            out = RUNS / task / 'mae' / f'G_corr_seed{seed}'
            jobs.append(dict(
                id=f'{task}_mae_G_corr_{seed}', phase='P1', task=task, backbone='G_corr',
                init='mae', seed=seed, out=str(out), requires=req,
                cmd=[PY, '-m', 'round4_no_cfo_qstudy.downstream_r4', '--mode', 'finetune',
                     '--config', str(cfg), '--manifest', str(manifest), '--backbone', 'G_corr',
                     '--pretrained', str(pt), '--out', str(out), '--seed', str(seed)] + eval_flag))
        # G_corr scratch radar (RML scratch reused from old SSM)
        if task == 'radar':
            for seed in SEEDS:
                out = RUNS / 'radar' / 'scratch' / f'G_corr_seed{seed}'
                jobs.append(dict(
                    id=f'radar_scratch_G_corr_{seed}', phase='P1', task='radar', backbone='G_corr',
                    init='scratch', seed=seed, out=str(out), requires=[],
                    cmd=[PY, '-m', 'round4_no_cfo_qstudy.downstream_r4', '--mode', 'scratch',
                         '--config', str(RADAR_CFG), '--manifest', str(RADAR_MANIFEST),
                         '--backbone', 'G_corr', '--out', str(out), '--seed', str(seed),
                         '--eval-split', 'test']))

    # ---- P1 downstream: A / B / C fixed-Q finetunes (D = P0 G_Qlearn MAE finetune) ----
    fixedq_pt = {s: RUNS / 'pretrain' / f'G_Qlearn_fixedq_mae_seed{s}' / 'encoder_final.pt' for s in SEEDS}
    fixedq_req = {s: [str(RUNS / 'pretrain' / f'G_Qlearn_fixedq_mae_seed{s}' / 'DONE.json')] for s in SEEDS}
    learnq_pt = {s: ROOT / EXISTING_PT[('G_Qlearn', 'mae')].format(s=s) for s in SEEDS}
    for setting, use_pt, freeze, deps in (
        ('A_fixedq_fixedq', fixedq_pt, True, fixedq_req),   # fixed-Q pretrain -> fixed-Q finetune
        ('B_learnq_fixedq', learnq_pt, True, None),          # learn-Q pretrain -> fixed-Q finetune
        ('C_fixedq_learnq', fixedq_pt, False, fixedq_req),   # fixed-Q pretrain -> learn-Q finetune
    ):
        for task, cfg, manifest in (('radar', RADAR_CFG, RADAR_MANIFEST),
                                    ('rml', RML_CFG, RML_MANIFEST)):
            eval_flag = ['--eval-split', 'test'] if task == 'radar' else []
            for seed in SEEDS:
                out = RUNS / task / setting / f'G_Qlearn_seed{seed}'
                cmd = [PY, '-m', 'round4_no_cfo_qstudy.downstream_r4', '--mode', 'finetune',
                       '--config', str(cfg), '--manifest', str(manifest), '--backbone', 'G_Qlearn',
                       '--pretrained', str(use_pt[seed]), '--out', str(out), '--seed', str(seed)]
                if freeze:
                    cmd.append('--freeze-aux')
                cmd += eval_flag
                jobs.append(dict(
                    id=f'{task}_{setting}_G_Qlearn_{seed}', phase='P1', task=task,
                    backbone='G_Qlearn', init=setting, seed=seed, out=str(out),
                    requires=(deps[seed] if deps else []), cmd=cmd))
    return jobs


def done(job):
    return Path(job['out']) / 'DONE.json'


def main():
    jobs = build_jobs()
    manifest_path = R4 / 'reports' / 'run_manifest.csv'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['id', 'phase', 'task', 'backbone', 'init', 'seed', 'gpu', 'status',
                  'started', 'finished', 'duration_s', 'exit_code', 'out']

    def ready(job):
        return all(Path(r).exists() for r in job['requires'])

    # Merge history from a previous orchestrator run (queueing is DONE.json-based, so a stale
    # manifest never blocks a retry; it only preserves the log).
    rows = {}
    if manifest_path.exists():
        with manifest_path.open(newline='') as f:
            for row in csv.DictReader(f):
                if row and row.get('id'):
                    rows[row['id']] = row

    def flush():
        with manifest_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for j in jobs:
                r = rows.get(j['id'])
                if r is None:
                    r = {k: '' for k in fieldnames}
                    r.update(id=j['id'], phase=j['phase'], task=j['task'], backbone=j['backbone'],
                             init=j['init'], seed=j['seed'], out=j['out'])
                w.writerow(r)

    n_done = sum(1 for j in jobs if done(j).exists())
    print(f'[orchestrate] {len(jobs)} jobs total, {n_done} already done, '
          f'{len(jobs) - n_done} to run', flush=True)

    active = {}  # gpu -> (job, proc, started)
    while True:
        # Reap finished processes.
        for gpu in list(active):
            job, proc, started = active[gpu]
            if proc.poll() is None:
                continue
            finished = time.strftime('%Y-%m-%d %H:%M:%S')
            rc = proc.returncode
            status = 'done' if (rc == 0 and done(job).exists()) else ('done_no_marker' if rc == 0 else 'failed')
            rows[job['id']] = dict(id=job['id'], phase=job['phase'], task=job['task'],
                                   backbone=job['backbone'], init=job['init'], seed=job['seed'],
                                   gpu=gpu, status=status, started=started, finished=finished,
                                   duration_s='', exit_code=rc, out=job['out'])
            del active[gpu]
            flush()
            print(f"[finish] gpu={gpu} {job['id']} rc={rc} {status} {finished}", flush=True)

        # Fill free GPUs with the highest-priority ready job (FIFO over build order).
        for gpu in GPUS:
            if gpu in active:
                continue
            active_jobs = {a[0]['id'] for a in active.values()}
            job = None
            for j in jobs:
                if j['id'] in active_jobs or done(j).exists():
                    continue
                if rows.get(j['id'], {}).get('status') == 'failed':
                    continue  # do not auto-retry a failed job; fix and clear the row first
                if ready(j):
                    job = j
                    break
            if job is None:
                continue
            cmd = job['cmd'] + ['--device', f'cuda:{gpu}']
            log = R4 / 'logs' / f"{job['id']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            lf = log.open('a')
            lf.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} START gpu={gpu}\n# {' '.join(cmd)}\n")
            lf.flush()
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
            started = time.strftime('%Y-%m-%d %H:%M:%S')
            active[gpu] = (job, proc, started)
            print(f"[start] gpu={gpu} {job['id']} pid={proc.pid} {started}", flush=True)

        if not active and all(done(j).exists() or rows.get(j['id'], {}).get('status') == 'failed'
                              for j in jobs):
            break
        time.sleep(5 if active else 20)

    flush()
    n_fail = sum(1 for r in rows.values() if r.get('status') == 'failed')
    n_done = sum(1 for j in jobs if done(j).exists())
    print(f'[orchestrate] DONE. {len(jobs)} jobs, done={n_done}, failures={n_fail}', flush=True)


if __name__ == '__main__':
    main()
