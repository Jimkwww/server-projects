#!/usr/bin/env python3
"""public_baselines_rml_radar_v2 scheduler.

Runs inside tmux, survives VSCode/SSH disconnect. 3-GPU queue with 3 concurrent jobs per
GPU (efficiency run is exclusive). Priority: P0 3-seed > P2 efficiency > P1 30ep > P1 60ep.
Hard deadline protection: stop launching new jobs at T-1h, finalize at deadline.
State is persisted (state/scheduler_state.json); completed jobs (result.json) are skipped on
restart, interrupted jobs re-enter pending. Trains are resumable (train.py --resume).
"""
import json, os, signal, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parent
PY = '/home/user_wzk/miniconda3/envs/SFM/bin/python'
CST = timezone(timedelta(hours=8))
DEADLINE = datetime(2026, 11, 28, 8, 50, 0, tzinfo=CST)          # hard stop
NO_NEW_AFTER = DEADLINE - timedelta(hours=1)                      # 07:50 CST
MAX_PER_GPU = 3
NGPU = 3
STATE = ROOT / 'state' / 'scheduler_state.json'
LOG_DIR = ROOT / 'logs'
RUNS = ROOT / 'runs'


def now():
    return datetime.now(CST)


def build_jobs():
    jobs = []
    def add(jid, prio, task, model, seed, lr, epochs):
        jobs.append(dict(id=jid, kind='train', prio=prio, task=task, model=model,
                         seed=seed, lr=lr, epochs=epochs))
    for seed in (42, 2026):                       # P0
        for task in ('rml', 'radar'):
            for model in ('q', 'structured', 'emind', 'wavesfm'):
                add(f'p0_s{seed}_{task}_{model}', 0, task, model, seed, 3e-4, 30)
    for task in ('rml', 'radar'):                 # P1 30ep lr1e-4
        for model in ('q', 'structured', 'emind', 'wavesfm'):
            add(f'p1_s17_lr1e4_30_{task}_{model}', 2, task, model, 17, 1e-4, 30)
    for lr in (3e-4, 1e-4):                       # P1 60ep
        for task in ('rml', 'radar'):
            for model in ('q', 'structured', 'emind', 'wavesfm'):
                add(f'p1_s17_lr{lr:g}_60_{task}_{model}', 3, task, model, 17, lr, 60)
    jobs.append(dict(id='p2_efficiency', kind='efficiency', prio=1, exclusive=True))
    return jobs


def out_dir(j):
    if j['kind'] == 'efficiency':
        return RUNS / 'efficiency.json'
    return RUNS / j['task'] / j['model'] / f"s{j['seed']}_lr{j['lr']:g}_ep{j['epochs']}"


def is_done(j):
    if j['kind'] == 'efficiency':
        return (RUNS / 'efficiency.json').exists()
    return (out_dir(j) / 'result.json').exists()


def cmd(j, gpu):
    if j['kind'] == 'efficiency':
        return [PY, '-u', 'efficiency.py', '--gpu', f'cuda:{gpu}', '--out', 'runs/efficiency.json']
    return [PY, '-u', 'train.py', '--task', j['task'], '--model', j['model'], '--gpu', f'cuda:{gpu}',
            '--seed', str(j['seed']), '--lr', str(j['lr']), '--epochs', str(j['epochs']),
            '--out', 'runs', '--resume']


class Scheduler:
    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.jobs = build_jobs()
        self.jmap = {j['id']: j for j in self.jobs}
        self.state = {}
        if STATE.exists():
            self.state = json.loads(STATE.read_text())
        # slots per gpu occupied by running procs
        self.procs = []  # dict(job, gpu, proc, log_path, retries, started)

    def job_status(self, jid):
        return self.state.get(jid, {}).get('status', 'pending')

    def running_count(self, gpu):
        n = 0
        for p in self.procs:
            if p['proc'].poll() is None and p['gpu'] == gpu:
                n += 1
        return n

    def gpu_slots(self):
        return {g: MAX_PER_GPU - self.running_count(g) for g in range(NGPU)}

    def any_exclusive_running(self):
        return any(p['job']['kind'] == 'efficiency' and p['proc'].poll() is None for p in self.procs)

    def reap(self):
        still = []
        for p in self.procs:
            rc = p['proc'].poll()
            if rc is None:
                still.append(p)
                continue
            jid = p['job']['id']
            ok = is_done(p['job'])
            st = self.state.setdefault(jid, {})
            if ok:
                st['status'] = 'done'
            else:
                st['retries'] = st.get('retries', 0) + 1
                if st['retries'] >= 2:
                    st['status'] = 'failed'
                else:
                    st['status'] = 'pending'  # retry
            st['exit_code'] = rc
            st['finished'] = now().isoformat()
            print(f'[{now().strftime("%H:%M:%S")}] {jid}: {"done" if ok else "failed(rc=%d)"%rc} '
                  f'-> {st["status"]}', flush=True)
        self.procs = still

    def launch(self):
        slots = self.gpu_slots()
        pending = [j for j in self.jobs
                   if self.job_status(j['id']) == 'pending' and not is_done(j)]
        pending.sort(key=lambda j: (j['prio'], j['id']))
        for j in pending:
            if self.any_exclusive_running():
                break
            if j['kind'] == 'efficiency':
                # exclusive: need a fully empty gpu
                free = [g for g in range(NGPU) if self.running_count(g) == 0]
                if not free:
                    continue
                gpu = free[0]
                if slots[gpu] <= 0:
                    continue
            else:
                cands = [g for g in range(NGPU) if slots[g] > 0]
                if not cands:
                    break
                gpu = min(cands, key=lambda g: self.running_count(g))
            log_path = LOG_DIR / f"{j['id']}.log"
            f = open(log_path, 'a')
            proc = subprocess.Popen(cmd(j, gpu), stdout=f, stderr=subprocess.STDOUT,
                                    cwd=str(ROOT), start_new_session=True)
            self.procs.append(dict(job=j, gpu=gpu, proc=proc, log_path=str(log_path),
                                   retries=self.state.get(j['id'], {}).get('retries', 0),
                                   started=now().isoformat()))
            st = self.state.setdefault(j['id'], {})
            st.update(status='running', gpu=gpu, pid=proc.pid, started=now().isoformat(),
                      retries=self.state.get(j['id'], {}).get('retries', 0),
                      cmd=' '.join(cmd(j, gpu)))
            slots[gpu] -= 1
            print(f'[{now().strftime("%H:%M:%S")}] launch {j["id"]} on cuda:{gpu} pid={proc.pid} '
                  f'(prio {j["prio"]})', flush=True)

    def save_state(self):
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(self.state, indent=2))

    def all_done(self):
        return all(is_done(j) or self.job_status(j['id']) == 'done' for j in self.jobs)

    def run(self):
        print(f'[{now().strftime("%H:%M:%S")}] scheduler start; deadline {DEADLINE.strftime("%H:%M")} '
              f'CST; no-new after {NO_NEW_AFTER.strftime("%H:%M")}', flush=True)
        while True:
            self.reap()
            if now() < NO_NEW_AFTER:
                self.launch()
            self.save_state()
            # stop condition
            if self.all_done():
                print(f'[{now().strftime("%H:%M:%S")}] all jobs done', flush=True)
                break
            if now() >= DEADLINE:
                print(f'[{now().strftime("%H:%M:%S")}] HARD DEADLINE reached', flush=True)
                break
            if now() >= NO_NEW_AFTER and not self.procs:
                print(f'[{now().strftime("%H:%M:%S")}] no-new window and nothing running', flush=True)
                break
            time.sleep(20)
        # finalize
        self.save_state()
        print(f'[{now().strftime("%H:%M:%S")}] finalizing (may terminate running procs)', flush=True)
        # terminate running procs at deadline
        for p in self.procs:
            if p['proc'].poll() is None:
                try:
                    os.killpg(p['proc'].pid, signal.SIGTERM)
                except Exception as e:
                    print(f'  kill {p["job"]["id"]}: {e}', flush=True)
        time.sleep(5)
        self.save_state()


if __name__ == '__main__':
    Scheduler().run()
