"""Run a list of shell commands across GPUs (one worker per GPU), logging each job.

Job list file: one command per line. Each line may contain the token @@GPU@@ which is replaced
by the worker's GPU index. Empty lines and lines starting with '#' are skipped. A job is
skipped if its command already produced the marker file named on the line as 'done: <path>'.

Usage: python dispatch.py --jobs jobs.txt --gpus 3 --logdir logs --retries 0
"""
import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--gpus', type=int, default=3)
    ap.add_argument('--logdir', required=True)
    ap.add_argument('--retries', type=int, default=0)
    a = ap.parse_args()

    logdir = Path(a.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for line in Path(a.jobs).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # optional 'done: <marker>' suffix -> skip if marker exists
        marker = None
        if ' done: ' in line:
            line, marker = line.split(' done: ', 1)
            line = line.strip()
        if marker and Path(marker).exists():
            continue
        jobs.append((len(jobs), line))

    total = len(jobs)
    print(f'[dispatch] {total} jobs on {a.gpus} GPUs', flush=True)
    if total == 0:
        return

    q = queue.Queue()
    for j in jobs:
        q.put(j)
    results = []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                idx, cmd = q.get_nowait()
            except queue.Empty:
                return
            cmd = cmd.replace('@@GPU@@', str(gpu))
            t0 = time.time()
            status = 'ok'
            tail = ''
            for attempt in range(a.retries + 1):
                try:
                    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if r.returncode == 0:
                        status = 'ok'
                        break
                    status = f'fail({r.returncode})'
                    tail = (r.stdout + r.stderr)[-800:]
                except Exception as e:
                    status = f'exc({e})'
                    tail = str(e)
                if attempt < a.retries:
                    time.sleep(10 * (attempt + 1))
            dur = time.time() - t0
            with lock:
                results.append((idx, status, dur))
            (logdir / f'job_{idx:04d}.log').write_text(cmd + '\n\n' + status + f' ({dur:.1f}s)\n' + tail)
            with lock:
                print(f'[{idx + 1}/{total}] {status} {dur:.1f}s', flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(a.gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    fails = [r for r in results if r[1] != 'ok']
    print(f'[dispatch] done: {total - len(fails)}/{total} ok, {len(fails)} failed', flush=True)
    for idx, status, dur in sorted(fails):
        print(f'  FAIL job {idx}: {status} ({dur:.1f}s)', flush=True)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
