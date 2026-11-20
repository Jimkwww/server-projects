"""Poll the orchestrator manifest until all jobs are done/failed, then print a summary.

Intended to be run as a lightweight background watcher (sleep loop, no GPU): exits when the
queue finishes so the parent session is re-invoked to produce the final report.
"""
import csv
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = ROOT / 'round4_no_cfo_qstudy' / 'reports' / 'run_manifest.csv'


def snapshot():
    if not MANIFEST.exists():
        return None
    rows = list(csv.DictReader(MANIFEST.open()))
    c = Counter(r['status'] for r in rows)
    pending = len(rows) - c.get('done', 0) - c.get('failed', 0) - c.get('done_no_marker', 0)
    return len(rows), c, pending


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    last = None
    while True:
        s = snapshot()
        if s is None:
            time.sleep(interval)
            continue
        total, c, pending = s
        line = f"[watch] total={total} done={c.get('done',0)} failed={c.get('failed',0)} pending={pending}"
        if line != last:
            print(line, flush=True)
            last = line
        if total > 0 and pending == 0 and c.get('done', 0) + c.get('failed', 0) + c.get('done_no_marker', 0) >= total:
            print(f'[watch] queue finished: {dict(c)}', flush=True)
            if c.get('failed', 0):
                rows = list(csv.DictReader(MANIFEST.open()))
                for r in rows:
                    if r['status'] == 'failed':
                        print(f"[watch] FAILED {r['id']} (exit {r['exit_code']})", flush=True)
            return 0
        time.sleep(interval)


if __name__ == '__main__':
    main()
