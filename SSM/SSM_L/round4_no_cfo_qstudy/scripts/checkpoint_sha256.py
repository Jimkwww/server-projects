"""Record SHA256 of every round-4 checkpoint (best.pt, encoder_final.pt) for provenance.

Writes reports/checkpoint_sha256.csv. Weights stay on the server; the hash list travels with
the package so results can be tied back to exact artifacts.
"""
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R4 = ROOT / 'round4_no_cfo_qstudy'
RUNS = R4 / 'runs'
OUT = R4 / 'reports' / 'checkpoint_sha256.csv'


def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ckpts = sorted(set(list(RUNS.rglob('best.pt')) + list(RUNS.rglob('encoder_final.pt'))))
    with OUT.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rel_path', 'bytes', 'sha256'])
        for p in ckpts:
            w.writerow([str(p.relative_to(ROOT)), p.stat().st_size, sha256(p)])
    print(f'[sha256] wrote {OUT} ({len(ckpts)} checkpoints)')


if __name__ == '__main__':
    main()
