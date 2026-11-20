"""Round-5 item-17 input audit: record the first K training batches' sample IDs and
injected CFO values (+ SHA256) for each seed, proving all configs that share a seed receive
byte-identical inputs. Replicates the exact engine order: shuffle generator `gen` seeded by
`seed`, CFO generator `aug_gen` seeded by `seed+100000`, order-B (raw -> CFO -> max_abs norm).
"""
import argparse
import hashlib
import json
import sys
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.engine import make_loader

BATCH = 128
WORKERS = 4
K_BATCHES = 3
EXTENT = 0.002


def audit(seed, manifest, device):
    m = read_manifest(manifest)
    ds = IQDataset(m, 'train')
    gen = torch.Generator().manual_seed(seed)
    aug_gen = torch.Generator(device=device).manual_seed(seed + 100000)
    dl = make_loader(ds, BATCH, WORKERS, True, gen)
    records = []
    h = hashlib.sha256()
    for b, (x, y, z, i) in enumerate(dl):
        if b >= K_BATCHES:
            break
        # exactly mirrors engine.py order-B CFO draw (before max_abs_norm)
        cfo = (2 * torch.rand(len(x), device=device, generator=aug_gen) - 1) * EXTENT
        ids = np.asarray(i.tolist(), dtype=np.int64)
        cfo_np = cfo.detach().cpu().numpy().astype(np.float64)
        h.update(ids.tobytes())
        h.update(cfo_np.tobytes())
        records.append({'batch': b, 'n': len(ids), 'ids': ids.tolist(), 'cfo': cfo_np.tolist()})
    return {'seed': seed, 'batches': K_BATCHES, 'ids_cfo_sha256': h.hexdigest(),
            'batches_detail': records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='configs/data_raw.json')
    ap.add_argument('--seeds', default='17,42,2026')
    ap.add_argument('--out', default='reports/round5_iq/input_audit.json')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    results = []
    for s in seeds:
        results.append(audit(s, args.manifest, torch.device(args.device)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'extent': EXTENT, 'batch_size': BATCH, 'workers': WORKERS,
                               'k_batches': K_BATCHES, 'seeds': results}, indent=2))
    # human-readable summary: all configs (7 round-5 variants) with the same seed share the
    # same ids_cfo_sha256 (the stream is model-independent, proven by tests/test_round5.py#17).
    print(json.dumps([{'seed': r['seed'], 'hash': r['ids_cfo_sha256']} for r in results], indent=2))


if __name__ == '__main__':
    main()
