#!/usr/bin/env python
"""Efficiency benchmark for the G_Q_corr round: G, G_corr, G_Q_corr, G_Q_corr_frozen.

One seed per variant (architecture identical across seeds), eval mode, one device, L=1024:
  real-scalar parameter count; batch=1 latency (median ms); batch=64 throughput (samples/s);
  inference peak memory (MiB). Includes the aux branch online computation (ComplexRecurBranch
  parallel_scan + state_corr_pack), warmup + GPU sync. Weights loaded from best.pt.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssm_exp.model import Classifier, max_abs_norm, parameter_count

ALLOWED_MISSING = {'aux_mean', 'aux_std'}


def time_forward(model, x, iters, warmup=5):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


def bench_run(run_dir, device, b64_iters=20, b1_iters=50):
    pt = Path(run_dir) / 'best.pt'
    ck = torch.load(pt, map_location='cpu', weights_only=False)
    cfg = ck['config']['model']
    seed = ck['config']['train']['seed']
    model = Classifier(cfg, seed=seed).to(device)
    missing = set(model.state_dict().keys()) - set(ck['model'].keys())
    extra = set(ck['model'].keys()) - set(model.state_dict().keys())
    if (missing - ALLOWED_MISSING) or extra:
        raise RuntimeError(f'{run_dir}: state_dict mismatch missing={sorted(missing - ALLOWED_MISSING)} '
                           f'extra={sorted(extra)}')
    model.load_state_dict(ck['model'], strict=False)
    variant = cfg['variant']
    params = parameter_count(model)

    x1 = max_abs_norm(torch.randn(1, 2, 1024, device=device))
    lat_b1 = time_forward(model, x1, b1_iters) * 1e3

    x64 = max_abs_norm(torch.randn(64, 2, 1024, device=device))
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(3):
            model(x64)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(b64_iters):
            model(x64)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    sps = 64 * b64_iters / dt
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {'variant': variant, 'params': params, 'latency_b1_ms': lat_b1,
            'throughput_b64_sps': sps, 'inference_peak_mem_mb': peak}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True, help='comma-separated run dirs (one per variant)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default='reports/global_qcorr/benchmark.json')
    a = p.parse_args()
    device = torch.device(a.device)
    results = []
    for rd in a.runs.split(','):
        if not rd.strip():
            continue
        r = bench_run(rd.strip(), device)
        results.append(r)
        print(json.dumps(r), flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
