#!/usr/bin/env python
"""Round-5 IQ efficiency benchmark: one seed per variant (architecture-identical across seeds).

Measures, in eval mode on one device (CUDA warmup + sync):
  real-scalar parameter count; batch=1 latency (median); batch=64 throughput (samples/s)
  + inference peak memory. L=1024 only (the round-5 protocol length).

Notes:
  - The aux branch (Q_linear/Q_corr/R_corr) runs on the continuous 1024-sample raw IQ
    regardless of the main-path patch size. ComplexRecurBranch uses parallel_scan;
    RealRecurBranch (R_corr) uses an explicit per-sample recurrence loop over 1024 steps,
    so its wall-clock is dominated by that Python loop, not by a fused kernel.
  - The SSM main path uses eager PyTorch O(N log N) parallel_scan, NOT a fused SSM
    kernel, so throughput here is not representative of a production fused implementation.
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


def time_forward(model, x, iters, warmup=5):
    model.eval()
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
    allowed = {'aux_mean', 'aux_std'}  # pre-round-5 baselines lack these buffers
    if (missing - allowed) or extra:
        raise RuntimeError(f'{run_dir}: unexpected state_dict mismatch '
                           f'missing={sorted(missing - allowed)} extra={sorted(extra)}')
    model.load_state_dict(ck['model'], strict=False)
    variant = cfg['variant']
    params = parameter_count(model)

    # batch=1 latency at L=1024
    x1 = max_abs_norm(torch.randn(1, 2, 1024, device=device))
    lat_b1 = time_forward(model, x1, b1_iters) * 1e3  # ms

    # batch=64 throughput + peak inference memory
    x64 = max_abs_norm(torch.randn(64, 2, 1024, device=device))
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(3):
        model(x64)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(b64_iters):
        model(x64)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = 64 * b64_iters / dt
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MiB

    return {'variant': variant, 'params': params, 'latency_b1_ms': lat_b1,
            'throughput_b64_sps': sps, 'inference_peak_mem_mb': peak}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True, help='comma-separated run dirs (one per variant)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default='reports/round5_iq/benchmark.json')
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
