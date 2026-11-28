"""P2 same-GPU exclusive efficiency benchmark (FP32, current implementations, no kernel tuning).

One process per (task, model): the process exits after measuring, so CUDA memory is fully
released and peak-allocated/reserved are clean per-model numbers. Measures batch-1 latency
(median/mean/std), throughput at batch64 in samples/s, token count (measured from the
encoder, incl. cls token where the model prepends one), peak allocated/reserved memory.
Full frontend+encoder+pooling+head, eval() + inference_mode, warmup, cuda sync, repeats.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train import make_model, TASKS
from ssm_exp.model import max_abs_norm

BATCH = 64

def get_tokens(name, model, L, device):
    x = torch.randn(1, L, 2, device=device)
    with torch.inference_mode():
        if name == 'structured':
            enc = model[0]
            xx = x / (x.abs().max(dim=1, keepdim=True).values.max(dim=-1, keepdim=True).values + 1e-6)
            return int(enc(xx)['sequence'].shape[1])
        if name == 'q':
            enc = model[0]
            xn = max_abs_norm(x.permute(0, 2, 1))
            return int(enc.forward_tokens(xn).shape[1])
        if name == 'emind':
            return (L + 4) // 4          # patch4 + cls, = emind_forward n
        if name == 'wavesfm':
            return L // 16 + 1           # segment16 + cls token (cls_token=True, global_pool='token')

def bench_latency(fwd, x):
    for _ in range(10):
        fwd(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fwd(x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.mean(ts)), float(np.std(ts))

def bench_throughput(fwd, x, seconds=8.0):
    torch.cuda.synchronize()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fwd(x)
        n += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    samples_per_s = n * x.shape[0] / dt
    return samples_per_s, dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['rml', 'radar'])
    ap.add_argument('--model', required=True, choices=['q', 'structured', 'emind', 'wavesfm'])
    ap.add_argument('--gpu', default='cuda:0')
    ap.add_argument('--out', default='efficiency.json')
    a = ap.parse_args()
    device = a.gpu
    torch.manual_seed(17)
    dname, nc, L, _ = TASKS[a.task]

    model, fwd, params = make_model(a.model, L, nc, device, seed=17)
    if isinstance(model, tuple):
        model[0].eval(); model[1].eval()
    else:
        model.eval()
    npar = sum(p.numel() * (2 if p.is_complex() else 1) for p in params)
    tok = get_tokens(a.model, model, L, device)

    x1 = torch.randn(1, L, 2, device=device)
    xb = torch.randn(BATCH, L, 2, device=device)

    # batch-1 latency + peak mem at batch1
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        lat_med, lat_mean, lat_std = bench_latency(fwd, x1)
    mem1_alloc = torch.cuda.max_memory_allocated(device) / 1e6
    mem1_resv = torch.cuda.max_memory_reserved(device) / 1e6

    # throughput at batch64 + peak mem at batch64
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        thr, dt = bench_throughput(fwd, xb, seconds=8.0)
    memb_alloc = torch.cuda.max_memory_allocated(device) / 1e6
    memb_resv = torch.cuda.max_memory_reserved(device) / 1e6

    row = {
        'task': a.task, 'model': a.model, 'L': L, 'tokens': tok,
        'batch1_ms_median': lat_med, 'batch1_ms_mean': lat_mean, 'batch1_ms_std': lat_std,
        'throughput_batch64_samples_s': thr, 'throughput_batch': BATCH,
        'peak_allocated_MB_batch1': mem1_alloc, 'peak_reserved_MB_batch1': mem1_resv,
        'peak_allocated_MB_batch64': memb_alloc, 'peak_reserved_MB_batch64': memb_resv,
        'params': npar,
        'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'precision': 'fp32',
        'allow_tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
        'allow_tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
        'note': 'attention sdpa (EMind patched), SSM parallel scan (structured/Q), no kernel tuning',
    }
    Path(a.out).write_text(json.dumps(row, indent=2))
    print(f'[{a.task}/{a.model}] L{L} tok{tok} b1 {lat_med:.2f}ms thr {thr:.0f} samples/s '
          f'mem_b1 {mem1_alloc:.0f}MB mem_b64 {memb_alloc:.0f}MB', flush=True)

if __name__ == '__main__':
    main()
