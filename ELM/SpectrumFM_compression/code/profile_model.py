"""
模型效率测量脚本。

测量指标：参数量（总/可训练）、模型大小（磁盘+按当前 dtype 的理论值）、
FLOPs（实测 + 论文公式口径）、推理延迟 batch=1、吞吐 batch=256、
显存峰值（b1/b256）、参数显存。
支持 FP16 推理（--half）。结果追加到 Results/profiling/benchmark.csv，每个配置一行。

用法（在 code/ 目录下）：
    python profile_model.py --name FP32 --ckpt Checkpoint/amc/Our.pt --f1 0.6474
    python profile_model.py --name FP16 --ckpt Checkpoint/amc/Our_fp16.pt --f1 <fp16的f1> --half
"""
import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from Model.model import ConformerClassifier


def measure_latency(model, x, warmup=10, iters=100):
    """cuda Event 计时：warmup 后测 iters 次，返回 (均值ms, 标准差ms)。"""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            model(x)
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return float(np.mean(times)), float(np.std(times))


def paper_formula_flops(N=128, d=256, h=4, dfeed=512, k=3, L=16):
    """按论文公式 (30) 计算 encoder 的理论 FLOPs（论文 2.03G 的 encoder 部分）。

    OMHSA = 4Nd^2 + 2hN^2dh（Q/K/V/O 投影 + attention 矩阵乘）
    OFFN  = 2Nd*dfeed（每个 FFN 两层线性）
    OConv = Ndk（注意论文只算了 depthwise 卷积，漏了两个 1×1 pointwise）
    Olayer = OMHSA + 2*OFFN + OConv
    代入论文参数约得 1.75G；论文声称 2.03G 是额外加了 GRU 等未公式化部分。
    """
    dh = d // h
    omhsa = 4 * N * d**2 + 2 * h * N**2 * dh
    offn = 2 * N * d * dfeed
    oconv = N * d * k
    return L * (omhsa + 2 * offn + oconv)


def main():
    parser = argparse.ArgumentParser(description="测量模型效率")
    parser.add_argument("--name", type=str, default="FP32", help="配置名（csv 第一列）")
    parser.add_argument("--ckpt", type=str, default="Checkpoint/amc/Our.pt", help="模型权重路径")
    parser.add_argument("--f1", type=float, default=0.6474, help="该配置的 F1（来自 eval.py 实测）")
    parser.add_argument("--half", action="store_true", help="FP16 推理")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    print(f"[1/5] 加载模型（{args.name} | {args.ckpt} | {'FP16' if args.half else 'FP32'}）...")
    model = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024).to(device)
    model.load_state_dict(torch.load(args.ckpt))
    if args.half:
        model = model.half()          # 先加载 FP32 再转（顺序铁律）
    model.eval()

    print("[2/5] 参数量与模型大小...")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    disk_mb = os.path.getsize(args.ckpt) / 1024**2
    theory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # 按当前 dtype

    print("[3/5] FLOPs（实测 + 论文公式口径）...")
    x1 = torch.randn(1, 128, 2, device=device)
    if args.half:
        x1 = x1.half()
    with FlopCounterMode(display=False) as fcm:
        model(x1)
    flops = fcm.get_total_flops()
    flops_paper = paper_formula_flops()

    print("[4/5] 推理延迟与吞吐量...")
    lat_b1, lat_b1_std = measure_latency(model, x1)
    x256 = torch.randn(256, 128, 2, device=device)
    if args.half:
        x256 = x256.half()
    lat_b256, _ = measure_latency(model, x256, warmup=5, iters=30)
    throughput = 256 / (lat_b256 / 1000.0)                   # samples/s

    print("[5/5] 显存占用...")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(x1)
    mem_b1 = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(x256)
    mem_b256 = torch.cuda.max_memory_allocated() / 1024**2
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    row = {
        "model": args.name,
        "params_M": round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
        "size_disk_MB": round(disk_mb, 1),
        "size_theory_MB": round(theory_mb, 1),
        "flops_G": round(flops / 1e9, 2),
        "flops_paper_G": round(flops_paper / 1e9, 2),
        "latency_b1_ms": round(lat_b1, 3),
        "latency_b1_std_ms": round(lat_b1_std, 3),
        "throughput_b256_sps": round(throughput, 1),
        "gpu_mem_b1_MB": round(mem_b1, 1),
        "gpu_mem_b256_MB": round(mem_b256, 1),
        "param_mem_MB": round(param_mem, 1),
        "f1": args.f1,
    }

    print("\n" + "=" * 56)
    for k, v in row.items():
        print(f"  {k:24s}: {v}")
    print("=" * 56)

    out_path = "Results/profiling/benchmark.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    existed = os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not existed:
            writer.writeheader()
        writer.writerow(row)
    print(f"结果已追加到 {out_path}")


if __name__ == "__main__":
    main()
