"""
延迟-batch 曲线：一次运行测多个 batch 档位（1/8/16/32/64/128/256）。

输出每档：延迟 ms、每样本摊薄 ms、吞吐 samples/s —— 展示「batch 越大每样本成本越低」
的完整形状，以及 b1 的 launch 税 vs 大 batch 的计算主导的分界。

用法（等 GPU 空闲时跑）：
  python latency_curve.py              # FP32 基线
  python latency_curve.py --half       # FP16 对比
"""
import argparse

import numpy as np
import torch

from Model.model import ConformerClassifier
from int8_static import merge_lora


def measure_latency(model, x, warmup=10, iters=30):
    """cuda Event 计时：warmup + 多次平均，返回 (均值ms, 标准差ms)。"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--half", action="store_true", help="FP16 对比")
    parser.add_argument("--ckpt", type=str, default="Checkpoint/amc/Our.pt",
                        help="原始结构模型权重（Our.pt / Our_fp16.pt）；剪枝/QAT 产物结构不同需配套加载逻辑，不在本脚本支持范围")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    model = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = merge_lora(model)                      # 与压缩实验同口径
    model = model.to(device).eval()
    if args.half:
        model = model.half()

    batches = [1, 8, 16, 32, 64, 128, 256]
    print(f"模型: {args.ckpt}（{'FP16' if args.half else 'FP32'}）\n")
    print(f"  {'batch':>6s} {'延迟ms':>10s} {'±std':>8s} {'每样本ms':>9s} {'吞吐/s':>9s}")
    for b in batches:
        x = torch.randn(b, 128, 2, device=device)
        if args.half:
            x = x.half()
        iters = 30 if b <= 32 else 15          # 大 batch 单次慢，少测几次
        lat, std = measure_latency(model, x, warmup=10, iters=iters)
        per_sample = lat / b
        throughput = 1000.0 / per_sample
        print(f"  {b:6d} {lat:10.2f} {std:8.2f} {per_sample:9.2f} {throughput:9.0f}")

    print("\n读法：每样本 ms 随 batch 增大而下降 = 固定开销（launch 税）被摊薄；")
    print("      吞吐趋平 = GPU 算力达到饱和。")


if __name__ == "__main__":
    main()
