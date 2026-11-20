"""
6 模型 × 7 档延迟曲线对比（GPU，推理）。

模型：FP32 / FP16 / 剪60% / 剪40%+FP16 / 纯 INT4 QAT / 剪40%+INT4 QAT
档位：batch = 1/8/16/32/64/128/256

前置：save_models.py 的产物（prune60.pt / prune40.pt / qat_int4.pt / qat_int4_prune40.pt）
用法：python latency_curves_all.py
"""
import csv
import numpy as np
import torch
import torch.nn as nn

from Model.model import ConformerClassifier, DepthwiseConv
from int8_static import merge_lora
from prune_ffn import prune_model
from qat_modules import QuantAwareLinear, QuantAwareConv1x1


def replace_quant_aware(model, nbits=4):
    count = 0
    for name, module in list(model.named_modules()):
        parent_name, _, attr = name.rpartition(".")
        parent = model if not parent_name else model.get_submodule(parent_name)
        if isinstance(module, nn.Linear):
            setattr(parent, attr, QuantAwareLinear(module, nbits=nbits))
            count += 1
        elif isinstance(module, DepthwiseConv):
            module.pointwise1 = QuantAwareConv1x1(module.pointwise1, nbits=nbits)
            module.pointwise2 = QuantAwareConv1x1(module.pointwise2, nbits=nbits)
            count += 2
    return model


def build_model(name):
    """按名字重建模型（与 save_models.py 的产物一一对应）。"""
    base = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    base.load_state_dict(torch.load("Checkpoint/amc/Our.pt", map_location="cpu"))
    base = merge_lora(base)

    if name == "fp32":
        return base
    if name == "fp16":
        return base.half()
    if name == "prune60":
        prune_model(base, keep_ratio=0.40, mode="magnitude")
        base.load_state_dict(torch.load("Checkpoint/prune60.pt", map_location="cpu"))
        return base
    if name == "prune40_fp16":
        prune_model(base, keep_ratio=0.60, mode="magnitude")
        base.load_state_dict(torch.load("Checkpoint/prune40.pt", map_location="cpu"))
        return base.half()
    if name == "qat_int4":
        replace_quant_aware(base, 4)
        base.load_state_dict(torch.load("Checkpoint/qat_int4.pt", map_location="cpu"))
        return base
    if name == "qat_int4_prune40":
        prune_model(base, keep_ratio=0.60, mode="magnitude")
        replace_quant_aware(base, 4)
        base.load_state_dict(torch.load("Checkpoint/qat_int4_prune40.pt", map_location="cpu"))
        return base
    raise ValueError(name)


def measure_latency(model, x, warmup=10, iters=30):
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
    return float(np.mean(times))


def main():
    device = torch.device("cuda:0")
    models = [("FP32", "fp32"), ("FP16", "fp16"), ("剪60%", "prune60"),
              ("剪40%+FP16", "prune40_fp16"), ("纯INT4QAT", "qat_int4"),
              ("剪40%+QAT", "qat_int4_prune40")]
    batches = [1, 8, 16, 32, 64, 128, 256]

    print("加载并测量 6 个模型 × 7 档...\n")
    # results[name] -> dict batch -> 每样本 ms
    results = {}
    for label, name in models:
        m = build_model(name).to(device).eval()
        dt = next(m.parameters()).dtype       # 输入 dtype 跟随模型（FP16 模型必须吃 FP16 输入）
        per_sample = {}
        for b in batches:
            x = torch.randn(b, 128, 2, device=device, dtype=dt)
            iters = 30 if b <= 32 else 15
            lat = measure_latency(m, x, warmup=10, iters=iters)
            per_sample[b] = lat / b
        results[label] = per_sample
        print(f"  {label:14s} 完成")

    print("\n" + "=" * 90)
    print("每样本延迟（ms）：")
    header = f"  {'batch':>5s}" + "".join(f"{l:>14s}" for l, _ in models)
    print(header)
    for b in batches:
        row = f"  {b:5d}"
        for label, _ in models:
            row += f"{results[label][b]:14.2f}"
        print(row)

    print("\n吞吐（samples/s）：")
    print(header)
    for b in batches:
        row = f"  {b:5d}"
        for label, _ in models:
            row += f"{1000.0 / results[label][b]:14.0f}"
        print(row)
    print("=" * 90)

    # 落盘 CSV（长表格式，方便画图）：model, batch, latency_ms_per_sample, throughput_sps
    out_path = "Results/profiling/speed_curves.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "batch", "latency_ms_per_sample", "throughput_sps"])
        for label, _ in models:
            for b in batches:
                w.writerow([label, b, f"{results[label][b]:.4f}",
                            f"{1000.0 / results[label][b]:.1f}"])
    print(f"\n已保存速度曲线数据: {out_path}")


if __name__ == "__main__":
    main()
