"""
torchao packed 算子基准（公正速度对比的正式实验）。

四个配置（全部 torch.compile）：
  ① FP32            基线
  ② BF16            对照（分离 bf16 本身的收益）
  ③ INT8 权重-only   官方 packed 实现
  ④ BF16 + INT4 packed (tile_packed_to_4d)  官方 4bit 实现

每个配置测：b1 延迟 / b256 吞吐 / F1（全测试集）/ state_dict 体积。
对照：手写版全 INT8 完全体（17.1MB / F1 0.6510）与手写 INT4（部署 12.2MB / QAT F1 0.6494）。
"""
import io
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.model import ConformerClassifier
from dataset import AMCDataset
from amc import process_data
from int8_static import merge_lora
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat


def measure(model, x, warmup=10, iters=50):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            model(x)
            ends[i].record()
        torch.cuda.synchronize()
        return float(np.mean([s.elapsed_time(e) for s, e in zip(starts, ends)]))


def evaluate_f1(model, loader, device, dtype):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for value, label, _ in tqdm(loader, desc="  F1评估", leave=False):
            out = model(value.to(device=device, dtype=dtype))
            yt.extend(label.numpy())
            yp.extend(out.argmax(dim=1).cpu().numpy())
    return f1_score(np.array(yt), np.array(yp), average="weighted")


def build_fp32():
    m = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    m.load_state_dict(torch.load("Checkpoint/amc/Our.pt", map_location="cpu"))
    return merge_lora(m)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="all",
                        help="all 或配置名（FP32/BF16/INT8wo/BF16+INT4packed），只跑指定配置")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    print("[1/2] 加载数据...")
    _, test_dict = process_data()
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    print("[2/2] 配置依次测量（含编译）...")
    results = []
    configs = [
        ("FP32", "fp32"),
        ("BF16", "bf16"),
        ("INT8wo", "int8wo"),
        ("BF16+INT4packed", "int4"),
    ]
    if args.only != "all":
        configs = [c for c in configs if c[0] == args.only]
        if not configs:
            raise ValueError(f"未知配置: {args.only}")
    for label, kind in configs:
        t0 = time.time()
        model = build_fp32()
        dtype = torch.float32
        if kind == "bf16":
            model = model.to(torch.bfloat16)
            dtype = torch.bfloat16
        elif kind == "int8wo":
            model = model.to(device)          # 先搬设备再量化
            quantize_(model, Int8WeightOnlyConfig())
        elif kind == "int4":
            model = model.to(device).to(torch.bfloat16)   # 打包算子需要 CUDA 上的权重
            dtype = torch.bfloat16
            quantize_(model, Int4WeightOnlyConfig(int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D))
        model = model.to(device).eval()

        # 编译（失败则退化为 eager，如实记录）
        cmodel = model
        compiled = "compile"
        try:
            cmodel = torch.compile(model)
            with torch.no_grad():
                cmodel(torch.randn(1, 128, 2, device=device, dtype=dtype))
        except Exception as e:
            cmodel = model
            compiled = f"eager({type(e).__name__})"

        x1 = torch.randn(1, 128, 2, device=device, dtype=dtype)
        x256 = torch.randn(256, 128, 2, device=device, dtype=dtype)
        b1 = measure(cmodel, x1)
        b256 = measure(cmodel, x256, warmup=5, iters=15)

        # 体积
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        size_mb = buf.getbuffer().nbytes / 1024**2

        f1 = evaluate_f1(model, test_loader, device, dtype)
        results.append((label, compiled, b1, b256, size_mb, f1, time.time() - t0))
        print(f"  {label}: b1 {b1:.2f}ms | b256 {256/(b256/1000):.0f}/s | {size_mb:.1f}MB | F1 {f1:.4f} | {compiled}")

    print("\n" + "=" * 92)
    print(f"  {'配置':18s} {'b1延迟':>8s} {'b256吞吐':>10s} {'体积':>8s} {'F1':>8s} {'模式':>8s}")
    for label, compiled, b1, b256, size_mb, f1, _ in results:
        print(f"  {label:18s} {b1:7.2f}ms {256/(b256/1000):9.0f}/s {size_mb:7.1f}MB {f1:8.4f} {compiled:>8s}")
    print("=" * 92)
    print("对照（手写版）：全 INT8 完全体 17.1MB / F1 0.6510；INT4 QAT 部署 12.2MB / F1 0.6494")


if __name__ == "__main__":
    main()
