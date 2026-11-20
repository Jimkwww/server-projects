"""
torch.compile 算子融合实验：验证能否砍掉 b1 延迟的 kernel launch 税。

原理：torch.compile（Inductor）会把多个小算子融合成大 kernel → kernel 数量骤降
     → b1 延迟的固定开销（launch 税，~95%）被砍。
对比：编译前 vs 编译后的 b1 延迟 / b256 吞吐（FP32 基线 + 终极配置）。
"""
import numpy as np
import torch

from Model.model import ConformerClassifier
from int8_static import merge_lora


def measure_latency(model, x, warmup=10, iters=50):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        with torch.no_grad():
            for i in range(iters):
                starts[i].record()
                model(x)
                ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return float(np.mean(times)), float(np.std(times))


def build_fp32():
    m = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    m.load_state_dict(torch.load("Checkpoint/amc/Our.pt", map_location="cpu"))
    return merge_lora(m)


def main():
    device = torch.device("cuda:0")
    print("[1/4] 构建 FP32 模型...")
    model = build_fp32().to(device).eval()

    x1 = torch.randn(1, 128, 2, device=device)
    x256 = torch.randn(256, 128, 2, device=device)

    print("[2/4] 编译前基线...")
    b1, b1std = measure_latency(model, x1)
    b256, _ = measure_latency(model, x256, warmup=5, iters=15)
    print(f"  编译前: b1 {b1:.2f}ms (±{b1std:.2f}) | b256 每样本 {b256/256:.3f}ms | 吞吐 {256/(b256/1000):.0f}/s")

    print("[3/4] torch.compile 编译（首次含编译时间，约 1-3 分钟）...")
    import time
    t0 = time.time()
    cmodel = torch.compile(model)
    with torch.no_grad():
        cmodel(x1)          # 触发编译
        cmodel(x256)
    torch.cuda.synchronize()
    print(f"  编译完成（{time.time()-t0:.0f}s）")

    print("[4/4] 编译后测量...")
    b1c, b1cstd = measure_latency(cmodel, x1)
    b256c, _ = measure_latency(cmodel, x256, warmup=5, iters=15)
    print(f"  编译后: b1 {b1c:.2f}ms (±{b1cstd:.2f}) | b256 每样本 {b256c/256:.3f}ms | 吞吐 {256/(b256c/1000):.0f}/s")

    print("\n" + "=" * 60)
    print(f"  b1 延迟:  {b1:.2f}ms → {b1c:.2f}ms（{b1/b1c:.2f}×）")
    print(f"  b256 吞吐: {256/(b256/1000):.0f}/s → {256/(b256c/1000):.0f}/s（{(256/(b256c/1000))/(256/(b256/1000)):.2f}×）")
    print("=" * 60)


if __name__ == "__main__":
    main()
