"""
快速对照：merge 后 FP32 vs 手写静态 INT8 的体积和延迟。

为什么单独写这个：完整 int8_static.py 的 F1 评估要 51 分钟（手写 int32 矩阵乘无优化），
而 F1 已测得 0.6474（零损失），无需重跑。这里只补测被对象混淆 bug 污染的体积和延迟。
"""
import copy
import io

import torch

from Model.model import ConformerClassifier
import int8_static

print("[1/3] 构建模型 + merge + 分离两个独立副本...")
model = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
model.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))
model.eval()
model = int8_static.merge_lora(model)

model_fp32 = copy.deepcopy(model)              # 独立副本 1：FP32 对照
model_quant = copy.deepcopy(model)             # 独立副本 2：量化版
model_quant = int8_static.replace_with_manual_quant(model_quant)

print("[2/3] 快速校准（随机数据，2 批——体积/延迟测量不依赖严格校准）+ finalize...")
model_quant.eval()
with torch.no_grad():
    model_quant(torch.randn(256, 128, 2))
    model_quant(torch.randn(256, 128, 2))
for m in model_quant.modules():
    if isinstance(m, int8_static.ManualQuantLinear):
        m.finalize()

print("[3/3] 体积 + 延迟对比...")
buf32, buf8 = io.BytesIO(), io.BytesIO()
torch.save(model_fp32.state_dict(), buf32)
torch.save(model_quant.state_dict(), buf8)
print(f"  体积: FP32 {buf32.getbuffer().nbytes/1024**2:.1f}MB → 手写INT8 {buf8.getbuffer().nbytes/1024**2:.1f}MB"
      f"（{buf32.getbuffer().nbytes/buf8.getbuffer().nbytes:.2f}×）")

x1 = torch.randn(1, 128, 2)
l32, _ = int8_static.measure_cpu_latency(model_fp32, x1, warmup=3, iters=10)
l8, _ = int8_static.measure_cpu_latency(model_quant, x1, warmup=3, iters=10)
print(f"  batch=1  : FP32 {l32:.2f}ms → INT8 {l8:.2f}ms（{l32/l8:.2f}×）")

x64 = torch.randn(64, 128, 2)
l32b, _ = int8_static.measure_cpu_latency(model_fp32, x64, warmup=1, iters=3)
l8b, _ = int8_static.measure_cpu_latency(model_quant, x64, warmup=1, iters=3)
print(f"  batch=64 : FP32 {l32b:.2f}ms → INT8 {l8b:.2f}ms（{l32b/l8b:.2f}×）")
