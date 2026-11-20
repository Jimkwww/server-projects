"""
INT8 动态量化热身实验（CPU，fbgemm 后端）。

做四件事：
  1. 构建 FP32 模型（CPU）并加载 Our.pt
  2. quantize_dynamic 把 nn.Linear + nn.GRU 动态量化（权重 INT8、激活运行时现算 scale）
  3. 对比：state_dict 体积、CPU 延迟（batch=1 / 256）
  4. 全测试集评估 F1，对比 FP32 基线

注意：torch 的 eager 模式 INT8 量化是 CPU 特性；GPU 上的 INT8 需走 TensorRT/torchao（延伸学习）。
"""
import io
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.model import ConformerClassifier
from dataset import AMCDataset
from amc import process_data


def measure_cpu_latency(model, x, warmup=5, iters=30):
    """CPU 延迟测量：warmup + 多次平均，返回 (均值ms, 标准差ms)。"""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def main():
    print("[1/5] 加载数据（约 1-3 分钟，无输出属正常）...")
    _, test_dict = process_data()
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    print("[2/5] 构建 FP32 模型（CPU）并加载权重...")
    model_fp32 = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)   # 不 .to(device)，全程 CPU
    model_fp32.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))
    model_fp32.eval()

    print("[3/5] 动态量化（nn.Linear + nn.GRU → INT8）...")
    from torch.ao.quantization import quantize_dynamic
    model_int8 = quantize_dynamic(model_fp32, {nn.Linear, nn.GRU}, dtype=torch.qint8)
    model_int8.eval()
    print(f"      GRU 替换后:        {type(model_int8.gru).__name__}")
    print(f"      classifier 替换后: {type(model_int8.classifier).__name__}")

    # 体积对比（序列化到内存，不落盘）
    buf32, buf8 = io.BytesIO(), io.BytesIO()
    torch.save(model_fp32.state_dict(), buf32)
    torch.save(model_int8.state_dict(), buf8)
    size32, size8 = buf32.getbuffer().nbytes / 1024**2, buf8.getbuffer().nbytes / 1024**2
    print(f"      state_dict 体积: FP32 {size32:.1f}MB → INT8 {size8:.1f}MB（{size32/size8:.2f}×）")

    print("[4/5] CPU 延迟对比（FP32 vs INT8）...")
    x1 = torch.randn(1, 128, 2)
    x256 = torch.randn(256, 128, 2)
    l32_b1, _ = measure_cpu_latency(model_fp32, x1)
    l8_b1, _ = measure_cpu_latency(model_int8, x1)
    l32_b256, _ = measure_cpu_latency(model_fp32, x256, warmup=3, iters=10)
    l8_b256, _ = measure_cpu_latency(model_int8, x256, warmup=3, iters=10)
    print(f"      batch=1  : FP32 {l32_b1:.2f}ms → INT8 {l8_b1:.2f}ms（{l32_b1/l8_b1:.2f}×）")
    print(f"      batch=256: FP32 {l32_b256:.2f}ms → INT8 {l8_b256:.2f}ms（{l32_b256/l8_b256:.2f}×）")

    print("[5/5] F1 评估（INT8 动态量化，CPU 全测试集）...")
    y_true, y_pred = [], []
    with torch.no_grad():
        for value, label, _ in tqdm(test_loader, desc="评估 INT8"):
            out = model_int8(value)          # 输入保持浮点，动态量化自动处理
            y_true.extend(label.numpy())
            y_pred.extend(out.argmax(dim=1).numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print("\n" + "=" * 46)
    print("  INT8 动态量化（CPU 推理）:")
    print(f"  Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision (weighted): {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Recall (weighted): {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  F1 Score (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("  FP32 基线参考: Acc 0.6357 / P 0.7144 / R 0.6357 / F1 0.6474")
    print("=" * 46)


if __name__ == "__main__":
    main()
