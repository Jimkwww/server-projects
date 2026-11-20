"""
INT8 动态量化完整版：LoRA merge + 全 Linear/GRU 动态量化。

对比之前的动态实验（未 merge → 31.0MB），merge 消除 LoRA 参数后预期 ~26MB。
这是动态量化的最佳形态：merge（部署标准预处理）+ quantize_dynamic（一行全量化）。
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
from int8_static import merge_lora
from int8_dynamic import measure_cpu_latency


def main():
    print("[1/5] 加载数据（约 1-3 分钟）...")
    _, test_dict = process_data()
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    print("[2/5] 构建模型 + LoRA merge...")
    model_fp32 = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    model_fp32.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))
    model_fp32.eval()
    model_fp32 = merge_lora(model_fp32)          # ← 关键：先 merge（消除 LoRA 参数）

    print("[3/5] 全动态量化（所有 nn.Linear + nn.GRU）...")
    from torch.ao.quantization import quantize_dynamic
    model_int8 = quantize_dynamic(model_fp32, {nn.Linear, nn.GRU}, dtype=torch.qint8)
    model_int8.eval()

    # 体积对比
    buf32, buf8 = io.BytesIO(), io.BytesIO()
    torch.save(model_fp32.state_dict(), buf32)
    torch.save(model_int8.state_dict(), buf8)
    size32, size8 = buf32.getbuffer().nbytes / 1024**2, buf8.getbuffer().nbytes / 1024**2
    print(f"      state_dict 体积: FP32(merge) {size32:.1f}MB → INT8全动态 {size8:.1f}MB（{size32/size8:.2f}×）")

    print("[4/5] CPU 延迟对比...")
    x1 = torch.randn(1, 128, 2)
    x256 = torch.randn(256, 128, 2)
    l32_b1, _ = measure_cpu_latency(model_fp32, x1)
    l8_b1, _ = measure_cpu_latency(model_int8, x1)
    l32_b256, _ = measure_cpu_latency(model_fp32, x256, warmup=3, iters=10)
    l8_b256, _ = measure_cpu_latency(model_int8, x256, warmup=3, iters=10)
    print(f"      batch=1  : FP32 {l32_b1:.2f}ms → INT8 {l8_b1:.2f}ms（{l32_b1/l8_b1:.2f}×）")
    print(f"      batch=256: FP32 {l32_b256:.2f}ms → INT8 {l8_b256:.2f}ms（{l32_b256/l8_b256:.2f}×）")

    print("[5/5] F1 评估（全测试集，CPU）...")
    y_true, y_pred = [], []
    with torch.no_grad():
        for value, label, _ in tqdm(test_loader, desc="评估 INT8全动态"):
            out = model_int8(value)
            y_true.extend(label.numpy())
            y_pred.extend(out.argmax(dim=1).numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print("\n" + "=" * 46)
    print("  INT8 全动态（merge 后，CPU 推理）:")
    print(f"  Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision (weighted): {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Recall (weighted): {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  F1 Score (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("  参考: FP32 0.6474 / 旧动态(未merge) 0.6525 / 31.0MB")
    print("=" * 46)


if __name__ == "__main__":
    main()
