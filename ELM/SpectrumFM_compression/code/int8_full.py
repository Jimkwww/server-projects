"""
INT8 完全体：卷积手写静态 + Linear/GRU 动态量化（merge 后）。

量化地图（最终版）：
  ✅ 卷积 pointwise1/pointwise2 → 手写静态 INT8（ManualQuantConv1x1，1×1 卷积 = 等价矩阵乘）
  ✅ Linear（attention/FFN/classifier）+ GRU → 动态量化（quantize_dynamic）
  ❌ depthwise 卷积、BN、LayerNorm、相对位置 → 保持 FP32（太小或敏感）

与「merge 全动态」实验的唯一差异 = 卷积是否量化 → 干净对比卷积量化的贡献。
预期体积 ~17MB（逼近理论下限）、F1 微掉。
"""
import io
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.model import ConformerClassifier, DepthwiseConv
from dataset import AMCDataset
from amc import process_data
from torch.ao.quantization import MinMaxObserver
from int8_static import merge_lora
from int8_dynamic import measure_cpu_latency


class ManualQuantConv1x1(nn.Module):
    """1×1 卷积的手写静态量化：等价矩阵乘（x[B,C,L] → [B,L,C] @ W^T → [B,L,O] → permute）。

    校准模式（act_scale 未定）：浮点 conv + observer 记录
    量化模式：激活 per-tensor 量化 → int32 矩阵乘 → 反量化（含 zero_point 修正项）
    """

    def __init__(self, conv, nbits=8):
        super().__init__()
        self.nbits = nbits
        self.weight_fp = conv.weight.detach()               # [C_out, C_in]
        self.register_buffer("bias", conv.bias.detach()
                             if conv.bias is not None else torch.zeros(conv.out_channels))
        self.register_buffer("qweight", torch.empty(0, dtype=torch.int8))
        self.register_buffer("w_scale", torch.tensor(0.0))
        self.register_buffer("act_scale", torch.tensor(0.0))
        self.register_buffer("act_zp", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("correction", torch.empty(0))
        self.observer = MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine)

    def forward(self, x):
        if self.act_scale.item() == 0.0:
            self.observer(x.detach())
            return F.conv1d(x, self.weight_fp, self.bias)
        B, C, L = x.shape
        qx = torch.round(x / self.act_scale.item() + self.act_zp.item()).clamp(0, 255).to(torch.int32)
        out = qx.permute(0, 2, 1) @ self.qweight.to(torch.int32).T          # [B, L, C_out]
        out = (out - self.correction.view(1, 1, -1)) * (self.act_scale.item() * self.w_scale.item()) \
            + self.bias.view(1, 1, -1)
        return out.permute(0, 2, 1)                                          # [B, C_out, L]

    def finalize(self):
        self.act_scale, zp = self.observer.calculate_qparams()
        self.act_zp = torch.tensor(int(zp), dtype=torch.int32)
        w_max = self.weight_fp.abs().max()
        qmax = 2 ** (self.nbits - 1) - 1          # 8bit→127, 4bit→7
        self.w_scale = torch.tensor((w_max / qmax).item())
        self.qweight = torch.round(self.weight_fp / self.w_scale.item()).clamp(-(qmax + 1), qmax).to(torch.int8)
        self.correction = (self.act_zp.item() * self.qweight.to(torch.int32).sum(dim=1)).to(torch.int32)
        self.weight_fp = None
        print(f"      卷积量化一层: a_scale={self.act_scale.item():.4f} a_zp={self.act_zp.item()} "
              f"w_scale={self.w_scale.item():.4f}")


def replace_convs(model):
    """量化地图落地：DepthwiseConv 的 pointwise1/pointwise2 → ManualQuantConv1x1。"""
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, DepthwiseConv):
            module.pointwise1 = ManualQuantConv1x1(module.pointwise1)
            module.pointwise2 = ManualQuantConv1x1(module.pointwise2)
            count += 2
    print(f"      卷积量化替换: {count} 个 pointwise 卷积（16 层 × 2）")
    return model


def main():
    print("[1/6] 加载数据（约 1-3 分钟）...")
    train_dict, test_dict = process_data()
    calib_values = torch.from_numpy(train_dict["value"][:512]).float()
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    print("[2/6] 构建模型 + merge + 卷积替换...")
    model = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    model.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))
    model.eval()
    model = merge_lora(model)
    model = replace_convs(model)

    print("[3/6] 校准（512 训练样本，卷积 observer 统计激活范围）...")
    with torch.no_grad():
        for i in range(0, len(calib_values), 256):
            model(calib_values[i:i + 256])

    print("[4/6] finalize 卷积（固定 scale/zp、量化权重）...")
    for module in model.modules():
        if isinstance(module, ManualQuantConv1x1):
            module.finalize()

    print("[5/6] Linear + GRU 动态量化...")
    from torch.ao.quantization import quantize_dynamic
    model = quantize_dynamic(model, {nn.Linear, nn.GRU}, dtype=torch.qint8)
    model.eval()

    # 体积（对照：merge 后未量化的 FP32 模型）
    buf32, buf8 = io.BytesIO(), io.BytesIO()
    fp32_ref = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    fp32_ref.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))
    fp32_ref = merge_lora(fp32_ref)
    torch.save(fp32_ref.state_dict(), buf32)
    torch.save(model.state_dict(), buf8)
    size32, size8 = buf32.getbuffer().nbytes / 1024**2, buf8.getbuffer().nbytes / 1024**2
    print(f"      state_dict 体积: FP32(merge) {size32:.1f}MB → INT8完全体 {size8:.1f}MB（{size32/size8:.2f}×）")

    print("[6/6] CPU 延迟对比 + F1 评估...")
    x1 = torch.randn(1, 128, 2)
    l32_b1, _ = measure_cpu_latency(fp32_ref, x1)
    l8_b1, _ = measure_cpu_latency(model, x1)
    print(f"      batch=1: FP32 {l32_b1:.2f}ms → INT8完全体 {l8_b1:.2f}ms（{l32_b1/l8_b1:.2f}×）")

    y_true, y_pred = [], []
    with torch.no_grad():
        for value, label, _ in tqdm(test_loader, desc="评估 INT8完全体"):
            out = model(value)
            y_true.extend(label.numpy())
            y_pred.extend(out.argmax(dim=1).numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print("\n" + "=" * 46)
    print("  INT8 完全体（卷积手写静态 + Linear/GRU 动态，CPU）:")
    print(f"  Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision (weighted): {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Recall (weighted): {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  F1 Score (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("  参考: FP32 0.6474 / merge全动态 ~26MB / 理论下限 ~17MB")
    print("=" * 46)


if __name__ == "__main__":
    main()
