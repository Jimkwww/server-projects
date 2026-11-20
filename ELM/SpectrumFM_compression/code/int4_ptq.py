"""
INT4 PTQ 实验：全手写 4bit 权重量化（W4A8）+ GRU 动态 INT8。

动机：QAT 的对照基线——先看 PTQ-INT4 掉多少 F1，之后 QAT 再拉回。
量化地图：
  ✅ 所有 Linear（attention/FFN/classifier）→ 手写 4bit（ManualQuantLinear(nbits=4)）
  ✅ 卷积 pointwise → 手写 4bit（ManualQuantConv1x1(nbits=4)）
  ✅ GRU → 动态 INT8（torch quantize_dynamic）
  ❌ depthwise/BN/LN/相对位置 → FP32

注意体积口径：torch 无原生 int4 dtype，本实验 qweight 用 int8 存（值域 -8~7），
存储不省；理论体积 8.6MB 需位打包（两个 int4 挤一个 int8）。本实验重点测 F1。
"""
import io

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.model import ConformerClassifier, DepthwiseConv
from dataset import AMCDataset
from amc import process_data
from int8_static import merge_lora, ManualQuantLinear
from int8_full import ManualQuantConv1x1


def replace_all_with_manual(model, nbits=4):
    """量化地图落地：所有 nn.Linear 和卷积 pointwise 换成手写 nbits 量化版。"""
    count = 0
    for name, module in list(model.named_modules()):
        parent_name, _, attr = name.rpartition(".")
        parent = model if not parent_name else model.get_submodule(parent_name)
        if isinstance(module, nn.Linear):
            setattr(parent, attr, ManualQuantLinear(module, nbits=nbits))
            count += 1
        elif isinstance(module, DepthwiseConv):
            module.pointwise1 = ManualQuantConv1x1(module.pointwise1, nbits=nbits)
            module.pointwise2 = ManualQuantConv1x1(module.pointwise2, nbits=nbits)
            count += 2
    print(f"      INT4 量化替换: {count} 个模块（Linear + 卷积 pointwise）")
    return model


def main():
    print("[1/6] 加载数据（约 1-3 分钟）...")
    train_dict, test_dict = process_data()
    calib_values = torch.from_numpy(train_dict["value"][:512]).float()
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    print("[2/6] 构建模型 + merge + INT4 替换...")
    model = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    model.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))
    model.eval()
    model = merge_lora(model)
    model = replace_all_with_manual(model, nbits=4)

    print("[3/6] 校准（512 训练样本）...")
    with torch.no_grad():
        for i in range(0, len(calib_values), 256):
            model(calib_values[i:i + 256])

    print("[4/6] finalize（固定 scale、量化 4bit 权重）...")
    for module in model.modules():
        if isinstance(module, (ManualQuantLinear, ManualQuantConv1x1)):
            module.finalize()

    print("[5/6] GRU 动态 INT8...")
    from torch.ao.quantization import quantize_dynamic
    model = quantize_dynamic(model, {nn.GRU}, dtype=torch.qint8)
    model.eval()

    # 体积（存储用 int8，非位打包；理论 4bit 体积另算）
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size_now = buf.getbuffer().nbytes / 1024**2
    print(f"      state_dict 体积: {size_now:.1f}MB（int8 存储；理论 4bit 位打包 ~8.6MB）")

    print("[6/6] F1 评估（全测试集，CPU）...")
    y_true, y_pred = [], []
    with torch.no_grad():
        for value, label, _ in tqdm(test_loader, desc="评估 INT4"):
            out = model(value)
            y_true.extend(label.numpy())
            y_pred.extend(out.argmax(dim=1).numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print("\n" + "=" * 46)
    print("  INT4 PTQ（W4A8 手写，CPU 推理）:")
    print(f"  Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision (weighted): {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Recall (weighted): {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  F1 Score (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("  参考: FP32 0.6474 / INT8完全体 0.6510（零损失）/ INT4 预期掉 2~6 点")
    print("=" * 46)


if __name__ == "__main__":
    main()
