"""
INT4 QAT（量化感知训练）实验。

与 PTQ 的对照：PTQ 是"事后量化"（校准后 scale 固定、无训练）；
QAT 是"训练中体验量化"（fake quantize + STE，小 lr 全量微调几个 epoch）。

流程：merge → 替换 QuantAware 模块（4bit）→ 全量小 lr 微调 → finalize → 推理态评估
用法：python int4_qat.py --epochs 5 [--prune-ratio 0.40]（--prune-ratio 用于剪枝+QAT 组合）
"""
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.model import ConformerClassifier, DepthwiseConv
from dataset import AMCDataset
from amc import process_data
from int8_static import merge_lora
from qat_modules import QuantAwareLinear, QuantAwareConv1x1


def replace_with_quant_aware(model, nbits=4):
    """把所有 nn.Linear 和卷积 pointwise 换成 QuantAware 训练版。"""
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
    print(f"      QuantAware 替换: {count} 个模块")
    return model


def evaluate(model, test_loader, device):
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for value, label, _ in tqdm(test_loader, desc="  评估", leave=False):
            out = model(value.to(device))
            y_true.extend(label.numpy())
            y_pred.extend(out.argmax(dim=1).cpu().numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="weighted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4, help="小学习率（模型已收敛，只适应量化）")
    parser.add_argument("--prune-ratio", type=float, default=0.0,
                        help="剪枝比例（剪掉多少）：0.60 = 剪60%、0.70 = 剪70%。0 = 纯 QAT 不剪枝")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[1/5] 加载数据 + 构建模型... 设备: {device}")
    train_dict, test_dict = process_data()
    train_loader = DataLoader(AMCDataset(train_dict), batch_size=256, shuffle=True)
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    model = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    model.load_state_dict(torch.load("Checkpoint/amc/Our.pt", map_location="cpu"))
    model.eval()
    model = merge_lora(model)

    if args.prune_ratio > 0:
        from prune_ffn import prune_model
        prune_model(model, keep_ratio=1 - args.prune_ratio, mode="magnitude")
        print(f"      组合实验：先剪枝 {int(args.prune_ratio*100)}%（magnitude）")
        params = sum(p.numel() for p in model.parameters())
        print(f"      剪枝后参数: {params/1e6:.2f}M")

    model = replace_with_quant_aware(model, nbits=4)

    print(f"[2/5] QAT 训练（{args.epochs} epochs, lr={args.lr}）...")
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(args.epochs):
        model.train()                     # 训练态：fake quantize + STE
        for value, label in tqdm(train_loader, desc=f"  训练 {epoch+1}/{args.epochs}", leave=False):
            value, label = value.to(device), label.to(device)
            loss = criterion(model(value), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc, f1 = evaluate(model, test_loader, device)
        print(f"      epoch {epoch+1}: Acc {acc:.4f} / F1 {f1:.4f}（训练态评估）")

    print("[3/5] finalize（冻结 scale、量化权重为 INT4）...")
    for module in model.modules():
        if isinstance(module, (QuantAwareLinear, QuantAwareConv1x1)):
            module.finalize()
    model.eval()

    # 注：GRU 保持 FP32（QAT 实验重点看 F1；GRU 动态 INT8 可后期叠加，会强制 CPU 慢评估）

    print("[5/5] 推理态评估（真量化，GPU）...")
    acc, f1 = evaluate(model, test_loader, device)
    p = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 50)
    print(f"  INT4 QAT{' + 剪枝' if args.prune_ratio > 0 else ''} 结果:")
    print(f"  Overall Accuracy: {acc:.4f}")
    print(f"  F1 Score (weighted): {f1:.4f}")
    print(f"  参数量: {p/1e6:.2f}M（剪枝前 16.75M）")
    print("  参考: FP32 0.6474 / INT8完全体 0.6510 / INT4-PTQ 待今晚结果")
    print("=" * 50)


if __name__ == "__main__":
    main()
