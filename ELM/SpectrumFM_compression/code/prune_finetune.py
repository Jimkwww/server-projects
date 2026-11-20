"""
实验 B：剪枝后的 LoRA 微调恢复。

对实验 A 的 4 个剪枝配置（magnitude/random × 剪25%/50%）各做 LoRA 微调，
看精度恢复情况——裁决「magnitude 判据的价值是否在微调后才显现」。

流程：剪枝 → inject_lora 注入 → 冻结非 LoRA 参数 → 微调 N 个 epoch → 评估
"""
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.model import ConformerClassifier, inject_lora
from dataset import AMCDataset
from amc import process_data
from int8_static import merge_lora
from prune_ffn import prune_model, evaluate


def finetune(model, train_loader, test_loader, device, epochs=3):
    """注入 LoRA + 冻结主干 + 微调。返回每个 epoch 的 (acc, f1)。"""
    inject_lora(model.encoder,
                target_modules=["query", "key", "value", "output", "fc1", "fc2"],
                r=16, alpha=32)
    # inject_lora 新建的 lora_A/lora_B 默认在 CPU，必须再 .to(device) 一起搬到 GPU
    model.to(device)
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        for value, label in tqdm(train_loader, desc=f"    训练 {epoch+1}/{epochs}", leave=False):
            value, label = value.to(device), label.to(device)
            loss = criterion(model(value), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc, f1 = evaluate(model, test_loader, device)
        history.append((acc, f1))
        print(f"      epoch {epoch+1}/{epochs}: Acc {acc:.4f} / F1 {f1:.4f}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="每个配置微调的 epoch 数")
    parser.add_argument("--configs", type=str, default="all",
                        help="all（4 个配置）/ 25（只剪25%的两个）/ 50")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] 加载数据 + 构建基线模型... 设备: {device}")
    train_dict, test_dict = process_data()
    train_loader = DataLoader(AMCDataset(train_dict), batch_size=256, shuffle=True)
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    base = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    base.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))
    base.eval()
    base = merge_lora(base)

    acc0, f10 = evaluate(copy.deepcopy(base), test_loader, device)
    print(f"      FP32 基线: Acc {acc0:.4f} / F1 {f10:.4f}")

    # 选择配置
    all_configs = [("magnitude", 0.75, 25), ("random", 0.75, 25),
                   ("magnitude", 0.50, 50), ("random", 0.50, 50)]
    if args.configs == "25":
        all_configs = all_configs[:2]
    elif args.configs == "50":
        all_configs = all_configs[2:]

    print(f"[2/4] 裸剪 + [3/4] LoRA 微调（{args.epochs} epochs/配置）...")
    rows = []
    for mode, keep_ratio, pct in all_configs:
        print(f"  --- {mode} 剪{pct}% ---")
        m = copy.deepcopy(base)
        prune_model(m, keep_ratio, mode=mode)
        acc_pruned, f1_pruned = evaluate(m, test_loader, device)
        print(f"      裸剪: Acc {acc_pruned:.4f} / F1 {f1_pruned:.4f}")
        history = finetune(m, train_loader, test_loader, device, epochs=args.epochs)
        rows.append((mode, pct, acc_pruned, f1_pruned, history))

    print("[4/4] 最终对比表...")
    print("\n" + "=" * 78)
    print(f"  {'配置':24s} {'裸剪F1':>8s} {'微调后F1':>9s} {'恢复Δ':>8s} {'vs基线':>8s}")
    print(f"  {'FP32 基线':24s} {'—':>8s} {'—':>9s} {'—':>8s} {f10:8.4f}")
    for mode, pct, _, f1_pruned, history in rows:
        f1_ft = history[-1][1]
        print(f"  {mode + ' 剪' + str(pct) + '%':24s} {f1_pruned:8.4f} {f1_ft:9.4f} "
              f"{f1_ft - f1_pruned:+8.4f} {f1_ft - f10:+8.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
