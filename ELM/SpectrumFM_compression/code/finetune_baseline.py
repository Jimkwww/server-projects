"""
对照实验：未压缩 FP32 模型的 LoRA 微调（5 epochs × 5 seeds）。

科学问题：压缩配置普遍"比基线好 0.25~0.55 点"，这是压缩的功劳还是「多训了 5 epochs」的功劳？
设计：对未压缩的 merge 后模型做与剪枝恢复完全相同的微调流程（同 seed 配对），
     若它也涨到 ~0.650 → 提升来自训练本身，压缩的真实贡献是「不损精度」；
     若不动（≈0.6474）→ 压缩确有正则化/扰动收益。

产物：Results/pruning/baseline_finetune.json
"""
import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from Model.model import ConformerClassifier
from dataset import AMCDataset
from amc import process_data
from int8_static import merge_lora
from prune_ffn import evaluate
from prune_multiseed import finetune_seeded


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[1/3] 加载数据 + 构建基线... 设备: {device}")
    train_dict, test_dict = process_data()
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    base = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    base.load_state_dict(torch.load("Checkpoint/amc/Our.pt", map_location="cpu"))
    base.eval()
    base = merge_lora(base)

    acc0, f10 = evaluate(copy.deepcopy(base), test_loader, device)
    print(f"      基线（未微调）: Acc {acc0:.4f} / F1 {f10:.4f}")

    print("[2/3] 5 seeds LoRA 微调（与剪枝恢复同流程、同 seed）...")
    f1s = []
    for seed in range(1, 6):
        m = copy.deepcopy(base)
        acc, f1 = finetune_seeded(m, train_dict, test_loader, device, seed, epochs=5)
        f1s.append(f1)
        print(f"      seed {seed}: F1 {f1:.4f}")

    f1s = np.array(f1s)
    print("[3/3] 结论...")
    print("\n" + "=" * 60)
    print(f"  基线（未微调）: {f10:.4f}")
    print(f"  微调后 5 seeds: {f1s.mean():.4f} ± {f1s.std():.4f}")
    print(f"  提升: {f1s.mean() - f10:+.4f}")
    if f1s.mean() - f10 > 2 * f1s.std():
        print("  → 提升显著：微调本身带来真实增益（压缩的贡献=不损精度）")
    else:
        print("  → 提升不显著：压缩配置的超基线属于噪声带")
    print("  参考：剪枝配置多 seed 均值 +0.25~0.55 点")
    print("=" * 60)

    out = "Results/pruning/baseline_finetune.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"baseline_f1": f10, "finetuned_f1s": [float(x) for x in f1s]}, f, indent=2)
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
