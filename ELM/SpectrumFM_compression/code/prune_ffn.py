"""
实验 A：FFN 中间维度结构化剪枝（magnitude vs random 对照）。

剪枝对象：16 层 × 2 个 FFN 的中间 512 个神经元
判据（magnitude）：每个中间神经元的「入边 + 出边」L1 范数之和
                    = |fc1.weight[i]| 的行范数 + |fc2.weight[:, i]| 的列范数
对照（random）  ：随机选同样数量的神经元（判断 magnitude 判据有没有信息量）
实现：成对切片重建——fc1 删行、fc2 删列，生成真正更小的稠密模型（不是置零）
流程：剪完【直接评估】F1（不微调，先看裸掉多少），GPU 上评估（快）
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.model import ConformerClassifier
from dataset import AMCDataset
from amc import process_data
from int8_static import merge_lora


def prune_ffn_importance(ffn, keep_ratio, mode="magnitude", seed=0):
    """按判据剪一个 FFN 的中间维度，成对切片重建 fc1/fc2。

    ffn 结构：fc1(256→512) → GELU → fc2(512→256)
    中间 512 个神经元被剪到 512×keep_ratio 个。
    """
    fc1, fc2 = ffn.fc1, ffn.fc2
    n = fc1.out_features                       # 512
    k = int(n * keep_ratio)

    if mode == "magnitude":
        # 判据：每个中间神经元的入边（fc1 行）+ 出边（fc2 列）的 L1 范数
        importance = fc1.weight.abs().sum(dim=1) + fc2.weight.abs().sum(dim=0)   # [n]
        keep = importance.argsort(descending=True)[:k]
    elif mode == "random":
        g = torch.Generator().manual_seed(seed)
        keep = torch.randperm(n, generator=g)[:k]
    else:
        raise ValueError(mode)

    keep = keep.sort()[0]                      # 排好序，保留原相对顺序（队伍裁员不换序）

    # 成对重建：fc1 删行（输出侧）、fc2 删列（输入侧）；fc2.bias 是输出侧，不动
    new_fc1 = nn.Linear(fc1.in_features, k, bias=fc1.bias is not None)
    new_fc1.weight.data = fc1.weight.data[keep].clone()
    if fc1.bias is not None:
        new_fc1.bias.data = fc1.bias.data[keep].clone()

    new_fc2 = nn.Linear(k, fc2.out_features, bias=fc2.bias is not None)
    new_fc2.weight.data = fc2.weight.data[:, keep].clone()
    if fc2.bias is not None:
        new_fc2.bias.data = fc2.bias.data.clone()

    ffn.fc1, ffn.fc2 = new_fc1, new_fc2


def prune_model(model, keep_ratio, mode="magnitude", seed=0):
    """对整个模型的所有 FFN 剪枝（16 层 × feed_forward1/2 = 32 个 FFN）。

    注意：encoder.layers 的第 0 个元素是 InputProjection（作者的设计），
    要跳过它，只对 ConformerEncoderLayer 剪。
    """
    from Model.model import ConformerEncoderLayer
    for layer in model.encoder.layers:
        if not isinstance(layer, ConformerEncoderLayer):
            continue
        for ffn_name in ("feed_forward1", "feed_forward2"):
            prune_ffn_importance(getattr(layer, ffn_name), keep_ratio, mode, seed)
    return model


def evaluate(model, test_loader, device):
    """GPU 上跑全测试集，返回 (accuracy, F1)。"""
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] 加载数据 + 构建模型（merge LoRA）... 设备: {device}")
    _, test_dict = process_data()
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    base = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    base.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))
    base.eval()
    base = merge_lora(base)

    print("[2/4] FP32 基线评估...")
    acc0, f10 = evaluate(copy.deepcopy(base), test_loader, device)
    print(f"      基线: Acc {acc0:.4f} / F1 {f10:.4f} / 参数 {sum(p.numel() for p in base.parameters())/1e6:.2f}M")

    print("[3/4] 四个剪枝配置（magnitude/random × 剪25%/50%）...")
    configs = [
        ("magnitude", 0.75),   # 保留 75%（剪 25%）
        ("random",    0.75),
        ("magnitude", 0.50),   # 保留 50%（剪 50%）
        ("random",    0.50),
    ]
    results = []
    for mode, keep_ratio in configs:
        m = copy.deepcopy(base)
        prune_model(m, keep_ratio, mode=mode)
        params = sum(p.numel() for p in m.parameters())
        acc, f1 = evaluate(m, test_loader, device)
        pruned_pct = int((1 - keep_ratio) * 100)
        print(f"      {mode:9s} 剪{pruned_pct:2d}%: Acc {acc:.4f} / F1 {f1:.4f} / 参数 {params/1e6:.2f}M")
        results.append((mode, pruned_pct, acc, f1, params))

    print("[4/4] 对比表（裸剪、未微调）...")
    print("\n" + "=" * 70)
    print(f"  {'配置':26s} {'Acc':>8s} {'F1':>8s} {'ΔF1':>8s} {'参数':>8s}")
    print(f"  {'FP32 基线':26s} {acc0:8.4f} {f10:8.4f} {'—':>8s} {sum(p.numel() for p in base.parameters())/1e6:7.2f}M")
    for mode, pct, acc, f1, params in results:
        print(f"  {mode + ' 剪' + str(pct) + '%':26s} {acc:8.4f} {f1:8.4f} {f1-f10:+8.4f} {params/1e6:7.2f}M")
    print("=" * 70)


if __name__ == "__main__":
    main()
