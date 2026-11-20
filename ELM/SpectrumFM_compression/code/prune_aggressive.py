"""
激进剪枝实验：60% / 70% 剪枝比例，探测这个模型剪枝的真实上限。

与实验 A/B 的差别：剪枝比例更激进（此前 25%/50% 全部恢复了，说明上限未到）。
同样流程：裸剪评估 → LoRA 微调 5 epochs → 对比恢复情况。
"""
import copy

import torch
from torch.utils.data import DataLoader

from Model.model import ConformerClassifier
from dataset import AMCDataset
from amc import process_data
from int8_static import merge_lora
from prune_ffn import prune_model, evaluate
from prune_finetune import finetune


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[1/3] 加载数据 + 构建基线... 设备: {device}")
    train_dict, test_dict = process_data()
    train_loader = DataLoader(AMCDataset(train_dict), batch_size=256, shuffle=True)
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    base = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    base.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))
    base.eval()
    base = merge_lora(base)

    acc0, f10 = evaluate(copy.deepcopy(base), test_loader, device)
    print(f"      FP32 基线: Acc {acc0:.4f} / F1 {f10:.4f}")

    print("[2/3] 裸剪 + LoRA 微调（5 epochs）...")
    configs = [("magnitude", 0.40, 60), ("random", 0.40, 60),
               ("magnitude", 0.30, 70), ("random", 0.30, 70)]
    rows = []
    for mode, keep_ratio, pct in configs:
        print(f"  --- {mode} 剪{pct}% ---")
        m = copy.deepcopy(base)
        prune_model(m, keep_ratio, mode=mode)
        acc_p, f1_p = evaluate(m, test_loader, device)
        params = sum(p.numel() for p in m.parameters())
        print(f"      裸剪: Acc {acc_p:.4f} / F1 {f1_p:.4f} / 参数 {params/1e6:.2f}M")
        history = finetune(m, train_loader, test_loader, device, epochs=5)
        rows.append((mode, pct, f1_p, history, params))

    print("[3/3] 对比表（含此前 25%/50% 的历史结果）...")
    print("\n" + "=" * 80)
    print(f"  {'配置':24s} {'裸剪F1':>8s} {'微调后F1':>9s} {'恢复Δ':>8s} {'vs基线':>8s} {'参数':>8s}")
    print(f"  {'FP32 基线':24s} {'—':>8s} {'—':>9s} {'—':>8s} {f10:8.4f} {sum(p.numel() for p in base.parameters())/1e6:7.2f}M")
    print(f"  {'[历史] mag 剪25%':24s} {'0.6429':>8s} {'0.6573':>9s} {'—':>8s} {'+0.0099':>8s} {'14.65':>7}M")
    print(f"  {'[历史] mag 剪50%':24s} {'0.6102':>8s} {'0.6538':>9s} {'—':>8s} {'+0.0065':>8s} {'12.55':>7}M")
    for mode, pct, f1_p, history, params in rows:
        f1_ft = history[-1][1]
        print(f"  {mode + ' 剪' + str(pct) + '%':24s} {f1_p:8.4f} {f1_ft:9.4f} "
              f"{f1_ft - f1_p:+8.4f} {f1_ft - f10:+8.4f} {params/1e6:7.2f}M")
    print("=" * 80)


if __name__ == "__main__":
    main()
