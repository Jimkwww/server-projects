"""
多 seed 配对验证：对裁决性剪枝配置做多次独立微调，报告 mean±std。

设计：
- 4 个配置（magnitude/random × 剪60%/70%）× N 个 seed
- 配对设计：同一 seed 下 magnitude 和 random 共用相同的 LoRA 初始化与 shuffle 顺序，
  两者的差异纯粹来自剪枝判据（剔除微调随机性的共同部分）
- 输出：每配置 F1 的 mean±std + 配对差（mag−rand，同 seed 相减）的 mean±std

用法：python prune_multiseed.py --seeds 5 --epochs 5
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from Model.model import ConformerClassifier, inject_lora
from dataset import AMCDataset
from amc import process_data
from int8_static import merge_lora
from prune_ffn import prune_model, evaluate


def finetune_seeded(model, train_dict, test_loader, device, seed, epochs=5):
    """固定 seed 的 LoRA 微调：seed 控制 LoRA 初始化、shuffle、dropout。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    inject_lora(model.encoder,
                target_modules=["query", "key", "value", "output", "fc1", "fc2"],
                r=16, alpha=32)
    model.to(device)
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # 按 seed 重建 loader：同一 seed 的 shuffle 顺序一致（配对设计的关键）
    train_loader = DataLoader(AMCDataset(train_dict), batch_size=256, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))

    for epoch in range(epochs):
        model.train()
        for value, label in train_loader:
            value, label = value.to(device), label.to(device)
            loss = criterion(model(value), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    acc, f1 = evaluate(model, test_loader, device)
    return acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="每个配置的独立运行次数")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[1/3] 加载数据 + 构建基线... seeds={args.seeds}, epochs={args.epochs}")
    train_dict, test_dict = process_data()
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    base = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    base.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))
    base.eval()
    base = merge_lora(base)

    acc0, f10 = evaluate(copy.deepcopy(base), test_loader, device)
    print(f"      FP32 基线: Acc {acc0:.4f} / F1 {f10:.4f}")

    print("[2/3] 多 seed 微调（配对设计：同 seed 共用随机源）...")
    configs = [("magnitude", 0.40, 60), ("random", 0.40, 60),
               ("magnitude", 0.30, 70), ("random", 0.30, 70)]
    results = {}   # (mode, pct) -> {"pruned_f1": x, "seed_f1s": [...]}
    seeds = list(range(1, args.seeds + 1))
    out_path = "Results/pruning/multiseed_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def save_results():
        """增量落盘：每测完一个 seed 就写一次，防止长跑中断丢结果。"""
        data = {
            "baseline_f1": f10,
            "configs": {f"{mode}_{pct}": {"pruned_f1": v["pruned_f1"], "seed_f1s": v["seed_f1s"]}
                        for (mode, pct), v in results.items()},
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    # 续跑：读已有结果，跳过已完成的 seed（昨晚中断后不浪费）
    if os.path.exists(out_path):
        with open(out_path) as f:
            old = json.load(f)
        for key, v in old.get("configs", {}).items():
            mode, pct = key.split("_")
            results[(mode, int(pct))] = {"pruned_f1": v["pruned_f1"], "seed_f1s": v["seed_f1s"]}
            done = len(v["seed_f1s"])
            if done:
                print(f"  续跑: {mode} 剪{pct}% 已完成 {done} 个 seed，跳过")
        f10 = old.get("baseline_f1", f10)

    for mode, keep_ratio, pct in configs:
        # 剪枝一次（magnitude 确定；random 用固定 seed 保证可复现）
        pruned = copy.deepcopy(base)
        prune_model(pruned, keep_ratio, mode=mode, seed=0)
        acc_p, f1_p = evaluate(pruned, test_loader, device)
        if (mode, pct) not in results:
            results[(mode, pct)] = {"pruned_f1": f1_p, "seed_f1s": []}
        done_n = len(results[(mode, pct)]["seed_f1s"])
        print(f"  {mode} 剪{pct}% 裸剪: F1 {results[(mode, pct)]['pruned_f1']:.4f}（已完成 {done_n}/{args.seeds} seeds）")
        for seed in seeds[done_n:]:           # 只跑未完成的 seed
            m = copy.deepcopy(pruned)
            _, f1 = finetune_seeded(m, train_dict, test_loader, device, seed, epochs=args.epochs)
            results[(mode, pct)]["seed_f1s"].append(f1)
            print(f"      seed {seed}: F1 {f1:.4f}")
            save_results()          # 每个 seed 测完立即落盘
        save_results()

    print("[3/3] 汇总：mean ± std 与配对差异...")
    print("\n" + "=" * 78)
    print(f"  {'配置':22s} {'微调F1 (mean±std)':>22s} {'vs基线':>8s}")
    print(f"  {'FP32 基线':22s} {'—':>22s} {f10:8.4f}")
    for mode, pct in [(k[0], k[2]) for k in configs]:   # configs 三元组 (mode, keep_ratio, pct)，取 pct 做 key
        f1s = results[(mode, pct)]["seed_f1s"]
        print(f"  {mode + ' 剪' + str(pct) + '%':22s} "
              f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}      {np.mean(f1s) - f10:+8.4f}")

    print("\n  --- 配对差异（magnitude − random，同 seed 相减，剔除共同随机性）---")
    for pct in (60, 70):
        diffs = [m - r for m, r in zip(results[("magnitude", pct)]["seed_f1s"],
                                       results[("random", pct)]["seed_f1s"])]
        print(f"  剪{pct}%: {np.mean(diffs):+.4f} ± {np.std(diffs):.4f}"
              f"（{'显著' if abs(np.mean(diffs)) > 2 * np.std(diffs) else '不显著'}）")
    print("=" * 78)


if __name__ == "__main__":
    main()
