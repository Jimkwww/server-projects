"""
保存关键压缩模型（供 profile_model.py 测效率 / 复现推理）。

产物（Checkpoint/）：
  prune60.pt           剪 60%（magnitude，merge 后）      —— 秒级，无需数据
  prune40.pt           剪 40%（magnitude，merge 后）      —— 秒级；剪40%+FP16 由此模型现场 .half()
  qat_int4.pt          纯 INT4 QAT（finalize 后）         —— 需训练 5 epochs
  qat_int4_prune40.pt  剪 40% + INT4 QAT（finalize 后）   —— 需训练 5 epochs

用法：
  python save_models.py            # 全部（约 1 小时：两个 QAT 各 ~30 分钟）
  python save_models.py --quick    # 只存剪枝模型（秒级）

加载注意：这些 checkpoint 是「压缩后结构」的 state_dict，加载时需按同样流程
重建结构（merge/剪枝/QuantAware 替换），再 load_state_dict——profile 脚本里配套实现。
"""
import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Model.model import ConformerClassifier, DepthwiseConv
from dataset import AMCDataset
from amc import process_data
from int8_static import merge_lora
from prune_ffn import prune_model
from qat_modules import QuantAwareLinear, QuantAwareConv1x1


def build_base():
    model = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    model.load_state_dict(torch.load("Checkpoint/amc/Our.pt", map_location="cpu"))
    model.eval()
    return merge_lora(model)


def replace_quant_aware(model, nbits=4):
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


def train_qat(model, train_loader, device, epochs=5, lr=2e-4):
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for value, label in train_loader:
            value, label = value.to(device), label.to(device)
            loss = criterion(model(value), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"      QAT epoch {epoch + 1}/{epochs} 完成")
    for module in model.modules():
        if isinstance(module, (QuantAwareLinear, QuantAwareConv1x1)):
            module.finalize()
    model.eval()
    return model


def save(model, path):
    torch.save(model.state_dict(), path)
    print(f"      已保存: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只保存剪枝模型（秒级，不训练 QAT）")
    parser.add_argument("--qat-epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("[1/4] 剪枝模型（magnitude，merge 后）...")
    m = build_base()
    prune_model(m, keep_ratio=0.40, mode="magnitude")
    save(m, "Checkpoint/prune60.pt")

    m = build_base()
    prune_model(m, keep_ratio=0.60, mode="magnitude")      # 剪 40%：keep 0.60
    save(m, "Checkpoint/prune40.pt")

    if args.quick:
        print("（--quick 模式，跳过 QAT 模型）")
        return

    print("[2/4] 纯 INT4 QAT 模型（训练 + finalize）...")
    train_dict, _ = process_data()
    train_loader = DataLoader(AMCDataset(train_dict), batch_size=256, shuffle=True)
    device = torch.device(args.device)

    m = build_base()
    m = replace_quant_aware(m, nbits=4)
    m = train_qat(m, train_loader, device, epochs=args.qat_epochs)
    save(m, "Checkpoint/qat_int4.pt")

    print("[3/4] 剪 40% + INT4 QAT 模型...")
    m = build_base()
    prune_model(m, keep_ratio=0.60, mode="magnitude")      # 剪 40%：keep 0.60
    m = replace_quant_aware(m, nbits=4)
    m = train_qat(m, train_loader, device, epochs=args.qat_epochs)
    save(m, "Checkpoint/qat_int4_prune40.pt")

    print("\n全部保存完成。")


if __name__ == "__main__":
    main()
