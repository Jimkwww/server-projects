"""
独立评估脚本：加载训练好的微调模型，在测试集上计算总体指标（OA/Precision/Recall/F1）。
支持 FP16 推理（--half）。

用法（在 code/ 目录下）：
    python eval.py            # FP32
    python eval.py --half     # FP16
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from Model.model import ConformerClassifier
from dataset import AMCDataset
from amc import process_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--half", action="store_true", help="FP16 推理")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] 设备: {device}，精度: {'FP16' if args.half else 'FP32'}")

    print("[2/4] 加载并预处理数据（约 1-3 分钟，无输出属正常）...")
    _, test_dict = process_data()
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    print("[3/4] 构建模型...")
    model = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024).to(device)
    model.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))   # 先加载 FP32
    if args.half:
        model = model.half()                                     # 再转 FP16
        print("      模型已转 FP16")

    print("[4/4] 评估中...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for value, label, _ in tqdm(test_loader, desc="评估"):
            value = value.to(device)
            if args.half:
                value = value.half()          # 输入必须同步转 FP16
            out = model(value)
            y_true.extend(label.numpy())
            y_pred.extend(out.argmax(dim=1).cpu().numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    print("\n" + "=" * 46)
    print(f"  样本数: {len(y_true)}")
    print(f"  Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision (weighted): {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Recall (weighted): {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  F1 Score (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("=" * 46)
    print("FP32 基线参考: Acc 0.6357 / P 0.7144 / R 0.6357 / F1 0.6474")


if __name__ == "__main__":
    main()
