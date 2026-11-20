"""
独立评估脚本：加载训练好的微调模型，在测试集上计算总体指标（OA/Precision/Recall/F1）。

用法（必须在 code/ 目录下）：
    python eval.py

说明：
    - 复用 amc.py 的 process_data() 加载数据（random_state=42 保证测试集与训练时一致）
    - 加载 Checkpoint/amc/Our.pt（微调后的完整模型，不是预训练权重 pretrain_model_2018.pt）
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from Model.model import ConformerClassifier
from dataset import AMCDataset
from amc import process_data          # 复用 amc.py 的数据预处理


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] 设备: {device}")

    # 1. 数据（process_data 里的 Python 循环归一化约 1-3 分钟，无输出属正常）
    print("[2/4] 加载并预处理数据（约 1-3 分钟，无输出属正常）...")
    _, test_dict = process_data()
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    # 2. 建模型（构造时会加载预训练权重 + 注入 LoRA）
    print("[3/4] 构建模型...")
    model = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024).to(device)

    # 3. 加载微调后的 checkpoint（覆盖构造时的预训练权重）
    ckpt = "Checkpoint/amc/Our.pt"
    model.load_state_dict(torch.load(ckpt))
    print(f"      已加载微调模型: {ckpt}")

    # 4. 评估
    print("[4/4] 评估中...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for value, label, _ in tqdm(test_loader, desc="评估"):
            value = value.to(device)
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
    print("论文 TABLE III (RML2016.10A 'Our'): Precision 0.7529 / Recall 0.6372 / F1 0.6620")


if __name__ == "__main__":
    main()
