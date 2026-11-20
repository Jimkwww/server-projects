#!/usr/bin/env python3
"""ResNet50 图像分类推理测试

用 torchvision 预训练的 ResNet50 对一张图片做 ImageNet 分类，
输出 top-5 预测类别和置信度，验证 GPU 推理是否正常工作。

用法:
    python resnet_inference.py <图片路径>
"""

import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


def main(image_path):
    # 1. 选择设备：有 GPU 用 GPU，否则退回 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[设备] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[设备] CPU")

    # 2. 加载预训练权重（首次运行会自动从网上下载权重，约 100MB）
    print("[加载] 正在加载 ResNet50 预训练权重 ...")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model = model.to(device)
    model.eval()

    # 3. 预处理：权重自带对应的预处理流程（缩放/归一化等）
    preprocess = weights.transforms()

    # 4. 加载并预处理图片
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # 5. 推理
    with torch.no_grad():
        logits = model(img_tensor)
    probs = F.softmax(logits[0], dim=0)

    # 6. top-5 结果
    top5_prob, top5_idx = torch.topk(probs, 5)
    categories = weights.meta["categories"]

    print(f"\n[结果] 图片: {image_path}")
    print(f"{'排名':<4} {'类别':<30} {'置信度':>8}")
    for rank, (idx, p) in enumerate(zip(top5_idx, top5_prob), start=1):
        print(f"{rank:<4} {categories[idx.item()]:<30} {p.item():>8.4f}")

    print(f"\n预测结果: {categories[top5_idx[0].item()]}  (置信度 {top5_prob[0].item():.4f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python resnet_inference.py <图片路径>")
        sys.exit(1)
    main(sys.argv[1])
