"""把 FP32 微调模型转成 FP16 并保存（体积减半的证据）。

用法（在 code/ 目录下）：
    python convert_to_fp16.py
"""
import torch

from Model.model import ConformerClassifier

model = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
model.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))   # FP32 加载
model = model.half()                                          # 再转 FP16（顺序铁律：先加载后转）
torch.save(model.state_dict(), "Checkpoint/amc/Our_fp16.pt")
print("FP16 checkpoint 已保存: Checkpoint/amc/Our_fp16.pt")
