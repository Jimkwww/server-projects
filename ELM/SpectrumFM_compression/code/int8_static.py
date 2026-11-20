"""
INT8 静态量化（PTQ）实验——手写量化版。

为什么手写：torch 2.14 里 eager 模式 convert 的 QuantStub 有 dispatch bug（已弃用无人修），
且手写能把量化公式（scale/zero_point/分桶/修正项）全部显式落地，教学价值更高。

流程（PTQ 三段式，概念与 torch 官方流程一一对应）：
  1. 替换：目标 Linear → ManualQuantLinear（内部带 observer）
  2. 校准：跑 512 个训练样本，observer 记录每层激活范围（对应 torch 的 prepare+校准）
  3. finalize：从 observer 取 scale/zero_point，量化权重（对应 torch 的 convert）

量化公式（前文概念的直接落地）：
  激活（非对称，quint8）：x_q = round(x / a_scale + a_zp)，范围 [0, 255]
  权重（对称，qint8）  ：w_q = round(w / w_scale)，w_scale = max(|w|)/127，范围 [-128, 127]
  反量化（含 zero_point 修正项）：
      out = (x_q @ w_q^T − a_zp · Σw_q) × a_scale × w_scale + bias
                          ↑ 非对称量化的硬件代价：矩阵乘要处理 zero_point 修正项
"""
import io
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.model import ConformerClassifier, LoRALinear
from dataset import AMCDataset
from amc import process_data
from torch.ao.quantization import MinMaxObserver


class ManualQuantLinear(nn.Module):
    """手动静态量化 Linear。

    校准阶段：浮点前向 + observer 记录激活范围（act_scale 未填时）
    推理阶段：激活量化 → int32 整数矩阵乘 → 反量化（含 zero_point 修正）
    """

    def __init__(self, linear, nbits=8):
        super().__init__()
        self.nbits = nbits
        self.weight_fp = linear.weight.detach()                        # 普通属性：校准用，finalize 后释放
        self.register_buffer("bias", linear.bias.detach()
                             if linear.bias is not None else torch.zeros(linear.out_features))
        self.register_buffer("qweight", torch.empty(0, dtype=torch.int8))   # finalize 后填入
        self.register_buffer("w_scale", torch.tensor(0.0))
        self.register_buffer("act_scale", torch.tensor(0.0))
        self.register_buffer("act_zp", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("correction", torch.empty(0))             # zp·Σw_q 修正项
        self.observer = MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine)

    def forward(self, x):
        if self.act_scale.item() == 0.0:
            # 校准阶段：浮点前向 + observer 顺带记录
            self.observer(x.detach())
            return F.linear(x, self.weight_fp, self.bias)
        # 量化推理
        qx = torch.round(x / self.act_scale.item() + self.act_zp.item()).clamp(0, 255).to(torch.int32)
        out = qx @ self.qweight.to(torch.int32).T                       # int32 累加
        return (out - self.correction) * (self.act_scale.item() * self.w_scale.item()) + self.bias

    def finalize(self):
        """校准后调用：固定激活 scale/zero_point，量化权重，预计算修正项。"""
        self.act_scale, zp = self.observer.calculate_qparams()
        self.act_zp = torch.tensor(int(zp), dtype=torch.int32)
        w_max = self.weight_fp.abs().max()
        qmax = 2 ** (self.nbits - 1) - 1          # 8bit→127, 4bit→7
        self.w_scale = torch.tensor((w_max / qmax).item())
        self.qweight = torch.round(self.weight_fp / self.w_scale.item()).clamp(-(qmax + 1), qmax).to(torch.int8)
        self.correction = (self.act_zp.item() * self.qweight.to(torch.int32).sum(dim=1)).to(torch.int32)
        self.weight_fp = None                                            # 释放 FP32 权重（体积已转移给 int8）
        print(f"      量化一层: a_scale={self.act_scale.item():.4f} a_zp={self.act_zp.item()} "
              f"w_scale={self.w_scale.item():.4f}")


def merge_lora(model):
    """LoRALinear → nn.Linear：LoRA 增量合并进原权重（推理数学等价）。"""
    count = 0
    for name, module in list(model.named_modules()):
        parent_name, _, attr = name.rpartition(".")
        parent = model if not parent_name else model.get_submodule(parent_name)
        if isinstance(module, LoRALinear):
            delta = (module.lora_A.T @ module.lora_B.T) * module.scaling      # [in, out]
            new_linear = nn.Linear(module.linear.in_features, module.linear.out_features,
                                   bias=module.linear.bias is not None)
            with torch.no_grad():
                new_linear.weight.copy_(module.linear.weight + delta.T)
                if module.linear.bias is not None:
                    new_linear.bias.copy_(module.linear.bias)
            setattr(parent, attr, new_linear)
            count += 1
    print(f"      LoRA merge: {count} 个 LoRALinear 已合并为 nn.Linear")
    return model


def replace_with_manual_quant(model):
    """量化地图落地：把 FFN 的 fc1/fc2 和分类头替换为 ManualQuantLinear。"""
    count = 0
    for name, module in list(model.named_modules()):
        parent_name, _, attr = name.rpartition(".")
        parent = model if not parent_name else model.get_submodule(parent_name)
        if isinstance(module, nn.Linear) and (attr in ("fc1", "fc2") or attr == "classifier"):
            setattr(parent, attr, ManualQuantLinear(module))
            count += 1
    print(f"      量化替换: {count} 个 Linear → ManualQuantLinear（FFN fc1/fc2 + 分类头）")
    return model


def measure_cpu_latency(model, x, warmup=5, iters=30):
    """CPU 延迟测量：warmup + 多次平均，返回 (均值ms, 标准差ms)。"""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def main():
    print("[1/6] 加载数据（约 1-3 分钟）...")
    train_dict, test_dict = process_data()
    calib_values = torch.from_numpy(train_dict["value"][:512]).float()   # 校准集：训练集抽 512 个
    test_loader = DataLoader(AMCDataset(test_dict), batch_size=256)

    print("[2/6] 构建模型 + LoRA merge...")
    model_fp32 = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    model_fp32.load_state_dict(torch.load("Checkpoint/amc/Our.pt"))
    model_fp32.eval()
    model_fp32 = merge_lora(model_fp32)

    print("[3/6] 量化替换（量化地图）...")
    # 注意：replace 是 in-place 修改，必须 deepcopy 分离出独立副本，否则
    # model_fp32 和 model_quant 指向同一对象，FP32 对照数据会全被量化污染。
    import copy
    model_quant = copy.deepcopy(model_fp32)
    model_quant = replace_with_manual_quant(model_quant)

    print("[4/6] 校准（512 个样本跑一遍，observer 统计每层激活范围）...")
    model_quant.eval()
    with torch.no_grad():
        for i in range(0, len(calib_values), 256):
            model_quant(calib_values[i:i + 256])

    print("[5/6] finalize（固定 scale/zero_point，量化权重）...")
    for module in model_quant.modules():
        if isinstance(module, ManualQuantLinear):
            module.finalize()

    # 体积对比（量化模型 state_dict：qweight int8 + bias/scale 浮点）
    buf32, buf8 = io.BytesIO(), io.BytesIO()
    torch.save(model_fp32.state_dict(), buf32)
    torch.save(model_quant.state_dict(), buf8)
    size32, size8 = buf32.getbuffer().nbytes / 1024**2, buf8.getbuffer().nbytes / 1024**2
    print(f"      state_dict 体积: FP32 {size32:.1f}MB → INT8手写 {size8:.1f}MB（{size32/size8:.2f}×）")

    print("[6/6] CPU 延迟对比 + F1 评估...")
    x1 = torch.randn(1, 128, 2)
    x256 = torch.randn(256, 128, 2)
    l32_b1, _ = measure_cpu_latency(model_fp32, x1)
    l8_b1, _ = measure_cpu_latency(model_quant, x1)
    l32_b256, _ = measure_cpu_latency(model_fp32, x256, warmup=3, iters=10)
    l8_b256, _ = measure_cpu_latency(model_quant, x256, warmup=3, iters=10)
    print(f"      batch=1  : FP32 {l32_b1:.2f}ms → INT8 {l8_b1:.2f}ms（{l32_b1/l8_b1:.2f}×）")
    print(f"      batch=256: FP32 {l32_b256:.2f}ms → INT8 {l8_b256:.2f}ms（{l32_b256/l8_b256:.2f}×）")

    y_true, y_pred = [], []
    with torch.no_grad():
        for value, label, _ in tqdm(test_loader, desc="评估 INT8手写"):
            out = model_quant(value)
            y_true.extend(label.numpy())
            y_pred.extend(out.argmax(dim=1).numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print("\n" + "=" * 46)
    print("  INT8 静态量化（手写 PTQ，CPU 推理）:")
    print(f"  Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision (weighted): {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Recall (weighted): {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  F1 Score (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("  FP32 基线参考: Acc 0.6357 / P 0.7144 / R 0.6357 / F1 0.6474")
    print("=" * 46)


if __name__ == "__main__":
    main()
