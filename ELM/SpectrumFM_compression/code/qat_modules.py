"""
QAT（量化感知训练）核心模块：fake quantize + STE 的手写实现。

与 PTQ 手写模块（int8_static.py）的对比：
  - PTQ 推理版：校准后 scale 固定、权重转 int8 存储、整数矩阵乘
  - QAT 训练版：权重保持 FP32（master weight）、前向 fake quantize（模拟量化误差）、
                STE 直通梯度、训练后 finalize 转成真量化

STE 的一行实现（核心中的核心）：
  x_q = x + (round(x/s) - x/s).detach() * s
  前向：x_q = round(x/s)*s，有真实量化误差
  反向：detach 掐断量化部分的梯度，梯度直通 x
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import MovingAverageMinMaxObserver


class QuantAwareLinear(nn.Module):
    """量化感知训练的 Linear（nbits 权重 + 8bit 激活）。

    训练态：fake quantize + STE，observer 移动平均更新激活 scale
    推理态（finalize 后）：转成真量化（复用 PTQ 的 int32 矩阵乘路径）
    """

    def __init__(self, linear, nbits=4):
        super().__init__()
        self.nbits = nbits
        # master weight 保持 FP32（训练中被 STE 梯度更新）
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone()
                                 if linear.bias is not None else torch.zeros(linear.out_features))
        self.act_observer = MovingAverageMinMaxObserver(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine)
        # 量化参数注册为 buffer：进 state_dict，保存/加载后可恢复 finalize 状态
        # （教训：普通属性不进 state_dict——之前的 QAT checkpoint 因此丢失量化参数）
        self.register_buffer("_qweight", torch.empty(linear.out_features, linear.in_features, dtype=torch.int8))
        self.register_buffer("_w_scale", torch.tensor(0.0))
        self.register_buffer("_a_scale", torch.tensor(0.0))
        self.register_buffer("_a_zp", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("_correction", torch.empty(linear.out_features, dtype=torch.int32))
        # 纯 Python 属性（不进 state_dict）：
        # _finalized —— finalize 状态标志，forward 用 bool 判断，避免逐层 GPU→CPU 同步
        # _qw_f32   —— int8 权重转 float32 的缓存，首次推理前向一次性建好，避免每步重转
        self._finalized = False
        self._qw_f32 = None

    def forward(self, x):
        # 分流依据是 finalize 状态标志（Python bool，无逐层 GPU→CPU 同步），
        # 而不是 train/eval 模式：训练中的评估（eval 模式）也走 fake quantize；
        # finalize / checkpoint 加载之后才走真量化推理。
        if not self._finalized:
            # ===== fake quantize + STE（训练与训练中评估） =====
            qmax = 2 ** (self.nbits - 1) - 1                     # 4bit→7, 8bit→127
            w_scale = self.weight.detach().abs().max() / qmax
            # 权重 fake quantize（STE：反向直通）
            w_q = self.weight + (torch.round(self.weight / w_scale)
                                 - self.weight / w_scale).detach() * w_scale
            # 激活 fake quantize（observer 移动平均统计范围）
            if self.training:                                       # 只在训练时更新统计，
                self.act_observer(x.detach())                       # 评估时用现有范围（标准 QAT 行为）
            a_scale, a_zp = self.act_observer.calculate_qparams()   # 取当前范围
            x_q = torch.round(x / a_scale + a_zp).clamp(0, 255)
            x_fq = (x_q - a_zp) * a_scale                          # 反量化回浮点（有误差）
            return F.linear(x_fq, w_q, self.bias)

        # ===== 推理态：finalize 后的真量化 =====
        # 注意：矩阵乘用 float32（GPU 不支持 int32 baddbmm；整数转浮点精确无损，
        # 量化误差来自值的离散化而非算子 dtype，数学等价）
        if self._qw_f32 is None or self._qw_f32.device != self._qweight.device:
            self._qw_f32 = self._qweight.to(torch.float32)   # 首次前向一次性缓存，避免每步重转
        qx = torch.round(x / self._a_scale + self._a_zp).clamp(0, 255).to(torch.float32)
        out = qx @ self._qw_f32.T
        return (out - self._correction) * (self._a_scale * self._w_scale) + self.bias

    def finalize(self):
        """训练结束：冻结 scale，把 master weight 量化成 int8（容器装 nbits 值）。"""
        a_scale, zp = self.act_observer.calculate_qparams()
        self._a_scale = a_scale.to(self.weight.device)
        self._a_zp = torch.tensor(int(zp), dtype=torch.int32)
        qmax = 2 ** (self.nbits - 1) - 1
        w_max = self.weight.detach().abs().max()
        self._w_scale = torch.tensor((w_max / qmax).item())
        self._qweight = torch.round(self.weight.detach() / self._w_scale) \
            .clamp(-(qmax + 1), qmax).to(torch.int8)
        self._correction = (self._a_zp * self._qweight.to(torch.int32).sum(dim=1)).to(torch.int32)
        self._finalized = True
        self._qw_f32 = None        # 权重已更新，失效缓存，由 forward 一次性重建

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        # 纯 Python 属性 _finalized/_qw_f32 不进 state_dict，加载后必须从 buffer 补推一次
        # （map_location="cpu" 加载时 .item() 无 GPU 同步成本）
        self._finalized = bool(self._a_scale.item() != 0.0)
        self._qw_f32 = None        # 由 forward 在目标设备上一次性重建


class QuantAwareConv1x1(nn.Module):
    """量化感知训练的 1×1 卷积（与 QuantAwareLinear 同理，矩阵乘视角）。"""

    def __init__(self, conv, nbits=4):
        super().__init__()
        self.nbits = nbits
        self.weight = nn.Parameter(conv.weight.detach().clone())   # [C_out, C_in]
        self.bias = nn.Parameter(conv.bias.detach().clone()
                                 if conv.bias is not None else torch.zeros(conv.out_channels))
        self.act_observer = MovingAverageMinMaxObserver(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine)
        self.register_buffer("_qweight", torch.empty(conv.out_channels, conv.in_channels, dtype=torch.int8))
        self.register_buffer("_w_scale", torch.tensor(0.0))
        self.register_buffer("_a_scale", torch.tensor(0.0))
        self.register_buffer("_a_zp", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("_correction", torch.empty(conv.out_channels, dtype=torch.int32))
        self._finalized = False
        self._qw_f32 = None

    def forward(self, x):
        if not self._finalized:
            qmax = 2 ** (self.nbits - 1) - 1
            w_scale = self.weight.detach().abs().max() / qmax
            w_q = self.weight + (torch.round(self.weight / w_scale)
                                 - self.weight / w_scale).detach() * w_scale
            if self.training:
                self.act_observer(x.detach())
            a_scale, a_zp = self.act_observer.calculate_qparams()
            x_q = torch.round(x / a_scale + a_zp).clamp(0, 255)
            x_fq = (x_q - a_zp) * a_scale
            return F.conv1d(x_fq, w_q, self.bias)

        B, C, L = x.shape
        if self._qw_f32 is None or self._qw_f32.device != self._qweight.device:
            self._qw_f32 = self._qweight.to(torch.float32)   # 首次前向一次性缓存，避免每步重转
        qx = torch.round(x / self._a_scale + self._a_zp).clamp(0, 255).to(torch.float32)
        out = qx.permute(0, 2, 1) @ self._qw_f32.T
        out = (out - self._correction.view(1, 1, -1)) * (self._a_scale * self._w_scale) \
            + self.bias.view(1, 1, -1)
        return out.permute(0, 2, 1)

    def finalize(self):
        a_scale, zp = self.act_observer.calculate_qparams()
        self._a_scale = a_scale.to(self.weight.device)
        self._a_zp = torch.tensor(int(zp), dtype=torch.int32)
        qmax = 2 ** (self.nbits - 1) - 1
        w_max = self.weight.detach().abs().max()
        self._w_scale = torch.tensor((w_max / qmax).item())
        self._qweight = torch.round(self.weight.detach() / self._w_scale) \
            .clamp(-(qmax + 1), qmax).to(torch.int8).squeeze(-1)     # [out,in,1] → [out,in]
        self._correction = (self._a_zp * self._qweight.to(torch.int32).sum(dim=1)).to(torch.int32)
        self._finalized = True
        self._qw_f32 = None        # 权重已更新，失效缓存，由 forward 一次性重建

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self._finalized = bool(self._a_scale.item() != 0.0)
        self._qw_f32 = None        # 由 forward 在目标设备上一次性重建
