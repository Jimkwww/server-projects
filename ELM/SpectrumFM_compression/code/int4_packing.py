"""
INT4 位打包：两个 4bit 值挤进一个字节，兑现真实体积（0.5 字节/参数量化权重）。

对象：qat_int4.pt / qat_int4_prune40.pt 的 QuantAware 模块权重（int8 容器存 -8~7）
内容：① 打包（uint8：高4位+低4位）② 解包验证数值一致 ③ 真实体积统计
输出：对比表（原 state_dict vs 打包后）+ 体积成分账

注意体积口径：GRU / bias / 归一化 / scale 保持 FP32（本脚本如实统计，
不把「全参数 4bit」的理论值当真——之前的 ~8.6MB 是简化估算，实际 ≈ 11MB 级）。
"""
import io

import torch
import torch.nn as nn

from Model.model import ConformerClassifier, DepthwiseConv
from int8_static import merge_lora
from prune_ffn import prune_model
from qat_modules import QuantAwareLinear, QuantAwareConv1x1


def pack_int4(qweight):
    """int8 容器（值 -8~7）→ uint8 位打包（两个 4bit 挤一个字节）。"""
    flat = qweight.flatten().to(torch.int16)
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype)])
    u = (flat + 8).to(torch.uint8)          # -8~7 → 0~15
    return ((u[0::2] << 4) | (u[1::2] & 0xF))


def unpack_int4(packed, numel):
    """uint8 packed → int8 还原（含奇数补零截断）。"""
    high = (packed >> 4).to(torch.int16) - 8
    low = (packed & 0xF).to(torch.int16) - 8
    flat = torch.stack([high, low], dim=1).flatten()[:numel]
    return flat.to(torch.int8)


def build_qat(name):
    base = ConformerClassifier(2, 256, 4, 16, 512, 11, 1024)
    base.load_state_dict(torch.load("Checkpoint/amc/Our.pt", map_location="cpu"))
    base = merge_lora(base)
    if name == "qat_int4_prune40":
        prune_model(base, keep_ratio=0.60, mode="magnitude")
    for n, module in list(base.named_modules()):
        parent_name, _, attr = n.rpartition(".")
        parent = base if not parent_name else base.get_submodule(parent_name)
        if isinstance(module, nn.Linear):
            setattr(parent, attr, QuantAwareLinear(module, nbits=4))
        elif isinstance(module, DepthwiseConv):
            module.pointwise1 = QuantAwareConv1x1(module.pointwise1, nbits=4)
            module.pointwise2 = QuantAwareConv1x1(module.pointwise2, nbits=4)
    base.load_state_dict(torch.load(f"Checkpoint/{name}.pt", map_location="cpu"))
    return base


def main():
    for name in ("qat_int4", "qat_int4_prune40"):
        print(f"════ {name}.pt ════")
        model = build_qat(name)
        state = model.state_dict()

        # ① 原始体积
        buf = io.BytesIO()
        torch.save(state, buf)
        size_orig = buf.getbuffer().nbytes / 1024**2

        # ② 打包所有量化权重 + 验证
        qw_bits_orig = 0    # 量化权重的原始字节（int8 容器）
        qw_bits_packed = 0  # 打包后字节
        ok_all = True
        for n, module in model.named_modules():
            if isinstance(module, (QuantAwareLinear, QuantAwareConv1x1)):
                q = module._qweight
                packed = pack_int4(q)
                restored = unpack_int4(packed, q.numel())
                ok = torch.equal(restored.reshape(q.shape), q)
                ok_all &= ok
                qw_bits_orig += q.numel()
                qw_bits_packed += packed.numel()

        # ③ 打包后的 state_dict 体积（量化权重替换为 packed uint8）
        state_packed = dict(state)
        for n, module in model.named_modules():
            if isinstance(module, (QuantAwareLinear, QuantAwareConv1x1)):
                state_packed[n + "._qweight"] = pack_int4(module._qweight)
        buf2 = io.BytesIO()
        torch.save(state_packed, buf2)
        size_packed = buf2.getbuffer().nbytes / 1024**2

        print(f"  量化权重: {qw_bits_orig/1e6:.2f}M 参数")
        print(f"    原存储(int8 容器): {qw_bits_orig/1024**2:.1f}MB → 打包后: {qw_bits_packed/1024**2:.1f}MB")
        print(f"    解包验证数值一致: {'✅' if ok_all else '❌'}")
        print(f"  state_dict 体积: {size_orig:.1f}MB → 打包后 {size_packed:.1f}MB（{size_orig/size_packed:.2f}×）")
        print()


if __name__ == "__main__":
    main()
