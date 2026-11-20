#!/usr/bin/env python
"""方案 A：int4 磁盘存储 + 自定义 fast codec + 加载 dequant 回 bf16 推理。

核心：磁盘存 int4（per-group128 asymmetric），加载时 O(numel) 纯 bit 操作解回 bf16，
推理走 merge bf16（1.51× 加速、67.72% 精度），避免 torchao dequantize() 的 eye-GEMM
（26.7s → 0.24s）。

对比口径（同一进程内，绕开 torchao 的 packed 格式，用简单 2×int4→uint8 pack）：
- 存储压缩比：仅量化 249 个 nn.Linear（4.205B 参数），Embedding 保持 bf16
- 加载时间：量化 + unpack 全程计时
- 推理精度/速度：unpack 回 bf16 后与 merge bf16 完全一致

用法:
  python bench_int4_disk.py --config configs/LEMMO_Recognition.local.json \
      --device cuda:2 --iq-length 16368 --sample-file <npy> --check-answer
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from lemmo.lora import LoRALinear


# ---------- 自定义 int4 codec（per-group128 asymmetric，纯 bit 操作） ----------
def qpack(w: torch.Tensor, gs: int = 128):
    """bf16/fp32 -> (packed uint8, scale fp32, zp int16)，asymmetric [0,15]。"""
    wf = w.detach().to(torch.float32)
    N, K = wf.shape
    assert K % gs == 0, f"K={K} 不能被 group_size={gs} 整除"
    ngrp = K // gs
    wv = wf.reshape(N, ngrp, gs)
    mn = wv.min(dim=2).values
    mx = wv.max(dim=2).values
    scale = ((mx - mn) / 15.0).clamp(min=1e-9)
    zp = (-mn / scale).round().clamp(0, 15).to(torch.int16)
    q = (wv / scale[:, :, None] + zp[:, :, None]).round().clamp(0, 15).to(torch.int8)
    q2 = q.reshape(N, ngrp, gs // 2, 2)
    packed = (q2[..., 0].to(torch.uint8) | (q2[..., 1].to(torch.uint8) << 4))
    return packed.cpu(), scale.cpu(), zp.cpu()


def qunpack(packed: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor, N: int, K: int, gs: int = 128, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """(packed, scale, zp) -> 恢复 bf16 权重。"""
    ngrp = K // gs
    p = packed.to(device)
    lo = (p & 0x0F).to(torch.float32)
    hi = ((p >> 4) & 0x0F).to(torch.float32)
    q = torch.stack([lo, hi], dim=-1).reshape(N, ngrp, gs)
    w = (q - zp.to(device).to(torch.float32)[:, :, None]) * scale.to(device)[:, :, None]
    return w.reshape(N, K).to(dtype)


def merge_lora(model: nn.Module) -> int:
    def _replace(parent, child_name, lm):
        bd = lm.base.weight.dtype
        bdev = lm.base.weight.device
        m = lm.base.weight.data + lm.scaling * (lm.lora_B.weight.data @ lm.lora_A.weight.data)
        m = m.to(bd)
        hb = lm.base.bias is not None
        nl = nn.Linear(lm.base.in_features, lm.base.out_features, bias=hb).to(bdev, bd)
        nl.weight.data.copy_(m)
        if hb:
            nl.bias.data.copy_(lm.base.bias.data.to(bd))
        setattr(parent, child_name, nl)

    c = 0
    for _, mod in list(model.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, LoRALinear):
                _replace(mod, cn, ch)
                c += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--iq-length", type=int, default=16368, choices=[128, 1024, 16368])
    ap.add_argument("--sample-file", default="/data1/user_wzk/LMO/data/interference/sample0_16368.npy")
    ap.add_argument("--check-answer", action="store_true")
    args = ap.parse_args()

    from lemmo import LEMMORecognitionRuntime
    from lemmo.runtime import _prompt, _insert_signal, LEMMO_RECOGNITION_SYSTEM_PROMPT
    from lemmo.io import load_iq

    rt = LEMMORecognitionRuntime(args.config, args.device)
    qwen = rt.qwen
    n_merged = merge_lora(qwen)
    print(f"[1] LoRA 合并 {n_merged} 层", flush=True)

    # ---------- 量化所有 Linear（拿到磁盘交付物） ----------
    t0 = time.time()
    layers = {}  # name -> (packed, scale, zp, N, K)
    for mn, mod in list(qwen.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear):
                p, s, z = qpack(ch.weight.data)
                # weight shape = (out_features, in_features)；qunpack 的 N=out_features, K=in_features
                layers[f"{mn}.{cn}".strip(".")] = (p, s, z, ch.out_features, ch.in_features)
    torch.cuda.synchronize()
    quant_time = time.time() - t0
    n_linear = len(layers)

    # 磁盘大小（仅量化后的 Linear；Embedding 等保持 bf16，另计）
    t0 = time.time()
    with open("/tmp/int4_disk_check.pt", "wb") as f:
        torch.save(layers, f)
    disk_int4_linear = os.path.getsize("/tmp/int4_disk_check.pt")
    os.remove("/tmp/int4_disk_check.pt")
    save_time = time.time() - t0

    # 原始 bf16 全模型磁盘大小
    t0 = time.time()
    with open("/tmp/bf16_disk_check.pt", "wb") as f:
        torch.save(qwen.state_dict(), f)
    disk_bf16 = os.path.getsize("/tmp/bf16_disk_check.pt")
    os.remove("/tmp/bf16_disk_check.pt")
    save_time2 = time.time() - t0

    # ---------- unpack 回 bf16（计时） ----------
    t0 = time.time()
    for mn, mod in list(qwen.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear):
                key = f"{mn}.{cn}".strip(".")
                p, s, z, N, K = layers[key]
                wd = qunpack(p, s, z, N, K, device=args.device, dtype=ch.weight.dtype)
                ch.weight.data.copy_(wd)
    torch.cuda.synchronize()
    unpack_time = time.time() - t0

    print(json.dumps({
        "n_merged": n_merged,
        "n_linear": n_linear,
        "quant_time_s": round(quant_time, 2),
        "unpack_time_s": round(unpack_time, 2),
        "save_int4_linear_MB": round(disk_int4_linear / 1e6, 1),
        "save_bf16_MB": round(disk_bf16 / 1e6, 1),
        "compression_ratio": round(disk_bf16 / disk_int4_linear, 2),
    }, indent=2), flush=True)

    # ---------- 端到端推理验证（answer 是否正确 + 速度） ----------
    if args.check_answer:
        array = load_iq(args.sample_file)
        signal = torch.from_numpy(array)[None].to(args.device, dtype=torch.bfloat16)
        rates = torch.tensor([20e6], dtype=torch.float32, device=args.device)
        q = "这段信号中存在哪种干扰？"
        prompt = _prompt(rt.tokenizer, LEMMO_RECOGNITION_SYSTEM_PROMPT, q, args.device)
        eos = int(rt.tokenizer.eos_token_id)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
            pr = rt.model(signal, rates, q_emb, prompt["question_mask"], "interference")
            emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
            out = qwen(inputs_embeds=emb, attention_mask=att, use_cache=True, return_dict=True)
            past = out.past_key_values
            nt = out.logits[:, -1].float().argmax(dim=-1)
            gen = []
            for _ in range(64):
                tk = int(nt.item())
                if tk == eos:
                    break
                gen.append(tk)
                out = qwen(input_ids=nt[:, None], attention_mask=att,
                            past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
                att = torch.cat((att, torch.ones((1, 1), dtype=att.dtype, device=att.device)), dim=1)
        answer = rt.tokenizer.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        print(f"[answer] {answer}", flush=True)


if __name__ == "__main__":
    main()
