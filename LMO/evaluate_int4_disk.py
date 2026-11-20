#!/usr/bin/env python
"""方案 A 完整精度评测：int4 磁盘存储 + 加载 unpack 回 bf16 推理。

流程：LEMMORecognitionRuntime → merge LoRA → 用 HQQ raw_output 对 249 个 Linear 量化
（与 torchao Int4WeightOnlyConfig(use_hqq=True) 同一套 scale/zero，保证量化精度一致）
→ pack 2×int4→uint8 存磁盘 → unpack+dequant 回 bf16 → 跑 2212 条官方12类评测。

结果 JSON 含：accuracy、per_class、elapsed、peak_mem、磁盘大小（int4 packed vs bf16）、
量化+unpack 计时。用于坐实"精度无损 + 存储压缩>2× + 加速>1.5×（推理走 merge bf16）"。

用法:
  python evaluate_int4_disk.py --parquets <3个parquet逗号分隔> \
      --config configs/LEMMO_Recognition.local.json --device cuda:2 \
      --sample-per-class 200 --output results/interference_full12_int4_disk.json
"""
import argparse
import json
import os
import tempfile
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OFFICIAL_12 = [
    "AM_Jamming", "Comb_Jamming", "FM_Jamming", "Multi_Tone",
    "Narrowband", "No_Interference", "Protocol_Bluetooth", "Protocol_WiFi",
    "Pulse_DME", "Single_Tone", "Sweep_Chirp", "Wideband",
]

ALIASES = {
    "AM_Jamming": ["调幅", "am"],
    "Comb_Jamming": ["梳状", "comb"],
    "FM_Jamming": ["调频", "fm"],
    "Multi_Tone": ["多音", "multi tone", "multi-tone", "多频"],
    "Narrowband": ["窄带", "narrowband"],
    "No_Interference": ["无干扰", "没有干扰", "不存在干扰", "no interference"],
    "Protocol_Bluetooth": ["蓝牙", "bluetooth"],
    "Protocol_WiFi": ["wifi", "wi-fi", "wi fi"],
    "Pulse_DME": ["dme", "脉冲"],
    "Single_Tone": ["单音", "single tone", "single-tone", "单频"],
    "Sweep_Chirp": ["扫频", "chirp", "线性调频"],
    "Wideband": ["宽带", "wideband"],
}


def label_in_answer(answer, label):
    a = answer.lower()
    for kw in ALIASES.get(label, [label.lower()]):
        if kw.lower() in a:
            return True
    return False


def merge_lora(model: nn.Module) -> int:
    from lemmo.lora import LoRALinear

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


def quantize_linear_hqq(lin: nn.Linear, gs: int = 128):
    """用 HQQ raw_output 量化一个 Linear，返回 (packed uint8, scale fp16, zero fp16, N, K)。"""
    from torchao.quantization.quant_primitives import choose_qparams_and_quantize_affine_hqq
    w = lin.weight.data
    N, K = w.shape
    W_q, scale, zero, _ = choose_qparams_and_quantize_affine_hqq(
        w, nbits=4, group_size=gs, axis=1, compute_dtype=torch.float16,
        device=w.device, raw_output=True,
    )
    # W_q: uint8 [N, K]; scale/zero: fp16 [N, K//gs]
    ngrp = K // gs
    q2 = W_q.reshape(N, ngrp, gs // 2, 2)
    packed = (q2[..., 0] | (q2[..., 1] << 4)).cpu()  # uint8 [N, ngrp, gs//2]
    return packed, scale.cpu(), zero.cpu()


def dequant_linear_hqq(packed, scale, zero, N, K, gs=128, device="cuda", dtype=torch.bfloat16):
    """(packed, scale, zero) -> 恢复 bf16 权重。"""
    ngrp = K // gs
    p = packed.to(device)
    lo = (p & 0x0F).to(torch.float32)
    hi = ((p >> 4) & 0x0F).to(torch.float32)
    W_q = torch.stack([lo, hi], dim=-1).reshape(N, ngrp, gs)  # uint8->float32
    w = (W_q - zero.to(device).float()[:, :, None]) * scale.to(device).float()[:, :, None]
    return w.reshape(N, K).to(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquets", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--question", default="这段信号中存在哪种干扰？")
    ap.add_argument("--sample-per-class", type=int, default=200)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from lemmo import LEMMORecognitionRuntime

    paths = [p.strip() for p in args.parquets.split(",") if p.strip()]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df = df[df["label_value"].isin(OFFICIAL_12)]
    if args.sample_per_class:
        sampled = []
        for lab in OFFICIAL_12:
            sub = df[df["label_value"] == lab]
            sampled.append(sub.sample(n=min(len(sub), args.sample_per_class), random_state=args.seed))
        df = pd.concat(sampled, ignore_index=True)
    print(f"eval set: {len(df)} rows", flush=True)

    runtime = LEMMORecognitionRuntime(args.config, args.device)
    n_merged = merge_lora(runtime.qwen)
    print(f"[merge] 合并 {n_merged} 层 LoRA", flush=True)

    # ---------- 量化所有 Linear 到磁盘格式 ----------
    t0 = time.time()
    layers = {}
    for mn, mod in list(runtime.qwen.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear):
                packed, scale, zero = quantize_linear_hqq(ch, args.group_size)
                layers[f"{mn}.{cn}".strip(".")] = (packed, scale, zero, ch.out_features, ch.in_features)
    torch.cuda.synchronize()
    quant_time = time.time() - t0
    n_linear = len(layers)

    # 磁盘大小
    with tempfile.TemporaryDirectory() as d:
        p_int4 = os.path.join(d, "int4.pt")
        p_bf16 = os.path.join(d, "bf16.pt")
        torch.save(layers, p_int4)
        torch.save(runtime.qwen.state_dict(), p_bf16)
        disk_int4 = os.path.getsize(p_int4)
        disk_bf16 = os.path.getsize(p_bf16)

    # ---------- unpack 回 bf16（计时） ----------
    t0 = time.time()
    for mn, mod in list(runtime.qwen.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear):
                key = f"{mn}.{cn}".strip(".")
                packed, scale, zero, N, K = layers[key]
                wd = dequant_linear_hqq(packed, scale, zero, N, K, args.group_size,
                                        device=args.device, dtype=ch.weight.dtype)
                ch.weight.data.copy_(wd)
    torch.cuda.synchronize()
    unpack_time = time.time() - t0

    print(json.dumps({
        "n_merged": n_merged, "n_linear": n_linear,
        "quant_time_s": round(quant_time, 2), "unpack_time_s": round(unpack_time, 2),
        "disk_int4_MB": round(disk_int4 / 1e6, 1), "disk_bf16_MB": round(disk_bf16 / 1e6, 1),
        "compression_ratio": round(disk_bf16 / disk_int4, 2),
    }, ensure_ascii=False), flush=True)

    # ---------- 完整评测（推理走 merge bf16） ----------
    torch.cuda.reset_peak_memory_stats()
    results = []
    t0 = time.time()
    tmpdir = tempfile.mkdtemp(prefix="lmo_int4disk_")
    for i, row in df.iterrows():
        iq = np.stack(row["iq"]).astype(np.float32)
        label = row["label_value"]
        sample_id = row["sample_id"]
        sr = float(row["sample_rate"])
        tmp_path = os.path.join(tmpdir, f"{i}.npy")
        np.save(tmp_path, iq)
        ans = runtime.answer(tmp_path, sr, "interference", args.question, args.max_new_tokens)
        results.append({"sample_id": sample_id, "label": label,
                        "prediction": ans, "correct": label_in_answer(ans, label)})
        if (i + 1) % 200 == 0 or (i + 1) == len(df):
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"[{i+1}/{len(df)}] acc={acc:.4f}", flush=True)

    acc = sum(r["correct"] for r in results) / len(results)
    elapsed = time.time() - t0
    mem_gb = torch.cuda.max_memory_allocated() / 1e9

    per = defaultdict(lambda: [0, 0])
    for r in results:
        per[r["label"]][1] += 1
        if r["correct"]:
            per[r["label"]][0] += 1
    per_acc = {l: {"correct": c, "total": t, "acc": c / t if t else 0}
               for l, (c, t) in sorted(per.items())}

    summary = {
        "scheme": "int4_disk (merge + int4存储 + unpack回bf16推理)",
        "total": len(results), "correct": sum(r["correct"] for r in results),
        "accuracy": acc, "elapsed_sec": elapsed,
        "avg_sec_per_sample": elapsed / len(results) if results else 0,
        "peak_mem_GB": mem_gb,
        "disk_int4_linear_MB": disk_int4 / 1e6,
        "disk_bf16_MB": disk_bf16 / 1e6,
        "compression_ratio": disk_bf16 / disk_int4,
        "quant_time_s": quant_time, "unpack_time_s": unpack_time,
        "per_class": per_acc,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== [int4_disk] accuracy: {acc:.4f} ({summary['correct']}/{summary['total']}) ===")
    print(f"=== elapsed {elapsed:.1f}s, avg {elapsed/len(results) if results else 0:.3f}s/sample, "
          f"mem {mem_gb:.2f}GB ===")
    print(f"=== disk int4 {disk_int4/1e9:.2f}GB vs bf16 {disk_bf16/1e9:.2f}GB = "
          f"{disk_bf16/disk_int4:.2f}x, quant {quant_time:.1f}s, unpack {unpack_time:.1f}s ===")
    for l in OFFICIAL_12:
        if l in per_acc:
            p = per_acc[l]
            print(f"  {l:22s} {p['correct']:4d}/{p['total']:4d} = {p['acc']:.4f}")
    print(f"=== saved to {args.output} ===")


if __name__ == "__main__":
    main()
