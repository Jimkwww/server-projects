#!/usr/bin/env python
"""LEMMO Q16 interference 量化精度评测（bf16 / int8 / int4 对照）。

在 evaluate_multi.py 基础上加 --quant 开关：加载 LEMMORecognitionRuntime 后对
rt.qwen 做 torchao 权重量化，再跑同一批采样样本，输出 accuracy + 每类 + 显存 + 磁盘大小。
"""
import argparse
import json
import os
import tempfile
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquets", required=True, help="comma-separated paths")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--question", default="这段信号中存在哪种干扰？")
    ap.add_argument("--sample-per-class", type=int, default=200, help="0=全部")
    ap.add_argument("--quant", default="bf16", choices=["bf16", "int8", "int4", "merge_bf16", "merge_int8", "merge_int4"])
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
    print(f"eval set: {len(df)} rows, quant={args.quant}", flush=True)

    runtime = LEMMORecognitionRuntime(args.config, args.device)

    # LoRA 合并
    from lemmo.lora import LoRALinear
    import torch.nn as nn
    def _merge_lora(model):
        def _replace(parent, child_name, lora_mod):
            base_dtype = lora_mod.base.weight.dtype
            base_dev = lora_mod.base.weight.device
            merged = lora_mod.base.weight.data + lora_mod.scaling * (
                lora_mod.lora_B.weight.data @ lora_mod.lora_A.weight.data
            )
            merged = merged.to(base_dtype)
            has_bias = lora_mod.base.bias is not None
            new_lin = nn.Linear(lora_mod.base.in_features, lora_mod.base.out_features, bias=has_bias)
            new_lin = new_lin.to(device=base_dev, dtype=base_dtype)
            new_lin.weight.data.copy_(merged)
            if has_bias:
                new_lin.bias.data.copy_(lora_mod.base.bias.data.to(base_dtype))
            setattr(parent, child_name, new_lin)
        c = 0
        for module_name, module in list(model.named_modules()):
            for child_name, child in list(module.named_children()):
                if isinstance(child, LoRALinear):
                    _replace(module, child_name, child)
                    c += 1
        return c

    do_merge = args.quant.startswith("merge_")
    if do_merge:
        n = _merge_lora(runtime.qwen)
        print(f"[merge] 合并 {n} 层 LoRA", flush=True)
        quant = args.quant.replace("merge_", "")

    # 量化
    if args.quant == "int8" or (do_merge and quant == "int8"):
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        quantize_(runtime.qwen, Int8WeightOnlyConfig())
    elif args.quant == "int4" or (do_merge and quant == "int4"):
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
        from torchao.quantization.quant_primitives import ZeroPointDomain
        # torchao 0.11.0 的 int4 默认布局在 bf16 权重上有 zero_point dtype bug
        # （scale 是 fp32 但 zero_point 硬编码 bf16），用 HQQ 路径绕过。
        quantize_(runtime.qwen, Int4WeightOnlyConfig(use_hqq=True, zero_point_domain=ZeroPointDomain.FLOAT))

    # 磁盘大小（序列化 qwen state_dict）
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "qwen.pt")
        torch.save(runtime.qwen.state_dict(), p)
        disk_gb = os.path.getsize(p) / 1e9

    torch.cuda.reset_peak_memory_stats()
    results = []
    t0 = time.time()
    tmpdir = tempfile.mkdtemp(prefix="lmo_quant_")
    for i, row in df.iterrows():
        iq = np.stack(row["iq"]).astype(np.float32)
        label = row["label_value"]
        sample_id = row["sample_id"]
        sr = float(row["sample_rate"])
        tmp_path = os.path.join(tmpdir, f"{i}.npy")
        np.save(tmp_path, iq)
        ans = runtime.answer(tmp_path, sr, "interference", args.question, args.max_new_tokens)
        results.append({
            "sample_id": sample_id, "label": label,
            "prediction": ans, "correct": label_in_answer(ans, label),
        })
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
        "quant": args.quant,
        "total": len(results),
        "correct": sum(r["correct"] for r in results),
        "accuracy": acc,
        "elapsed_sec": elapsed,
        "avg_sec_per_sample": elapsed / len(results) if results else 0,
        "peak_mem_GB": mem_gb,
        "qwen_disk_GB": disk_gb,
        "per_class": per_acc,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== [{args.quant}] accuracy: {acc:.4f} ({summary['correct']}/{summary['total']}) ===")
    print(f"=== elapsed {elapsed:.1f}s, avg {elapsed/len(results) if results else 0:.3f}s/sample, mem {mem_gb:.2f}GB, disk {disk_gb:.2f}GB ===")
    for l in OFFICIAL_12:
        if l in per_acc:
            p = per_acc[l]
            print(f"  {l:22s} {p['correct']:4d}/{p['total']:4d} = {p['acc']:.4f}")
    print(f"=== saved to {args.output} ===")


if __name__ == "__main__":
    main()
