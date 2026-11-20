#!/usr/bin/env python
"""Qwen3.5-4B 架构量化可行性探查。

只读分析，不加载到 GPU，输出：
1. 每个命名模块的类型/形状/参数量/dtype
2. 按类别聚合：Linear(可量化) vs Mamba/SSM/Conv(需特殊处理) vs Embedding/Norm(其他)
3. 每层 layer_types 分布（full_attention vs linear_attention）

用法:
  python probe_arch.py --qwen /data1/user_wzk/LMO/models/Qwen --out arch_map.json
"""
import argparse
import json
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoConfig


def classify(name, module):
    """返回该模块的分类标签。"""
    mn = type(module).__name__.lower()
    n = name.lower()
    if "embed" in n or isinstance(module, torch.nn.Embedding):
        return "embedding"
    if "norm" in mn or "layernorm" in mn or "rmsnorm" in mn:
        return "norm"
    if isinstance(module, torch.nn.Linear):
        # 排除 mamba/ssm 内部的 linear？先标记为 linear，后续看名字
        return "linear"
    if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Conv2d):
        return "conv"
    if "mamba" in n or "ssm" in n or "mixer" in n or "conv" in n:
        return "ssm_or_conv"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.qwen, local_files_only=True)
    tc = cfg.text_config
    print("=== text_config ===")
    print("model_type:", cfg.model_type)
    print("hidden:", tc.hidden_size, "layers:", tc.num_hidden_layers)
    print("heads:", tc.num_attention_heads, "kv_heads:", tc.num_key_value_heads,
          "head_dim:", tc.head_dim)
    print("intermediate:", tc.intermediate_size, "vocab:", tc.vocab_size)
    print("layer_types:", getattr(tc, "layer_types", None))

    model = AutoModelForCausalLM.from_pretrained(
        args.qwen, local_files_only=True,
        dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )

    agg = defaultdict(lambda: {"params": 0, "modules": 0, "bytes": 0})
    rows = []
    for name, module in model.named_modules():
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        if n_params == 0:
            continue
        cat = classify(name, module)
        n_bytes = n_params * 2  # bf16
        agg[cat]["params"] += n_params
        agg[cat]["modules"] += 1
        agg[cat]["bytes"] += n_bytes
        rows.append({
            "name": name,
            "type": type(module).__name__,
            "cat": cat,
            "params": n_params,
            "shape": str(tuple(module.weight.shape)) if hasattr(module, "weight") and module.weight is not None else None,
        })

    total_params = sum(a["params"] for a in agg.values())
    print("\n=== 按类别聚合 ===")
    for cat in sorted(agg, key=lambda c: -agg[c]["params"]):
        a = agg[cat]
        print(f"  {cat:14s} params={a['params']:>12,} ({a['params']/total_params*100:5.1f}%)  "
              f"modules={a['modules']:>4d}  bf16_bytes={a['bytes']/2**30:.2f}GB")

    print(f"\n  TOTAL params = {total_params:,}  (~{total_params*2/2**30:.2f}GB bf16)")

    # 每层 linear_attention / full_attention 分布
    print("\n=== 层类型分布 (per model.layers.N) ===")
    per_layer = {}
    for r in rows:
        parts = r["name"].split(".")
        # model.layers.N...
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
            try:
                idx = int(parts[2])
            except ValueError:
                continue
            per_layer.setdefault(idx, {"linear": 0, "ssm_or_conv": 0, "conv": 0, "norm": 0})
    for i in sorted(per_layer):
        # 判断该层是否含 mamba/ssm
        has_ssm = any(r["name"].startswith(f"model.layers.{i}.") and
                      r["cat"] in ("ssm_or_conv", "conv") for r in rows)
        kind = "linear_attention(含SSM/conv)" if has_ssm else "full_attention"
        print(f"  layer {i:2d}: {kind}")

    with open(args.out, "w") as f:
        json.dump({
            "text_config": {k: getattr(tc, k) for k in dir(tc) if not k.startswith("_") and isinstance(getattr(tc, k), (int, float, str, bool, list, tuple))},
            "aggregate": {k: v for k, v in agg.items()},
            "total_params": total_params,
            "modules": rows,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
