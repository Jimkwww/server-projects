#!/usr/bin/env python
"""R3 解码单步变量消融：定位 decode 单步时间到底花在哪。

分别测：
  A) attention_mask 增长 cat vs None
  B) DynamicCache vs StaticCache
  C) .item() 同步 vs 不同步
  D) 纯 CUDA 时间（不含任何 host 同步）

用法:
  python bench_decode_step.py --config configs/LEMMO_Recognition.local.json --device cuda:2
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lemmo.lora import LoRALinear

PARSQUETS = {
    16368: "/data1/user_wzk/LMO/data/interference/gnss_jamming_test.parquet",
}


def merge_lora(model: nn.Module) -> int:
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

    count = 0
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                _replace(module, child_name, child)
                count += 1
    return count


def load_iq_mem(array):
    value = np.asarray(array, dtype=np.float32, order="C")
    scale = float(np.max(np.abs(value)))
    return np.ascontiguousarray(value / scale)


def timeit(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return float(ms.mean()), float(np.percentile(ms, 50)), float(np.percentile(ms, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()

    from lemmo import LEMMORecognitionRuntime
    from lemmo.runtime import _prompt, _insert_signal, LEMMO_RECOGNITION_SYSTEM_PROMPT

    runtime = LEMMORecognitionRuntime(args.config, args.device)
    dev = runtime.device
    tok = runtime.tokenizer
    qwen = runtime.qwen
    model = runtime.model

    n = merge_lora(qwen)
    print(f"LoRA 合并 {n} 层", flush=True)

    df = pd.read_parquet(PARSQUETS[16368])
    row = df.iloc[0]
    arr = load_iq_mem(np.stack(row["iq"]).astype(np.float32))
    sr = float(row["sample_rate"])
    signal = torch.from_numpy(arr)[None].to(dev, dtype=torch.bfloat16)
    rates = torch.tensor([sr], dtype=torch.float32, device=dev)
    prompt = _prompt(tok, LEMMO_RECOGNITION_SYSTEM_PROMPT, "这段信号中存在哪种干扰？", dev)

    def do_prefill(static=False):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
            pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
            emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
            if static:
                from transformers.cache_utils import StaticCache
                cache = StaticCache(config=qwen.config, max_cache_len=att.shape[1] + 128)
                out = qwen(inputs_embeds=emb, attention_mask=att, past_key_values=cache, use_cache=True, return_dict=True)
            else:
                out = qwen(inputs_embeds=emb, attention_mask=att, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), out.past_key_values, att

    results = {}

    # ---- A) attention_mask 增长 cat vs None（DynamicCache）----
    def decode_am_cat():
        nt, past, a = do_prefill(static=False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(20):
                out = qwen(input_ids=nt[:, None], attention_mask=a, past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
                a = torch.cat((a, torch.ones((1, 1), dtype=a.dtype, device=a.device)), dim=1)

    def decode_am_none():
        nt, past, a = do_prefill(static=False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(20):
                out = qwen(input_ids=nt[:, None], attention_mask=None, past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)

    r = timeit(decode_am_cat, iters=20, warmup=3)
    results["A_dynamic_am_cat_20tok"] = r
    r = timeit(decode_am_none, iters=20, warmup=3)
    results["A_dynamic_am_none_20tok"] = r
    print(f"A) DynamicCache am_cat 20tok: {r[0]:.3f}ms | am_none 20tok: {results['A_dynamic_am_cat_20tok'][0]:.3f}ms", flush=True)

    # ---- B) StaticCache ----
    def decode_static():
        nt, past, a = do_prefill(static=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(20):
                out = qwen(input_ids=nt[:, None], attention_mask=None, past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)

    r = timeit(decode_static, iters=20, warmup=3)
    results["B_static_am_none_20tok"] = r
    print(f"B) StaticCache am_none 20tok: {r[0]:.3f}ms", flush=True)

    # ---- C) 纯 CUDA 时间（无 host 同步，用 CUDA event 包住 kernel）----
    # 单步 decode 的纯 GPU 时间：连续 launch 10 步，只记录首尾 event
    def decode_gpu_only():
        nt, past, a = do_prefill(static=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(10):
                out = qwen(input_ids=nt[:, None], attention_mask=None, past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)

    # 用 event 测纯 GPU 吞吐（10 步连续，无 per-step 同步）
    for _ in range(3):
        decode_gpu_only()
    torch.cuda.synchronize()
    gpu_times = []
    for _ in range(10):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        nt, past, a = do_prefill(static=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(10):
                out = qwen(input_ids=nt[:, None], attention_mask=None, past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
        e.record()
        torch.cuda.synchronize()
        gpu_times.append(s.elapsed_time(e) / 10.0)
    gpu_mean = float(np.mean(gpu_times))
    results["C_gpu_only_per_step"] = gpu_mean
    print(f"C) 纯 GPU 单步（10步连续无同步）: {gpu_mean:.3f}ms", flush=True)

    # ---- D) 逐 token .item() 同步的额外开销 ----
    def decode_with_item():
        nt, past, a = do_prefill(static=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(20):
                out = qwen(input_ids=nt[:, None], attention_mask=None, past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                t = int(out.logits[:, -1].float().argmax(dim=-1).item())
                nt = torch.tensor([[t]], device=dev)

    r = timeit(decode_with_item, iters=20, warmup=3)
    results["D_static_am_none_with_item_20tok"] = r
    print(f"D) StaticCache am_none + item() 20tok: {r[0]:.3f}ms", flush=True)

    print("\n=== 汇总 ===", flush=True)
    for k, v in results.items():
        if isinstance(v, tuple):
            print(f"{k}: mean={v[0]:.3f} p50={v[1]:.3f} p95={v[2]:.3f}", flush=True)
        else:
            print(f"{k}: {v:.3f}", flush=True)


if __name__ == "__main__":
    main()
