#!/usr/bin/env python
"""R3 诚实对比：真实贪心 decode（含 .item() + eos 判断）下 eager vs torch.compile。

关键：之前的 bench_compile_decode 循环无 .item() 同步，测的是 GPU 上限(0.71ms/tok)；
      bench_cudagraph 循环含 .item()，测的是真实行为(14.5ms/tok)。
      差 20 倍 = host↔GPU 逐 token 同步 latency。

这里在"真实贪心 decode"语义下（每 token .item() 判断 eos）对比：
  A) eager + .item()
  B) torch.compile(reduce-overhead) + .item()
  C) torch.compile(default) + .item()

每个都跑完整 20 token（或用 eos 提前终止），测 wall clock 和 GPU time。

用法:
  python bench_real_decode.py --config configs/LEMMO_Recognition.local.json --device cuda:2
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()

    from lemmo import LEMMORecognitionRuntime
    from lemmo.runtime import _prompt, _insert_signal, LEMMO_RECOGNITION_SYSTEM_PROMPT
    from transformers.cache_utils import StaticCache

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
    eos = int(tok.eos_token_id)

    MAX_NEW = 64

    def do_prefill():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
            pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
            emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
            cache = StaticCache(config=qwen.config, max_cache_len=att.shape[1] + MAX_NEW)
            out = qwen(inputs_embeds=emb, attention_mask=att, past_key_values=cache, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), cache

    # 真实贪心 decode：每 token .item() 判断 eos
    def real_decode(fwd, nt, past):
        gen = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(MAX_NEW):
                t = int(nt.item())  # host 同步拿 token
                if t == eos:
                    break
                gen.append(t)
                out = fwd(input_ids=nt[:, None], attention_mask=None,
                          past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
        return nt, past, gen

    def time_decode(fwd, iters=15, warmup=3):
        for _ in range(warmup):
            nt, past = do_prefill()
            real_decode(fwd, nt, past)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        lens = []
        for i in range(iters):
            nt, past = do_prefill()
            starts[i].record()
            nt, past, gen = real_decode(fwd, nt, past)
            ends[i].record()
            lens.append(len(gen))
        torch.cuda.synchronize()
        ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
        return ms, lens

    # A) eager
    nt, past = do_prefill()
    ms_a, lens_a = time_decode(qwen, iters=15)
    print(f"A) eager  真实decode: mean={ms_a.mean():.2f}ms p50={np.percentile(ms_a,50):.2f} | 平均生成 {np.mean(lens_a):.1f} token | {ms_a.mean()/np.mean(lens_a):.2f}ms/tok", flush=True)

    # B) torch.compile reduce-overhead
    try:
        nt, past = do_prefill()
        compiled_ro = torch.compile(qwen, mode="reduce-overhead", fullgraph=False)
        # warmup 触发编译 + graph 捕获
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(6):
                out = compiled_ro(input_ids=nt[:, None], attention_mask=None,
                                  past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
        torch.cuda.synchronize()
        ms_b, lens_b = time_decode(compiled_ro, iters=15)
        print(f"B) compile(reduce-overhead) 真实decode: mean={ms_b.mean():.2f}ms p50={np.percentile(ms_b,50):.2f} | 平均生成 {np.mean(lens_b):.1f} token | {ms_b.mean()/np.mean(lens_b):.2f}ms/tok", flush=True)
    except Exception as e:
        print(f"B) compile reduce-overhead 失败: {type(e).__name__}: {str(e)[:300]}", flush=True)

    # C) torch.compile default
    try:
        nt, past = do_prefill()
        compiled_d = torch.compile(qwen, mode="default", fullgraph=False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(6):
                out = compiled_d(input_ids=nt[:, None], attention_mask=None,
                                 past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
        torch.cuda.synchronize()
        ms_c, lens_c = time_decode(compiled_d, iters=15)
        print(f"C) compile(default) 真实decode: mean={ms_c.mean():.2f}ms p50={np.percentile(ms_c,50):.2f} | 平均生成 {np.mean(lens_c):.1f} token | {ms_c.mean()/np.mean(lens_c):.2f}ms/tok", flush=True)
    except Exception as e:
        print(f"C) compile default 失败: {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
