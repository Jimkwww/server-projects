#!/usr/bin/env python
"""R3 端到端：merge bf16 的 eager vs torch.compile 完整 answer 一致性 + e2e 计时。

集成：
- signal path（EM 编码器 + bridge）
- prefill（StaticCache）
- 真实贪心 decode（含 .item() + eos 判断）
- 对比 eager vs torch.compile(default)

先跑单样本验证 answer 一致，再测 20 次 e2e 计时。

用法:
  python bench_e2e_compile.py --config configs/LEMMO_Recognition.local.json --device cuda:2
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
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--question", default="这段信号中存在哪种干扰？")
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
    prompt = _prompt(tok, LEMMO_RECOGNITION_SYSTEM_PROMPT, args.question, dev)
    eos = int(tok.eos_token_id)

    # 预计算 signal path + prefill（一次，复用）
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
        pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
        emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
        max_cache_len = att.shape[1] + args.max_new_tokens

    def prefill():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            cache = StaticCache(config=qwen.config, max_cache_len=max_cache_len)
            out = qwen(inputs_embeds=emb, attention_mask=att, past_key_values=cache, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), cache

    def greedy(fwd, nt, past):
        gen = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(args.max_new_tokens):
                t = int(nt.item())
                if t == eos:
                    break
                gen.append(t)
                out = fwd(input_ids=nt[:, None], attention_mask=None,
                          past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
        return tok.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip(), len(gen)

    # --- 答案一致性 ---
    nt, past = prefill()
    ans_eager, n_eager = greedy(qwen, nt, past)
    print(f"[eager] answer={ans_eager!r} ({n_eager} tok)", flush=True)

    compiled = torch.compile(qwen, mode="default", fullgraph=False)
    nt, past = prefill()
    # warmup 触发编译
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(6):
            out = compiled(input_ids=nt[:, None], attention_mask=None,
                           past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            nt = out.logits[:, -1].float().argmax(dim=-1)
    torch.cuda.synchronize()
    nt, past = prefill()
    ans_comp, n_comp = greedy(compiled, nt, past)
    print(f"[compile] answer={ans_comp!r} ({n_comp} tok)", flush=True)
    print(f"答案一致: {ans_eager == ans_comp}", flush=True)

    # --- e2e 计时（signal path + prefill + decode 全包）---
    def e2e_eager():
        nt, past = prefill()
        return greedy(qwen, nt, past)

    def e2e_comp():
        nt, past = prefill()
        return greedy(compiled, nt, past)

    for name, fn in [("eager", e2e_eager), ("compile", e2e_comp)]:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        iters = 20
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
        print(f"[e2e {name}] mean={ms.mean():.2f}ms p50={np.percentile(ms,50):.2f} p95={np.percentile(ms,95):.2f}", flush=True)


if __name__ == "__main__":
    main()
