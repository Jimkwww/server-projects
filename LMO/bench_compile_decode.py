#!/usr/bin/env python
"""R3: torch.compile 只编译 decode（StaticCache + attention_mask=None 消除动态形状）。

之前 torch.compile 失败原因：
- reduce-overhead 模式 segfault（GatedDeltaNet cache 状态）
- default 模式 dtype 报错（autocast 隐式转换 fp32/bf16）
- recompile limit（4D mask 形状变化，DynamicCache cat）

现在用 StaticCache(is_compileable=True) + am=None 消除动态形状，只编译 decode 单步。

用法:
  python bench_compile_decode.py --config configs/LEMMO_Recognition.local.json --device cuda:2
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


def timeit_decode(step_fn, n_tok, iters, warmup):
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        step_fn()
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

    MAX_NEW = 32

    def do_prefill():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
            pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
            emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
            cache = StaticCache(config=qwen.config, max_cache_len=att.shape[1] + MAX_NEW)
            out = qwen(inputs_embeds=emb, attention_mask=att, past_key_values=cache, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), cache, att.shape[1]

    N_TOK = 20

    # ---- 基线：eager decode ----
    def eager_step(nt, past):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = qwen(input_ids=nt[:, None], attention_mask=None,
                        past_key_values=past, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), out.past_key_values

    # ---- torch.compile decode ----
    print("\n=== 尝试 torch.compile decode ===", flush=True)
    compile_modes = ["default", "reduce-overhead", "max-autotune-no-cudagraphs"]
    for mode in compile_modes:
        try:
            print(f"\n--- mode={mode} ---", flush=True)
            # 重新 prefill，用全新 cache
            nt, past, L = do_prefill()
            compiled = torch.compile(qwen, mode=mode, fullgraph=False)

            # warmup + 触发编译
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(5):
                    out = compiled(input_ids=nt[:, None], attention_mask=None,
                                   past_key_values=past, use_cache=True, return_dict=True)
                    past = out.past_key_values
                    nt = out.logits[:, -1].float().argmax(dim=-1)
            torch.cuda.synchronize()

            # 测速
            def compiled_step():
                nonlocal nt, past
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = compiled(input_ids=nt[:, None], attention_mask=None,
                                   past_key_values=past, use_cache=True, return_dict=True)
                    past = out.past_key_values
                    nt = out.logits[:, -1].float().argmax(dim=-1)

            r = timeit_decode(compiled_step, N_TOK, iters=20, warmup=3)
            print(f"compile[{mode}] decode: mean={r[0]:.2f}ms/{N_TOK}tok = {r[0]/N_TOK:.2f}ms/tok", flush=True)
        except Exception as e:
            print(f"compile[{mode}] 失败: {type(e).__name__}: {str(e)[:300]}", flush=True)
            # 重新加载 cache，避免 cache 被编译污染
            nt, past, L = do_prefill()

    # ---- 基线 eager 测速（最后测，避免被编译影响）----
    nt, past, L = do_prefill()
    def eager_loop():
        nonlocal nt, past
        nt, past = eager_step(nt, past)
    r = timeit_decode(eager_loop, N_TOK, iters=20, warmup=3)
    print(f"\n[eager baseline] decode: mean={r[0]:.2f}ms/{N_TOK}tok = {r[0]/N_TOK:.2f}ms/tok", flush=True)


if __name__ == "__main__":
    main()
