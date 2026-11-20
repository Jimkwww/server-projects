#!/usr/bin/env python
"""R2: LoRA 离线合并 + 耗时对比。

把 Qwen 里所有 LoRALinear 合并回 nn.Linear（W = base + scaling * lora_B @ lora_A），
消除 decode 每 token 的 496 次额外小 matmul。不保存权重文件，纯 in-place。

先做精度 sanity（合并前后 answer 一致性），再测 prefill/decode/端到端。

用法:
  python bench_lora_merge.py --config configs/LEMMO_Recognition.local.json \
      --iq-length 16368 --device cuda:2 --iters 30
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lemmo.lora import LoRALinear

PARSQUETS = {
    128: "/data1/user_wzk/LMO/data/interference/owl_init_test.parquet",
    1024: "/data1/user_wzk/LMO/data/interference/em_infer_comm_test.parquet",
    16368: "/data1/user_wzk/LMO/data/interference/gnss_jamming_test.parquet",
}


def merge_lora(model: nn.Module) -> int:
    """把 model 里所有 LoRALinear 合并回 nn.Linear，返回合并数量。"""
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


def timeit(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return float(ms.mean()), float(np.percentile(ms, 50)), float(np.percentile(ms, 95))


def load_iq_mem(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32, order="C")
    scale = float(np.max(np.abs(value)))
    return np.ascontiguousarray(value / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--iq-length", type=int, required=True, choices=[128, 1024, 16368])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--question", default="这段信号中存在哪种干扰？")
    args = ap.parse_args()

    from lemmo import LEMMORecognitionRuntime
    from lemmo.runtime import _prompt, _insert_signal, _greedy_generate, LEMMO_RECOGNITION_SYSTEM_PROMPT

    runtime = LEMMORecognitionRuntime(args.config, args.device)
    dev = runtime.device
    tok = runtime.tokenizer
    qwen = runtime.qwen
    model = runtime.model

    # 合并前 sanity answer
    df = pd.read_parquet(PARSQUETS[args.iq_length])
    row = df.iloc[0]
    arr = load_iq_mem(np.stack(row["iq"]).astype(np.float32))
    label = row["label_value"]
    sr = float(row["sample_rate"])
    signal = torch.from_numpy(arr)[None].to(dev, dtype=torch.bfloat16)
    rates = torch.tensor([sr], dtype=torch.float32, device=dev)
    prompt = _prompt(tok, LEMMO_RECOGNITION_SYSTEM_PROMPT, args.question, dev)
    eos = int(tok.eos_token_id)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
        pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
        emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
    ans_before = _greedy_generate(qwen, tok, emb, att, args.max_new_tokens)
    print(f"[sanity] 合并前 answer = {ans_before!r} (label={label})", flush=True)

    # 合并前计时
    def prefill_fn():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = qwen(inputs_embeds=emb, attention_mask=att, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), out.past_key_values

    prefill_next, prefill_past = prefill_fn()
    def decode_one():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = qwen(input_ids=prefill_next[:, None], attention_mask=att,
                        past_key_values=prefill_past, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), out.past_key_values

    p_before = timeit(prefill_fn, args.iters)
    d_before = timeit(decode_one, args.iters)
    print(f"[R0] prefill mean={p_before[0]:.3f}ms  decode_one mean={d_before[0]:.3f}ms", flush=True)

    # 合并
    n_merged = merge_lora(qwen)
    print(f"[R2] 合并 LoRA 层数: {n_merged}", flush=True)

    # 合并后 sanity（需要重新跑一遍 answer，因为 cache 已失效）
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
        pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
        emb2, att2 = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
    ans_after = _greedy_generate(qwen, tok, emb2, att2, args.max_new_tokens)
    print(f"[sanity] 合并后 answer = {ans_after!r}", flush=True)
    match = (ans_before == ans_after)
    print(f"[sanity] answer 一致性: {'一致 ✓' if match else '不一致 ✗'}", flush=True)

    # 合并后计时
    prefill_next2, prefill_past2 = prefill_fn()
    def decode_one2():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = qwen(input_ids=prefill_next2[:, None], attention_mask=att2,
                        past_key_values=prefill_past2, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), out.past_key_values

    p_after = timeit(prefill_fn, args.iters)
    d_after = timeit(decode_one2, args.iters)
    print(f"[R2] prefill mean={p_after[0]:.3f}ms  decode_one mean={d_after[0]:.3f}ms", flush=True)

    summary = {
        "n_merged": n_merged,
        "answer_match": bool(match),
        "ans_before": ans_before,
        "ans_after": ans_after,
        "prefill_ms": {"before": round(p_before[0], 3), "after": round(p_after[0], 3)},
        "decode_one_ms": {"before": round(d_before[0], 3), "after": round(d_after[0], 3)},
        "prefill_speedup": round(p_before[0] / p_after[0], 3),
        "decode_speedup": round(d_before[0] / d_after[0], 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
