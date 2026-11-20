#!/usr/bin/env python
"""R3 探测：merge 后 decode 单步 CPU/CUDA 占比 + StaticCache 可行性。

1) 用 torch.autograd.profiler 精确测 merge 后 decode 单步的 Self CPU vs Self CUDA
2) 测试 StaticCache（预分配，混合 full_attention + linear_attention）能否正常 decode
3) 测每步各算子 CPU 时间分布，确认 Python 调度开销占比

用法:
  python probe_r3.py --config configs/LEMMO_Recognition.local.json --device cuda:2
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

    # 预计算 prefill，拿到 fresh cache
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
        pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
        emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
        out = qwen(inputs_embeds=emb, attention_mask=att, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_token = out.logits[:, -1].float().argmax(dim=-1)
        att_len = att.shape[1]

    print(f"\n=== prefill 后 cache 类型: {type(past).__name__} ===", flush=True)
    print(f"seq_len={att_len}, 已生成 token = 第一个 {int(next_token.item())}", flush=True)

    # --- 实验1: profiler 测 decode 单步 ---
    print("\n=== 实验1: decode 单步 profiler (merge bf16) ===", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # warmup
        for _ in range(3):
            out = qwen(input_ids=next_token[:, None], attention_mask=att,
                        past_key_values=past, use_cache=True, return_dict=True)
            next_token2 = out.logits[:, -1].float().argmax(dim=-1)
            att = torch.cat((att, torch.ones((1, 1), dtype=att.dtype, device=att.device)), dim=1)
            past = out.past_key_values
            next_token = next_token2
        torch.cuda.synchronize()

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                out = qwen(input_ids=next_token[:, None], attention_mask=att,
                            past_key_values=past, use_cache=True, return_dict=True)
                next_token2 = out.logits[:, -1].float().argmax(dim=-1)
                att = torch.cat((att, torch.ones((1, 1), dtype=att.dtype, device=att.device)), dim=1)
                past = out.past_key_values
                next_token = next_token2
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25), flush=True)

    # --- 实验2: StaticCache 可行性 ---
    print("\n=== 实验2: StaticCache 可行性 ===", flush=True)
    try:
        from transformers.cache_utils import StaticCache
        # 重新 prefill 到 StaticCache
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb2 = qwen.get_input_embeddings()(prompt["question_ids"])
            pr2 = model(signal, rates, q_emb2, prompt["question_mask"], "interference")
            emb2, att2 = _insert_signal(qwen.get_input_embeddings(), prompt, pr2)
            max_cache_len = att2.shape[1] + 64
            static_cache = StaticCache(config=qwen.config, max_cache_len=max_cache_len)
            out2 = qwen(inputs_embeds=emb2, attention_mask=att2, past_key_values=static_cache,
                        use_cache=True, return_dict=True)
            static_past = out2.past_key_values
            nt2 = out2.logits[:, -1].float().argmax(dim=-1)
            print(f"StaticCache prefill OK, seq_len={static_past.get_seq_length()}", flush=True)
            # decode 一步
            out3 = qwen(input_ids=nt2[:, None], attention_mask=att2,
                         past_key_values=static_past, use_cache=True, return_dict=True)
            print(f"StaticCache decode OK, seq_len={static_past.get_seq_length()}", flush=True)
            print(f"layer_types 数量: {len(static_past.layers)}, 每层类型: {[type(l).__name__ for l in static_past.layers[:4]]}...", flush=True)
            print(f"is_compileable: {static_past.is_compileable}", flush=True)
    except Exception as e:
        import traceback
        print(f"StaticCache 失败: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
