#!/usr/bin/env python
"""R3 收尾：prefill 也 compile，看能否再压 e2e 加速比。

当前状态（evaluate_final.py）：
- 只 compile 了 decode（10.22ms/tok）
- prefill 还是 eager（merge 后 ~20.4ms）

这里测 prefill 也 compile 的效果。prefill 用 StaticCache + 固定形状，理论上可 compile。
注意：prefill 的 attention_mask 形状随输入长度变化，但固定 16368 点样本时形状固定。

用法:
  python bench_prefill_compile.py --config configs/LEMMO_Recognition.local.json --device cuda:2
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lemmo.lora import LoRALinear

PARSQUETS = {
    16368: "/data1/user_wzk/LMO/data/interference/gnss_jamming_test.parquet",
}


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
    eos = int(tok.eos_token_id)

    n = merge_lora(qwen)
    print(f"LoRA 合并 {n} 层", flush=True)

    df = pd.read_parquet(PARSQUETS[16368])
    row = df.iloc[0]
    arr = load_iq_mem(np.stack(row["iq"]).astype(np.float32))
    sr = float(row["sample_rate"])
    signal = torch.from_numpy(arr)[None].to(dev, dtype=torch.bfloat16)
    rates = torch.tensor([sr], dtype=torch.float32, device=dev)
    prompt = _prompt(tok, LEMMO_RECOGNITION_SYSTEM_PROMPT, "这段信号中存在哪种干扰？", dev)
    MAX_CACHE = 2048

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
        pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
        emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)

    # 尝试 prefill compile：qwen(inputs_embeds=...) 首次调用
    print("=== 尝试 prefill compile ===", flush=True)
    for mode in ["default", "reduce-overhead"]:
        try:
            compiled = torch.compile(qwen, mode=mode, fullgraph=False)
            # warmup 触发编译
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(5):
                    cache = StaticCache(config=qwen.config, max_cache_len=MAX_CACHE)
                    out = compiled(inputs_embeds=emb, attention_mask=att, past_key_values=cache, use_cache=True, return_dict=True)
            torch.cuda.synchronize()
            # 测速
            iters = 15
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for i in range(iters):
                cache = StaticCache(config=qwen.config, max_cache_len=MAX_CACHE)
                starts[i].record()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = compiled(inputs_embeds=emb, attention_mask=att, past_key_values=cache, use_cache=True, return_dict=True)
                ends[i].record()
            torch.cuda.synchronize()
            ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
            print(f"compile[{mode}] prefill: mean={ms.mean():.2f}ms p50={np.percentile(ms,50):.2f}", flush=True)
        except Exception as e:
            print(f"compile[{mode}] prefill 失败: {type(e).__name__}: {str(e)[:300]}", flush=True)

    # eager prefill 基线
    iters = 15
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        cache = StaticCache(config=qwen.config, max_cache_len=MAX_CACHE)
        starts[i].record()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = qwen(inputs_embeds=emb, attention_mask=att, past_key_values=cache, use_cache=True, return_dict=True)
        ends[i].record()
    torch.cuda.synchronize()
    ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    print(f"[eager] prefill: mean={ms.mean():.2f}ms p50={np.percentile(ms,50):.2f}", flush=True)


if __name__ == "__main__":
    main()
