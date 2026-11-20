#!/usr/bin/env python
"""R3 关键优化：消除 decode 循环逐 token 的 host↔GPU 同步（.item()）。

贪心 decode 的 token 序列是确定的，只依赖前一个 token 的 logits，不依赖 eos 判断本身。
因此可以：GPU 上连续生成 max_new_tokens 个 token（全程无 .item() 同步），最后一次性同步，
找第一个 eos 截断。把"每 token 一次 host 同步"变成"整个 decode 一次同步"。

对比三种（真实语义等价，贪心）：
  A) 官方逐 token .item()
  B) 无同步 eager + eos 延后截断
  C) 无同步 compile + eos 延后截断

用法:
  python bench_noitem_decode.py --config configs/LEMMO_Recognition.local.json --device cuda:2
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
    ap.add_argument("--max-new-tokens", type=int, default=64)
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

    def prefill():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            cache = StaticCache(config=qwen.config, max_cache_len=MAX_CACHE)
            out = qwen(inputs_embeds=emb, attention_mask=att, past_key_values=cache, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), cache

    def decode_sync(fwd, nt, past):
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
        return tok.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    def decode_nosync(fwd, nt, past):
        tokens = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(args.max_new_tokens):
                tokens.append(nt)
                out = fwd(input_ids=nt[:, None], attention_mask=None,
                          past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
            seq = torch.cat(tokens, dim=0).tolist()
        gen = []
        for t in seq:
            if t == eos:
                break
            gen.append(t)
        return tok.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    # ---- 答案一致性 ----
    nt, past = prefill()
    ans_sync = decode_sync(qwen, nt, past)
    nt, past = prefill()
    ans_nosync = decode_nosync(qwen, nt, past)
    print(f"[A sync]        answer={ans_sync!r}", flush=True)
    print(f"[B nosync]      answer={ans_nosync!r}", flush=True)
    print(f"A==B: {ans_sync == ans_nosync}", flush=True)

    compiled = torch.compile(qwen, mode="default", fullgraph=False)
    nt, past = prefill()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(6):
            out = compiled(input_ids=nt[:, None], attention_mask=None,
                           past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            nt = out.logits[:, -1].float().argmax(dim=-1)
    torch.cuda.synchronize()
    nt, past = prefill()
    ans_comp = decode_nosync(compiled, nt, past)
    print(f"[C nosync+comp] answer={ans_comp!r}", flush=True)
    print(f"A==C: {ans_sync == ans_comp}", flush=True)

    # ---- 测速 ----
    def bench(fn, fwd):
        for _ in range(3):
            nt, past = prefill()
            fn(fwd, nt, past)
        torch.cuda.synchronize()
        iters = 20
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            nt, past = prefill()
            starts[i].record()
            fn(fwd, nt, past)
            ends[i].record()
        torch.cuda.synchronize()
        return np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])

    ms_a = bench(decode_sync, qwen)
    ms_b = bench(decode_nosync, qwen)
    ms_c = bench(decode_nosync, compiled)

    print(f"\n=== decode 计时（8 token，真实贪心语义）===", flush=True)
    print(f"A) 官方逐 token .item(): mean={ms_a.mean():.2f}ms ({ms_a.mean()/8:.2f}ms/tok)", flush=True)
    print(f"B) 无同步 eager:       mean={ms_b.mean():.2f}ms ({ms_b.mean()/8:.2f}ms/tok)", flush=True)
    print(f"C) 无同步 compile:     mean={ms_c.mean():.2f}ms ({ms_c.mean()/8:.2f}ms/tok)", flush=True)
    print(f"加速 B vs A = {ms_a.mean()/ms_b.mean():.2f}x, C vs A = {ms_a.mean()/ms_c.mean():.2f}x", flush=True)


if __name__ == "__main__":
    main()
