#!/usr/bin/env python
"""干净端到端 benchmark v2：统一口径对比 R0(原始) vs merge / merge_int8 / merge_int4。

完全自包含：每次迭代都重新 prefill + 全新贪心 decode，无共享可变 cache 状态。
绕开磁盘 IO（不 np.save）。测信号路径 / prefill / decode 全程 / 端到端。

用法:
  python bench_e2e.py --config configs/LEMMO_Recognition.local.json \
      --device cuda:2 --iq-length 16368 --iters 15 --combo merge_int4
"""
import argparse
import json
import os
import tempfile

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


def load_iq_mem(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32, order="C")
    scale = float(np.max(np.abs(value)))
    return np.ascontiguousarray(value / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--iq-length", type=int, required=True, choices=[128, 1024, 16368])
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--question", default="这段信号中存在哪种干扰？")
    ap.add_argument("--combo", required=True, choices=["bf16", "merge", "merge_int8", "merge_int4"])
    args = ap.parse_args()

    from lemmo import LEMMORecognitionRuntime
    from lemmo.runtime import _prompt, _insert_signal, LEMMO_RECOGNITION_SYSTEM_PROMPT

    do_merge = args.combo != "bf16"
    quant = None
    if "int8" in args.combo:
        quant = "int8"
    elif "int4" in args.combo:
        quant = "int4"

    runtime = LEMMORecognitionRuntime(args.config, args.device)
    dev = runtime.device
    tok = runtime.tokenizer
    qwen = runtime.qwen
    model = runtime.model

    if do_merge:
        n = merge_lora(qwen)
        print(f"[{args.combo}] LoRA 合并 {n} 层", flush=True)

    if quant == "int8":
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        quantize_(qwen, Int8WeightOnlyConfig())
    elif quant == "int4":
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
        from torchao.quantization.quant_primitives import ZeroPointDomain
        quantize_(qwen, Int4WeightOnlyConfig(use_hqq=True, zero_point_domain=ZeroPointDomain.FLOAT))

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "qwen.pt")
        torch.save(qwen.state_dict(), p)
        disk_gb = os.path.getsize(p) / 1e9

    df = pd.read_parquet(PARSQUETS[args.iq_length])
    row = df.iloc[0]
    arr = load_iq_mem(np.stack(row["iq"]).astype(np.float32))
    label = row["label_value"]
    sr = float(row["sample_rate"])
    signal = torch.from_numpy(arr)[None].to(dev, dtype=torch.bfloat16)
    rates = torch.tensor([sr], dtype=torch.float32, device=dev)
    prompt = _prompt(tok, LEMMO_RECOGNITION_SYSTEM_PROMPT, args.question, dev)
    eos = int(tok.eos_token_id)

    # 每阶段都是完全自包含，fresh cache
    def signal_path():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
            return model(signal, rates, q_emb, prompt["question_mask"], "interference")

    def prefill():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
            pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
            emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
            out = qwen(inputs_embeds=emb, attention_mask=att, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), out.past_key_values, att

    def decode_full():
        # fresh prefill + fresh decode
        nt, past, a = prefill()
        gen = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(args.max_new_tokens):
                tk = int(nt.item())
                if tk == eos:
                    break
                gen.append(tk)
                out = qwen(input_ids=nt[:, None], attention_mask=a,
                            past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
                a = torch.cat((a, torch.ones((1, 1), dtype=a.dtype, device=a.device)), dim=1)
        return len(gen)

    def e2e():
        nt, past, a = prefill()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(args.max_new_tokens):
                tk = int(nt.item())
                if tk == eos:
                    break
                out = qwen(input_ids=nt[:, None], attention_mask=a,
                            past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
                a = torch.cat((a, torch.ones((1, 1), dtype=a.dtype, device=a.device)), dim=1)

    torch.cuda.reset_peak_memory_stats()
    sp = timeit(signal_path, args.iters)
    pf = timeit(prefill, args.iters)
    dc = timeit(decode_full, max(3, args.iters // 2))
    ee = timeit(e2e, max(3, args.iters // 2))
    torch.cuda.synchronize()
    mem_gb = torch.cuda.max_memory_allocated() / 1e9

    # 单独跑一次完整 decode 拿输出 token 数和答案
    nt, past, a = prefill()
    gen = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.max_new_tokens):
            tk = int(nt.item())
            if tk == eos:
                break
            gen.append(tk)
            out = qwen(input_ids=nt[:, None], attention_mask=a,
                        past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            nt = out.logits[:, -1].float().argmax(dim=-1)
            a = torch.cat((a, torch.ones((1, 1), dtype=a.dtype, device=a.device)), dim=1)
    answer = tok.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    summary = {
        "combo": args.combo,
        "label": label,
        "answer": answer,
        "n_output_tokens": len(gen),
        "signal_path_ms": {"mean": round(sp[0], 3), "p50": round(sp[1], 3), "p95": round(sp[2], 3)},
        "prefill_ms": {"mean": round(pf[0], 3), "p50": round(pf[1], 3), "p95": round(pf[2], 3)},
        "decode_full_ms": {"mean": round(dc[0], 3), "p50": round(dc[1], 3), "p95": round(dc[2], 3)},
        "e2e_ms": {"mean": round(ee[0], 3), "p50": round(ee[1], 3), "p95": round(ee[2], 3)},
        "peak_mem_GB": round(mem_gb, 3),
        "qwen_disk_GB": round(disk_gb, 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
