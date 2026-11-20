#!/usr/bin/env python
"""R3 决定性测量：decode 是否 GPU-bound + CUDA Graph 实际收益。

修正 bench_decode_step.py 的两个 bug：
1. C 实验 prefill 混入计时 → 拆开
2. decode_with_item 的 input_ids 3D bug

干净测量：
- decode-only wall clock（prefill 在计时外，StaticCache，am=None）
- 纯 GPU 时间（event 包住 N 步 decode，prefill 在外）
- 尝试手动 CUDA graph 捕获 decode 单步
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

    MAX_NEW = 32

    def do_prefill():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
            pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
            emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
            cache = StaticCache(config=qwen.config, max_cache_len=att.shape[1] + MAX_NEW)
            out = qwen(inputs_embeds=emb, attention_mask=att, past_key_values=cache, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), cache, att.shape[1]

    # ---- 干净 decode-only wall clock ----
    def decode_n(nt, past, n_tok):
        gen = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(n_tok):
                out = qwen(input_ids=nt[:, None], attention_mask=None,
                            past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
                gen.append(int(nt.item()))
        return nt, past, gen

    # warmup
    nt, past, L = do_prefill()
    decode_n(nt, past, 5)
    torch.cuda.synchronize()

    # wall clock：20 token，prefill 在外
    N_TOK = 20
    iters = 15
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        nt, past, L = do_prefill()
        starts[i].record()
        decode_n(nt, past, N_TOK)
        ends[i].record()
    torch.cuda.synchronize()
    wall_ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    print(f"[wall clock] decode {N_TOK} tok: mean={wall_ms.mean():.2f} p50={np.percentile(wall_ms,50):.2f} → {wall_ms.mean()/N_TOK:.2f}ms/tok", flush=True)

    # ---- 纯 GPU 时间：prefill 在外，只包 decode ----
    gpu_times = []
    for _ in range(iters):
        nt, past, L = do_prefill()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        decode_n(nt, past, N_TOK)
        e.record()
        torch.cuda.synchronize()
        gpu_times.append(s.elapsed_time(e))
    gpu_ms = np.array(gpu_times)
    print(f"[GPU only] decode {N_TOK} tok: mean={gpu_ms.mean():.2f} p50={np.percentile(gpu_ms,50):.2f} → {gpu_ms.mean()/N_TOK:.2f}ms/tok", flush=True)
    print(f"→ CPU launch overhead = wall - GPU = {wall_ms.mean()-gpu_ms.mean():.2f}ms over {N_TOK} tok = {(wall_ms.mean()-gpu_ms.mean())/N_TOK:.2f}ms/tok", flush=True)

    # ---- CUDA Graph 捕获尝试 ----
    print("\n=== CUDA Graph 捕获 ===", flush=True)
    try:
        nt, past, L = do_prefill()
        # 静态输入 buffer
        static_ids = nt.clone().detach()
        static_logits = None
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # warmup on a side stream
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    out = qwen(input_ids=static_ids[:, None], attention_mask=None,
                                past_key_values=past, use_cache=True, return_dict=True)
                    static_logits = out.logits[:, -1].float().argmax(dim=-1)
            torch.cuda.current_stream().wait_stream(s)

            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                out = qwen(input_ids=static_ids[:, None], attention_mask=None,
                            past_key_values=past, use_cache=True, return_dict=True)
                static_logits = out.logits[:, -1].float().argmax(dim=-1)
        print("CUDA Graph 捕获成功", flush=True)

        # 跑 graph：复制新 input_ids 到 static buffer，replay
        def graph_step(nt):
            static_ids.copy_(nt)
            graph.replay()
            return static_logits.clone()

        # warmup
        nt, past, L = do_prefill()
        for _ in range(5):
            nt2 = graph_step(nt)
            nt = nt2
        torch.cuda.synchronize()

        g_starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        g_ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            nt, past, L = do_prefill()
            g_starts[i].record()
            for _ in range(N_TOK):
                nt = graph_step(nt)
            g_ends[i].record()
        torch.cuda.synchronize()
        g_ms = np.array([s.elapsed_time(e) for s, e in zip(g_starts, g_ends)])
        print(f"[CUDA Graph] decode {N_TOK} tok: mean={g_ms.mean():.2f} → {g_ms.mean()/N_TOK:.2f}ms/tok", flush=True)
        print(f"→ CUDA Graph 相对 wall clock 加速: {wall_ms.mean()/g_ms.mean():.2f}×", flush=True)
    except Exception as e:
        import traceback
        print(f"CUDA Graph 失败: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
