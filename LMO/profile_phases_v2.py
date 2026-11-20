#!/usr/bin/env python
"""LEMMO interference 推理耗时分解 + profiler（BF16/INT8/INT4 对照）。

严格复现 LEMMORecognitionRuntime.answer() 的推理逻辑，但拆开各阶段，
绕开 evaluate_quant.py 里每样本写 npy 的磁盘 IO。用 torch.cuda.Event 计时。

阶段（answer() 内部对应）：
  load        : load_iq 数值归一化（一次性，不进每样本计时）
  to_device   : numpy -> cuda bf16 tensor
  encoder     : signal.encode() = P16 encoder(12层FLA) + resampler
  bridge      : bridge + task_adapter + projector
  prefill     : qwen 第一次 forward（inputs_embeds，建立 cache/state）
  decode      : 逐 token 贪心（含 EOS，统计真实输出 token 数）

用法:
  python profile_phases_v2.py --config configs/LEMMO_Recognition.local.json \
      --iq-length 16368 --quant bf16 --iters 30 --mode timing --device cuda:2
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

PARSQUETS = {
    128: "/data1/user_wzk/LMO/data/interference/owl_init_test.parquet",
    1024: "/data1/user_wzk/LMO/data/interference/em_infer_comm_test.parquet",
    16368: "/data1/user_wzk/LMO/data/interference/gnss_jamming_test.parquet",
}


def load_iq_mem(array: np.ndarray) -> np.ndarray:
    """复现 lemmo.io.load_iq 的归一化，但输入已是内存数组（不写文件）。"""
    value = np.asarray(array, dtype=np.float32, order="C")
    if not np.isfinite(value).all():
        raise ValueError("IQ contains NaN or Inf")
    scale = float(np.max(np.abs(value)))
    if not np.isfinite(scale) or scale < 1e-12:
        raise ValueError("IQ is empty or has zero amplitude")
    return np.ascontiguousarray(value / scale)


def get_signal(iq_length: int):
    """从 parquet 取一个真实样本，返回内存归一化后的 [1,L,2] numpy。"""
    df = pd.read_parquet(PARSQUETS[iq_length])
    row = df.iloc[0]
    iq = np.stack(row["iq"]).astype(np.float32)
    label = row["label_value"]
    sr = float(row["sample_rate"])
    arr = load_iq_mem(iq)
    return arr, label, sr


def timeit(fn, iters, warmup=3):
    """用 cuda.Event 计时，返回 (mean_ms, p50_ms, p95_ms, all_ms)。"""
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
    return float(ms.mean()), float(np.percentile(ms, 50)), float(np.percentile(ms, 95)), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--iq-length", type=int, required=True, choices=[128, 1024, 16368])
    ap.add_argument("--quant", default="bf16", choices=["bf16", "int8", "int4"])
    ap.add_argument("--question", default="这段信号中存在哪种干扰？")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--mode", default="timing", choices=["timing", "profiler"])
    ap.add_argument("--output", default=None, help="JSON 输出路径")
    args = ap.parse_args()

    from lemmo import LEMMORecognitionRuntime
    from lemmo.runtime import (
        _prompt, _insert_signal, _greedy_generate, LEMMO_RECOGNITION_SYSTEM_PROMPT,
    )

    print(f"=== 加载 runtime (device={args.device}) ===", flush=True)
    runtime = LEMMORecognitionRuntime(args.config, args.device)
    dev = runtime.device
    tok = runtime.tokenizer
    qwen = runtime.qwen
    model = runtime.model

    # 量化（同 evaluate_quant.py）
    if args.quant == "int8":
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        quantize_(qwen, Int8WeightOnlyConfig())
        print("=== int8 量化完成 ===", flush=True)
    elif args.quant == "int4":
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
        from torchao.quantization.quant_primitives import ZeroPointDomain
        quantize_(qwen, Int4WeightOnlyConfig(use_hqq=True, zero_point_domain=ZeroPointDomain.FLOAT))
        print("=== int4 量化完成 ===", flush=True)

    # 取样本（内存，不写文件）
    arr, label, sr = get_signal(args.iq_length)
    signal = torch.from_numpy(arr)[None].to(dev, dtype=torch.bfloat16)
    rates = torch.tensor([sr], dtype=torch.float32, device=dev)
    prompt = _prompt(tok, LEMMO_RECOGNITION_SYSTEM_PROMPT, args.question, dev)
    eos = int(tok.eos_token_id)
    max_new = args.max_new_tokens
    n_text_tokens = int(prompt["input_ids"].shape[1])
    n_signal_tokens = args.iq_length // 16
    print(f"=== 样本: len={args.iq_length} label={label} sr={sr} ===", flush=True)
    print(f"=== 文本token={n_text_tokens} 信号token={n_signal_tokens} prefill总长={n_text_tokens + n_signal_tokens} ===", flush=True)

    # 预计算一次，拿到 prefill 的 cache 供 decode 复现
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
        tokens, mask = model.signal.encode(signal, rates)
        bridge = model.signal.bridge(tokens, mask, q_emb, prompt["question_mask"])
        adapted = model.task_adapters["interference"](bridge["conditioned_queries"])
        projected = model.signal.bridge.projector(model.signal.bridge.output_norm(adapted))
        embeddings, attention = _insert_signal(qwen.get_input_embeddings(), prompt, projected)
        prefill_out = qwen(inputs_embeds=embeddings, attention_mask=attention, use_cache=True, return_dict=True)

    prefill_past = prefill_out.past_key_values
    prefill_next = prefill_out.logits[:, -1].float().argmax(dim=-1)

    # ---- 各阶段 fn ----
    def fn_to_device():
        return torch.from_numpy(arr)[None].to(dev, dtype=torch.bfloat16)

    def fn_encoder():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model.signal.encode(signal, rates)

    def fn_bridge():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            t, m = model.signal.encode(signal, rates)
            b = model.signal.bridge(t, m, q_emb, prompt["question_mask"])
            a = model.task_adapters["interference"](b["conditioned_queries"])
            return model.signal.bridge.projector(model.signal.bridge.output_norm(a))

    def fn_prefill():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = qwen(inputs_embeds=embeddings, attention_mask=attention, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), out.past_key_values

    def fn_decode_one():
        """固定 cache 的单步 decode（稳态每 token 耗时）。"""
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = qwen(
                input_ids=prefill_next[:, None],
                attention_mask=attention,
                past_key_values=prefill_past,
                use_cache=True,
                return_dict=True,
            )
            return out.logits[:, -1].float().argmax(dim=-1), out.past_key_values

    def fn_decode_full():
        """完整贪心 decode（含 EOS），返回 token 数。"""
        gen = []
        nt = prefill_next
        past = prefill_past
        att = attention
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(max_new):
                tk = int(nt.item())
                if tk == eos:
                    break
                gen.append(tk)
                out = qwen(
                    input_ids=nt[:, None],
                    attention_mask=att,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
                att = torch.cat((att, torch.ones((1, 1), dtype=att.dtype, device=att.device)), dim=1)
        return len(gen)

    # sanity：完整 answer 复现，确认输出合理
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        qe = qwen.get_input_embeddings()(prompt["question_ids"])
        pr = model(signal, rates, qe, prompt["question_mask"], "interference")
        emb2, att2 = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
    ans = _greedy_generate(qwen, tok, emb2, att2, max_new)
    print(f"=== [sanity] answer = {ans!r} (label={label}) ===", flush=True)

    if args.mode == "profiler":
        print("=== profiler 模式（3 次完整 answer）===", flush=True)
        from torch.profiler import profile, ProfilerActivity
        def full_answer():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                qe = qwen.get_input_embeddings()(prompt["question_ids"])
                pr = model(signal, rates, qe, prompt["question_mask"], "interference")
                eb, at = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
            _greedy_generate(qwen, tok, eb, at, max_new)
        full_answer()  # 预热
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, profile_memory=False) as prof:
            for _ in range(3):
                full_answer()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
        return

    # ---- 计时 ----
    print(f"=== 计时 {args.iters} 次 ===", flush=True)
    stages = {}
    for name, fn in [
        ("to_device", fn_to_device),
        ("encoder", fn_encoder),
        ("bridge", fn_bridge),
        ("prefill", fn_prefill),
        ("decode_one", fn_decode_one),
        ("decode_full", fn_decode_full),
    ]:
        mean, p50, p95, allms = timeit(fn, args.iters)
        stages[name] = {"mean_ms": round(mean, 3), "p50_ms": round(p50, 3), "p95_ms": round(p95, 3)}
        print(f"  {name:12s} mean={mean:8.3f}ms  p50={p50:8.3f}  p95={p95:8.3f}", flush=True)

    # 统计 decode token 数（多次取中位数）
    tok_counts = []
    for _ in range(10):
        tok_counts.append(fn_decode_full())
    torch.cuda.synchronize()
    n_tokens = int(np.median(tok_counts))
    print(f"=== 输出 token 数（中位）: {n_tokens} ===", flush=True)

    enc_ms = stages["encoder"]["mean_ms"]
    bridge_ms = stages["bridge"]["mean_ms"]
    prefill_ms = stages["prefill"]["mean_ms"]
    decode_full_ms = stages["decode_full"]["mean_ms"]
    signal_ms = enc_ms + bridge_ms
    qwen_ms = prefill_ms + decode_full_ms
    total_ms = signal_ms + qwen_ms
    p = qwen_ms / total_ms * 100

    summary = {
        "quant": args.quant,
        "iq_length": args.iq_length,
        "n_text_tokens": n_text_tokens,
        "n_signal_tokens": n_signal_tokens,
        "n_output_tokens": n_tokens,
        "label": label,
        "stages": stages,
        "signal_ms": round(signal_ms, 3),
        "qwen_ms": round(qwen_ms, 3),
        "total_ms": round(total_ms, 3),
        "qwen_ratio_pct": round(p, 2),
        "decode_per_token_ms": round(decode_full_ms / n_tokens, 3) if n_tokens else None,
    }
    print("\n===== 汇总 =====", flush=True)
    print(f"信号路径(encoder+bridge): {signal_ms:.3f} ms  ({signal_ms/total_ms*100:.1f}%)")
    print(f"Qwen prefill             : {prefill_ms:.3f} ms  ({prefill_ms/total_ms*100:.1f}%)")
    print(f"Qwen decode ({n_tokens}tok): {decode_full_ms:.3f} ms  ({decode_full_ms/total_ms*100:.1f}%)")
    print(f"端到端                   : {total_ms:.3f} ms")
    print(f"Qwen 耗时占比 p          : {p:.1f}%")
    print(f"Amdahl 上限（Qwen无限快）: {total_ms/signal_ms:.2f}×", flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"=== 已保存 {args.output} ===", flush=True)


if __name__ == "__main__":
    main()
