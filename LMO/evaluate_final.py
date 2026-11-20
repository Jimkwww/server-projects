#!/usr/bin/env python
"""最终组合方案：int4 磁盘存储(2.40×) + merge bf16 推理 + torch.compile decode(1.68×)。

在 evaluate_compile.py 基础上加 int4 磁盘存储（复用 evaluate_int4_disk.py 的 HQQ raw_output
+ pack/unpack），推理走 merge bf16 + compile decode。三项硬指标全部同口径复测。

流程：Runtime → merge LoRA → HQQ 量化 249 Linear → pack → 算磁盘大小 →
      unpack 回 bf16 → compile decode → 跑 2212 条官方12类。

用法:
  python evaluate_final.py --parquets "<3个parquet>" --config configs/LEMMO_Recognition.local.json \
      --device cuda:2 --output results/interference_full12_final.json
"""
import argparse
import json
import os
import tempfile
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lemmo.lora import LoRALinear

OFFICIAL_12 = [
    "AM_Jamming", "Comb_Jamming", "FM_Jamming", "Multi_Tone",
    "Narrowband", "No_Interference", "Protocol_Bluetooth", "Protocol_WiFi",
    "Pulse_DME", "Single_Tone", "Sweep_Chirp", "Wideband",
]

ALIASES = {
    "AM_Jamming": ["调幅", "am"],
    "Comb_Jamming": ["梳状", "comb"],
    "FM_Jamming": ["调频", "fm"],
    "Multi_Tone": ["多音", "multi tone", "multi-tone", "多频"],
    "Narrowband": ["窄带", "narrowband"],
    "No_Interference": ["无干扰", "没有干扰", "不存在干扰", "no interference"],
    "Protocol_Bluetooth": ["蓝牙", "bluetooth"],
    "Protocol_WiFi": ["wifi", "wi-fi", "wi fi"],
    "Pulse_DME": ["dme", "脉冲"],
    "Single_Tone": ["单音", "single tone", "single-tone", "单频"],
    "Sweep_Chirp": ["扫频", "chirp", "线性调频"],
    "Wideband": ["宽带", "wideband"],
}


def label_in_answer(answer, label):
    a = answer.lower()
    for kw in ALIASES.get(label, [label.lower()]):
        if kw.lower() in a:
            return True
    return False


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


def quantize_linear_hqq(lin: nn.Linear, gs: int = 128):
    from torchao.quantization.quant_primitives import choose_qparams_and_quantize_affine_hqq
    w = lin.weight.data
    N, K = w.shape
    W_q, scale, zero, _ = choose_qparams_and_quantize_affine_hqq(
        w, nbits=4, group_size=gs, axis=1, compute_dtype=torch.float16,
        device=w.device, raw_output=True,
    )
    ngrp = K // gs
    q2 = W_q.reshape(N, ngrp, gs // 2, 2)
    packed = (q2[..., 0] | (q2[..., 1] << 4)).cpu()
    return packed, scale.cpu(), zero.cpu()


def dequant_linear_hqq(packed, scale, zero, N, K, gs=128, device="cuda", dtype=torch.bfloat16):
    ngrp = K // gs
    p = packed.to(device)
    lo = (p & 0x0F).to(torch.float32)
    hi = ((p >> 4) & 0x0F).to(torch.float32)
    W_q = torch.stack([lo, hi], dim=-1).reshape(N, ngrp, gs)
    w = (W_q - zero.to(device).float()[:, :, None]) * scale.to(device).float()[:, :, None]
    return w.reshape(N, K).to(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquets", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--question", default="这段信号中存在哪种干扰？")
    ap.add_argument("--sample-per-class", type=int, default=200)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile-mode", default="default")
    args = ap.parse_args()

    from lemmo import LEMMORecognitionRuntime
    from lemmo.runtime import _prompt, _insert_signal, LEMMO_RECOGNITION_SYSTEM_PROMPT
    from transformers.cache_utils import StaticCache

    paths = [p.strip() for p in args.parquets.split(",") if p.strip()]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df = df[df["label_value"].isin(OFFICIAL_12)]
    if args.sample_per_class:
        sampled = []
        for lab in OFFICIAL_12:
            sub = df[df["label_value"] == lab]
            sampled.append(sub.sample(n=min(len(sub), args.sample_per_class), random_state=args.seed))
        df = pd.concat(sampled, ignore_index=True)
    print(f"eval set: {len(df)} rows", flush=True)

    runtime = LEMMORecognitionRuntime(args.config, args.device)
    dev = runtime.device
    tok = runtime.tokenizer
    qwen = runtime.qwen
    model = runtime.model
    eos = int(tok.eos_token_id)

    n = merge_lora(qwen)
    print(f"[merge] 合并 {n} 层 LoRA", flush=True)

    # ---------- int4 磁盘存储 ----------
    # 量化所有 nn.Linear 的 weight（bias 保留 bf16，不量化）
    t0 = time.time()
    layers = {}
    linear_paths = []
    for mn, mod in list(qwen.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear):
                path = f"{mn}.{cn}".strip(".")
                linear_paths.append(path)
                packed, scale, zero = quantize_linear_hqq(ch, args.group_size)
                layers[path] = (packed, scale, zero, ch.out_features, ch.in_features)
    torch.cuda.synchronize()
    quant_time = time.time() - t0

    full_sd = runtime.qwen.state_dict()
    # state_dict 里对应 Linear 层的 .weight key
    linear_weight_keys = {p + ".weight" for p in linear_paths}
    # 非 int4 部分 = state_dict 去掉 Linear weight 后的所有 key
    # （含 bias、embed_tokens、lm_head 若为 Linear 则其 weight 已入 int4、norm、dt_bias、A_log 等）
    nonlin_sd = {k: v for k, v in full_sd.items() if k not in linear_weight_keys}

    with tempfile.TemporaryDirectory() as d:
        p_int4 = os.path.join(d, "int4_linear.pt")
        torch.save(layers, p_int4)
        disk_int4 = os.path.getsize(p_int4)
        p_nonlin = os.path.join(d, "nonlinear.pt")
        torch.save(nonlin_sd, p_nonlin)
        disk_nonlin = os.path.getsize(p_nonlin)
        disk_full = disk_int4 + disk_nonlin
        p_bf16 = os.path.join(d, "bf16.pt")
        torch.save(full_sd, p_bf16)
        disk_bf16 = os.path.getsize(p_bf16)

    # ---------- unpack 回 bf16 ----------
    t0 = time.time()
    for mn, mod in list(qwen.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear):
                key = f"{mn}.{cn}".strip(".")
                packed, scale, zero, N, K = layers[key]
                wd = dequant_linear_hqq(packed, scale, zero, N, K, args.group_size,
                                        device=dev, dtype=ch.weight.dtype)
                ch.weight.data.copy_(wd)
    torch.cuda.synchronize()
    unpack_time = time.time() - t0

    compression = disk_bf16 / disk_full
    print(json.dumps({
        "n_merged": n, "n_linear": len(layers),
        "quant_time_s": round(quant_time, 2), "unpack_time_s": round(unpack_time, 2),
        "disk_int4_linear_MB": round(disk_int4 / 1e6, 1),
        "disk_nonlinear_MB": round(disk_nonlin / 1e6, 1),
        "disk_full_MB": round(disk_full / 1e6, 1),
        "disk_bf16_MB": round(disk_bf16 / 1e6, 1),
        "compression_ratio": round(compression, 2),
    }, ensure_ascii=False), flush=True)

    # ---------- compile decode 评测 ----------
    MAX_CACHE = 2048
    compiled_qwen = torch.compile(qwen, mode=args.compile_mode, fullgraph=False)

    def prefill(emb, att):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            cache = StaticCache(config=qwen.config, max_cache_len=MAX_CACHE)
            out = qwen(inputs_embeds=emb, attention_mask=att, past_key_values=cache, use_cache=True, return_dict=True)
            return out.logits[:, -1].float().argmax(dim=-1), cache

    def greedy(nt, past):
        gen = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(args.max_new_tokens):
                t = int(nt.item())
                if t == eos:
                    break
                gen.append(t)
                out = compiled_qwen(input_ids=nt[:, None], attention_mask=None,
                                    past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
        return tok.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    def answer(array, sr, question):
        signal = torch.from_numpy(array)[None].to(dev, dtype=torch.bfloat16)
        rates = torch.tensor([sr], dtype=torch.float32, device=dev)
        prompt = _prompt(tok, LEMMO_RECOGNITION_SYSTEM_PROMPT, question, dev)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
            pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
            emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
        nt, past = prefill(emb, att)
        return greedy(nt, past)

    # warmup
    tmpdir = tempfile.mkdtemp(prefix="lmo_final_")
    first = df.iloc[0]
    iq0 = np.stack(first["iq"]).astype(np.float32)
    p0 = os.path.join(tmpdir, "warmup.npy")
    np.save(p0, iq0)
    from lemmo.io import load_iq, LEMMO_RECOGNITION_LENGTHS
    _ = answer(load_iq(p0, LEMMO_RECOGNITION_LENGTHS), float(first["sample_rate"]), args.question)
    torch.cuda.synchronize()
    print("[warmup] compile 完成", flush=True)

    torch.cuda.reset_peak_memory_stats()
    results = []
    t0 = time.time()
    for i, row in df.iterrows():
        iq = np.stack(row["iq"]).astype(np.float32)
        label = row["label_value"]
        sample_id = row["sample_id"]
        sr = float(row["sample_rate"])
        tmp_path = os.path.join(tmpdir, f"{i}.npy")
        np.save(tmp_path, iq)
        ans = answer(load_iq(tmp_path, LEMMO_RECOGNITION_LENGTHS), sr, args.question)
        results.append({"sample_id": sample_id, "label": label,
                        "prediction": ans, "correct": label_in_answer(ans, label)})
        if (i + 1) % 200 == 0 or (i + 1) == len(df):
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"[{i+1}/{len(df)}] acc={acc:.4f}", flush=True)

    acc = sum(r["correct"] for r in results) / len(results)
    elapsed = time.time() - t0
    mem_gb = torch.cuda.max_memory_allocated() / 1e9

    per = defaultdict(lambda: [0, 0])
    for r in results:
        per[r["label"]][1] += 1
        if r["correct"]:
            per[r["label"]][0] += 1
    per_acc = {l: {"correct": c, "total": t, "acc": c / t if t else 0}
               for l, (c, t) in sorted(per.items())}

    summary = {
        "scheme": "final (int4磁盘存储 + merge bf16推理 + torch.compile decode)",
        "compile_mode": args.compile_mode,
        "total": len(results), "correct": sum(r["correct"] for r in results),
        "accuracy": acc, "elapsed_sec": elapsed,
        "avg_sec_per_sample": elapsed / len(results) if results else 0,
        "peak_mem_GB": mem_gb,
        "disk_full_GB": disk_full / 1e9, "disk_bf16_GB": disk_bf16 / 1e9,
        "compression_ratio": compression,
        "quant_time_s": quant_time, "unpack_time_s": unpack_time,
        "per_class": per_acc,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== [final] accuracy: {acc:.4f} ({summary['correct']}/{summary['total']}) ===")
    print(f"=== elapsed {elapsed:.1f}s, avg {elapsed/len(results) if results else 0:.3f}s/sample, "
          f"mem {mem_gb:.2f}GB, disk {disk_full/1e9:.2f}GB vs bf16 {disk_bf16/1e9:.2f}GB = {compression:.2f}x ===")
    for l in OFFICIAL_12:
        if l in per_acc:
            p = per_acc[l]
            print(f"  {l:22s} {p['correct']:4d}/{p['total']:4d} = {p['acc']:.4f}")
    print(f"=== saved to {args.output} ===")


if __name__ == "__main__":
    main()
