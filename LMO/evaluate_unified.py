#!/usr/bin/env python
"""统一口径评测：所有方案在同一冻结采样集 sample_set_2212.json 上跑。

修复此前 bf16 与 merge 各自在脚本内 .sample(random_state=42) 采样导致
样本集不一致（sample_id 交集仅 1237/2212）的问题。

本脚本：
  - 只从 results/sample_set_2212.json 读样本 id（固定顺序，固定集合）
  - 按 sample_id 从 parquet 取 iq/label/sample_rate
  - 每个 --scheme 跑完全相同的 2212 条
  - 输出统一 JSON：accuracy/per_class/elapsed/avg/mem/disk + 完整逐条 results

scheme 取值：
  bf16            原始 online LoRA（baseline）
  merge_bf16      LoRA 合并，bf16 推理
  merge_compile   LoRA 合并 + torch.compile decode + StaticCache
  int4_disk       LoRA 合并 + HQQ int4 存储 + unpack 回 bf16 推理（eager）
  final           int4_disk + torch.compile decode（完整最终方案）

用法:
  python evaluate_unified.py --config configs/LEMMO_Recognition.local.json \
      --device cuda:2 --scheme final --output results/unified_final.json
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

PARSQUETS = [
    "/data1/user_wzk/LMO/data/interference/em_infer_comm_test.parquet",
    "/data1/user_wzk/LMO/data/interference/owl_init_test.parquet",
    "/data1/user_wzk/LMO/data/interference/gnss_jamming_test.parquet",
]


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
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--question", default="这段信号中存在哪种干扰？")
    ap.add_argument("--scheme", required=True,
                    choices=["bf16", "merge_bf16", "merge_compile", "int4_disk", "final"])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--sample-set", default="results/sample_set_2212.json")
    ap.add_argument("--compile-mode", default="default")
    args = ap.parse_args()

    from lemmo import LEMMORecognitionRuntime
    from lemmo.runtime import _prompt, _insert_signal, LEMMO_RECOGNITION_SYSTEM_PROMPT
    from lemmo.io import load_iq, LEMMO_RECOGNITION_LENGTHS
    from transformers.cache_utils import StaticCache

    # ---- 冻结采样集 ----
    with open(args.sample_set) as f:
        sample_set = json.load(f)
    print(f"[unified] 冻结采样集 {len(sample_set)} 条，scheme={args.scheme}", flush=True)

    df = pd.concat([pd.read_parquet(p) for p in PARSQUETS], ignore_index=True)
    by_id = {r.sample_id: r for r in df.itertuples()}

    runtime = LEMMORecognitionRuntime(args.config, args.device)
    dev = runtime.device
    tok = runtime.tokenizer
    qwen = runtime.qwen
    model = runtime.model
    eos = int(tok.eos_token_id)

    do_merge = args.scheme in ("merge_bf16", "merge_compile", "int4_disk", "final")
    do_compile = args.scheme in ("merge_compile", "final")
    do_int4 = args.scheme in ("int4_disk", "final")

    n_merged = 0
    if do_merge:
        n_merged = merge_lora(qwen)
        print(f"[unified] LoRA 合并 {n_merged} 层", flush=True)

    # ---- int4 磁盘存储 ----
    disk_full = disk_bf16 = compression = None
    quant_time = unpack_time = None
    layers = None
    if do_int4:
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

        full_sd = qwen.state_dict()
        linear_weight_keys = {p + ".weight" for p in linear_paths}
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
        compression = disk_bf16 / disk_full

        # unpack 回 bf16
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
        print(f"[unified] int4 存储 disk_full={disk_full/1e9:.2f}GB vs bf16 {disk_bf16/1e9:.2f}GB = {compression:.2f}x, "
              f"quant {quant_time:.1f}s, unpack {unpack_time:.1f}s", flush=True)
    else:
        # 非 int4 方案：磁盘口径 = merge 后的完整 state_dict（或 bf16 原始）
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "qwen.pt")
            torch.save(qwen.state_dict(), p)
            disk_bf16 = os.path.getsize(p)
        disk_full = disk_bf16
        compression = 1.0

    # ---- compile ----
    compiled_qwen = None
    if do_compile:
        compiled_qwen = torch.compile(qwen, mode=args.compile_mode, fullgraph=False)
        MAX_CACHE = 2048

    def prefill(emb, att):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if do_compile:
                cache = StaticCache(config=qwen.config, max_cache_len=MAX_CACHE)
                out = qwen(inputs_embeds=emb, attention_mask=att,
                           past_key_values=cache, use_cache=True, return_dict=True)
                return out.logits[:, -1].float().argmax(dim=-1), cache, None
            else:
                out = qwen(inputs_embeds=emb, attention_mask=att, use_cache=True, return_dict=True)
                return out.logits[:, -1].float().argmax(dim=-1), out.past_key_values, att

    def greedy(nt, past, att):
        gen = []
        fwd = compiled_qwen if do_compile else qwen
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(args.max_new_tokens):
                t = int(nt.item())
                if t == eos:
                    break
                gen.append(t)
                if do_compile:
                    # compile 分支：StaticCache 形状固定 + attention_mask=None（已验证语义）
                    out = fwd(input_ids=nt[:, None], attention_mask=None,
                              past_key_values=past, use_cache=True, return_dict=True)
                else:
                    # eager 分支：原版 _greedy_generate 逻辑，增长 attention_mask
                    att = torch.cat((att, torch.ones((1, 1), dtype=att.dtype, device=att.device)), dim=1)
                    out = fwd(input_ids=nt[:, None], attention_mask=att,
                              past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                nt = out.logits[:, -1].float().argmax(dim=-1)
        return tok.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    def answer(rec):
        r = by_id[rec["sample_id"]]
        arr = np.stack(r.iq).astype(np.float32)
        # 直接用内存，不落盘（绕开 np.save IO 污染）
        value = np.ascontiguousarray(arr)
        scale = float(np.max(np.abs(value)))
        arr_norm = np.ascontiguousarray(value / scale)
        signal = torch.from_numpy(arr_norm)[None].to(dev, dtype=torch.bfloat16)
        rates = torch.tensor([float(r.sample_rate)], dtype=torch.float32, device=dev)
        prompt = _prompt(tok, LEMMO_RECOGNITION_SYSTEM_PROMPT, args.question, dev)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
            pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
            emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
        nt, past, att_out = prefill(emb, att)
        return greedy(nt, past, att_out)

    # warmup（compile 需触发）
    first = sample_set[0]
    _ = answer(first)
    torch.cuda.synchronize()
    print("[unified] warmup 完成", flush=True)

    torch.cuda.reset_peak_memory_stats()
    results = []
    t0 = time.time()
    for i, rec in enumerate(sample_set):
        ans = answer(rec)
        correct = label_in_answer(ans, rec["label"])
        results.append({"sample_id": rec["sample_id"], "label": rec["label"],
                        "prediction": ans, "correct": correct})
        if (i + 1) % 200 == 0 or (i + 1) == len(sample_set):
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"[{i+1}/{len(sample_set)}] acc={acc:.4f}", flush=True)

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
        "scheme": args.scheme,
        "sample_set": os.path.basename(args.sample_set),
        "total": len(results),
        "correct": sum(r["correct"] for r in results),
        "accuracy": acc,
        "elapsed_sec": elapsed,
        "avg_sec_per_sample": elapsed / len(results) if results else 0,
        "peak_mem_GB": mem_gb,
        "disk_full_GB": disk_full / 1e9 if disk_full else None,
        "disk_bf16_GB": disk_bf16 / 1e9 if disk_bf16 else None,
        "compression_ratio": compression,
        "n_merged": n_merged,
        "quant_time_s": quant_time,
        "unpack_time_s": unpack_time,
        "per_class": per_acc,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== [{args.scheme}] accuracy: {acc:.4f} ({summary['correct']}/{summary['total']}) ===")
    print(f"=== elapsed {elapsed:.1f}s, avg {elapsed/len(results) if results else 0:.4f}s/sample, "
          f"mem {mem_gb:.2f}GB, disk {disk_full/1e9 if disk_full else 0:.2f}GB, "
          f"compression {compression if compression else 0:.2f}x ===")
    print(f"=== saved to {args.output} ===")


if __name__ == "__main__":
    main()
