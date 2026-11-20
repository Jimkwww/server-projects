#!/usr/bin/env python
"""R3 全量精度评测：merge bf16 + torch.compile(decode) 跑官方 12 类 2212 条。

与 evaluate_merge.py 完全同口径（含 np.save 磁盘 IO、同采样 seed、同 label 判定），
唯一差异：decode 用 torch.compile(default) + StaticCache + attention_mask=None。

输出 accuracy / per_class / elapsed / avg_sec_per_sample / peak_mem / disk。

用法:
  python evaluate_compile.py \
    --parquets "/data1/user_wzk/LMO/data/interference/em_infer_comm_test.parquet,/data1/user_wzk/LMO/data/interference/owl_init_test.parquet,/data1/user_wzk/LMO/data/interference/gnss_jamming_test.parquet" \
    --config configs/LEMMO_Recognition.local.json \
    --device cuda:2 --output results/interference_full12_merge_compile.json
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

    c = 0
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                _replace(module, child_name, child)
                c += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquets", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--question", default="这段信号中存在哪种干扰？")
    ap.add_argument("--sample-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune-no-cudagraphs"])
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

    # 磁盘大小（同 evaluate_merge.py 口径）
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "qwen.pt")
        torch.save(qwen.state_dict(), p)
        disk_gb = os.path.getsize(p) / 1e9

    # 固定 max_cache_len，保证 StaticCache 形状固定 → compile 只编译一次 decode
    MAX_CACHE = 2048

    # torch.compile 只包 decode（qwen 整体 compile，但 prefill 用 eager qwen.__call__）
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

    def answer_compile(array, sr, question):
        signal = torch.from_numpy(array)[None].to(dev, dtype=torch.bfloat16)
        rates = torch.tensor([sr], dtype=torch.float32, device=dev)
        prompt = _prompt(tok, LEMMO_RECOGNITION_SYSTEM_PROMPT, question, dev)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
            pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
            emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
        nt, past = prefill(emb, att)
        return greedy(nt, past)

    # warmup：触发 decode 编译（用第一条样本）
    tmpdir = tempfile.mkdtemp(prefix="lmo_compile_")
    first = df.iloc[0]
    iq0 = np.stack(first["iq"]).astype(np.float32)
    p0 = os.path.join(tmpdir, "warmup.npy")
    np.save(p0, iq0)
    a0 = load_iq_local(p0)
    _ = answer_compile(a0, float(first["sample_rate"]), args.question)
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
        ans = answer_compile(load_iq_local(tmp_path), sr, args.question)
        results.append({
            "sample_id": sample_id, "label": label,
            "prediction": ans, "correct": label_in_answer(ans, label),
        })
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
        "quant": f"merge_compile_{args.compile_mode}",
        "total": len(results),
        "correct": sum(r["correct"] for r in results),
        "accuracy": acc,
        "elapsed_sec": elapsed,
        "avg_sec_per_sample": elapsed / len(results) if results else 0,
        "peak_mem_GB": mem_gb,
        "qwen_disk_GB": disk_gb,
        "per_class": per_acc,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== [merge_compile] accuracy: {acc:.4f} ({summary['correct']}/{summary['total']}) ===")
    print(f"=== elapsed {elapsed:.1f}s, avg {elapsed/len(results) if results else 0:.3f}s/sample, mem {mem_gb:.2f}GB, disk {disk_gb:.2f}GB ===")
    for l in OFFICIAL_12:
        if l in per_acc:
            p = per_acc[l]
            print(f"  {l:22s} {p['correct']:4d}/{p['total']:4d} = {p['acc']:.4f}")
    print(f"=== saved to {args.output} ===")


def load_iq_local(path):
    """同 lemmo.io.load_iq 的归一化逻辑，但内联避免额外 import。"""
    from lemmo.io import load_iq, LEMMO_RECOGNITION_LENGTHS
    return load_iq(path, LEMMO_RECOGNITION_LENGTHS)


if __name__ == "__main__":
    main()
