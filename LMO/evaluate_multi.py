#!/usr/bin/env python
"""LEMMO Q16 interference 全口径评测（官方 12 类，多 parquet 采样）。

用法:
  python evaluate_multi.py \
      --parquets /data1/.../gnss_jamming_test.parquet,/tmp/em_infer_comm_test.parquet,/tmp/owl_init_test.parquet \
      --config configs/LEMMO_Recognition.local.json \
      --device cuda:2 --max-new-tokens 64 \
      --sample-per-class 200 \
      --output results/interference_full12_baseline.json

口径:
  - 官方 12 类来自 checkpoint label_maps（Protocol_ZigBee 是 OOV，排除）
  - 每类最多 --sample-per-class 条（0=全部）
  - 逐条写临时 npy 调 runtime.answer()
"""
import argparse
import json
import os
import tempfile
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquets", required=True, help="comma-separated paths")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--question", default="这段信号中存在哪种干扰？")
    ap.add_argument("--sample-per-class", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    from lemmo import LEMMORecognitionRuntime

    paths = [p.strip() for p in args.parquets.split(",") if p.strip()]
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        print(f"loaded {p}: {len(df)} rows", flush=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # 只保留官方 12 类
    before = len(df)
    df = df[df["label_value"].isin(OFFICIAL_12)]
    print(f"filter to official 12: {before} -> {len(df)} rows", flush=True)

    # 每类采样
    if args.sample_per_class:
        sampled = []
        for lab in OFFICIAL_12:
            sub = df[df["label_value"] == lab]
            n = min(len(sub), args.sample_per_class)
            sampled.append(sub.sample(n=n, random_state=42))
        df = pd.concat(sampled, ignore_index=True)
        print(f"after per-class sampling (<= {args.sample_per_class}): {len(df)} rows", flush=True)

    dist = Counter(df["label_value"])
    print("sample distribution:", dict(sorted(dist.items())), flush=True)

    runtime = LEMMORecognitionRuntime(args.config, args.device)
    print("runtime loaded", flush=True)

    results = []
    t0 = time.time()
    tmpdir = tempfile.mkdtemp(prefix="lmo_full12_")
    for i, row in df.iterrows():
        iq = np.stack(row["iq"]).astype(np.float32)
        label = row["label_value"]
        sample_id = row["sample_id"]
        sr = float(row["sample_rate"])

        tmp_path = os.path.join(tmpdir, f"{i}.npy")
        np.save(tmp_path, iq)

        ans = runtime.answer(tmp_path, sr, "interference", args.question, args.max_new_tokens)
        correct = label_in_answer(ans, label)
        results.append({
            "sample_id": sample_id,
            "label": label,
            "prediction": ans,
            "correct": correct,
            "sample_rate": sr,
            "length": len(iq),
        })
        if (i + 1) % 200 == 0 or (i + 1) == len(df):
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"[{i+1}/{len(df)}] acc={acc:.4f}", flush=True)

    acc = sum(r["correct"] for r in results) / len(results)
    elapsed = time.time() - t0

    # 每类正确率
    per = defaultdict(lambda: [0, 0])
    for r in results:
        per[r["label"]][1] += 1
        if r["correct"]:
            per[r["label"]][0] += 1
    per_acc = {l: {"correct": c, "total": t, "acc": c / t if t else 0}
               for l, (c, t) in sorted(per.items())}

    summary = {
        "total": len(results),
        "correct": sum(r["correct"] for r in results),
        "accuracy": acc,
        "elapsed_sec": elapsed,
        "avg_sec_per_sample": elapsed / len(results) if results else 0,
        "per_class": per_acc,
        "sample_per_class": args.sample_per_class,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== accuracy: {acc:.4f} ({summary['correct']}/{summary['total']}) ===")
    print(f"=== elapsed: {elapsed:.1f}s, avg {elapsed/len(results) if results else 0:.3f}s/sample ===")
    print("=== per-class ===")
    for l in OFFICIAL_12:
        if l in per_acc:
            p = per_acc[l]
            print(f"  {l:22s} {p['correct']:4d}/{p['total']:4d} = {p['acc']:.4f}")
    print(f"=== saved to {args.output} ===")


if __name__ == "__main__":
    main()
