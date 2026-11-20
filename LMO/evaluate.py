#!/usr/bin/env python
"""LEMMO Q16 interference 批量评测脚本（基线 + 后续量化对照）。

直接复用官方 LEMMORecognitionRuntime，避免重复造轮子。

用法:
  python evaluate.py --parquet <test.parquet> \
      --config configs/LEMMO_Recognition.local.json \
      --device cuda:2 --max-new-tokens 64 \
      --output results/interference_baseline.json

流程:
  1. 读 parquet（列 iq=[16368,2], label_value, sample_rate）
  2. 建 LEMMORecognitionRuntime（一次加载 Qwen + LoRA + 信号模型）
  3. 逐条把 iq 存临时 .npy，调 runtime.answer() 贪心解码
  4. 宽松匹配真实标签 vs 答案
  5. 输出 json: 每条 {sample_id, label, prediction, correct} + 汇总 accuracy
"""
import argparse
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd


def label_in_answer(answer, label):
    """宽松匹配：类别名/别名出现在答案里就算对。

    测试集 6 类真实标签 → 常见中文/英文别名：
      Pulse_DME       → dme / 脉冲
      Narrowband      → 窄带
      FM_Jamming      → 调频
      Sweep_Chirp     → 扫频 / 线性调频 / chirp
      AM_Jamming      → 调幅
      No_Interference → 无干扰 / 没有干扰
    """
    aliases = {
        "Pulse_DME": ["dme", "脉冲"],
        "Narrowband": ["窄带", "narrowband"],
        "FM_Jamming": ["调频", "fm"],
        "Sweep_Chirp": ["扫频", "chirp", "线性调频"],
        "AM_Jamming": ["调幅", "am"],
        "No_Interference": ["无干扰", "没有干扰", "不存在干扰", "no interference"],
    }
    a = answer.lower()
    for kw in aliases.get(label, [label.lower()]):
        if kw.lower() in a:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--question", default="这段信号中存在哪种干扰？")
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    from lemmo import LEMMORecognitionRuntime

    df = pd.read_parquet(args.parquet)
    if args.limit:
        df = df.head(args.limit)
    print(f"total samples: {len(df)}", flush=True)

    runtime = LEMMORecognitionRuntime(args.config, args.device)
    print("runtime loaded", flush=True)

    results = []
    t0 = time.time()
    tmpdir = tempfile.mkdtemp(prefix="lmo_eval_")
    for i, row in df.iterrows():
        iq = np.stack(row["iq"]).astype(np.float32)  # [16368, 2]
        label = row["label_value"]
        sample_id = row["sample_id"]
        sr = float(row["sample_rate"])

        # runtime.answer 接受文件路径，存临时 npy
        tmp_path = os.path.join(tmpdir, f"{i}.npy")
        np.save(tmp_path, iq)

        ans = runtime.answer(
            tmp_path,
            sr,
            "interference",
            args.question,
            args.max_new_tokens,
        )
        correct = label_in_answer(ans, label)
        results.append({
            "sample_id": sample_id,
            "label": label,
            "prediction": ans,
            "correct": correct,
        })
        acc = sum(r["correct"] for r in results) / len(results)
        print(f"[{i+1}/{len(df)}] label={label:16s} correct={correct}  "
              f"acc={acc:.3f}  ans={ans!r}", flush=True)

    acc = sum(r["correct"] for r in results) / len(results)
    elapsed = time.time() - t0
    summary = {
        "total": len(results),
        "correct": sum(r["correct"] for r in results),
        "accuracy": acc,
        "elapsed_sec": elapsed,
        "avg_sec_per_sample": elapsed / len(results) if results else 0,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n=== accuracy: {acc:.4f} ({summary['correct']}/{summary['total']}) ===")
    print(f"=== elapsed: {elapsed:.1f}s, avg {elapsed/len(results) if results else 0:.2f}s/sample ===")
    print(f"=== saved to {args.output} ===")


if __name__ == "__main__":
    main()
