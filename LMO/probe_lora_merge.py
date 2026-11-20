#!/usr/bin/env python
"""P0-2 排查 LoRA 合并精度变化：首 token logits + 后续序列 logits 对比。

核心问题：LoRA 合并（online LoRA 分支）在数学上应等价，但有限精度下可能不同。
本脚本量化这种差异到底有多大，并判断 +1.9pp 精度提升的真实来源。

方法：同一批固定样本，先跑 online LoRA（未合并），记录完整 logits 序列；
再 merge，跑同一批样本，记录 logits；对比：
  - 首 token logits 的 max-abs-diff 与 argmax 是否翻转
  - 后续各步 logits 的 max-abs-diff
  - 完整预测文本是否一致

顺序不可逆（merge 就地替换 LoRALinear），所以先 online 后 merge。

用法:
  python probe_lora_merge.py --config configs/LEMMO_Recognition.local.json \
      --device cuda:2 --n-samples 30
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lemmo.lora import LoRALinear

PARSQUETS = [
    "/data1/user_wzk/LMO/data/interference/em_infer_comm_test.parquet",
    "/data1/user_wzk/LMO/data/interference/owl_init_test.parquet",
    "/data1/user_wzk/LMO/data/interference/gnss_jamming_test.parquet",
]


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
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--question", default="这段信号中存在哪种干扰？")
    args = ap.parse_args()

    from lemmo import LEMMORecognitionRuntime
    from lemmo.runtime import _prompt, _insert_signal, LEMMO_RECOGNITION_SYSTEM_PROMPT

    # 固定样本集：从 sample_set_2212.json 取前 n 条（跨类均匀取会更好，这里按文件顺序）
    with open("results/sample_set_2212.json") as f:
        sample_set = json.load(f)

    df = pd.concat([pd.read_parquet(p) for p in PARSQUETS], ignore_index=True)
    by_id = {r.sample_id: r for r in df.itertuples()}

    # 跨 12 类均匀取 n 条，避免只取到单一长度
    from collections import OrderedDict
    picked = OrderedDict()
    for rec in sample_set:
        if rec["sample_id"] in picked:
            continue
        if len(picked) >= args.n_samples:
            break
        picked[rec["sample_id"]] = rec
    samples = list(picked.values())
    print(f"[probe] 取 {len(samples)} 条样本", flush=True)

    runtime = LEMMORecognitionRuntime(args.config, args.device)
    dev = runtime.device
    tok = runtime.tokenizer
    qwen = runtime.qwen
    model = runtime.model
    eos = int(tok.eos_token_id)

    # 预加载所有样本数据到内存
    loaded = []
    for rec in samples:
        r = by_id[rec["sample_id"]]
        arr = load_iq_mem(np.stack(r.iq).astype(np.float32))
        loaded.append((rec, arr, float(r.sample_rate)))
    print(f"[probe] 样本加载完成", flush=True)

    # 通用 forward：返回 (pred_text, logits_list[每步 (1,V) float32])
    def run_all():
        out_records = []
        for rec, arr, sr in loaded:
            signal = torch.from_numpy(arr)[None].to(dev, dtype=torch.bfloat16)
            rates = torch.tensor([sr], dtype=torch.float32, device=dev)
            prompt = _prompt(tok, LEMMO_RECOGNITION_SYSTEM_PROMPT, args.question, dev)
            logits_list = []
            gen = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
                pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
                emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
                out = qwen(inputs_embeds=emb, attention_mask=att, use_cache=True, return_dict=True)
                past = out.past_key_values
                logits = out.logits[:, -1].float()
                logits_list.append(logits.cpu())
                nt = logits.argmax(dim=-1)
                for _ in range(args.max_new_tokens):
                    t = int(nt.item())
                    if t == eos:
                        break
                    gen.append(t)
                    out = qwen(input_ids=nt[:, None], attention_mask=None,
                               past_key_values=past, use_cache=True, return_dict=True)
                    past = out.past_key_values
                    logits = out.logits[:, -1].float()
                    logits_list.append(logits.cpu())
                    nt = logits.argmax(dim=-1)
            pred = tok.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            out_records.append({"sample_id": rec["sample_id"], "label": rec["label"],
                                "pred": pred, "logits": logits_list})
        return out_records

    # 1) online LoRA（未合并）
    print("[probe] 跑 online LoRA ...", flush=True)
    online = run_all()

    # 2) merge
    n = merge_lora(qwen)
    print(f"[probe] LoRA 合并 {n} 层，跑 merge ...", flush=True)
    merged = run_all()

    # 3) 对比
    print("\n===== 对比结果 =====", flush=True)
    n_same_pred = 0
    n_first_flip = 0
    max_first_diff = 0.0
    max_all_diff = 0.0
    for o, m in zip(online, merged):
        assert o["sample_id"] == m["sample_id"]
        same = o["pred"] == m["pred"]
        if same:
            n_same_pred += 1
        # 首 token logits
        d_first = float((o["logits"][0] - m["logits"][0]).abs().max())
        max_first_diff = max(max_first_diff, d_first)
        o_arg = int(o["logits"][0].argmax())
        m_arg = int(m["logits"][0].argmax())
        if o_arg != m_arg:
            n_first_flip += 1
        # 全部 logits 逐步（长度可能不同，取 min）
        n_steps = min(len(o["logits"]), len(m["logits"]))
        for s in range(n_steps):
            d = float((o["logits"][s] - m["logits"][s]).abs().max())
            max_all_diff = max(max_all_diff, d)
    print(f"预测文本一致: {n_same_pred}/{len(samples)}")
    print(f"首 token argmax 翻转: {n_first_flip}/{len(samples)}")
    print(f"首 token logits max-abs-diff: {max_first_diff:.6e}")
    print(f"全部步骤 logits max-abs-diff: {max_all_diff:.6e}")
    # 输出几个不一致的样例
    for o, m in zip(online, merged):
        if o["pred"] != m["pred"]:
            print(f"  [不一致] {o['sample_id']} label={o['label']}")
            print(f"    online: {o['pred']!r}")
            print(f"    merged: {m['pred']!r}")


if __name__ == "__main__":
    main()
