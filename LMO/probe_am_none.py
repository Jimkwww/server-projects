#!/usr/bin/env python
"""最小复现：decode attention_mask=None 是否报错"""
import torch, numpy as np, pandas as pd
from lemmo import LEMMORecognitionRuntime
from lemmo.runtime import _prompt, _insert_signal, LEMMO_RECOGNITION_SYSTEM_PROMPT
from lemmo.lora import LoRALinear
import torch.nn as nn

runtime = LEMMORecognitionRuntime("configs/LEMMO_Recognition.local.json", "cuda:0")
qwen = runtime.qwen; model = runtime.model; tok = runtime.tokenizer; dev = runtime.device

def merge_lora(model):
    def _r(p, c, l):
        m = l.base.weight.data + l.scaling * (l.lora_B.weight.data @ l.lora_A.weight.data)
        m = m.to(l.base.weight.dtype)
        nl = nn.Linear(l.base.in_features, l.base.out_features, bias=l.base.bias is not None)
        nl = nl.to(device=l.base.weight.device, dtype=l.base.weight.dtype)
        nl.weight.data.copy_(m)
        if l.base.bias is not None:
            nl.bias.data.copy_(l.base.bias.data.to(l.base.weight.dtype))
        setattr(p, c, nl)
    for mn, mod in list(model.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, LoRALinear):
                _r(mod, cn, ch)

merge_lora(qwen)
print("merged")

df = pd.read_parquet("/data1/user_wzk/LMO/data/interference/gnss_jamming_test.parquet")
row = df.iloc[0]
arr = np.stack(row["iq"]).astype(np.float32)
sc = float(np.max(np.abs(arr)))
arr = np.ascontiguousarray(arr / sc)
signal = torch.from_numpy(arr)[None].to(dev, dtype=torch.bfloat16)
rates = torch.tensor([float(row["sample_rate"])], dtype=torch.float32, device=dev)
prompt = _prompt(tok, LEMMO_RECOGNITION_SYSTEM_PROMPT, "这段信号中存在哪种干扰？", dev)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    q_emb = qwen.get_input_embeddings()(prompt["question_ids"])
    pr = model(signal, rates, q_emb, prompt["question_mask"], "interference")
    emb, att = _insert_signal(qwen.get_input_embeddings(), prompt, pr)
    print("emb shape:", emb.shape, "att shape:", att.shape, flush=True)
    from transformers.cache_utils import StaticCache
    cache = StaticCache(config=qwen.config, max_cache_len=att.shape[1] + 128)
    out = qwen(inputs_embeds=emb, attention_mask=att, past_key_values=cache, use_cache=True, return_dict=True)
    print("prefill ok, seq_len=", cache.get_seq_length(), flush=True)
    nt = out.logits[:, -1].float().argmax(dim=-1)
    print("input_ids shape:", nt[:, None].shape, flush=True)
    out2 = qwen(input_ids=nt[:, None], attention_mask=None, past_key_values=cache, use_cache=True, return_dict=True)
    print("decode am_none ok", flush=True)
