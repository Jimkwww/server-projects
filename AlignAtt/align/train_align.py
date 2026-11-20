"""Supervised training for the relation-augmented global encoder (A0-A6).

Shares the exact training protocol of the frozen SwinE Global baseline
(emfm/train.py helpers: same optimizer, cosine schedule, seed, data split,
evaluation) so every variant is directly comparable with the fixed baseline
(from-scratch 0.3835 +/- 0.0034, 3 seeds).

The ONLY difference is the model: RelationGlobalEncoder / AlignClassifier with a
configurable ``relation`` block. ``relation.mode='none'`` reproduces the baseline
through the new code path (verified by align/test_correlation.py).
"""
import argparse
import json
import math
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from emfm.data import RML2018A, collate, sha256
from emfm.train import seed_all, dump, autocast, loader, evaluate, save_checkpoint
from .align_models import AlignClassifier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True, help="New directory per run")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.epochs is not None:
        cfg["training"]["epochs"] = a.epochs
    if a.batch_size is not None:
        cfg["training"]["batch_size"] = a.batch_size
    t = cfg["training"]
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    precision = t.get("precision", "fp32")
    digest = sha256(a.manifest)
    if digest != cfg["data"]["manifest_sha256"]:
        raise ValueError("Manifest hash differs from frozen configuration")
    seed_all(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)

    model = AlignClassifier(cfg["model"], cfg["classes"]).to(device)

    source_hashes = {str(f.relative_to(Path(__file__).parent)): sha256(f)
                     for f in sorted(Path(__file__).parent.rglob("*.py"))}
    rel_cfg = cfg["model"].get("relation", {})
    metadata = {"config": cfg, "mode": "train", "seed": a.seed,
                "manifest_sha256": digest, "source_hashes": source_hashes,
                "h5_path": str(Path(a.h5).resolve()), "manifest_path": str(Path(a.manifest).resolve()),
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "python": platform.python_version(), "torch": torch.__version__,
                "workers": a.workers,
                "align_relation": rel_cfg,
                "parameters": sum(v.numel() for v in model.parameters()),
                "encoder_parameters": sum(v.numel() for v in model.encoder.parameters()),
                "relation_parameters": model.relation_param_count(),
                "head_parameters": sum(v.numel() for v in model.head.parameters())}

    def dataset(subset, split):
        return RML2018A(a.h5, a.manifest, subset, split, cfg["data"]["normalization"])

    val_gen = torch.Generator().manual_seed(a.seed + 1)
    train_ds = dataset("cls_dev", "train")
    val_ds = dataset("cls_dev", "val")
    generator = torch.Generator().manual_seed(a.seed)
    train_dl = loader(train_ds, t["batch_size"], a.workers, True, generator)
    val_dl = loader(val_ds, t["batch_size"], a.workers, False, val_gen)

    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    total = t["epochs"] * len(train_dl)
    warmup = max(1, int(total * t["warmup_fraction"]))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    dump(out / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    best = -1.
    step = 0
    for epoch in range(t["epochs"]):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        count, correct, loss_sum = 0, 0, 0.
        for batch in train_dl:
            x, mask, y = (batch[k].to(device) for k in ("iq", "valid_mask", "label"))
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, precision):
                logits = model(x, mask)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            count += len(y)
            loss_sum += float(loss.detach()) * len(y)
            correct += int((logits.argmax(-1) == y).sum())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - begin
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        val = evaluate(model, val_dl, device, precision, cfg["classes"])
        improved = val["oa"] > best
        best = max(best, val["oa"])
        record = {"epoch": epoch, "step": step, "train_loss": loss_sum / count,
                  "train_oa": correct / count, "train_seconds": seconds,
                  "train_samples_s": count / seconds, "peak_train_allocated_bytes": peak,
                  "lr_next": scheduler.get_last_lr()[0], "validation": val,
                  "diagnostic_same_set": False}
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "validation"}) +
              f" val_loss={val['loss']:.5f} val_oa={val['oa']:.5f}", flush=True)
        state = {"metadata": metadata, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch, "best": best, "step": step,
                 "loader_rng": generator.get_state(), "val_rng": val_gen.get_state(),
                 "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                 "python_rng": random.getstate(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        save_checkpoint(out / "last.pt", state)
        if improved:
            save_checkpoint(out / "best.pt", state)
            dump(out / "best_metrics.json", {"epoch": epoch, **val})
    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
