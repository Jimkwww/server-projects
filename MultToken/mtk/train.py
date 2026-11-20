"""Supervised from-scratch training for TokenClassifier.

Shares the exact training protocol of the frozen ``SwinE/pretrain_v1/emfm/train.py``
(same seed helpers, optimizer, cosine schedule, evaluation, checkpointing, data
split). The only difference is the model: tokenizer + global backbone + head
instead of the baseline patch-embedding encoder.
"""
import argparse
import contextlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import RML2018A, collate, sha256
from .model import TokenClassifier


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def autocast(device, precision):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" else contextlib.nullcontext())


def loader(ds, batch, workers, shuffle, generator):
    kwargs = {"multiprocessing_context": "spawn"} if workers else {}
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      collate_fn=collate, worker_init_fn=worker_seed, generator=generator,
                      pin_memory=torch.cuda.is_available(), **kwargs)


@torch.no_grad()
def evaluate(model, batches, device, precision, classes):
    model.eval()
    cm = torch.zeros(classes, classes, dtype=torch.long)
    snr_stats = {}
    loss_sum, count = 0., 0
    for batch in batches:
        x, mask, y = (batch[k].to(device) for k in ("iq", "valid_mask", "label"))
        with autocast(device, precision):
            logits = model(x, mask)
            loss = nn.functional.cross_entropy(logits, y, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation loss")
        pred = logits.argmax(-1).cpu()
        target = batch["label"]
        cm += torch.bincount(target * classes + pred, minlength=classes ** 2).reshape(classes, classes)
        for snr in batch["snr"].unique():
            select = batch["snr"] == snr
            key = str(float(snr))
            stat = snr_stats.setdefault(key, [0, 0])
            stat[0] += int((pred[select] == target[select]).sum())
            stat[1] += int(select.sum())
        loss_sum += float(loss)
        count += len(y)
    tp = cm.diag().float()
    f1 = 2 * tp / (cm.sum(0) + cm.sum(1)).clamp_min(1)
    return {"n": count, "loss": loss_sum / count, "oa": float(tp.sum() / count),
            "macro_f1": float(f1.mean()), "confusion_matrix": cm.tolist(),
            "per_snr": {k: {"correct": v[0], "n": v[1], "oa": v[0] / v[1]}
                        for k, v in sorted(snr_stats.items(), key=lambda kv: float(kv[0]))}}


def save_checkpoint(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True, help="New directory per run")
    p.add_argument("--mode", choices=["smoke", "train", "evaluate"], default="smoke")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--checkpoint", help="Own trusted checkpoint, required for evaluate")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.epochs is not None:
        cfg["training"]["epochs"] = a.epochs
    if a.batch_size is not None:
        cfg["training"]["batch_size"] = a.batch_size
    if a.mode == "smoke":
        cfg["training"]["epochs"] = 1
    t = cfg["training"]
    if t["epochs"] < 1 or t["batch_size"] < 1 or a.workers < 0:
        p.error("epochs/batch_size must be positive; workers nonnegative")
    if a.mode == "evaluate" and not a.checkpoint:
        p.error("evaluate needs --checkpoint")
    device = torch.device(a.device)
    if device.type not in ("cuda", "cpu"):
        p.error("Supports server CPU/CUDA only")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    precision = t.get("precision", "fp32")
    if precision not in ("fp32", "bf16"):
        p.error("precision must be fp32 or bf16")
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        p.error("bf16 requested but unsupported; use fp32")
    digest = sha256(a.manifest)
    if digest != cfg["data"]["manifest_sha256"]:
        raise ValueError("Manifest hash differs from frozen configuration")
    seed_all(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)

    model = TokenClassifier(cfg, cfg["classes"]).to(device)
    tk = model.tokenizer

    source_hashes = {str(f.relative_to(Path(__file__).parent)): sha256(f)
                     for f in sorted(Path(__file__).parent.rglob("*.py"))}
    metadata = {"config": cfg, "mode": a.mode, "seed": a.seed,
                "manifest_sha256": digest, "source_hashes": source_hashes,
                "h5_path": str(Path(a.h5).resolve()), "manifest_path": str(Path(a.manifest).resolve()),
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "python": platform.python_version(), "torch": torch.__version__,
                "workers": a.workers,
                "parameters": sum(v.numel() for v in model.parameters()),
                "tokenizer_parameters": sum(v.numel() for v in tk.parameters()),
                "backbone_parameters": sum(v.numel() for v in model.backbone.parameters()),
                "head_parameters": sum(v.numel() for v in model.head.parameters())}

    def dataset(subset, split):
        return RML2018A(a.h5, a.manifest, subset, split, cfg["data"]["normalization"])

    val_gen = torch.Generator().manual_seed(a.seed + 1)
    if a.mode == "evaluate":
        state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        if state["metadata"]["mode"] != "train":
            raise ValueError("Only cls_dev training checkpoints support heldout evaluation")
        old = state["metadata"]
        if any(old["config"][k] != cfg[k] for k in ("tokenizer", "backbone", "data", "classes")):
            raise ValueError("Checkpoint config mismatch")
        model.load_state_dict(state["model"])
        batches = loader(dataset("cls_dev", "val"), t["batch_size"], a.workers, False, val_gen)
        result = evaluate(model, batches, device, precision, cfg["classes"])
        dump(out / "metadata.json", metadata)
        dump(out / "val_metrics.json", result)
        print(json.dumps(result), flush=True)
        return
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
                  "lr_next": scheduler.get_last_lr()[0], "validation": val}
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
