"""Stable-failure-condition diagnosis for MultToken T0 vs T2.

Evaluates the six best checkpoints (T0/T2 x seed {17,42,2026}) selected by
val OA, on the frozen cls_dev val split (fixed sample order), first reproducing
the baseline predictions and then under two families of *whole-signal* unitary
perturbations applied to the RAW I/Q before the exact same max_abs
preprocessing:

  * global phase rotation   x[n] * exp(j*phi),  phi in {0,45,...,315} deg
  * small additive freq. offset  x[n] * exp(j*2*pi*delta*n),
    delta = k / N,  k in {0, +-0.25, +-0.5, +-1, +-2},  N = 1024

Both perturbations are unit-modulus so the complex L2 energy is preserved, but
the per-component max_abs normalization scale is NOT rotation-invariant; the
perturbed raw signal is re-normalized with the identical formula used in
training (max abs component). Zero phase / zero offset feed the exact baseline
normalized input, so they reproduce the original predictions bitwise.

Outputs (all under failure_diagnosis/):
  failure_diagnosis_protocol.md   (written by the agent after this run)
  baseline_predictions.csv / .npz
  perturbation_predictions/*.npz / .csv
  failure_summary.csv
  perturbation_per_snr.csv / perturbation_per_class.csv
  charts/*.png
  analysis_outputs.json           (machine-readable aggregates for the report)
"""
import os
import sys
import json
import math
import hashlib
from pathlib import Path

import numpy as np
import torch

PROJ = Path("/home/user_wzk/wzk/projects/MultToken")
sys.path.insert(0, str(PROJ))

from mtk.model import TokenClassifier   # noqa: E402
from mtk.data import RML2018A, sha256   # noqa: E402

# ---------------------------------------------------------------------------
# Frozen inputs (matches baseline_audit.md / results.csv)
# ---------------------------------------------------------------------------
H5 = "/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5"
MANIFEST = "/home/user_wzk/wzk/projects/SwinE/manifest/rml2018a_manifest.csv"
N = 1024

OUT = PROJ / "failure_diagnosis"
PERTS_OUT = OUT / "perturbation_predictions"
CHARTS = OUT / "charts"
for d in (OUT, PERTS_OUT, CHARTS):
    os.makedirs(d, exist_ok=True)

# (family, tokenizer_name, seed, run_dir)
MODELS = [
    ("T0", "t0", 17,   "sup_t0_s17"),
    ("T0", "t0", 42,   "p2_t0_s42"),
    ("T0", "t0", 2026, "p2_t0_s2026"),
    ("T2", "t2", 17,   "sup_t2_s17"),
    ("T2", "t2", 42,   "p2_t2_s42"),
    ("T2", "t2", 2026, "p2_t2_s2026"),
]

PHASES = [0, 45, 90, 135, 180, 225, 270, 315]          # degrees
OFFSET_FACTORS = [0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]  # x 1/N

PERTURBATIONS = [("phase", float(p)) for p in PHASES] + \
                [("offset", float(k)) for k in OFFSET_FACTORS]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH = 128


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Data loading: ordered records + raw (un-normalized) + normalized I/Q
# ---------------------------------------------------------------------------
def load_val():
    ds = RML2018A(H5, MANIFEST, "cls_dev", "val", "max_abs")
    n = len(ds)
    ids = np.empty(n, dtype=np.int64)
    labels = np.empty(n, dtype=np.int64)
    snrs = np.empty(n, dtype=np.float32)
    norm = np.empty((n, N, 2), dtype=np.float32)
    raw = np.empty((n, N, 2), dtype=np.float32)
    scales = np.empty(n, dtype=np.float32)
    for i in range(n):
        r = ds[i]
        ids[i] = r["sample_id"]
        labels[i] = r["label"]
        snrs[i] = r["snr"]
        nq = r["iq"].numpy()                      # float32 [N,2]
        norm[i] = nq
        scales[i] = r["scale"]
        raw[i] = (nq * r["scale"]).astype(np.float32)
    ds.close()
    return ids, labels, snrs, norm, raw, scales


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_models():
    models = []
    for family, tk_name, seed, run_dir in MODELS:
        ckpt = PROJ / "runs" / run_dir / "best.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = state["metadata"]["config"]
        assert state["metadata"]["mode"] == "train"
        assert cfg["tokenizer"]["name"] == tk_name
        model = TokenClassifier(cfg, cfg["classes"])
        model.load_state_dict(state["model"])
        model.to(DEVICE).eval()
        models.append(dict(family=family, seed=seed, run_dir=run_dir,
                           ckpt=str(ckpt), cfg=cfg, model=model))
    return models


# ---------------------------------------------------------------------------
# Perturbation on RAW I/Q (float64 math, cast back to float32)
# ---------------------------------------------------------------------------
def apply_phase(raw, deg):
    z = raw[..., 0].astype(np.float64) + 1j * raw[..., 1].astype(np.float64)
    z = z * np.exp(1j * math.radians(deg))
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def apply_offset(raw, factor):
    delta = factor / N
    n = np.arange(N)
    ph = (2 * np.pi * delta * n).astype(np.float64)
    z = raw[..., 0].astype(np.float64) + 1j * raw[..., 1].astype(np.float64)
    z = z * np.exp(1j * ph)
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def normalize(x32):
    """Exact training formula: scale = max abs component (floor 1e-8)."""
    scale = max(float(np.abs(x32).max()), 1e-8)
    return (x32 / scale).astype(np.float32), scale


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model, x_norm):
    """x_norm [n,N,2] float32 -> logits [n,C] float32 on cpu."""
    n = x_norm.shape[0]
    logits = []
    for b in range(0, n, BATCH):
        xb = torch.from_numpy(x_norm[b:b + BATCH]).to(DEVICE)
        valid = torch.ones(xb.shape[0], xb.shape[1], dtype=torch.bool, device=DEVICE)
        logits.append(model(xb, valid).cpu())
    return torch.cat(logits, dim=0).numpy()


def softmax_rows(x):
    e = x - x.max(axis=1, keepdims=True)
    e = np.exp(e)
    return e / e.sum(axis=1, keepdims=True)


def confusion_matrix(pred, label, C=24):
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (label, pred), 1)
    return cm


def macro_f1(cm):
    tp = np.diag(cm).astype(np.float64)
    denom = np.maximum((cm.sum(0) + cm.sum(1)).astype(np.float64), 1.0)
    return float((2 * tp / denom).mean()), (2 * tp / denom)


def per_snr_oa(pred, label, snr):
    snrs = np.unique(snr)
    out = {}
    for s in snrs:
        sel = snr == s
        out[str(float(s))] = dict(n=int(sel.sum()),
                                  correct=int((pred[sel] == label[sel]).sum()),
                                  oa=float((pred[sel] == label[sel]).mean()))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading val data ...", flush=True)
    ids, labels, snrs, norm, raw, scales = load_val()
    ns = len(labels)
    print(f"val n={ns}, classes={labels.max()+1}, "
          f"snr range=[{snrs.min()},{snrs.max()}], unique snr={len(np.unique(snrs))}",
          flush=True)

    # --- class x snr = 5 check --------------------------------------------
    per_cell = {}
    for c, s in zip(labels, snrs):
        per_cell[(int(c), float(s))] = per_cell.get((int(c), float(s)), 0) + 1
    cell_counts = np.array(list(per_cell.values()))
    print(f"cells={len(per_cell)}, samples/cell min={cell_counts.min()} "
          f"max={cell_counts.max()} mean={cell_counts.mean():.2f}", flush=True)

    models = load_models()
    nm = len(models)
    print(f"Loaded {nm} models on {DEVICE}", flush=True)

    # ------------------------------------------------------------------
    # Step 1: reproduce baseline predictions (exact normalized input)
    # ------------------------------------------------------------------
    base_probs = np.zeros((nm, ns, 24), dtype=np.float32)
    base_logits = np.zeros((nm, ns, 24), dtype=np.float32)
    base_metrics = []
    for mi, m in enumerate(models):
        lg = predict_logits(m["model"], norm)
        base_logits[mi] = lg
        pr = softmax_rows(lg)
        base_probs[mi] = pr.astype(np.float32)
        pred = lg.argmax(1)
        cm = confusion_matrix(pred, labels)
        f1, _ = macro_f1(cm)
        oa = float((pred == labels).mean())
        mf1_reported = float(0)  # filled below from best_metrics
        # loss
        ce = -np.log(np.clip(pr[np.arange(ns), labels], 1e-12, 1.0))
        base_metrics.append(dict(family=m["family"], seed=m["seed"], oa=oa,
                                 macro_f1=f1, loss=float(ce.mean())))
        print(f"  {m['family']} s{m['seed']}: OA={oa:.8f} macroF1={f1:.8f} "
              f"loss={ce.mean():.6f}", flush=True)

    # compare vs best_metrics.json
    print("\n--- baseline reproduction vs best_metrics.json ---", flush=True)
    repro_ok = True
    for mi, m in enumerate(models):
        bm = json.loads((PROJ / "runs" / m["run_dir"] / "best_metrics.json").read_text())
        oa_ref, f1_ref = bm["oa"], bm["macro_f1"]
        d_oa = abs(base_metrics[mi]["oa"] - oa_ref)
        d_f1 = abs(base_metrics[mi]["macro_f1"] - f1_ref)
        flag = "OK" if d_oa < 1e-6 and d_f1 < 1e-6 else "MISMATCH"
        if flag != "OK":
            repro_ok = False
        print(f"  {m['family']} s{m['seed']}: dOA={d_oa:.3e} dF1={d_f1:.3e} {flag}",
              flush=True)
    print(f"reproduction {'PASS' if repro_ok else 'FAIL'}", flush=True)

    # --- save baseline per-sample -------------------------------------------------
    col_models = [f"{m['family']}_s{m['seed']}" for m in models]
    with open(OUT / "baseline_predictions.csv", "w") as f:
        f.write("sample_id,class_idx,snr,model,seed,checkpoint,pred,"
                "prob_true,ce_loss,correct\n")
        for mi, m in enumerate(models):
            lg = base_logits[mi]
            pr = base_probs[mi]
            pred = lg.argmax(1)
            ce = -np.log(np.clip(pr[np.arange(ns), labels], 1e-12, 1.0))
            for i in range(ns):
                f.write(f"{ids[i]},{labels[i]},{snrs[i]:.1f},{m['family']},"
                        f"{m['seed']},{m['ckpt']},{pred[i]},"
                        f"{pr[i, labels[i]]:.6f},{ce[i]:.6f},"
                        f"{int(pred[i] == labels[i])}\n")
    np.savez(OUT / "baseline_probs.npz",
             probs=base_probs.astype(np.float16), logits=base_logits,
             models=np.array(col_models), sample_id=ids,
             class_idx=labels, snr=snrs)
    # baseline confusion matrices + per-class + per-snr
    cm_all = np.zeros((nm, 24, 24), dtype=np.int64)
    for mi, m in enumerate(models):
        cm_all[mi] = confusion_matrix(base_logits[mi].argmax(1), labels)
    np.save(OUT / "baseline_confusion_matrices.npy", cm_all)
    with open(OUT / "baseline_per_class.csv", "w") as f:
        f.write("model,seed,class_idx,precision,recall,f1,support\n")
        for mi, m in enumerate(models):
            cm = cm_all[mi]
            for c in range(24):
                tp = cm[c, c]
                fp = cm[:, c].sum() - tp
                fn = cm[c, :].sum() - tp
                p = tp / (tp + fp) if (tp + fp) else 0.0
                r = tp / (tp + fn) if (tp + fn) else 0.0
                f1c = 2 * p * r / (p + r) if (p + r) else 0.0
                f.write(f"{m['family']},{m['seed']},{c},{p:.6f},{r:.6f},"
                        f"{f1c:.6f},{cm[c, :].sum()}\n")
    with open(OUT / "baseline_per_snr.csv", "w") as f:
        f.write("model,seed,snr,correct,n,oa\n")
        for mi, m in enumerate(models):
            ps = per_snr_oa(base_logits[mi].argmax(1), labels, snrs)
            for s in sorted(ps, key=float):
                f.write(f"{m['family']},{m['seed']},{s},{ps[s]['correct']},"
                        f"{ps[s]['n']},{ps[s]['oa']:.6f}\n")

    # ------------------------------------------------------------------
    # Steps 2-4: perturbations (raw -> perturb -> max_abs re-normalize)
    # ------------------------------------------------------------------
    npert = len(PERTURBATIONS)
    pert_probs = np.zeros((nm, npert, ns, 24), dtype=np.float16)
    pert_logits = np.zeros((nm, npert, ns, 24), dtype=np.float32)
    pert_scales = np.zeros((nm, npert), dtype=np.float32)  # unused, scales per sample below
    scale_ratio = np.zeros((npert, ns), dtype=np.float32)

    # baseline correct flags per model
    base_correct = np.stack([base_logits[mi].argmax(1) == labels for mi in range(nm)])

    # zero perturbation should reproduce exactly -> feed norm
    zero_phase_idx = PERTURBATIONS.index(("phase", 0.0))
    zero_offset_idx = PERTURBATIONS.index(("offset", 0.0))

    summary_rows = []
    per_snr_rows = []
    per_class_rows = []

    for pi, (ptype, pval) in enumerate(PERTURBATIONS):
        # build perturbed normalized input
        if (ptype == "phase" and pval == 0.0) or (ptype == "offset" and pval == 0.0):
            xp = norm  # exact baseline
            sratio = np.ones(ns, dtype=np.float32)
        else:
            if ptype == "phase":
                raw_p = apply_phase(raw, pval)
            else:
                raw_p = apply_offset(raw, pval)
            xp = np.empty_like(norm)
            for i in range(ns):
                xn, sc = normalize(raw_p[i])
                xp[i] = xn
                sratio[i] = sc / scales[i]
        scale_ratio[pi] = sratio

        for mi, m in enumerate(models):
            lg = predict_logits(m["model"], xp)
            pert_logits[mi, pi] = lg
            pr = softmax_rows(lg)
            pert_probs[mi, pi] = pr.astype(np.float16)
            pred = lg.argmax(1)
            ce = -np.log(np.clip(pr[np.arange(ns), labels], 1e-12, 1.0))
            cm = confusion_matrix(pred, labels)
            f1, _ = macro_f1(cm)
            oa = float((pred == labels).mean())
            bc = base_correct[mi]
            pc = pred == labels
            orig_ok_now_bad = int((bc & ~pc).sum())
            orig_bad_now_ok = int((~bc & pc).sum())
            changed = int((pred != base_logits[mi].argmax(1)).sum())
            base_pr = base_probs[mi][np.arange(ns), labels]
            base_ce = -np.log(np.clip(base_pr, 1e-12, 1.0))
            summary_rows.append(dict(
                family=m["family"], seed=m["seed"], pert_type=ptype,
                pert_value=pval, oa=oa, macro_f1=f1,
                oa_delta=oa - base_metrics[mi]["oa"],
                f1_delta=f1 - base_metrics[mi]["macro_f1"],
                loss=float(ce.mean()), loss_delta=float((ce - base_ce).mean()),
                prob_true=float(pr[np.arange(ns), labels].mean()),
                prob_true_delta=float((pr[np.arange(ns), labels] - base_pr).mean()),
                orig_correct_to_wrong=orig_ok_now_bad,
                orig_correct_to_wrong_ratio=(orig_ok_now_bad / max(1, int(bc.sum()))),
                orig_wrong_to_correct=orig_bad_now_ok,
                orig_wrong_to_correct_ratio=(orig_bad_now_ok / max(1, int((~bc).sum()))),
                pred_change=changed,
                pred_change_rate=changed / ns,
                low_snr_oa=float((pred[snrs <= 0] == labels[snrs <= 0]).mean()),
                high_snr_oa=float((pred[snrs > 0] == labels[snrs > 0]).mean()),
            ))
            # per-snr
            ps = per_snr_oa(pred, labels, snrs)
            for s in sorted(ps, key=float):
                per_snr_rows.append(dict(family=m["family"], seed=m["seed"],
                                         pert_type=ptype, pert_value=pval,
                                         snr=s, **ps[s]))
            # per-class f1
            _, f1c = macro_f1(cm)
            for c in range(24):
                tp = cm[c, c]
                fp = cm[:, c].sum() - tp
                fn = cm[c, :].sum() - tp
                p = tp / (tp + fp) if (tp + fp) else 0.0
                r = tp / (tp + fn) if (tp + fn) else 0.0
                f1_cc = 2 * p * r / (p + r) if (p + r) else 0.0
                per_class_rows.append(dict(family=m["family"], seed=m["seed"],
                                           pert_type=ptype, pert_value=pval,
                                           class_idx=c, f1=f1_cc,
                                           macro_f1_component=float(f1c[c]),
                                           support=int(cm[c, :].sum())))
        print(f"  done {ptype} {pval} ({pi+1}/{npert})", flush=True)

    # verify zero perturbations reproduced baseline exactly
    print("\n--- zero perturbation reproduction check ---", flush=True)
    for mi, m in enumerate(models):
        for pi in (zero_phase_idx, zero_offset_idx):
            d = np.abs(pert_logits[mi, pi] - base_logits[mi]).max()
            print(f"  {m['family']} s{m['seed']} "
                  f"{PERTURBATIONS[pi]}: max|dlogit|={d:.3e}", flush=True)

    # ------------------------------------------------------------------
    # Save perturbation outputs
    # ------------------------------------------------------------------
    np.savez(PERTS_OUT / "perturbation_probs.npz",
             probs=pert_probs, logits=pert_logits,
             models=np.array(col_models), sample_id=ids, class_idx=labels, snr=snrs,
             pert_type=np.array([p[0] for p in PERTURBATIONS]),
             pert_value=np.array([p[1] for p in PERTURBATIONS]))
    np.save(PERTS_OUT / "perturbation_scale_ratio.npy", scale_ratio)

    # per-sample perturbation summary csv (compact: pred, prob_true, loss, correct)
    with open(PERTS_OUT / "perturbation_summary.csv", "w") as f:
        f.write("sample_id,class_idx,snr,model,seed,pert_type,pert_value,"
                "pred,prob_true,ce_loss,correct,pred_changed\n")
        for mi, m in enumerate(models):
            base_pred = base_logits[mi].argmax(1)
            for pi, (ptype, pval) in enumerate(PERTURBATIONS):
                lg = pert_logits[mi, pi]
                pr = pert_probs[mi, pi].astype(np.float32)
                pred = lg.argmax(1)
                ce = -np.log(np.clip(pr[np.arange(ns), labels], 1e-12, 1.0))
                for i in range(ns):
                    f.write(f"{ids[i]},{labels[i]},{snrs[i]:.1f},{m['family']},"
                            f"{m['seed']},{ptype},{pval},{pred[i]},"
                            f"{pr[i, labels[i]]:.6f},{ce[i]:.6f},"
                            f"{int(pred[i] == labels[i])},"
                            f"{int(pred[i] != base_pred[i])}\n")

    # failure_summary.csv
    import pandas as pd
    pd.DataFrame(summary_rows).to_csv(OUT / "failure_summary.csv", index=False)
    pd.DataFrame(per_snr_rows).to_csv(PERTS_OUT / "perturbation_per_snr.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(PERTS_OUT / "perturbation_per_class.csv", index=False)

    # ------------------------------------------------------------------
    # Analysis aggregates -> analysis_outputs.json
    # ------------------------------------------------------------------
    an = {}

    # baseline per-model
    an["baseline"] = [dict(family=m["family"], seed=m["seed"],
                           oa=base_metrics[i]["oa"],
                           macro_f1=base_metrics[i]["macro_f1"])
                      for i, m in enumerate(models)]

    # family x seed baseline correct flags
    fam = {mi: m["family"] for mi, m in enumerate(models)}
    seed = {mi: m["seed"] for mi, m in enumerate(models)}

    # persistent wrong (wrong in all 3 seeds of a family)
    def idx_of(family):
        return [mi for mi in range(nm) if fam[mi] == family]

    def persistent_wrong(family):
        mi_s = idx_of(family)
        wrong = np.ones(ns, dtype=bool)
        for mi in mi_s:
            wrong &= (base_correct[mi] == 0)
        return wrong

    t0_wrong_all = persistent_wrong("T0")
    t2_wrong_all = persistent_wrong("T2")
    an["t0_persistent_wrong"] = dict(n=int(t0_wrong_all.sum()),
                                     sample_ids=ids[t0_wrong_all].tolist())
    an["t2_persistent_wrong"] = dict(n=int(t2_wrong_all.sum()),
                                     sample_ids=ids[t2_wrong_all].tolist())

    # T0 wrong & T2 correct, all 3 seeds (T2 robustly fixes)
    t0_idx = idx_of("T0"); t2_idx = idx_of("T2")
    for s in (17, 42, 2026):
        mi0 = next(i for i in t0_idx if seed[i] == s)
        mi2 = next(i for i in t2_idx if seed[i] == s)
        an.setdefault("per_seed_t0_wrong_t2_correct", {})[s] = int(
            ((base_correct[mi0] == 0) & (base_correct[mi2] == 1)).sum())

    # common-correct sets (fixed, per seed) + their composition + failure rates
    an["common_correct"] = {}
    for s in (17, 42, 2026):
        mi0 = next(i for i in t0_idx if seed[i] == s)
        mi2 = next(i for i in t2_idx if seed[i] == s)
        common = (base_correct[mi0] == 1) & (base_correct[mi2] == 1)
        # composition
        comp = {}
        for c, sn in zip(labels[common], snrs[common]):
            comp.setdefault(int(c), {}).setdefault(float(sn), 0)
            comp[int(c)][float(sn)] += 1
        low = int((snrs[common] <= 0).sum())
        high = int((snrs[common] > 0).sum())
        an["common_correct"][s] = dict(n=int(common.sum()), low=low, high=high,
                                       n_class=len(set(labels[common].tolist())))
        # failure rate on this set per perturbation, per family
        fr = {}
        for fam_ in ("T0", "T2"):
            mi = next(i for i in idx_of(fam_) if seed[i] == s)
            for pi, (ptype, pval) in enumerate(PERTURBATIONS):
                pc = pert_logits[mi, pi].argmax(1) == labels
                fail = int((common & ~pc).sum())
                fr.setdefault(fam_, {})[f"{ptype}:{pval}"] = fail / max(1, int(common.sum()))
        an["common_correct"][s]["failure_rate"] = fr

    # recurrent confusion pairs (off-diagonal, present in all 3 seeds of family)
    an["confusion_pairs"] = {}
    for fam_ in ("T0", "T2"):
        mi_s = idx_of(fam_)
        cms = [cm_all[mi] for mi in mi_s]
        pairs = {}
        for cm in cms:
            for c in range(24):
                for p in range(24):
                    if c != p and cm[c, p] > 0:
                        pairs.setdefault((c, p), []).append(int(cm[c, p]))
        rec = {f"{c}->{p}": v for (c, p), v in pairs.items() if len(v) == 3}
        top = sorted(pairs.items(), key=lambda kv: -sum(kv[1]))[:15]
        an["confusion_pairs"][fam_] = dict(
            recurrent_all3=rec,
            top15=[dict(src=k[0][0], dst=k[0][1], counts=k[1]) for k in top])

    # per perturbation, family-level same-direction degradation flags
    an["degradation_flags"] = {}
    for pi, (ptype, pval) in enumerate(PERTURBATIONS):
        for fam_ in ("T0", "T2"):
            mi_s = idx_of(fam_)
            drops = []
            for mi in mi_s:
                pc = pert_logits[mi, pi].argmax(1) == labels
                drops.append((pc.mean() - base_correct[mi].mean()))
            an["degradation_flags"].setdefault(f"{ptype}:{pval}", {})[fam_] = dict(
                per_seed_drop=drops, all_negative=all(d < 0 for d in drops),
                mean_drop=float(np.mean(drops)))

    # paired bootstrap CIs for headline: OA drop per model for key perturbations
    rng = np.random.default_rng(0)
    def paired_boot(a, b, n_iter=2000):
        n = len(a)
        out = np.empty(n_iter)
        for k in range(n_iter):
            s = rng.integers(0, n, n)
            out[k] = a[s].mean() - b[s].mean()
        return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

    an["bootstrap"] = {}
    key_pert = [("phase", 180.0), ("phase", 90.0),
                ("offset", 2.0), ("offset", -2.0), ("offset", 1.0)]
    for ptype, pval in key_pert:
        if (ptype, pval) not in PERTURBATIONS:
            continue
        pi = PERTURBATIONS.index((ptype, pval))
        for mi, m in enumerate(models):
            pc = pert_logits[mi, pi].argmax(1) == labels
            lo, hi = paired_boot(base_correct[mi].astype(float), pc.astype(float))
            an["bootstrap"].setdefault(f"{ptype}:{pval}", {})[f"{m['family']}_s{m['seed']}"] = \
                dict(lo=lo, hi=hi, mean=float(base_correct[mi].mean() - pc.mean()))

    # spectrum check: mean power spectrum of raw complex val signal + band-edge energy
    rng2 = np.random.default_rng(1)
    sub = rng2.choice(ns, min(ns, 600), replace=False)
    zs = raw[sub, :, 0].astype(np.float64) + 1j * raw[sub, :, 1].astype(np.float64)
    spec = np.abs(np.fft.fft(zs, axis=1)) ** 2
    mean_spec = spec.mean(axis=0)
    freqs = np.fft.fftfreq(N)
    total = mean_spec.sum()
    edge = float(mean_spec[np.abs(freqs) > 0.45].sum() / total)
    an["spectrum"] = dict(
        n_samples=len(sub), total_energy=float(total),
        band_edge_energy_fraction_gt045=float(edge),
        max_offset_cycles_per_sample=float(max(abs(k) for k in OFFSET_FACTORS) / N),
        max_offset_total_phase_cycles=float(max(abs(k) for k in OFFSET_FACTORS)))

    dump(OUT / "analysis_outputs.json", an)
    print("\nSaved analysis_outputs.json", flush=True)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fam_colors = {"T0": "tab:blue", "T2": "tab:red"}
    seed_styles = {17: "-", 42: "--", 2026: "-."}

    def oa_of(mi, pi):
        return float((pert_logits[mi, pi].argmax(1) == labels).mean())

    def f1_of(mi, pi):
        cm = confusion_matrix(pert_logits[mi, pi].argmax(1), labels)
        return macro_f1(cm)[0]

    # --- phase response ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ph_idx = [PERTURBATIONS.index(("phase", float(p))) for p in PHASES]
    for mi, m in enumerate(models):
        oas = [oa_of(mi, pi) for pi in ph_idx]
        f1s = [f1_of(mi, pi) for pi in ph_idx]
        axes[0].plot(PHASES, oas, label=f"{m['family']} s{m['seed']}",
                     color=fam_colors[m['family']], linestyle=seed_styles[m['seed']],
                     marker="o", markersize=3)
        axes[1].plot(PHASES, f1s, label=f"{m['family']} s{m['seed']}",
                     color=fam_colors[m['family']], linestyle=seed_styles[m['seed']],
                     marker="o", markersize=3)
    axes[0].set(xlabel="global phase (deg)", ylabel="OA", title="Phase response — OA")
    axes[1].set(xlabel="global phase (deg)", ylabel="macro-F1", title="Phase response — macro-F1")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(CHARTS / "phase_response.png", dpi=130)
    plt.close(fig)

    # --- offset response ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    off_idx = [PERTURBATIONS.index(("offset", float(k))) for k in OFFSET_FACTORS]
    for mi, m in enumerate(models):
        oas = [oa_of(mi, pi) for pi in off_idx]
        axes[0].plot(OFFSET_FACTORS, oas, label=f"{m['family']} s{m['seed']}",
                     color=fam_colors[m['family']], linestyle=seed_styles[m['seed']],
                     marker="o", markersize=3)
        # split by direction (drop relative to 0)
        base_o = oa_of(mi, off_idx[0])
        axes[1].plot(OFFSET_FACTORS, [o - base_o for o in oas],
                     label=f"{m['family']} s{m['seed']}",
                     color=fam_colors[m['family']], linestyle=seed_styles[m['seed']],
                     marker="o", markersize=3)
    axes[0].set(xlabel="offset factor k (delta = k/N)", ylabel="OA",
                title="Frequency-offset response — OA")
    axes[1].set(xlabel="offset factor k (delta = k/N)", ylabel="OA − OA(0)",
                title="Frequency-offset response — ΔOA")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(CHARTS / "offset_response.png", dpi=130)
    plt.close(fig)

    # --- confusion pairs (family-level mean cm, log1p) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, fam_ in zip(axes, ("T0", "T2")):
        cms = np.stack([cm_all[mi] for mi in idx_of(fam_)])
        mean_cm = cms.mean(axis=0)
        im = ax.imshow(np.log1p(mean_cm), cmap="viridis", aspect="auto")
        ax.set(title=f"{fam_} mean confusion (log1p)", xlabel="pred", ylabel="true")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(CHARTS / "confusion_matrices.png", dpi=130)
    plt.close(fig)

    # --- T0 vs T2 paired degradation + common-correct failure ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    allp = PERTURBATIONS
    for fam_ in ("T0", "T2"):
        drops = []
        for pi in range(npert):
            dd = []
            for mi in idx_of(fam_):
                dd.append(base_correct[mi].mean() - oa_of(mi, pi))
            drops.append(float(np.mean(dd)))
        axes[0].plot(range(npert), drops, label=fam_, color=fam_colors[fam_],
                     marker="o", markersize=3)
    axes[0].set_xticks(range(npert))
    axes[0].set_xticklabels([f"{t}:{v}" for t, v in allp], rotation=90, fontsize=6)
    axes[0].set(ylabel="mean OA drop", title="T0 vs T2 mean degradation")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].legend()
    # common-correct failure rate (averaged over seeds) at each perturbation
    for fam_ in ("T0", "T2"):
        frs = []
        for pi in range(npert):
            vals = []
            for s in (17, 42, 2026):
                cc = an["common_correct"][s]
                fr = cc["failure_rate"][fam_][f"{allp[pi][0]}:{allp[pi][1]}"]
                vals.append(fr)
            frs.append(float(np.mean(vals)))
        axes[1].plot(range(npert), frs, label=fam_, color=fam_colors[fam_], marker="o",
                     markersize=3)
    axes[1].set_xticks(range(npert))
    axes[1].set_xticklabels([f"{t}:{v}" for t, v in allp], rotation=90, fontsize=6)
    axes[1].set(ylabel="failure rate on common-correct set", title="Common-correct set")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "t0_vs_t2_paired.png", dpi=130)
    plt.close(fig)

    # --- spectrum check ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, mean_spec / total, lw=0.7)
    ax.set(xlabel="cycles/sample", ylabel="normalized mean power",
           title=f"Mean raw-IQ spectrum (n={len(sub)}), band-edge frac={edge:.4f}")
    ax.axvline(0.45, color="r", ls="--", lw=0.7)
    ax.axvline(-0.45, color="r", ls="--", lw=0.7)
    fig.tight_layout()
    fig.savefig(CHARTS / "spectrum_check.png", dpi=130)
    plt.close(fig)

    print("Done. Outputs in", OUT, flush=True)


if __name__ == "__main__":
    main()
