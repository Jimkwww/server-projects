"""Connectivity / reachability analysis for unequal-window schedules (brief §4).

For each layer l in order, builds the allowed attention matrix A_l and the
input-dependency relation R_l via boolean matrix multiplication
    R_{l+1} = (A_l @ R_l) > 0,  R_0 = identity.
A_l includes the diagonal (the residual self-connection), so each layer's own
self path is covered. Analysis is about encoder token propagation only; the
final global mean pool is NOT counted as a global interaction.

Reports, per layer: reachable-input count per output token (min/mean/max and
fraction of N), how many output tokens cover all inputs, per-input influence
counts, plus the layer each output token first reaches full coverage (or -1).
Writes a JSON report and prints a readable text summary (partition diagram +
downsampled A_l / final-R heatmaps).

Run from the MixT root:  python -m emfm.connectivity ...
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .unequal_window_models import rotate_right, window_intervals, allowed_matrix


def schedule(window_list, rotation, depth):
    """List of (rotation_index, rotated_window_list, intervals) per layer."""
    k = len(window_list)
    out = []
    for l in range(depth):
        r = l % k if rotation == "cyclic" else 0
        wl = rotate_right(window_list, r)
        out.append((r, wl, window_intervals(wl, sum(window_list))))
    return out


def analyze(window_list, rotation, depth):
    n = sum(window_list)
    layers = []
    R = np.eye(n, dtype=np.int64)
    first_full = np.full(n, -1, dtype=np.int64)
    A_mats = []
    for l in range(depth):
        r = l % len(window_list) if rotation == "cyclic" else 0
        wl = rotate_right(window_list, r)
        A = allowed_matrix(wl, n).numpy().astype(np.int64)
        A_mats.append(A)
        R = (A @ R) > 0
        reach = R.sum(axis=1)
        influence = R.sum(axis=0)
        full = reach == n
        first_full = np.where((first_full == -1) & full, l, first_full)
        layers.append({
            "layer": l,
            "rotation_index": r,
            "window_list": wl,
            "intervals": [[int(s), int(e)] for s, e in window_intervals(wl, n)],
            "reachable_min": int(reach.min()),
            "reachable_mean": float(reach.mean()),
            "reachable_max": int(reach.max()),
            "reachable_frac_mean": float(reach.mean() / n),
            "full_coverage_tokens": int(full.sum()),
            "per_input_influence_min": int(influence.min()),
            "per_input_influence_mean": float(influence.mean()),
            "per_input_influence_max": int(influence.max()),
        })
    return {
        "n": n, "window_list": list(window_list), "rotation": rotation, "depth": depth,
        "sum_w2": int(sum(w * w for w in window_list)),
        "layers": layers,
        "first_full_coverage_layer": [int(v) for v in first_full],
        "final_reachable_frac": float((R.sum(axis=1) == n).mean()) if depth else 1.0,
        "final_R": R.tolist() if n <= 256 else None,
        "A_l": [A.tolist() for A in A_mats] if n <= 256 else None,
    }


def downsample_ascii(M, target=40, chars=" .:*#"):
    n = M.shape[0]
    step = max(1, int(np.ceil(n / target)))
    lines = []
    for i in range(0, n, step):
        row = []
        for j in range(0, n, step):
            block = M[i:i + step, j:j + step]
            frac = float(block.mean()) if block.size else 0.0
            row.append(chars[int(frac * (len(chars) - 1) + 0.5)])
        lines.append("".join(row))
    return "\n".join(lines)


def partition_diagram(window_list, n, target=64):
    """One-line ASCII diagram: each window proportional to its width."""
    segs = []
    for w in window_list:
        segs.append("#" * max(1, round(w / n * target)))
    return "|".join(segs)


def render(report, target=40):
    n = report["n"]
    lines = [f"N={n}  W={report['window_list']}  rotation={report['rotation']}  "
             f"depth={report['depth']}  sum(w^2)={report['sum_w2']}",
             f"global-baseline reference: sum(w^2)/N^2 = {report['sum_w2'] / n / n:.3f} "
             f"({report['sum_w2'] / n / n * 100:.1f}% of full N^2 attention budget)",
             ""]
    lines.append("per-layer partition (r = rotation index):")
    for L in report["layers"]:
        lines.append(f"  L{L['layer']} r={L['rotation_index']}: "
                     f"{partition_diagram(L['window_list'], n)}")
    lines.append("")
    lines.append("reachability stats per layer:")
    hdr = (f"{'L':>3} {'r':>3} {'reach_min':>9} {'reach_mean':>10} "
           f"{'reach_max':>9} {'frac':>6} {'full_cov':>8} {'infl_mean':>9}")
    lines.append(hdr)
    for L in report["layers"]:
        lines.append(f"{L['layer']:>3} {L['rotation_index']:>3} "
                     f"{L['reachable_min']:>9} {L['reachable_mean']:>10.1f} "
                     f"{L['reachable_max']:>9} {L['reachable_frac_mean']:>6.3f} "
                     f"{L['full_coverage_tokens']:>8} {L['per_input_influence_mean']:>9.1f}")
    lines.append("")
    lines.append(f"first layer each output token reaches full coverage "
                 f"(-1 = never within depth): {report['first_full_coverage_layer']}")
    lines.append(f"final reachable fraction (full-coverage output tokens): "
                 f"{report['final_reachable_frac']:.3f}")
    if report["A_l"] is not None:
        lines.append("")
        for L in report["layers"]:
            lines.append(f"A_{L['layer']} heatmap (downsampled {target}x{target}):")
            lines.append(downsample_ascii(np.array(report["A_l"][L['layer']]), target))
        lines.append("")
        lines.append(f"final R heatmap (downsampled {target}x{target}):")
        lines.append(downsample_ascii(np.array(report["final_R"]), target))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window-list", default="4,8,16,24,32,44",
                   help="comma-separated window lengths (sum must equal N)")
    p.add_argument("--rotation", choices=["cyclic", "fixed"], default="cyclic")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--output", required=True, help="JSON output path")
    a = p.parse_args()
    wl = [int(x) for x in a.window_list.split(",") if x.strip()]
    if any(w < 1 for w in wl):
        p.error("window lengths must be positive")
    report = analyze(wl, a.rotation, a.depth)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(render(report))
    print(f"\nJSON written to {out}")


if __name__ == "__main__":
    main()
