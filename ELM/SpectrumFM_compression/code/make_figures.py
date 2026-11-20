"""
生成 PPT 所需的全部图（matplotlib Agg 后端，无需显示器），输出到 figures/ 目录。

图清单：
  1. param_composition   参数构成（饼图，18.06M 含 LoRA）
  2. volume_compression  体积压缩演进（柱状图，含部署口径）
  3. f1_accuracy         各配置精度 F1（柱状图 + 基线参考线）
  4. pruning_curve       剪枝曲线（裸剪 vs LoRA 微调后，mag/random）
  5. multiseed_box       多 seed 箱线图 + 对照实验（读 JSON）
  6. qat_vs_ptq          INT4 PTQ vs QAT vs 基线
  7. speed_curves        速度曲线（6 模型 × 7 档，读 speed_curves.csv）
  8. torchao_compare     torchao 官方实现对比（b1 / b256）

依赖：matplotlib, pandas。速度曲线图需先跑 latency_curves_all.py 生成 CSV。
注：系统无中文字体，图内文字暂用英文（PPT 幻灯片文字用中文）。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# 统一配色（压缩手段 → 颜色）
C_FP32 = "#4c72b0"     # 蓝
C_FP16 = "#dd8452"     # 橙
C_INT8 = "#55a868"     # 绿
C_INT4 = "#c44e52"     # 红
C_PRUNE = "#8172b2"    # 紫
C_ULTIMATE = "#ccb974"  # 黄褐
C_GRAY = "#888888"


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ---------------------------------------------------------------- 1 参数构成
def fig_param_composition():
    labels = ["FFN Linear", "Attention Linear", "Convolution",
              "LoRA A/B", "GRU", "Others\n(LN/relpos/cls)"]
    sizes = [8.41, 4.20, 3.18, 1.31, 0.79, 0.17]   # M 参数（合并前 18.06M）
    colors = [C_PRUNE, C_FP32, C_FP16, C_ULTIMATE, C_INT8, C_GRAY]
    total = sum(sizes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    # 左：饼图
    wedges, _, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct="%.0f%%", startangle=90,
        pctdistance=0.72, labeldistance=1.08,
        textprops={"fontsize": 9})
    for t in autotexts:
        t.set_color("white"); t.set_fontsize(8)
    ax1.set_title("Parameter Composition (18.06M total)")

    # 右：柱状图（绝对值）
    y = np.arange(len(labels))[::-1]
    ax2.barh(y, sizes, color=colors)
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=9)
    for i, v in enumerate(sizes):
        ax2.text(v + 0.15, y[i], f"{v:.2f}M", va="center", fontsize=8)
    ax2.set_xlabel("Parameters (M)")
    ax2.set_title("Absolute Size per Component")
    ax2.set_xlim(0, 10)
    fig.tight_layout()
    save(fig, "1_param_composition.png")


# ---------------------------------------------------------------- 2 体积压缩
def fig_volume_compression():
    # (名称, 体积 MB, 颜色, 是否部署口径)
    data = [
        ("FP32 baseline", 69.2, C_FP32, False),
        ("FP16", 34.7, C_FP16, False),
        ("INT8 dynamic", 31.0, C_INT8, False),
        ("INT8 static(65L)", 40.3, C_INT8, False),
        ("merge full-dyn", 26.0, C_INT8, False),
        ("INT8 complete", 17.1, C_INT8, False),
        ("INT4 QAT (deploy)", 12.2, C_INT4, True),
        ("Ultimate (deploy)", 10.5, C_ULTIMATE, True),
    ]
    names = [d[0] for d in data]
    vals = [d[1] for d in data]
    colors = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(np.arange(len(data))[::-1], vals, color=colors)
    ax.set_yticks(np.arange(len(data))[::-1]); ax.set_yticklabels(names, fontsize=9)
    for i, (v, d) in enumerate(zip(vals, data)):
        label = f"{v:.1f}MB ({69.2 / v:.1f}x)"
        if d[3]:
            label += " *"
        ax.text(v + 0.8, len(data) - 1 - i, label, va="center", fontsize=8)
    ax.axvline(69.2, color=C_GRAY, ls="--", lw=0.8)
    ax.set_xlabel("Model size (MB, log scale)")
    ax.set_xscale("log")
    ax.set_xlim(8, 120)
    ax.set_title("Model Size Compression (baseline 69.2MB)  * = deployment size")
    fig.tight_layout()
    save(fig, "2_volume_compression.png")


# ---------------------------------------------------------------- 3 精度 F1
def fig_f1_accuracy():
    data = [
        ("FP32 baseline", 0.6474, C_FP32),
        ("FP16", 0.6475, C_FP16),
        ("INT8 dynamic", 0.6525, C_INT8),
        ("INT8 static(65L)", 0.6474, C_INT8),
        ("merge full-dyn", 0.6504, C_INT8),
        ("INT8 complete", 0.6510, C_INT8),
        ("INT4 PTQ", 0.6401, C_INT4),
        ("INT4 QAT", 0.6494, C_INT4),
        ("Ultimate", 0.6426, C_ULTIMATE),
    ]
    names = [d[0] for d in data]
    vals = [d[1] for d in data]
    colors = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, vals, color=colors)
    ax.axhline(0.6474, color=C_GRAY, ls="--", lw=1, label="baseline F1=0.6474")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0004, f"{v:.4f}",
                ha="center", fontsize=7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("F1 (weighted)")
    ax.set_ylim(0.62, 0.665)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Accuracy (F1) per Configuration")
    fig.tight_layout()
    save(fig, "3_f1_accuracy.png")


# ---------------------------------------------------------------- 4 剪枝曲线
def fig_pruning_curve():
    prune_pct = [0, 25, 50, 60, 70]
    # 裸剪（未微调）
    bare_mag = [0.6474, 0.6429, 0.6102, 0.5845, 0.5447]
    bare_rnd = [0.6474, 0.6439, 0.6170, 0.5874, 0.5506]
    # LoRA 微调后（25/50 为单次；60/70 为多 seed 均值）
    ft_mag = [0.6525, 0.6573, 0.6538, 0.6499, 0.6507]
    ft_rnd = [0.6525, 0.6534, 0.6526, 0.6529, 0.6511]
    ft_mag_err = [0.0051, 0, 0, 0.0070, 0.0053]
    ft_rnd_err = [0.0051, 0, 0, 0.0042, 0.0024]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(prune_pct, bare_mag, "o-", color=C_PRUNE, label="bare, magnitude")
    ax.plot(prune_pct, bare_rnd, "s--", color=C_FP32, label="bare, random")
    ax.errorbar(prune_pct, ft_mag, yerr=ft_mag_err, fmt="o-", color=C_PRUNE,
                lw=2, label="+LoRA finetune, magnitude", capsize=3)
    ax.errorbar(prune_pct, ft_rnd, yerr=ft_rnd_err, fmt="s--", color=C_FP32,
                lw=2, label="+LoRA finetune, random", capsize=3)
    ax.axhline(0.6474, color=C_GRAY, ls=":", lw=1, label="baseline F1=0.6474")
    ax.set_xlabel("FFN width pruned (%)")
    ax.set_ylabel("F1 (weighted)")
    ax.set_ylim(0.50, 0.67)
    ax.set_xticks(prune_pct)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Structured Pruning: Bare vs +LoRA Finetune (5 epochs)")
    fig.tight_layout()
    save(fig, "4_pruning_curve.png")


# ---------------------------------------------------------------- 5 多 seed 箱线
def fig_multiseed_box():
    with open("Results/pruning/multiseed_results.json") as f:
        ms = json.load(f)
    with open("Results/pruning/baseline_finetune.json") as f:
        bl = json.load(f)

    data = [
        ("mag\n60%", ms["configs"]["magnitude_60"]["seed_f1s"]),
        ("rand\n60%", ms["configs"]["random_60"]["seed_f1s"]),
        ("mag\n70%", ms["configs"]["magnitude_70"]["seed_f1s"]),
        ("rand\n70%", ms["configs"]["random_70"]["seed_f1s"]),
        ("control\n(no prune)", bl["finetuned_f1s"]),
    ]
    labels = [d[0] for d in data]
    values = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bp = ax.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
    palette = [C_PRUNE, C_FP32, C_PRUNE, C_FP32, C_GRAY]
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    # 散点
    for i, v in enumerate(values):
        x = np.full(len(v), i + 1) + np.random.uniform(-0.08, 0.08, len(v))
        ax.scatter(x, v, s=18, color="k", alpha=0.6, zorder=3)
    ax.axhline(ms["baseline_f1"], color=C_GRAY, ls="--", lw=1,
               label=f"baseline F1={ms['baseline_f1']:.4f}")
    ax.set_ylabel("F1 (weighted)")
    ax.set_ylim(0.63, 0.67)
    ax.legend(fontsize=8)
    ax.set_title("Multi-seed Validation (5 seeds each) + Control Experiment")
    fig.tight_layout()
    save(fig, "5_multiseed_box.png")


# ---------------------------------------------------------------- 6 QAT vs PTQ
def fig_qat_vs_ptq():
    names = ["FP32 baseline", "INT4 PTQ", "INT4 QAT"]
    vals = [0.6474, 0.6401, 0.6494]
    colors = [C_FP32, C_GRAY, C_INT4]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(names, vals, color=colors, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0006, f"{v:.4f}",
                ha="center", fontsize=10)
    ax.axhline(0.6474, color=C_GRAY, ls="--", lw=1)
    ax.set_ylabel("F1 (weighted)")
    ax.set_ylim(0.635, 0.655)
    ax.set_title("INT4: QAT recovers PTQ drop (+0.93pt)")
    fig.tight_layout()
    save(fig, "6_qat_vs_ptq.png")


# ---------------------------------------------------------------- 7 速度曲线
def fig_speed_curves():
    csv_path = "Results/profiling/speed_curves.csv"
    if not os.path.exists(csv_path):
        print("  ! speed_curves.csv 不存在，跳过速度曲线图（先跑 latency_curves_all.py）")
        return
    import csv as _csv
    models = []      # 有序模型名
    table = {}       # model -> {batch: (lat_ms, thr)}
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            m = row["model"]
            if m not in table:
                table[m] = {}
                models.append(m)
            table[m][int(row["batch"])] = (float(row["latency_ms_per_sample"]),
                                           float(row["throughput_sps"]))
    batches = sorted({b for m in table for b in table[m]})
    cmap = {m: c for m, c in zip(models,
            [C_FP32, C_FP16, C_PRUNE, C_ULTIMATE, C_INT4, C_INT8])}
    # 服务器无中文字体，图例转英文
    rename = {"剪60%": "prune60%", "剪40%+FP16": "prune40%+FP16",
              "纯INT4QAT": "INT4QAT", "剪40%+QAT": "prune40%+QAT"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in models:
        xs = batches
        lat = [table[m][b][0] for b in xs]
        thr = [table[m][b][1] for b in xs]
        lbl = rename.get(m, m)
        ax1.plot(xs, lat, "o-", color=cmap[m], label=lbl, lw=1.5, ms=4)
        ax2.plot(xs, thr, "o-", color=cmap[m], label=lbl, lw=1.5, ms=4)
    ax1.set_xscale("log"); ax2.set_xscale("log")
    ax1.set_xlabel("batch size"); ax1.set_ylabel("latency / sample (ms)")
    ax2.set_xlabel("batch size"); ax2.set_ylabel("throughput (samples/s)")
    ax1.set_title("Latency per Sample vs Batch")
    ax2.set_title("Throughput vs Batch")
    ax1.legend(fontsize=7, loc="upper right"); ax2.legend(fontsize=7, loc="lower right")
    fig.suptitle("Speed Curves (6 models x 7 batches, GPU A100)", y=1.02)
    fig.tight_layout()
    save(fig, "7_speed_curves.png")


# ---------------------------------------------------------------- 8 torchao 对比
def fig_torchao_compare():
    names = ["FP32", "BF16", "INT8 w/o", "BF16+INT4pack"]
    b1 = [6.85, 5.98, 6.70, 11.85]
    thr = [2995, 9806, 3016, 1151]
    colors = [C_FP32, C_FP16, C_INT8, C_INT4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    b1bars = ax1.bar(names, b1, color=colors)
    for b, v in zip(b1bars, b1):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}", ha="center", fontsize=8)
    ax1.set_ylabel("b1 latency (ms)"); ax1.set_title("Latency (batch=1)")
    ax1.axhline(6.85, color=C_GRAY, ls="--", lw=0.8)

    thrbars = ax2.bar(names, thr, color=colors)
    for b, v in zip(thrbars, thr):
        ax2.text(b.get_x() + b.get_width() / 2, v + 60, f"{v}", ha="center", fontsize=8)
    ax2.set_ylabel("throughput (samples/s)"); ax2.set_title("Throughput (batch=256)")
    ax2.axhline(2995, color=C_GRAY, ls="--", lw=0.8)
    fig.suptitle("torchao Official Impl (all torch.compile)", y=1.02)
    fig.tight_layout()
    save(fig, "8_torchao_compare.png")


if __name__ == "__main__":
    print("生成 PPT 图...")
    fig_param_composition()
    fig_volume_compression()
    fig_f1_accuracy()
    fig_pruning_curve()
    fig_multiseed_box()
    fig_qat_vs_ptq()
    fig_speed_curves()
    fig_torchao_compare()
    print("完成。")
