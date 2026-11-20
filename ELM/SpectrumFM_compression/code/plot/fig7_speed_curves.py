"""图 7：速度曲线（6 模型 × 7 档 batch，延迟 + 吞吐）。

数据已内联（原 speed_curves.csv），改数字直接改下面 DATA 即可；
想重测真实数据再跑 latency_curves_all.py 生成 CSV，然后回填到这里。

用法：
    python plot/fig7_speed_curves.py

想调整的地方：
    FIG / DPI   —— 尺寸、分辨率
    DATA        —— {模型: ([延迟 ms/样本], [吞吐 samples/s])}，与 BATCHES 对齐
    CMAP        —— 模型 → 颜色
    RENAME      —— 图例名
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体（思源黑体，装在 ~/.local/share/fonts/）
for _f in (Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf",
           Path.home() / ".local/share/fonts/NotoSansCJKsc-Bold.otf"):
    if _f.exists():
        font_manager.fontManager.addfont(str(_f))
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.titlesize"] = 16     # 子图标题
plt.rcParams["axes.labelsize"] = 14     # X/Y 轴标签
plt.rcParams["xtick.labelsize"] = 12    # 刻度数字
plt.rcParams["ytick.labelsize"] = 12    # 刻度数字

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# -------------------------- 可调参数 --------------------------
FIG = (10, 5)
DPI = 400
C_FP32 = "#4c72b0"
C_FP16 = "#dd8452"
C_PRUNE = "#8172b2"
C_ULTIMATE = "#ccb974"
C_INT4 = "#c44e52"
C_INT8 = "#55a868"
# -------------------------- 数据 --------------------------
BATCHES = [1, 8, 16, 32, 64, 128, 256]
# 每行：模型 -> (各 batch 延迟 ms/样本, 各 batch 吞吐 samples/s)
DATA = {
    "FP32":       ([8.4597, 1.4908, 0.9598, 0.7243, 0.5610, 0.5105, 0.4999],
                   [118.2, 670.8, 1041.8, 1380.6, 1782.4, 1958.7, 2000.2]),
    "FP16":       ([7.2729, 0.9432, 0.5346, 0.3331, 0.2425, 0.1965, 0.1842],
                   [137.5, 1060.2, 1870.7, 3002.5, 4123.9, 5090.1, 5427.4]),
    "剪60%":      ([8.3686, 1.3464, 0.8524, 0.6466, 0.4718, 0.4273, 0.4147],
                   [119.5, 742.7, 1173.2, 1546.4, 2119.6, 2340.4, 2411.7]),
    "剪40%+FP16": ([7.0591, 1.0338, 0.6005, 0.3860, 0.2630, 0.2194, 0.2060],
                   [141.7, 967.3, 1665.3, 2591.0, 3802.4, 4556.9, 4854.6]),
    "纯INT4QAT":  ([14.6484, 2.4750, 1.5477, 1.1206, 0.8886, 0.8683, 0.8611],
                   [68.3, 404.0, 646.1, 892.4, 1125.4, 1151.7, 1161.3]),
    "剪40%+QAT":  ([14.2186, 2.3523, 1.4394, 1.0428, 0.7729, 0.7465, 0.7561],
                   [70.3, 425.1, 694.7, 959.0, 1293.9, 1339.6, 1322.5]),
}
CMAP = {
    "FP32": C_FP32, "FP16": C_FP16, "剪60%": C_PRUNE,
    "剪40%+FP16": C_ULTIMATE, "纯INT4QAT": C_INT4, "剪40%+QAT": C_INT8,
}
RENAME = {"剪60%": "剪枝 60%", "剪40%+FP16": "剪枝 40% + FP16",
          "纯INT4QAT": "纯 INT4 QAT", "剪40%+QAT": "剪枝 40% + QAT"}
# ----------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG)
for m, (lat, thr) in DATA.items():
    lbl = RENAME.get(m, m)
    ax1.plot(BATCHES, lat, "o-", color=CMAP[m], label=lbl, lw=1.5, ms=4)
    ax2.plot(BATCHES, thr, "o-", color=CMAP[m], label=lbl, lw=1.5, ms=4)
ax1.set_xscale("log")
ax2.set_xscale("log")
ax1.set_xlabel("batch 大小")
ax1.set_ylabel("单样本延迟 (ms)")
ax2.set_xlabel("batch 大小")
ax2.set_ylabel("吞吐 (samples/s)")
ax1.set_title("单样本延迟 vs batch")
ax2.set_title("吞吐 vs batch")
ax1.grid(True, ls="--", alpha=0.4)
ax2.grid(True, ls="--", alpha=0.4)
ax1.legend(fontsize=11, loc="upper right")
ax2.legend(fontsize=11, loc="upper left")
fig.suptitle("速度曲线（6 模型 × 7 档 batch，GPU A100）", y=1.02)

fig.tight_layout()
path = OUT / "7_speed_curves.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"✓ {path}")
