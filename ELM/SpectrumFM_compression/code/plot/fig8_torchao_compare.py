"""图 8：torchao 官方实现对比（b1 延迟 + b256 吞吐）。

用法：
    python plot/fig8_torchao_compare.py

想调整的地方：
    FIG / DPI   —— 尺寸、分辨率
    names       —— 柱子标签
    b1 / thr    —— b1 延迟(ms) 与 b256 吞吐(samples/s)
    colors      —— 颜色
    ax1.axhline(6.85) / ax2.axhline(2995) —— 基线参考线
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
FIG = (11, 5)
DPI = 400
C_FP32 = "#4c72b0"
C_FP16 = "#dd8452"
C_INT8 = "#dbb740"
C_INT4 = "#c03c40"
C_GRAY = "#888888"
# -------------------------- 数据 --------------------------
names = ["FP32", "BF16", "INT8 仅权重", "BF16+INT4打包"]
b1 = [6.85, 5.98, 6.70, 11.85]
thr = [2995, 9806, 3016, 1151]
colors = [C_FP32, C_FP16, C_INT8, C_INT4]
# ----------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG)

b1bars = ax1.bar(names, b1, color=colors)
for b, v in zip(b1bars, b1):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}", ha="center", fontsize=12)
ax1.set_ylabel("单样本延迟 (ms)")
ax1.set_title("延迟（batch=1）")
ax1.axhline(6.85, color=C_GRAY, ls="--", lw=0.8)

thrbars = ax2.bar(names, thr, color=colors)
for b, v in zip(thrbars, thr):
    ax2.text(b.get_x() + b.get_width() / 2, v + 60, f"{v}", ha="center", fontsize=12)
ax2.set_ylabel("吞吐 (samples/s)")
ax2.set_title("吞吐（batch=256）")
ax2.axhline(2995, color=C_GRAY, ls="--", lw=0.8)
fig.suptitle("torchao 官方实现（均 torch.compile）", y=1.02)

fig.tight_layout()
path = OUT / "8_torchao_compare.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"✓ {path}")
