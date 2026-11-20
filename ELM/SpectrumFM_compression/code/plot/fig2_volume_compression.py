"""图 2：体积压缩演进（横向柱状图，对数轴）。

用法：
    python plot/fig2_volume_compression.py

想调整的地方：
    FIG / DPI     —— 尺寸、分辨率
    data          —— 每行 (名称, 体积MB, 颜色, 是否部署口径)
    ax.set_xlim(8, 120) —— 对数轴范围
    BASELINE      —— 基线体积（画参考线 + 算压缩比）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
FIG = (7, 5.5)
DPI = 400
C_FP32 = "#4c72b0"
C_FP16 = "#dd8452"
C_INT8 = "#ebd140"
C_INT4 = "#74c2cc"
C_ULTIMATE = "#56da56"
BASELINE = 69.2   # FP32 基线体积 MB
# -------------------------- 数据 --------------------------
# (名称, 体积 MB, 颜色)
data = [
    ("FP32 基线",         69.2, C_FP32, False),
    ("FP16",              34.7, C_FP16, False),
    ("INT8 linear层动态",         31.0, C_INT8, False),
    ("INT8 linear层静态",    40.3, C_INT8, False),
    ("Merge INT8动态",        26.0, C_INT8, False),
    ("INT8 完全",         17.1, C_INT8, False),
    ("INT4 QAT",  12.2, C_INT4, True),
    ("剪40%+INT4 QAT",      10.5, C_ULTIMATE, True),
]
# ----------------------------------------------------------------

names = [d[0] for d in data]
vals = [d[1] for d in data]
colors = [d[2] for d in data]

fig, ax = plt.subplots(figsize=FIG)
ax.barh(np.arange(len(data))[::-1], vals, color=colors)
ax.set_yticks(np.arange(len(data))[::-1])
ax.set_yticklabels(names, fontsize=12)
for i, (v, d) in enumerate(zip(vals, data)):
    label = f"{v:.1f}MB ({BASELINE / v:.1f}×)"
    ax.text(v + 0.8, len(data) - 1 - i, label, va="center", fontsize=11)
ax.axvline(BASELINE, color="#888888", ls="--", lw=0.8)
ax.set_xlabel("模型体积 (MB，对数轴)")
ax.set_xscale("log")
ax.set_xlim(8, 120)
ax.grid(True, ls="--", alpha=0.4)
ax.set_title(f"模型体积压缩（基线 {BASELINE}MB）")

fig.tight_layout()
path = OUT / "2_volume_compression.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"✓ {path}")
