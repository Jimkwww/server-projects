"""图 3：各配置精度 F1（柱状图 + 基线参考线）。

用法：
    python plot/fig3_f1_accuracy.py

想调整的地方：
    FIG / DPI    —— 尺寸、分辨率
    data         —— (名称, F1, 颜色)
    BASELINE     —— 基线 F1（虚线参考线）
    ax.set_ylim(0.62, 0.665) —— Y 轴范围
    rotation=40  —— X 轴标签倾斜角度
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
C_INT8 = "#55a868"
C_INT4 = "#c44e52"
C_ULTIMATE = "#ccb974"
BASELINE = 0.6474
# -------------------------- 数据 --------------------------
data = [
    ("FP32 基线",        0.6474, C_FP32),
    ("FP16",             0.6475, C_FP16),
    ("INT8 动态",        0.6525, C_INT8),
    ("INT8 静态(65L)",   0.6474, C_INT8),
    ("合并全动态",       0.6504, C_INT8),
    ("INT8 完整",        0.6510, C_INT8),
    ("INT4 PTQ",         0.6401, C_INT4),
    ("INT4 QAT",         0.6494, C_INT4),
    ("终极",             0.6426, C_ULTIMATE),
]
# ----------------------------------------------------------------

names = [d[0] for d in data]
vals = [d[1] for d in data]
colors = [d[2] for d in data]

fig, ax = plt.subplots(figsize=FIG)
bars = ax.bar(names, vals, color=colors)
ax.axhline(BASELINE, color="#888888", ls="--", lw=1, label=f"基线 F1={BASELINE}")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0004, f"{v:.4f}",
            ha="center", fontsize=10)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=11)
ax.set_ylabel("F1 分数")
ax.set_ylim(0.62, 0.665)
ax.legend(fontsize=11, loc="lower right")
ax.set_title("各配置精度（F1）")

fig.tight_layout()
path = OUT / "3_f1_accuracy.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"✓ {path}")
