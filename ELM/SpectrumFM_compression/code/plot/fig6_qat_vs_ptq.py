"""图 6：INT4 PTQ vs QAT vs 基线。

用法：
    python plot/fig6_qat_vs_ptq.py

想调整的地方：
    FIG / DPI   —— 尺寸、分辨率
    names / vals —— 柱子的标签与 F1 值
    colors      —— 颜色
    ax.set_ylim(0.635, 0.655) —— Y 轴范围
    width=0.5   —— 柱子宽度
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
FIG = (6, 4)
DPI = 400
C_FP32 = "#4c72b0"
C_INT4 = "#da8422"
C_GRAY = "#888888"
BASELINE = 0.6474
# -------------------------- 数据 --------------------------
names = ["FP32 基线", "INT4 PTQ", "INT4 QAT"]
vals = [0.6474, 0.6401, 0.6494]
colors = [C_FP32, C_GRAY, C_INT4]
# ----------------------------------------------------------------

fig, ax = plt.subplots(figsize=FIG)
bars = ax.bar(names, vals, color=colors, width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0006, f"{v:.4f}",
            ha="center", fontsize=12)
ax.axhline(BASELINE, color=C_GRAY, ls="--", lw=1)
ax.set_ylabel("F1 分数")
ax.set_ylim(0.635, 0.655)
ax.grid(True, ls="--", alpha=0.4)
ax.set_title("INT4量化 - QAT/PTQ对比")

fig.tight_layout()
path = OUT / "6_qat_vs_ptq.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"✓ {path}")
