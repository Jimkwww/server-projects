"""图 1：参数构成（饼图 + 绝对值柱状图）。

用法：
    python plot/fig1_param_composition.py        # 从 code/ 目录
    python code/plot/fig1_param_composition.py   # 从项目根目录
（脚本内部用 __file__ 定位 figures/，从哪跑都行）

想调整的地方：
    FIG       —— 图整体尺寸（宽, 高）英寸
    DPI       —— 分辨率，越大越清晰、文件越大
    COLORS    —— 各组件颜色（与 labels 一一对应）
    sizes     —— 各组件参数量（M）
    ax1.set_title / ax2.set_title —— 标题文字
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

# 输出目录：code/figures/
OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# -------------------------- 可调参数 --------------------------
FIG = (12, 5)
DPI = 400
COLORS = ["#8172b2", "#4c72b0", "#dd8452", "#ccb974", "#55a868", "#888888"]
# -------------------------- 数据（M 参数）--------------------------
labels = ["FFN 线性层", "注意力线性层", "卷积层",
          "LoRA A/B", "GRU", "其他\n(LN/相对位置/CLS)"]
sizes = [8.41, 4.20, 3.18, 1.31, 0.79, 0.17]   # 合计 18.06M
# ----------------------------------------------------------------

total = sum(sizes)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG)

# 左：饼图
wedges, _, autotexts = ax1.pie(
    sizes, labels=labels, colors=COLORS, autopct="%.0f%%", startangle=90,
    pctdistance=0.72, labeldistance=1.08, textprops={"fontsize": 11})
for t in autotexts:
    t.set_color("white")
    t.set_fontsize(10)
ax1.set_title(f"参数构成（共 {total:.2f}M）")

# 右：柱状图（绝对值）
y = np.arange(len(labels))[::-1]
ax2.barh(y, sizes, color=COLORS)
ax2.set_yticks(y)
ax2.set_yticklabels(labels, fontsize=12)
for i, v in enumerate(sizes):
    ax2.text(v + 0.15, y[i], f"{v:.2f}M", va="center", fontsize=11)
ax2.set_xlabel("参数量 (M)")
ax2.set_title("各组件参数量")
ax2.set_xlim(0, 10)

fig.tight_layout()
path = OUT / "1_param_composition.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"✓ {path}")
