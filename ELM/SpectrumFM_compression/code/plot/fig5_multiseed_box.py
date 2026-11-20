"""图 5：多 seed 箱线图 + 对照实验。

用法：
    python plot/fig5_multiseed_box.py

想调整的地方：
    FIG / DPI   —— 尺寸、分辨率
    GROUPS      —— 每行 (标签, [5 个 seed 的 F1])
    BASELINE    —— 基线 F1（虚线参考线）
    palette     —— 各箱体的颜色
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
FIG = (9, 5)
DPI = 400
C_PRUNE = "#8172b2"
C_FP32 = "#4c72b0"
C_GRAY = "#888888"
BASELINE = 0.6474
# -------------------------- 数据 --------------------------
# (标签, 5 个 seed 的 F1)
GROUPS = [
    ("幅值\n60%",          [0.6388, 0.6504, 0.6480, 0.6515, 0.6607]),
    ("随机\n60%",          [0.6519, 0.6493, 0.6553, 0.6482, 0.6598]),
    ("幅值\n70%",          [0.6484, 0.6492, 0.6545, 0.6431, 0.6585]),
    ("随机\n70%",          [0.6542, 0.6500, 0.6499, 0.6479, 0.6537]),
    ("对照\n(不剪枝)",     [0.6494, 0.6502, 0.6471, 0.6546, 0.6615]),
]
palette = [C_PRUNE, C_FP32, C_PRUNE, C_FP32, C_GRAY]
# ----------------------------------------------------------------

labels = [g[0] for g in GROUPS]
values = [g[1] for g in GROUPS]

fig, ax = plt.subplots(figsize=FIG)
bp = ax.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
for patch, c in zip(bp["boxes"], palette):
    patch.set_facecolor(c)
    patch.set_alpha(0.55)
for i, v in enumerate(values):
    x = np.full(len(v), i + 1) + np.random.uniform(-0.08, 0.08, len(v))
    ax.scatter(x, v, s=18, color="k", alpha=0.6, zorder=3)
ax.axhline(BASELINE, color=C_GRAY, ls="--", lw=1, label=f"基线 F1={BASELINE:.4f}")
ax.set_ylabel("F1 分数")
ax.set_ylim(0.63, 0.67)
ax.legend(fontsize=11)
ax.set_title("多 seed 验证（各 5 个 seed）+ 对照实验")

fig.tight_layout()
path = OUT / "5_multiseed_box.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"✓ {path}")
