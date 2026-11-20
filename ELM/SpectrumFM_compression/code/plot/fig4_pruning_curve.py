"""图 4：剪枝曲线（裸剪 vs LoRA 微调，magnitude / random）。

用法：
    python plot/fig4_pruning_curve.py

想调整的地方：
    FIG / DPI         —— 尺寸、分辨率
    prune_pct         —— X 轴剪枝比例
    bare_mag / bare_rnd      —— 裸剪（未微调）F1
    ft_mag / ft_rnd          —— LoRA 微调后 F1（均为均值）
    M_BARE_MAG 等 M_*  —— 每条线的 marker（4 条线各不相同）
    C_PRUNE / C_FP32  —— 两条系列的颜色
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
plt.rcParams["axes.labelsize"] = 14     # 顺带把 X/Y 轴标签也调了
plt.rcParams["xtick.labelsize"] = 12    # 刻度数字
plt.rcParams["ytick.labelsize"] = 12    # 刻度数字

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# -------------------------- 可调参数 --------------------------
FIG = (6, 4.8)
DPI = 400
C_PRUNE = "#da8422"   # 紫色 = magnitude
C_FP32 = "#4c72b0"    # 蓝色 = random
C_GRAY = "#888888"
BASELINE = 0.6474
# -------------------------- 数据 --------------------------
prune_pct = [0, 25, 50, 60, 70]
# 裸剪（未微调）
bare_mag = [0.6474, 0.6429, 0.6102, 0.5845, 0.5447]
bare_rnd = [0.6474, 0.6439, 0.6170, 0.5874, 0.5506]
# LoRA 微调后（均为均值）
ft_mag = [0.6525, 0.6573, 0.6538, 0.6499, 0.6507]
ft_rnd = [0.6525, 0.6534, 0.6526, 0.6529, 0.6511]
# 4 条线 marker 各不相同
M_BARE_MAG = "o"
M_BARE_RND = "s"
M_FT_MAG = "^"
M_FT_RND = "D"
# ----------------------------------------------------------------

fig, ax = plt.subplots(figsize=FIG)
ax.plot(prune_pct, bare_mag, marker=M_BARE_MAG, ls="--", color=C_PRUNE,
        label="裸剪，幅值")
ax.plot(prune_pct, bare_rnd, marker=M_BARE_RND, ls="--", color=C_FP32,
        label="裸剪，随机")
ax.plot(prune_pct, ft_mag, marker=M_FT_MAG, ls="-", color=C_PRUNE, lw=2,
        label="LoRA 微调，幅值")
ax.plot(prune_pct, ft_rnd, marker=M_FT_RND, ls="-", color=C_FP32, lw=2,
        label="LoRA 微调，随机")
ax.axhline(BASELINE, color=C_GRAY, ls=":", lw=1, label=f"基线 F1={BASELINE}")
ax.grid(True, ls="--", alpha=0.4)
ax.set_xlabel("FFN 剪枝率 (%)")
ax.set_ylabel("F1 分数")
ax.set_ylim(0.50, 0.67)
ax.set_xticks(prune_pct)
ax.legend(fontsize=12, loc="lower left")
ax.set_title("结构化剪枝分析")

fig.tight_layout()
path = OUT / "4_pruning_curve.png"
fig.savefig(path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"✓ {path}")
