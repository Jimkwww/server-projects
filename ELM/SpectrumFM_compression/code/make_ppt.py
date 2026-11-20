#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组装 SpectrumFM 压缩实验 5 页汇报 PPT。

依赖：python-pptx。图来自 figures/（先跑 make_figures.py 生成）。
中文文本在 pptx 里仅存字符串+字体名，在用户本机（有中文字体）渲染，服务器无字体不影响。
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "SpectrumFM_压缩实验_汇报.pptx")

# ---- 配色（与 make_figures.py 一致）----
C_FP32 = "4c72b0"; C_FP16 = "dd8452"; C_INT8 = "55a868"
C_INT4 = "c44e52"; C_PRUNE = "8172b2"; C_ULT = "ccb974"; C_GRAY = "888888"
C_DARK = "1a1a1a"; C_TITLE = "20364d"; C_ACCENT = "4c72b0"
FONT = "Microsoft YaHei"

EMU_W, EMU_H = Inches(13.333), Inches(7.5)


def _rgb(hexs):
    return RGBColor.from_string(hexs.lstrip("#"))


def _set_font(run, size=14, bold=False, color=C_DARK, italic=False):
    f = run.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = _rgb(color)
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", FONT)


def add_text(slide, l, t, w, h, lines, anchor=MSO_ANCHOR.TOP):
    """lines: list of dict(text,size,bold,color,bullet,space_before)."""
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    first = True
    for ln in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = ln.get("align", PP_ALIGN.LEFT)
        if ln.get("space_before"):
            p.space_before = Pt(ln["space_before"])
        if ln.get("space_after"):
            p.space_after = Pt(ln["space_after"])
        segs = ln["text"] if isinstance(ln["text"], list) else [ln["text"]]
        for seg in segs:
            r = p.add_run()
            r.text = seg
            _set_font(r, ln.get("size", 14), ln.get("bold", False),
                      ln.get("color", C_DARK), ln.get("italic", False))
    return tb


def add_pic(slide, name, l, t, w=None, h=None):
    """按宽或高等比插入图片，返回实际占位尺寸。"""
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return None
    if w is not None:
        return slide.shapes.add_picture(p, Inches(l), Inches(t), width=Inches(w))
    return slide.shapes.add_picture(p, Inches(l), Inches(t), height=Inches(h))


def add_rect(slide, l, t, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    sp = slide.shapes.add_shape(shape, Inches(l), Inches(t), Inches(w), Inches(h))
    sp.fill.solid()
    sp.fill.fore_color.rgb = _rgb(fill)
    if line:
        sp.line.color.rgb = _rgb(line)
        sp.line.width = Pt(1.0)
    else:
        sp.line.fill.background()
    sp.shadow.inherit = False
    return sp


def add_title(slide, text, sub=None):
    add_rect(slide, 0, 0, 13.333, 0.86, "ffffff")
    add_rect(slide, 0, 0, 0.16, 0.86, C_ACCENT)
    add_rect(slide, 0.16, 0.82, 13.17, 0.04, C_ACCENT)
    add_text(slide, 0.4, 0.08, 11.5, 0.7,
             [dict(text=text, size=24, bold=True, color=C_TITLE)])
    if sub:
        add_text(slide, 0.4, 0.58, 12.6, 0.24,
                 [dict(text=sub, size=11, color=C_GRAY)])


def add_table(slide, l, t, w, data, col_w, header_fill=C_ACCENT,
              font_size=11, row_h=0.34, header_size=11.5):
    """data: list[list[str]]，首行为表头。"""
    rows, cols = len(data), len(data[0])
    shp = slide.shapes.add_table(rows, cols, Inches(l), Inches(t),
                                 Inches(w), Inches(row_h * rows))
    tbl = shp.table
    tbl.first_row = False
    tbl.horz_banding = False
    for c in range(cols):
        tbl.columns[c].width = Inches(col_w[c])
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.04)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(val)
            if i == 0:
                _set_font(r, header_size, True, "ffffff")
                cell.fill.solid(); cell.fill.fore_color.rgb = _rgb(header_fill)
            else:
                _set_font(r, font_size, False, C_DARK)
                cell.fill.solid()
                cell.fill.fore_color.rgb = _rgb("ffffff" if i % 2 else "eef2f7")
    return shp


prs = Presentation()
prs.slide_width = EMU_W
prs.slide_height = EMU_H
BLANK = prs.slide_layouts[6]

# =====================================================================
# Slide 1 —— 模型参数构成 + 每层可压缩方式
# =====================================================================
s = prs.slides.add_slide(BLANK)
add_title(s, "SpectrumFM 模型压缩 · 参数构成与压缩手段",
          "Conformer 混合 encoder（16 层，d=256，H=4，d_feed=512）+ LoRA ｜ 任务 AMC ｜ 18.06M 参数 / 69.2MB")

add_rect(s, 0.4, 1.05, 6.4, 5.9, "f7f9fc")
add_text(s, 0.55, 1.15, 6.1, 0.4, [dict(text="参数构成（FP32 口径，1 参数 = 4 字节）", size=15, bold=True, color=C_TITLE)])
comp = [
    ["组件", "参数量", "大小", "可采用的压缩方式"],
    ["FFN Linear (fc1/fc2×16)", "8.41M (47%)", "33.6MB", "量化 INT8/INT4 · 剪枝中间维"],
    ["Attention Linear (Q/K/V/O)", "4.20M (23%)", "16.8MB", "量化 · 剪注意力头(未做)"],
    ["卷积 (pointwise/depthwise)", "3.18M (18%)", "12.7MB", "pointwise 量化 · 剪通道"],
    ["LoRA A/B (微调时)", "1.31M (7%)", "5.2MB", "merge 后并入权重、消失"],
    ["GRU (双向 256→256)", "0.79M (4%)", "3.1MB", "动态量化"],
    ["其他 (LN/relpos/分类头)", "0.10M", "0.2MB", "保持 FP32"],
    ["合计", "18.06M", "69.2MB", "—"],
]
add_table(s, 0.55, 1.6, 6.1, comp, col_w=[2.05, 1.15, 0.9, 2.0], font_size=10.5, row_h=0.34)
add_text(s, 0.55, 4.9, 6.1, 1.9, [
    dict(text="三条关键规律", size=14, bold=True, color=C_TITLE),
    dict(text="① FFN 占 47%，是体积与剪枝的主战场；卷积 pointwise 是大头（可量化）。", size=11.5, space_before=4),
    dict(text="② GRU(3.1MB) 是唯一还以 FP32 实存的实质部分，量化它可再省 ~2.3MB。", size=11.5, space_before=4),
    dict(text="③ LoRA 只在微调期存在，部署 merge 后自动消失（省 5.2MB）。", size=11.5, space_before=4),
])

add_text(s, 7.05, 1.1, 5.9, 0.4, [dict(text="参数占比可视化", size=15, bold=True, color=C_TITLE)])
pic = add_pic(s, "1_param_composition.png", 7.05, 1.6, w=5.85)
if pic:
    add_text(s, 7.05, 4.15, 5.85, 0.35,
             [dict(text="左：各组件占比 ｜ 右：FFN 每层 2×FFN×16 层展开", size=10.5, color=C_GRAY)])
add_text(s, 7.05, 4.6, 5.85, 2.2, [
    dict(text="关键点：FFN + Attention + 卷积 = 88% 参数", size=12.5, bold=True, color=C_INT4, space_before=4),
    dict(text="三者均可量化；FFN 宽度可结构化剪枝（本项目的剪枝战场）。", size=11.5, space_before=4),
])

# =====================================================================
# Slide 2 —— 可采用的压缩手段
# =====================================================================
s = prs.slides.add_slide(BLANK)
add_title(s, "可采用的压缩手段",
          "FP16 半精度 · 静态量化 · 动态量化 · 结构化剪枝（随机 / 最大值）")

# 四个方法块（2×2）
blocks = [
    ("FP16 半精度", C_FP16, "权重/激活 32→16bit：体积 -50%，Tensor Core 加速，精度零损失。部署成本最低的“免费午餐”。"),
    ("静态量化（PTQ）", C_INT8, "用标定集预统计激活 scale/zero-point，权重+激活都 INT8；推理无标定开销，适合输入分布固定的场景。"),
    ("动态量化", C_INT4, "只量化权重，激活在推理时逐批动态算 scale。实现简单、对激活分布鲁棒，代价是每批多一步量化开销。"),
    ("结构化剪枝", C_PRUNE, "按 L1 范数（magnitude）或随机（random）整列删除 FFN 中间维神经元，真实省 FLOPs/显存。注意力头剪枝(4头)本项目未做，集中在 FFN 宽度。"),
]
bx, by, bw, bh = 0.4, 1.12, 6.25, 1.3
for i, (t, c, d) in enumerate(blocks):
    r = i // 2; cix = i % 2
    l = bx + cix * (bw + 0.25); tp = by + r * (bh + 0.18)
    add_rect(s, l, tp, 0.12, bh, c)
    add_rect(s, l + 0.12, tp, bw - 0.12, bh, "f7f9fc")
    add_text(s, l + 0.25, tp + 0.08, bw - 0.4, bh - 0.15, [
        dict(text=t, size=15, bold=True, color=c),
        dict(text=d, size=11, space_before=3),
    ])

# 底部两张佐证图（按高约束，避免溢出）
add_text(s, 0.4, 4.05, 6.25, 0.35,
         [dict(text="量化：INT4 上 QAT 比 PTQ 好 0.93 点", size=13, bold=True, color=C_TITLE)])
pic = add_pic(s, "6_qat_vs_ptq.png", 0.4, 4.45, h=2.75)
add_text(s, 4.6, 4.05, 5.9, 0.35,
         [dict(text="剪枝：25% 几乎无损，微调可全部恢复", size=13, bold=True, color=C_TITLE)])
pic = add_pic(s, "4_pruning_curve.png", 4.6, 4.45, h=2.75)

# =====================================================================
# Slide 3 —— 模型大小压缩实验
# =====================================================================
s = prs.slides.add_slide(BLANK)
add_title(s, "模型大小压缩实验",
          "体积从 69.2MB 一路压缩到 10.5MB（6.6×），精度几乎无损")

add_rect(s, 0.4, 1.05, 6.4, 5.9, "f7f9fc")
add_text(s, 0.55, 1.12, 6.1, 0.4, [dict(text="体积压缩排行榜（基线 69.2MB）", size=14.5, bold=True, color=C_TITLE)])
vol = [
    ["排名", "配置", "体积", "压缩率", "F1"],
    ["1", "剪40%+INT4 QAT+打包", "10.5MB", "6.6×", "0.6426"],
    ["2", "INT4 QAT+打包", "12.2MB", "5.7×", "0.6494"],
    ["3", "INT8 完全体", "17.1MB", "4.05×", "0.6510"],
    ["4", "merge 全动态", "26.0MB", "2.66×", "0.6504"],
    ["5", "INT8 动态", "31.0MB", "2.23×", "0.6525"],
    ["6", "FP16", "34.7MB", "2.0×", "0.6475"],
    ["—", "FP32 基线", "69.2MB", "1×", "0.6474"],
]
add_table(s, 0.55, 1.58, 6.1, vol, col_w=[0.5, 1.95, 0.95, 0.85, 0.8], font_size=10.5, row_h=0.32)
add_text(s, 0.55, 4.35, 6.1, 2.4, [
    dict(text="体积账怎么算出来的", size=13.5, bold=True, color=C_TITLE),
    dict(text="① 量化：INT8 每参数 4B→1B，INT4+位打包 4B→0.5B。", size=11, space_before=4),
    dict(text="② 打包 + 丢 master weight：部署只存 _qweight + 非量化部分。", size=11, space_before=4),
    dict(text="③ 剪枝：先砍 40% 参数，再量化，体积叠加。", size=11, space_before=4),
    dict(text="④ 剩余 FP32 = GRU 3.1MB（量化可再省 ~2.3MB）。", size=11, space_before=4),
])

add_pic(s, "2_volume_compression.png", 7.0, 1.15, w=5.85)
add_pic(s, "3_f1_accuracy.png", 7.0, 4.15, w=5.85)

# =====================================================================
# Slide 4 —— 模型推理加速实验
# =====================================================================
s = prs.slides.add_slide(BLANK)
add_title(s, "模型推理加速实验",
          "6 模型 × 7 档 batch 速度曲线 ｜ torchao 官方实现对照")

speed_png = os.path.join(FIG, "7_speed_curves.png")
if os.path.exists(speed_png):
    add_pic(s, "7_speed_curves.png", 0.4, 1.15, w=7.1)
else:
    add_rect(s, 0.4, 1.15, 7.1, 4.0, "f2f4f7")
    add_text(s, 0.4, 2.6, 7.1, 1.0, [
        dict(text="速度曲线图（待生成）", size=16, bold=True, color=C_GRAY, align=PP_ALIGN.CENTER),
        dict(text="跑 latency_curves_all.py 存 CSV 后，由 make_figures.py 生成", size=11, color=C_GRAY, align=PP_ALIGN.CENTER, space_before=4),
    ])
add_pic(s, "8_torchao_compare.png", 7.7, 1.15, w=5.25)

add_text(s, 0.4, 5.35, 12.6, 1.9, [
    dict(text="速度维度的五条结论", size=14, bold=True, color=C_TITLE),
    dict(text="① 唯一显著改善 b1 的手段是 torch.compile 融合（-31%，8.10→5.55ms）；量化/剪枝动不了 b1 的 kernel-launch 税。", size=11, space_before=4),
    dict(text="② 速度之王 = BF16+编译（9806/s），比 FP16 未编译高 50%。", size=11, space_before=4),
    dict(text="③ 手写量化推理态仍慢于 FP32：b1 ~1.7×、b256 吞吐 -34~42%（量化账本的 element-wise 碎片化开销）——公正对比须用官方算子。", size=11, space_before=4),
    dict(text="④ 这类“小而碎”模型：量化赚体积、不赚速度（小矩阵吃不到 Tensor Core 红利）。", size=11, space_before=4),
])

# =====================================================================
# Slide 5 —— 实验结果总结
# =====================================================================
s = prs.slides.add_slide(BLANK)
add_title(s, "实验结果总结",
          "压缩手段 × 收益各占一格，无单一方案全赢")

add_text(s, 0.4, 1.02, 12.5, 0.35, [dict(text="压缩手段 × 收益矩阵", size=15, bold=True, color=C_TITLE)])
mx = [
    ["压缩手段", "体积", "速度(b256)", "b1 延迟", "精度", "一句话结论"],
    ["FP16", "-50% 34.7MB", "+171% 5427/s", "-14%", "零损失", "免费午餐"],
    ["INT8 完全体", "-75% 17.1MB", "CPU 不加速", "—", "零损失", "体积逼近下限"],
    ["INT4 QAT+打包", "-82% 12.2MB", "无增益(-42%)", "慢1.7×", "+0.20", "QAT 拉回精度"],
    ["剪枝25~70%+微调", "-12~35% 参数", "+21%(剪60%)", "≈0%", "≥基线", "零代价剪枝"],
    ["剪40%+QAT+打包(终极)", "-85% 10.5MB", "—", "—", "-0.48", "体积-精度综合最优"],
    ["BF16+compile", "-53% 32.2MB", "+448% 9806/s", "-31%", "零损失", "速度之王"],
]
add_table(s, 0.4, 1.4, 12.5, mx, col_w=[2.0, 1.7, 1.7, 1.2, 1.3, 2.2, 2.4], font_size=10.5, row_h=0.38)

add_text(s, 0.4, 4.15, 7.2, 3.0, [
    dict(text="四条总结论", size=14.5, bold=True, color=C_TITLE),
    dict(text="① 没有单一方案全赢：体积归 INT4+打包、速度归 BF16+融合、b1 归融合、精度归 INT8/QAT——按部署约束组合。", size=11.5, space_before=5),
    dict(text="② 量化对小模型的收益是体积而非速度（小而碎结构吃不到 Tensor Core 红利）。", size=11.5, space_before=5),
    dict(text="③ 含训练的压缩实验必须多 seed + 未压缩对照组（单次 ±0.5 点可翻转结论）。", size=11.5, space_before=5),
    dict(text="④ QAT 价值实测兑现：INT4 上比 PTQ 好 0.93 点，是低比特量化必需品。", size=11.5, space_before=5),
])

add_rect(s, 7.8, 4.15, 5.15, 3.0, "fdf6ec")
add_text(s, 8.0, 4.3, 4.8, 2.7, [
    dict(text="终极形态（端侧部署推荐）", size=14, bold=True, color=C_ULT),
    dict(text="剪 40% + INT4 QAT + 位打包", size=13.5, bold=True, color=C_DARK, space_before=6),
    dict(text="10.5MB（6.6× 压缩）· F1 0.6426", size=12.5, bold=True, color=C_INT4, space_before=6),
    dict(text="体积-精度综合最优，全链路已验证。", size=11, color=C_GRAY, space_before=6),
    dict(text="可选后续：TensorRT 部署级引擎化（已学概念，时间富余再做）。", size=10.5, color=C_GRAY, space_before=8),
])

prs.save(OUT)
print("已生成:", OUT)
print("  速度曲线图:", "已包含" if os.path.exists(speed_png) else "未包含（缺 7_speed_curves.png）")
