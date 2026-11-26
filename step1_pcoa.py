#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1 —— PCoA 可视化：先看现象
================================================================================
推文逻辑第①步：把高维菌群组成用 PCoA 压到二维，肉眼看样本怎么分布。
同一份数据画两张图：
  左图 按「分组 IM/DN」着色 —— 如果疾病真的主导，点应该按颜色分开；
  右图 按「批次 B1/B2/B3」着色 —— 如果批次主导，点会按颜色抱成三团。

示例结果：左图颜色混在一起，右图三种颜色各自抱团 →
          "你以为的疾病差异，其实是批次"。

运行：python step1_pcoa.py
输出：output/step1_pcoa_by_group.png / step1_pcoa_by_batch.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from demo_common import make_demo_data, get_distance, pcoa, setup_cn_font

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

COLORS = ["#2E6DA4", "#C0504D", "#3C9A5F"]
MARKERS = ["o", "s", "^"]


def plot_panel(ax, co, ev, labels, title, color_map, marker_map):
    for g in sorted(set(labels)):
        ii = np.array([i for i, x in enumerate(labels) if x == g])
        ax.scatter(co[ii, 0], co[ii, 1], s=64, color=color_map[g],
                   marker=marker_map[g], edgecolor="white", linewidth=0.7,
                   label=f"{g} (n={len(ii)})", zorder=3, alpha=0.9)
        # 组质心
        ax.scatter(co[ii, 0].mean(), co[ii, 1].mean(), s=170,
                   color=color_map[g], marker="X", edgecolor="black",
                   linewidth=0.9, zorder=4)
    ax.set_xlabel(f"PCo1 ({100*ev[0]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PCo2 ({100*ev[1]:.1f}%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=True, fontsize=9, loc="best")


def main():
    has_cn = setup_cn_font()
    counts, meta = make_demo_data()
    D = get_distance(counts, "bray")
    co, ev = pcoa(D)

    group = meta["group"].values
    batch = meta["batch"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    # 左：按分组
    gc = {g: COLORS[i % 3] for i, g in enumerate(sorted(set(group)))}
    gm = {g: "o" for g in sorted(set(group))}
    plot_panel(axes[0], co, ev, group,
               "① 按疾病分组着色（IM / DN）" if has_cn else "1) Colored by group (IM/DN)",
               gc, gm)

    # 右：按批次
    bc = {b: COLORS[i % 3] for i, b in enumerate(sorted(set(batch)))}
    bm = {b: MARKERS[i % 3] for i, b in enumerate(sorted(set(batch)))}
    plot_panel(axes[1], co, ev, batch,
               "② 按测序批次着色（B1 / B2 / B3）" if has_cn else "2) Colored by batch (B1/B2/B3)",
               bc, bm)

    note = ("同一份数据：按分组看混杂，按批次看抱团" if has_cn
            else "Same data: mixed by group, clustered by batch")
    fig.suptitle(note + "  ·  Bray-Curtis PCoA", fontsize=13, y=1.01, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT, "step1_pcoa_by_group_and_batch.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("[完成] PCoA 图：", path)
    print(f"        前两轴解释率 PCo1={100*ev[0]:.1f}%  PCo2={100*ev[1]:.1f}%")
    print("        ← 左图按分组着色混杂；右图按批次着色明显抱团")


if __name__ == "__main__":
    main()
