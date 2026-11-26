#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3 —— pRDA / PVCA 方差分解：把混在一起的信号拆干净
================================================================================
推文逻辑第③步：PERMANOVA 分别告诉"批次多少、分组多少",但二者可能共线混在一起。
pRDA 把响应矩阵 Y（这里用 PCoA 坐标，也可用 Hellinger）对 分组+批次 做回归，
一次性拆成四块：

  ┌─────────────┬─────────────┬──────────────┬────────┐
  │ 分组独有     │ 批次独有     │ 二者共享(共线) │ 残差    │
  │ 控制批次后   │ 控制分组后   │              │ 未解释  │
  │ 还剩多少分组 │ 还剩多少批次 │              │        │
  └─────────────┴─────────────┴──────────────┴────────┘

并在"批次内置换分组 / 分组内置换批次"下做分层置换检验，得到偏效应 P 值。

示例结果：批次独有 ≈ 65%，分组独有 ≈ 12% → 扣掉批次后，疾病信号所剩无几。

运行：python step3_prda.py
输出：output/step3_prda_partition.png（堆叠条形图）
      output/step3_prda_result.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from demo_common import (make_demo_data, get_distance, pcoa, partition,
                         setup_cn_font)

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    has_cn = setup_cn_font()
    counts, meta = make_demo_data()
    group = meta["group"].values
    batch = meta["batch"].values

    D = get_distance(counts, "bray")
    Y, ev = pcoa(D)                      # 响应矩阵 = PCoA 坐标（全部正轴）

    res = partition(Y, group, batch, nperm=499, seed=1)

    L = lambda cn, en: cn if has_cn else en
    print("  R²(分组单独) = %.3f   R²(批次单独) = %.3f   R²(共同) = %.3f"
          % (res["r2_group"], res["r2_batch"], res["r2_both"]))
    print("  偏分组(控制批次) = %.3f  P=%.3f" % (res["partial_group"], res["p_group"]))
    print("  偏批次(控制分组) = %.3f  P=%.3f" % (res["partial_batch"], res["p_batch"]))
    print("  共享(共线) = %.3f   残差 = %.3f" % (res["shared"], res["resid"]))

    # 结果表
    row = {
        "响应": "PCoA坐标(Bray)", "样本数": Y.shape[0],
        "R2_分组单独": res["r2_group"], "R2_批次单独": res["r2_batch"],
        "R2_共同": res["r2_both"],
        "偏分组_控制批次": res["partial_group"], "偏分组_P": res["p_group"],
        "偏批次_控制分组": res["partial_batch"], "偏批次_P": res["p_batch"],
        "共享_共线": res["shared"], "残差": res["resid"],
    }
    pd.DataFrame([row]).to_csv(os.path.join(OUT, "step3_prda_result.csv"),
                               index=False, encoding="utf-8-sig")

    # 堆叠条形图
    pg = max(res["partial_group"], 0)
    pb = max(res["partial_batch"], 0)
    sh = max(res["shared"], 0)
    rs = max(res["resid"], 0)
    labels = [L("分组独有\n(控制批次)", "Group only\n| batch"),
              L("批次独有\n(控制分组)", "Batch only\n| group"),
              L("二者共享\n(共线)", "Shared\n(collinear)"),
              L("残差", "Residual")]
    vals = [pg, pb, sh, rs]
    colors = ["#2E6DA4", "#C0504D", "#8E6BA8", "#D9D9D9"]

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    left = 0.0
    for lab, v, c in zip(labels, vals, colors):
        ax.barh([0], [v], left=left, height=0.5, color=c, edgecolor="white")
        if v > 0.02:
            ax.text(left + v/2, 0, f"{100*v:.1f}%", ha="center", va="center",
                    fontsize=10, color="#333" if c == "#D9D9D9" else "white",
                    fontweight="bold")
        left += v
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(L("解释的变异比例", "Explained variance proportion"), fontsize=11)
    ax.set_title(L("pRDA 方差分解：批次是主要结构来源",
                   "pRDA variation partitioning: batch dominates"),
                 fontsize=12, fontweight="bold")
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors],
              labels, loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.35, top=0.82)
    p = os.path.join(OUT, "step3_prda_partition.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print("[完成]", p)


if __name__ == "__main__":
    main()
