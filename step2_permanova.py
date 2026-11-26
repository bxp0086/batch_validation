#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2 —— PERMANOVA 定量：给现象一个数字
================================================================================
推文逻辑第②步：PCoA 只是"看着像",PERMANOVA 才能回答"批次解释了多少变异"。

做两件事：
  A) 三种距离下的 R² 对比（bray / jaccard / euclid）
     - Bray-Curtis（丰度）：批次 vs 分组各解释多少
     - Jaccard（有无）：如果批次在 Jaccard 下也很大，说明批次连物种"在不在"都改变了
  B) 阴性对照（同批次内随机拆两组），看零分布
     - 真实批次 R² 应远高于随机拆分 R² → 批次差异是真信号，不是方法假象

R² 口径：距离平方分解 (SS_all - SS_within)/SS_all，与 sens_gates.permanova 一致。

运行：python step2_permanova.py
输出：output/step2_permanova_bar.png（R² 对比柱状图）
      output/step2_null_control.png（阴性对照零分布）
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from demo_common import (make_demo_data, get_distance, permanova,
                         setup_cn_font, jaccard_dist, bray_curtis, rel_abund)

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    has_cn = setup_cn_font()
    counts, meta = make_demo_data()
    group = meta["group"].values
    batch = meta["batch"].values

    # ---------- A) 三种距离下：批次 vs 分组 的 PERMANOVA R² ----------
    rows = []
    for metric in ["bray", "jaccard", "euclid"]:
        D = get_distance(counts, metric)
        rb, _, pb = permanova(D, batch, nperm=499, seed=1)
        rg, _, pg = permanova(D, group, nperm=499, seed=1)
        rows.append({"距离": metric, "批次R2": rb, "批次P": pb,
                     "分组R2": rg, "分组P": pg})
        print(f"  {metric:8s} 批次 R²={rb:.3f} (P={pb:.3f})  |  "
              f"分组 R²={rg:.3f} (P={pg:.3f})")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "step2_permanova_result.csv"),
              index=False, encoding="utf-8-sig")

    # 柱状图
    x = np.arange(len(df))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, df["批次R2"], w, label=(("批次 batch" if not has_cn else "测序批次")),
           color="#C0504D", edgecolor="white")
    ax.bar(x + w/2, df["分组R2"], w, label=(("分组 group" if not has_cn else "疾病分组")),
           color="#2E6DA4", edgecolor="white")
    for i, r in df.iterrows():
        ax.text(i - w/2, r["批次R2"] + 0.01, f"{r['批次R2']:.2f}",
                ha="center", fontsize=9)
        ax.text(i + w/2, r["分组R2"] + 0.01, f"{r['分组R2']:.2f}",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["距离"])
    ax.set_ylabel("PERMANOVA R²" if not has_cn else "解释变异比例 (R²)", fontsize=11)
    ax.set_ylim(0, 1.0)
    title = "批次 vs 分组：解释的群落变异（PERMANOVA）" if has_cn else "Batch vs Group: explained variation (PERMANOVA)"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    p1 = os.path.join(OUT, "step2_permanova_bar.png")
    fig.savefig(p1, dpi=220)
    plt.close(fig)
    print("[完成]", p1)

    # ---------- B) 阴性对照：同批次内随机拆两组 ----------
    rng = np.random.default_rng(7)
    # 用 B1 批次样本（同一批次内本应"无批次差异"）
    sub = counts.loc[:, meta.index[meta["batch"] == "B1"]]
    n = sub.shape[1]
    dj = jaccard_dist(sub >= 10)
    db = bray_curtis(rel_abund(sub))
    r2j_real, _, _ = permanova(dj, np.arange(n) % 2, 199, seed=1)  # 仅占位，下面用真实标签
    null_j, null_b = [], []
    for it in range(60):
        lab = np.zeros(n, dtype=int)
        perm = rng.permutation(n)
        lab[perm[:n//2]] = 0
        lab[perm[n//2:]] = 1
        rj, _, _ = permanova(dj, lab, 99, seed=it)
        rb2, _, _ = permanova(db, lab, 99, seed=it)
        null_j.append(rj)
        null_b.append(rb2)

    real_batch_bray = df.loc[df["距离"] == "bray", "批次R2"].values[0]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(null_b, bins=15, color="#8E6BA8", edgecolor="white", alpha=0.85,
            label=(("随机拆分 R² (null)" if not has_cn else "同批次随机拆分 R²（零分布）")))
    ax.axvline(np.median(null_b), color="#333", linestyle="--", linewidth=1.4,
               label=(("null median" if not has_cn else "零分布中位数")))
    ax.axvline(real_batch_bray, color="#C0504D", linewidth=2.2,
               label=(("real batch R²" if not has_cn else "真实批次 R²")))
    t = ("真实批次 R² 远超随机拆分 → 批次差异是真信号" if has_cn
         else "Real batch R² >> null → batch effect is real")
    ax.set_title(t, fontsize=12, fontweight="bold")
    ax.set_xlabel("Bray-Curtis R²", fontsize=11)
    ax.set_ylabel("频数" if has_cn else "Count", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    p2 = os.path.join(OUT, "step2_null_control.png")
    fig.savefig(p2, dpi=220)
    plt.close(fig)
    print("[完成]", p2)
    print(f"        随机拆分 R² 中位数 = {np.median(null_b):.3f}；"
          f"真实批次 R² = {real_batch_bray:.3f}")


if __name__ == "__main__":
    main()
