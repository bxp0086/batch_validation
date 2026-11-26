#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_common.py —— 批次效应推文示例的公共模块
================================================================================
只依赖 numpy / pandas / scipy，不依赖 skbio 或任何真实数据。
包含两部分：

  1) 模拟数据生成 make_demo_data()
     - 造一个「分组信号弱、批次信号强」的菌群计数表，正好演示推文的核心观点：
       "看起来像疾病的差异，其实主要是批次"。
     - 数据结构：行为菌种，列为样本；元数据含 分组(IM/DN) 与 批次(B1/B2/B3)。

  2) 与真实分析完全同口径的统计函数（复刻自 sens_gates.py）
     - rel_abund / bray_curtis / jaccard_dist / euclid_dist：相对丰度与三种距离
     - permanova：距离平方分解的 R²、伪 F、置换 P
     - pcoa：距离矩阵双中心化 → 主坐标与解释率
     - pRDA：dummy / r2_of / build_both / partition（含分层置换检验）

三步脚本 step1/step2/step3 都 import 本模块，保证口径一致。
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


# ------------------------------------------------------------------ 中文字体
def setup_cn_font():
    """尽力设置中文字体；返回是否找到。找不到时图仍可出，仅中文可能显示为方块。"""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    cn = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei",
          "WenQuanYi Micro Hei", "PingFang SC", "Arial Unicode MS"]
    avail = {f.name for f in fm.fontManager.ttflist}
    font = next((f for f in cn if f in avail), None)
    if font:
        plt.rcParams["font.sans-serif"] = [font]
        plt.rcParams["axes.unicode_minus"] = False
    return font is not None


# ------------------------------------------------------------------ 1. 模拟数据
def make_demo_data(n_per_group_batch=8, n_species=80, depth=5000, seed=42):
    """
    生成示例用的菌群计数表 + 元数据。

    设计意图（关键，决定了示例的"故事"）：
      - 分组 IM / DN 之间只放很弱的真实差异（约 8% 的菌有 1.3 倍变化）；
      - 批次 B1/B2/B3 之间放很强的系统偏移（约 30% 的菌有 2~3.5 倍变化）；
      - 3 个批次里 IM/DN 大致均衡（不完全正交，留一点共线更真实）。
    ==> PCoA 上样本会"按批次聚团、按分组混杂"，与推文主题一致。

    返回
      counts : DataFrame  菌种 × 样本 的计数（整数）
      meta   : DataFrame  样本 × [sample, group, batch, site]
    """
    rng = np.random.default_rng(seed)
    groups = ["IM", "DN"]
    batches = ["B1", "B2", "B3"]

    # —— 全局固定的结构（所有样本共享，这样批次/分组信号才不会被个体噪声淹没）——
    base = np.exp(rng.normal(0, 1.3, n_species))          # 各菌基线丰度
    # 每个批次一套固定倍率：约 35% 的菌被系统性抬高/压低（批次效应，强）
    batch_effect = {}
    for b in batches:
        eff = np.ones(n_species)
        pick = rng.choice(n_species, size=int(0.35 * n_species), replace=False)
        eff[pick] *= np.exp(rng.normal(0, 1.1, len(pick)))
        batch_effect[b] = eff
    # 每个分组一套固定倍率：约 8% 的菌小幅变化（真实分组信号，弱）
    group_effect = {}
    for g in groups:
        eff = np.ones(n_species)
        pick = rng.choice(n_species, size=int(0.08 * n_species), replace=False)
        eff[pick] *= np.exp(rng.normal(0, 0.5, len(pick)))
        group_effect[g] = eff

    rows, metas = [], []
    for g in groups:
        for b in batches:
            # 每(分组,批次)放 n_per_group_batch 个样本,组间/批次间大致均衡
            for k in range(n_per_group_batch):
                sid = f"{g}_{b}_{k+1:02d}"
                # 个体噪声是所有随机性的来源（较小），其余为固定结构
                noise = np.exp(rng.normal(0, 0.22, n_species))
                lam = base * batch_effect[b] * group_effect[g] * noise
                lam = lam / lam.sum()
                cnt = rng.multinomial(depth, lam)
                rows.append(cnt)
                metas.append({"sample": sid, "group": g, "batch": b,
                              "site": "SA", "patient": f"{g}_{b}_P{k//2+1}"})

    counts = pd.DataFrame(
        np.array(rows).T,
        index=[f"Species_{i+1:03d}" for i in range(n_species)],
        columns=[m["sample"] for m in metas],
    )
    meta = pd.DataFrame(metas).set_index("sample")
    return counts, meta


# ------------------------------------------------------------------ 2. 距离
def rel_abund(counts):
    """菌种×样本计数 → 相对丰度（按列归一）"""
    tot = counts.sum(axis=0).replace(0, np.nan)
    return counts.div(tot, axis=1).fillna(0.0)


def bray_curtis(prop):
    x = prop.T.values.astype(float)
    x = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
    D = np.zeros((x.shape[0], x.shape[0]))
    for k in range(x.shape[1]):
        v = x[:, k]
        if v.max() <= 0:
            continue
        D += np.abs(v[:, None] - v[None, :])
    D /= 2.0
    np.fill_diagonal(D, 0.0)
    return pd.DataFrame(D, index=prop.columns, columns=prop.columns)


def jaccard_dist(binary):
    """只看检出/未检出的 Jaccard 距离"""
    x = (np.asarray(binary, dtype=float) > 0).astype(float)
    inter = x.T @ x
    size = x.sum(axis=0)
    union = size[:, None] + size[None, :] - inter
    D = 1.0 - np.divide(inter, np.maximum(union, 1e-12))
    np.fill_diagonal(D, 0.0)
    index = binary.columns if hasattr(binary, "columns") else None
    return pd.DataFrame(D, index=index, columns=index)


def euclid_dist(counts):
    """log(相对丰度+eps) 的欧氏距离，与真实脚本口径一致"""
    P = rel_abund(counts)
    L = np.log10(P.values + 1e-6).T
    D = squareform(pdist(L))
    return pd.DataFrame(D, index=P.columns, columns=P.columns)


def get_distance(counts, metric, min_count=10):
    if metric == "bray":
        return bray_curtis(rel_abund(counts))
    if metric == "jaccard":
        return jaccard_dist(counts >= min_count)
    if metric == "euclid":
        return euclid_dist(counts)
    raise ValueError("未知距离：" + metric)


# ------------------------------------------------------------------ 3. PERMANOVA
def permanova(D, labels, nperm=999, seed=42):
    """基于距离矩阵的 PERMANOVA，返回 (R², 伪F, P)。"""
    labels = np.asarray(labels)
    M = np.asarray(D, dtype=float)
    n = len(labels)
    D2 = M ** 2
    groups = np.unique(labels)
    a = len(groups)
    if a < 2 or n - a < 1:
        return np.nan, np.nan, np.nan
    iu = np.triu_indices(n, 1)
    sst = D2[iu].sum() / n

    def ssw_of(lab):
        s = 0.0
        for g in groups:
            ii = np.where(lab == g)[0]
            if len(ii) < 2:
                continue
            s += D2[np.ix_(ii, ii)].sum() / 2.0 / len(ii)
        return s

    ssw = ssw_of(labels)
    F = ((sst - ssw) / (a - 1)) / max(ssw / (n - a), 1e-12)
    R2 = (sst - ssw) / sst
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(nperm):
        s = ssw_of(rng.permutation(labels))
        if ((sst - s) / (a - 1)) / max(s / (n - a), 1e-12) >= F:
            ge += 1
    return float(R2), float(F), float((1 + ge) / (1 + nperm))


# ------------------------------------------------------------------ 4. PCoA
def pcoa(D):
    """距离矩阵双中心化 → (坐标, 各轴解释率)。仅保留正特征值。"""
    M = np.asarray(D, dtype=float)
    A = -0.5 * (M ** 2)
    G = A - A.mean(0)[None, :] - A.mean(1)[:, None] + A.mean()
    vals, vecs = np.linalg.eigh(G)
    o = np.argsort(vals)[::-1]
    vals, vecs = vals[o], vecs[:, o]
    pos = vals > 1e-9
    co = vecs[:, pos] * np.sqrt(vals[pos])
    tot = vals[vals > 0].sum() or 1.0
    return co, vals[pos] / tot


# ------------------------------------------------------------------ 5. pRDA 方差分解
def dummy(labels, with_intercept=True):
    """标签 → 设计矩阵（截距 + k-1 哑变量）"""
    labels = np.asarray(labels)
    levels = sorted(set(labels))
    if len(levels) < 2:
        return None
    cols = [np.ones(len(labels))] if with_intercept else []
    for lv in levels[1:]:
        cols.append((labels == lv).astype(float))
    return np.column_stack(cols)


def r2_of(Y, X):
    """R² = trace(Yc'HYc)/trace(Yc'Yc)，QR 处理秩亏。X 需含截距。"""
    if X is None:
        return 0.0
    Yc = Y - Y.mean(axis=0, keepdims=True)
    sst = np.trace(Yc.T @ Yc)
    if sst <= 0:
        return np.nan
    Q, _ = np.linalg.qr(X)
    r = np.linalg.matrix_rank(X)
    Q = Q[:, :r]
    ssr = np.trace(Yc.T @ Q @ Q.T @ Yc)
    return float(np.clip(ssr / sst, 0, 1))


def build_both(Xg, Xb):
    """共享截距合并分组与批次设计矩阵"""
    if Xg is None:
        return Xb
    if Xb is None:
        return Xg
    return np.column_stack([Xg, Xb[:, 1:]])


def perm_within(labels, strata, rng):
    """在 strata 各层内部打乱 labels（保持分层结构不变）"""
    out = np.asarray(labels).copy()
    for s in np.unique(strata):
        ii = np.where(np.asarray(strata) == s)[0]
        if len(ii) > 1:
            out[ii] = rng.permutation(out[ii])
    return out


def partition(Y, group, batch, nperm=999, seed=42):
    """
    方差分解 + 分层置换检验。Y 为 n×p 响应矩阵（如 PCoA 坐标）。
    返回字典：各组 R²、偏效应、P 值、共享与残差。
    """
    Xg, Xb = dummy(group), dummy(batch)
    Xboth = build_both(Xg, Xb)
    r2g, r2b, r2both = r2_of(Y, Xg), r2_of(Y, Xb), r2_of(Y, Xboth)
    pg = r2both - r2b          # 控制批次后的分组偏效应
    pb = r2both - r2g          # 控制分组后的批次偏效应
    shared = r2g + r2b - r2both
    resid = 1 - r2both

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(nperm):                      # 检验偏分组：批次内置换分组
        Xgp = dummy(perm_within(group, batch, rng))
        if r2_of(Y, build_both(Xgp, Xb)) - r2b >= pg - 1e-12:
            ge += 1
    p_group = (1 + ge) / (1 + nperm)
    ge = 0
    for _ in range(nperm):                      # 检验偏批次：分组内置换批次
        Xbp = dummy(perm_within(batch, group, rng))
        if r2_of(Y, build_both(Xg, Xbp)) - r2g >= pb - 1e-12:
            ge += 1
    p_batch = (1 + ge) / (1 + nperm)

    return dict(r2_group=r2g, r2_batch=r2b, r2_both=r2both,
                partial_group=pg, partial_batch=pb,
                p_group=p_group, p_batch=p_batch,
                shared=shared, resid=resid)


if __name__ == "__main__":
    c, m = make_demo_data()
    print("模拟数据：%d 菌种 × %d 样本" % c.shape)
    print(m.head())
    print("分组分布:", m["group"].value_counts().to_dict())
    print("批次分布:", m["batch"].value_counts().to_dict())
