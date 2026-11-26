#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py —— 一键跑完推文示例的三步主线
================================================================================
  ① step1  PCoA 可视化      → 先看现象（按批次抱团、按分组混杂）
  ② step2  PERMANOVA 定量    → 给现象一个 R² 和 P 值 + 阴性对照
  ③ step3  pRDA/PVCA 分解    → 把批次与分组各自解释的变异拆干净

运行： python run_all.py
全部图与表输出到 output/ 目录，可直接取用做推文配图。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("step1_pcoa.py", "① PCoA 可视化"),
    ("step2_permanova.py", "② PERMANOVA 定量"),
    ("step3_prda.py", "③ pRDA/PVCA 方差分解"),
]


def main():
    print("=" * 72)
    print("批次效应推文示例 · 三步主线  PCoA → PERMANOVA → pRDA/PVCA")
    print("=" * 72)
    for script, title in STEPS:
        print("\n>>>", title, f"[{script}]")
        ret = subprocess.run([sys.executable, os.path.join(HERE, script)],
                             cwd=HERE)
        if ret.returncode != 0:
            print(f"[失败] {script} 退出码 {ret.returncode}，停止后续步骤")
            sys.exit(ret.returncode)
    print("\n" + "=" * 72)
    print("全部完成。产物在 output/：")
    outdir = os.path.join(HERE, "output")
    for f in sorted(os.listdir(outdir)):
        print("  -", f)
    print("=" * 72)


if __name__ == "__main__":
    main()
