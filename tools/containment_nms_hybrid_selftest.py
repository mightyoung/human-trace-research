#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""containment_nms 的 hybrid 扩展自测（C 方案接入生产的验收断言）。

跑法: python tools/containment_nms_hybrid_selftest.py    退出码 0 = 全过
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from containment_nms import containment_nms  # noqa: E402

FAIL = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + extra) if extra and not cond else ''}")
    if not cond:
        FAIL.append(name)


def main() -> int:
    print("[1] 向后兼容：不传新参数 = 原 keep=larger 行为")
    # 大框 0..400 高 400，内置小框 0..300 高 300（包含度 1.0）
    boxes = np.array([[0, 0, 100, 400], [10, 10, 90, 310]], dtype=float)
    scores = np.array([0.30, 0.90])
    k = containment_nms(boxes, scores, 0.85, "larger")
    check("larger 保留大框", list(k) == [0], f"got {list(k)}")
    k2 = containment_nms(boxes, scores, 0.85, "larger", hy_score_k=0.0)
    check("hy_score_k=0 与不传一致", list(k) == list(k2))

    print("[2] hybrid：小框分数够硬 → 小框顶上（305 f4435 实况）")
    # 复刻 305 f4435：#7 h862 sc0.330 吞 #2 h696 sc0.414
    boxes = np.array([[1095, 528, 1477, 1391], [1101, 535, 1478, 1231]], dtype=float)
    scores = np.array([0.330, 0.414])
    check("hybrid k=0.8 选中小框", list(containment_nms(boxes, scores, 0.85, "larger",
                                                     hy_score_k=0.8, hy_min_h=200)) == [1])
    check("hybrid k=0.8 但 min_h=800 不介入",
          list(containment_nms(boxes, scores, 0.85, "larger",
                               hy_score_k=0.8, hy_min_h=800)) == [0])

    print("[3] hybrid：小框分数不够硬 → 仍保留大框")
    scores2 = np.array([0.60, 0.30])   # 小框分低
    check("分数不足时不顶掉",
          list(containment_nms(boxes, scores2, 0.85, "larger",
                               hy_score_k=0.8, hy_min_h=0)) == [0])

    print("[4] min_h 只作用于「小框高度」")
    # 远景小框：高 60 / 40，分数 0.4 vs 0.5
    far = np.array([[0, 0, 40, 60], [2, 5, 38, 45]], dtype=float)
    sc = np.array([0.40, 0.50])
    check("远景(min_h=200) 不介入 → 保留大框",
          list(containment_nms(far, sc, 0.85, "larger", hy_score_k=0.8, hy_min_h=200)) == [0])
    check("远景(min_h=0) 介入 → 保留小框",
          list(containment_nms(far, sc, 0.85, "larger", hy_score_k=0.8, hy_min_h=0)) == [1])

    print("[5] 边界：空 / 单框 / 无包含关系")
    check("空输入", len(containment_nms(np.zeros((0, 4)), np.zeros(0))) == 0)
    check("单框", list(containment_nms(np.array([[0, 0, 10, 20]]), np.array([0.5]))) == [0])
    sep = np.array([[0, 0, 50, 100], [200, 0, 250, 100]], dtype=float)
    check("两个分开的人都不删",
          len(containment_nms(sep, np.array([0.5, 0.6]), 0.85, "larger",
                              hy_score_k=0.8, hy_min_h=200)) == 2)

    print("[6] 输出必须按面积降序（否则会改变 ByteTrack 输入顺序）")
    b = np.array([[0, 0, 60, 300], [0, 0, 50, 200], [0, 0, 40, 100]], dtype=float)  # 三层嵌套
    s = np.array([0.20, 0.50, 0.80])
    kk = containment_nms(b, s, 0.85, "larger", hy_score_k=0.8, hy_min_h=50)
    areas = [(b[i][2] - b[i][0]) * (b[i][3] - b[i][1]) for i in kk]
    check("面积降序", areas == sorted(areas, reverse=True), f"got {kk} areas={areas}")
    check("三层嵌套只留 1 个", len(kk) == 1, f"got {kk}")

    print("[7] keep=higher 不受 hybrid 影响（显式指定时才接管）")
    check("higher 仍按分数",
          list(containment_nms(boxes, np.array([0.30, 0.90]), 0.85, "higher")) == [1])

    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} 项 -> {FAIL}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
