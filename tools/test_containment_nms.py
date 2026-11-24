#!/usr/bin/env python3
"""Containment-NMS 的离线单元测试（不依赖 GPU / 模型）。

运行:
    python tools/test_containment_nms.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from containment_nms import containment_nms, dedup_detections  # noqa: E402


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    return cond


def main():
    ok = True

    # 1. 典型嵌套：小框完全在大框内，且小框分数更高 -> 应保留大框
    b = np.array([[100, 100, 200, 300], [110, 110, 190, 200]], float)
    s = np.array([0.33, 0.39])
    k = containment_nms(b, s, 0.85, "larger")
    ok &= check("嵌套对(小框分更高) -> 保留大框", k.tolist() == [0])

    # 2. 并排两人，互不包含 -> 都保留
    b = np.array([[100, 100, 150, 300], [200, 100, 250, 300]], float)
    k = containment_nms(b, np.array([0.5, 0.5]), 0.85, "larger")
    ok &= check("并排两人 -> 都保留", len(k) == 2)

    # 3. 半重叠两人（IoU 高但互不包含）-> 都保留
    b = np.array([[100, 100, 200, 300], [150, 100, 250, 300]], float)
    k = containment_nms(b, np.array([0.5, 0.5]), 0.85, "larger")
    ok &= check("半重叠两人 -> 都保留", len(k) == 2)

    # 4. 空输入
    k = containment_nms(np.zeros((0, 4)), np.zeros(0))
    ok &= check("空输入 -> 空结果", len(k) == 0)

    # 5. 单框
    k = containment_nms(np.array([[0, 0, 10, 10]]), np.array([0.5]))
    ok &= check("单框 -> 保留自身", k.tolist() == [0])

    # 6. 三层嵌套 -> 只剩最大框
    b = np.array([[100, 100, 300, 400], [110, 110, 250, 300], [120, 120, 200, 220]], float)
    k = containment_nms(b, np.array([0.3, 0.4, 0.5]), 0.85, "larger")
    ok &= check("三层嵌套 -> 只剩最大框", k.tolist() == [0])

    # 7. 阈值边界：小框 100% 在内
    b = np.array([[100, 100, 200, 200], [100, 100, 150, 200]], float)
    s = np.array([0.3, 0.5])
    ok &= check("thr=0.99 -> 仍去重", len(containment_nms(b, s, 0.99, "larger")) == 1)
    ok &= check("thr=1.01 -> 不去重", len(containment_nms(b, s, 1.01, "larger")) == 2)
    ok &= check("thr=0 -> 最激进(仍去重)", len(containment_nms(b, s, 0.0, "larger")) == 1)

    # 8. keep='higher' 时保留高分（小框）
    b = np.array([[100, 100, 200, 300], [110, 110, 190, 200]], float)
    s = np.array([0.33, 0.39])
    k = containment_nms(b, s, 0.85, "higher")
    ok &= check("keep=higher -> 保留小框(高分)", k.tolist() == [1])

    # 9. 结果与输入顺序无关（稳定性）
    b = np.array([[100, 100, 300, 400], [110, 110, 250, 300], [120, 120, 200, 220]], float)
    s = np.array([0.3, 0.4, 0.5])
    k1 = containment_nms(b, s, 0.85, "larger")
    perm = [2, 0, 1]
    k2 = containment_nms(b[perm], s[perm], 0.85, "larger")
    # 两个结果映射回原始下标后应一致
    ok &= check("输入顺序无关", sorted(k1.tolist()) == sorted([perm[i] for i in k2]))

    # 10. dedup_detections 返回值形状一致
    b = np.array([[100, 100, 200, 300], [110, 110, 190, 200]], float)
    s = np.array([0.33, 0.39])
    bb, ss, idx = dedup_detections(b, s, 0.85, "larger")
    ok &= check("dedup 返回形状一致", bb.shape[0] == ss.shape[0] == len(idx) == 1)

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
