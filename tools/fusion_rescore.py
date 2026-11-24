# -*- coding: utf-8 -*-
"""从嵌入缓存 npz 重新打分（T1/T3 通用）。

主协议 22 负对（与 DINOv3-B/Gemma2 历史评估一致）+ EXTRA 扩展对。
用法: python tools/fusion_rescore.py <cache.npz> [out.json]
"""
import sys
import json
import numpy as np

# 主评估协议（crops_s8 覆盖的 30 对）
POS = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
NEG = [(14, 1), (3, 1), (50, 11), (7, 1), (10, 1), (126, 18), (57, 11),
       (124, 18), (43, 11), (3, 4), (131, 24), (53, 11), (131, 28),
       (131, 23), (142, 26), (98, 26), (61, 14), (123, 19), (127, 19),
       (146, 21), (132, 21), (131, 21)]
# labels_v2 扩展：(123,18) 有 crop；(111,18) 无 crop
EXTRA = [(123, 18), (111, 18)]


def auc(p, n):
    return float(np.mean([(x > y) + 0.5 * (x == y) for x in p for y in n]))


def main():
    z = np.load(sys.argv[1], allow_pickle=False)
    keys = set(z.files)
    print(f"[rescore] {sys.argv[1]}  ({len(keys)} keys)")

    def cos(a, b):
        ka, kb = f"a{a}", f"b{b}"
        if ka not in keys or kb not in keys:
            return None
        return float(np.dot(z[ka], z[kb]))

    pv = [cos(a, b) for a, b in POS]
    nv = [cos(a, b) for a, b in NEG]
    miss = [p for p, v in zip(POS + NEG, pv + nv) if v is None]
    assert not miss, f"缺轨迹: {miss}"
    print(f"AUC = {auc(pv, nv):.4f}")
    print(f"POS: min={min(pv):.4f} mean={np.mean(pv):.4f} max={max(pv):.4f}")
    print(f"NEG: min={min(nv):.4f} mean={np.mean(nv):.4f} max={max(nv):.4f}")
    print(f"gap = minPOS - maxNEG = {min(pv) - max(nv):+.4f}")
    print()
    for (a, b), v in zip(POS, pv):
        print(f"  POS {a:>3}:{b:<3} {v:.4f}")
    for (a, b), v in zip(NEG, nv):
        print(f"  NEG {a:>3}:{b:<3} {v:.4f}")
    print()
    print("EXTRA（labels_v2 扩展）:")
    ex = {}
    for a, b in EXTRA:
        v = cos(a, b)
        if v is None:
            print(f"  --- {a}:{b}  无 crop")
        else:
            print(f"  EXT {a}:{b}  {v:.4f}")
            ex[f"{a}:{b}"] = v
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump({"pos": pv, "neg": nv, "auc": auc(pv, nv), "extra": ex},
                      f, ensure_ascii=False, indent=1)
        print(f"[rescore] -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
