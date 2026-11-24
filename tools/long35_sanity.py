# -*- coding: utf-8 -*-
"""长窗数据一致性校验：long35 与 v10 的重叠段（前 10 分钟）逐帧对照。

long35 切片与 v10 切片共享相同 -ss 起点 → 帧号 0..~14924 应为同一段视频。
同模型 + 同参数的确定性推理下：frame / count / track_ids 集合 应当一致。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ["316", "318", "320"]


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows[d["frame"]] = d
    return rows


def main():
    ok_all = True
    for cam in CAMS:
        p10 = os.path.join(ROOT, "output_crosscam", f"{cam}_detections.jsonl")
        pL = os.path.join(ROOT, "output_crosscam", "long35", f"{cam}_long35_detections.jsonl")
        if not os.path.exists(pL):
            print(f"[skip] {cam}: long35 数据不存在")
            continue
        r10, rL = load(p10), load(pL)
        common = sorted(set(r10) & set(rL))
        n_diff_c = n_diff_t = n_diff_b = 0
        first_diff = None
        for f in common:
            a, b = r10[f], rL[f]
            if a["count"] != b["count"]:
                n_diff_c += 1
            if set(a.get("track_ids", [])) != set(b.get("track_ids", [])):
                n_diff_t += 1
                if first_diff is None:
                    first_diff = (f, a.get("track_ids"), b.get("track_ids"))
            # boxes 对照（同帧同 tid 的框）
            mab = {t: bx for t, bx in zip(a.get("track_ids", []), a.get("boxes", []))}
            mbb = {t: bx for t, bx in zip(b.get("track_ids", []), b.get("boxes", []))}
            for t in set(mab) & set(mbb):
                if any(abs(u - v) > 0.6 for u, v in zip(mab[t], mbb[t])):
                    n_diff_b += 1
                    break
        print(f"== {cam}: 重叠帧 {len(common)} | count 不同 {n_diff_c} | "
              f"tid 集合不同 {n_diff_t} | 框不同 {n_diff_b}")
        if first_diff:
            print(f"   首个 tid 差异: frame={first_diff[0]}")
            print(f"     v10: {first_diff[1]}")
            print(f"     L35: {first_diff[2]}")
        if n_diff_c or n_diff_t or n_diff_b:
            ok_all = False
    print("\n[verdict]", "一致 ✓" if ok_all else "存在差异（见上）")


if __name__ == "__main__":
    main()
