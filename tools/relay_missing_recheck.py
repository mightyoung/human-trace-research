# -*- coding: utf-8 -*-
"""1181 missing 复核：把审计恢复的 7 条 FP 轨嵌入合并回 v4 集后重算接力缺嵌入。

- 构建 merged 嵌入目录 cradio_scale_v4c（v4 + 318×4 / 305×1 / 316×1 / 323×1 补编码）
- 复现 relay_rescore.do_score 逻辑（同 ent_frag/frag 检查），逐条缺因归类：
    frag_emb_absent → 缺因轨 ∈ {restored_fp(恢复集), reflection_removed(v1剔除), other}
- 自校验：v4 模式应复现官方 n_scored=2563 / n_missing=1181
- 输出：
    analysis/relay/relay_scored_cradio_v4c.json   同格式 v4c 打分（官方口径）
    analysis/relay/missing_recheck_v4c.json       诊断（救援数/剩余归因）

用法: python tools/relay_missing_recheck.py
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from crosscam_relay import OUT_DIR, NET_DIR, load_foot_trajs  # noqa: E402

CAMS = ["316", "305", "318", "320", "323", "343"]
PAT = "{cam}_emb_scale.npz"
SIM_THR = 0.943

V4 = os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale_v4")
V4C = os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale_v4c")
EXTRA = os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale_v4_extra")
EXTRAS = {"318": "318_extra4_emb.npz", "305": "305_extra1_emb.npz",
          "316": "316_extra1_emb.npz", "323": "323_extra1_emb.npz"}
RESTORED = {"316": {32}, "323": {193}, "318": {53, 82, 98, 135}, "305": {12},
            "320": set(), "343": set()}
TC = os.path.join(ROOT, "output_crosscam", "analysis", "track_cleaning",
                  "track_cleaning.json")


def key_to_tid(k):
    num = ""
    for ch in reversed(k):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def build_merged():
    os.makedirs(V4C, exist_ok=True)
    report = {}
    for cam in CAMS:
        src = os.path.join(V4, PAT.format(cam=cam))
        dst = os.path.join(V4C, PAT.format(cam=cam))
        z = np.load(src)
        data = {k: z[k] for k in z.files}
        n_before = len(data)
        ep = os.path.join(EXTRA, EXTRAS[cam]) if cam in EXTRAS else None
        if ep and os.path.exists(ep):
            ze = np.load(ep)
            for k in ze.files:
                data[k] = ze[k]
        np.savez(dst, **data)
        report[cam] = {"n_v4": n_before, "n_v4c": len(data), "added": len(data) - n_before}
    print("[merge] 合并嵌入：", {c: r["added"] for c, r in report.items()})
    return report


def run_score(emb_dir, restrict_restored_ok=True):
    """复现 do_score，返回 (scored, missing_details)"""
    spans, ent_frag = {}, {}
    for cam in CAMS:
        foot = load_foot_trajs(cam)
        spans[cam] = {tid: (float(fr[0]), float(fr[-1]))
                      for tid, (fr, u, v, h) in foot.items()}
    for cam in CAMS:
        with open(os.path.join(NET_DIR, f"net_tracks_{cam}.json"), encoding="utf-8") as f:
            d = json.load(f)
        for e in d["entities"]:
            tids = [t for t in e["tracks"] if t in spans[cam]]
            if not tids:
                continue
            tids.sort(key=lambda t: spans[cam][t][0])
            ent_frag[f"{cam}:{e['eid']}"] = (tids[0], tids[-1])

    embs = {}
    for cam in CAMS:
        p = os.path.join(emb_dir, PAT.format(cam=cam))
        d = np.load(p)
        embs[cam] = {key_to_tid(k): d[k] for k in d.files}

    tc = json.load(open(TC, encoding="utf-8"))
    removed = {c: {int(t) for t in tc["summary"][c]["flags"].keys()} for c in CAMS}

    with open(os.path.join(OUT_DIR, "relay_candidates.json"), encoding="utf-8") as f:
        cand = json.load(f)["candidates"]

    scored, missing = [], []
    for c in cand:
        ca, _ = c["a"].split(":")
        cb, _ = c["b"].split(":")
        if c["a"] not in ent_frag or c["b"] not in ent_frag:
            missing.append({"a": c["a"], "b": c["b"], "reason": "entity_no_frag"})
            continue
        fa = ent_frag[c["a"]][1]
        fb = ent_frag[c["b"]][0]
        va, vb = embs[ca].get(fa), embs[cb].get(fb)
        if va is None or vb is None:
            m = {"a": c["a"], "b": c["b"], "reason": "frag_emb_absent"}
            for cam, t, side in ((ca, fa, "a"), (cb, fb, "b")):
                if embs[cam].get(t) is None:
                    if t in RESTORED[cam]:
                        cat = "restored_fp"
                    elif t in removed[cam]:
                        cat = "reflection_removed"
                    else:
                        cat = "other"
                    m[f"missing_{side}"] = {"cam": cam, "tid": t, "cat": cat}
            missing.append(m)
            continue
        sim = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        rec = dict(c)
        rec["sim"] = round(sim, 4)
        rec["a_frag_last"] = fa
        rec["b_frag_first"] = fb
        scored.append(rec)
    scored.sort(key=lambda r: -r["sim"])
    return scored, missing


def main():
    merge_report = build_merged()

    print("\n[score] v4 (原版) ...")
    s4, m4 = run_score(V4)
    print(f"  n_scored={len(s4)} n_missing={len(m4)}")
    cats4 = {}
    for m in m4:
        for side in ("missing_a", "missing_b"):
            if side in m:
                cats4[m[side]["cat"]] = cats4.get(m[side]["cat"], 0) + 1
    print("  v4 缺因归类:", cats4)

    print("\n[score] v4c (合并恢复轨) ...")
    s4c, m4c = run_score(V4C)
    print(f"  n_scored={len(s4c)} n_missing={len(m4c)}")
    cats4c = {}
    for m in m4c:
        for side in ("missing_a", "missing_b"):
            if side in m:
                cats4c[m[side]["cat"]] = cats4c.get(m[side]["cat"], 0) + 1
    print("  v4c 缺因归类:", cats4c)

    # 救援明细
    key4 = {(r["a"], r["b"]) for r in s4}
    key4c = {(r["a"], r["b"]) for r in s4c}
    rescued = sorted(key4c - key4)
    print(f"\n[rescue] 由恢复轨救回的候选: {len(rescued)}")
    resc_rows = []
    for a, b in rescued:
        r = next(x for x in s4c if x["a"] == a and x["b"] == b)
        resc_rows.append({"a": a, "b": b, "sim": r["sim"],
                          "a_frag_last": r["a_frag_last"], "b_frag_first": r["b_frag_first"]})
    for r in resc_rows[:12]:
        print("  ", r)

    out = {
        "v4_ref": {"n_scored": len(s4), "n_missing": len(m4), "cats": cats4,
                   "official_match": len(s4) == 2563 and len(m4) == 1181},
        "v4c": {"n_scored": len(s4c), "n_missing": len(m4c), "cats": cats4c},
        "rescued_pairs": resc_rows,
        "merge_report": merge_report,
        "note": "v4c = v4 嵌入集 + 审计恢复的 7 条 FP 轨（318×4/305×1/316×1/323×1）",
    }
    p1 = os.path.join(OUT_DIR, "relay_scored_cradio_v4c.json")
    json.dump({"sim_thr": SIM_THR, "n_scored": len(s4c), "n_missing": len(m4c),
               "candidates": s4c}, open(p1, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    p2 = os.path.join(OUT_DIR, "missing_recheck_v4c.json")
    json.dump(out, open(p2, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {p1}\n[out] {p2}")


if __name__ == "__main__":
    main()
