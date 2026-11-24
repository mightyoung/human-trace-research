# -*- coding: utf-8 -*-
"""v2 迁移等价性对拍：旧档（_legacy_flat）vs 新库（CURRENT 代际）**逐位比对**。

对每个 store 做两层验证：
  1. 数据层：行集一致、每键元数据全字段一致、嵌入矩阵**逐位相等**（f16→f32 上转）；
  2. 检索层：抽样查询 × 8 种检索配置（默认/排除自身/排除机位/机位白名单/时窗/
     两级 nprobe/单级/topk25），新库 search 与"批次 A 数学的独立复刻（作用于旧档数据）"
     的 top-k 键序与分数**精确相等**。

用法：
  python tools/store_v2_equivalence.py                    # 全部 5 库
  python tools/store_v2_equivalence.py --stores store_cradio_v4 --queries 60
输出：
  output_crosscam/analysis/global_reid/store_v2_equivalence_report.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from global_reid_store import GlobalReIDStore                      # noqa: E402
from store_admin import inspect as admin_inspect, read_legacy_flat  # noqa: E402

BASE = os.path.join(ROOT, "output_crosscam", "analysis", "global_reid")
STORES = ["store_cradio", "store_cradio_v3", "store_cradio_v4",
          "store_d3b", "store_long35"]
OUT = os.path.join(BASE, "store_v2_equivalence_report.json")
_META_FIELDS = ("cam", "date", "tid", "t_first", "t_last", "h_med", "u", "v", "extra")


def _ref_search(En, Mn, keys, meta, q, *, topk=10, cams=None, exclude_cam=None,
                t_range=None, exclude_key=None, stage="auto",
                mrl_dim=None, nprobe=None):
    """批次 A 检索数学的独立复刻（作用于旧档数据；En/Mn 已归一化）。"""
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    q = q / (np.linalg.norm(q) + 1e-9)
    if topk == 0 or En.shape[0] == 0:
        return []
    allowed = []
    for i, k in enumerate(keys):
        m = meta[k]
        if cams is not None and m["cam"] not in cams:
            continue
        if exclude_cam is not None and m["cam"] == exclude_cam:
            continue
        if exclude_key is not None and k == exclude_key:
            continue
        if t_range is not None:
            t = m["t_first"] if m["t_first"] is not None else -1e9
            if not (t_range[0] <= t <= t_range[1]):
                continue
        allowed.append(i)
    if not allowed:
        return []
    ai = np.asarray(allowed)
    two = mrl_dim is not None and stage in ("two", "auto")
    if two:
        npb = int(nprobe or max(topk * 8, 64))
        npb = min(npb, len(ai))
        qm = q[:mrl_dim]
        qm = qm / (np.linalg.norm(qm) + 1e-9)
        sm = Mn[ai] @ qm
        if npb < len(ai):
            cand = ai[np.argpartition(-sm, npb - 1)[:npb]]
        else:
            cand = ai
        sf = En[cand] @ q
        oi = np.argsort(-sf)
        order = cand[oi]
        scores = sf[oi]
    else:
        s = En[ai] @ q
        oi = np.argsort(-s)
        order = ai[oi]
        scores = s[oi]
    out = []
    for i, sc in zip(order, scores):
        out.append({"key": keys[i], "score": float(sc)})
        if len(out) >= topk:
            break
    return out


def _configs(qk, cam, cams_two):
    return [
        ("default", dict(topk=10)),
        ("exclude_self", dict(topk=10, exclude_key=qk)),
        ("exclude_cam", dict(topk=10, exclude_cam=cam)),
        ("cams_two", dict(topk=10, cams=cams_two)),
        ("t_range", dict(topk=10, t_range=(0, 120))),
        ("two_np64", dict(topk=10, stage="two", nprobe=64)),
        ("single", dict(topk=10, stage="single")),
        ("topk25", dict(topk=25)),
    ]


def check_store(name, n_queries=40):
    root = os.path.join(BASE, name)
    legacy_dir = os.path.join(root, "_legacy_flat")
    res = {"store": name, "root": root}
    if not os.path.isdir(legacy_dir):
        res.update({"status": "SKIP", "reason": "无 _legacy_flat（未迁移或已清理）"})
        return res
    t0 = time.time()
    leg = read_legacy_flat(legacy_dir)
    st = GlobalReIDStore.open(root)
    n = st.stats()["n"]

    # ---- 1) 数据层 ----
    idx = {r["key"]: i for i, r in enumerate(leg["rows"])}
    order = sorted(idx)
    res["n_legacy"] = leg["version"]["n"]
    res["n_v2"] = n
    res["fields_ok"] = bool(n == leg["version"]["n"] == len(order))
    E_ref = np.stack([leg["E"][idx[k]] for k in order]).astype(np.float32)
    emb_ok = all(np.array_equal(st.rows[k], E_ref[i]) for i, k in enumerate(order))
    meta_ref = {r["key"]: r for r in leg["rows"]}
    meta_bad = []
    for k in order:
        for f in _META_FIELDS:
            a = st.meta[k].get(f)
            b = (meta_ref[k].get(f) if f != "extra" else meta_ref[k].get("extra_parsed"))
            if a != b:
                meta_bad.append((k, f, a, b))
                break
    by_cam_leg = {}
    for r in leg["rows"]:
        by_cam_leg[r["cam"]] = by_cam_leg.get(r["cam"], 0) + 1
    res["emb_bitwise_ok"] = bool(emb_ok)
    res["meta_ok"] = not meta_bad
    res["by_cam_ok"] = bool(by_cam_leg == st.stats()["by_cam"])
    res["data_all_ok"] = bool(res["fields_ok"] and emb_ok and not meta_bad
                              and res["by_cam_ok"])

    # ---- 2) 检索层 ----
    En = E_ref.copy()
    En /= (np.linalg.norm(En, axis=1, keepdims=True) + 1e-9)
    mrl_dim = st.mrl_dim
    Mn = None
    if mrl_dim is not None:
        Mn = E_ref[:, :mrl_dim].copy()
        Mn /= (np.linalg.norm(Mn, axis=1, keepdims=True) + 1e-9)
    meta_st = {k: st.meta[k] for k in order}
    step = max(1, n // max(1, n_queries))
    queries = order[::step][:n_queries]
    all_cams = sorted({st.meta[k]["cam"] for k in order})
    n_cmp = 0
    mismatches = []
    max_abs_diff = 0.0
    for qk in queries:
        cam = st.meta[qk]["cam"]
        others = [c for c in all_cams if c != cam]
        cams_two = others[:2] if others else [cam]
        q = st.rows[qk]
        for cname, cfg in _configs(qk, cam, cams_two):
            r_new = st.search(q, **cfg)
            r_ref = _ref_search(En, Mn, order, meta_ref, q, mrl_dim=mrl_dim, **cfg)
            n_cmp += 1
            got = [(h["key"], h["score"]) for h in r_new]
            want = [(h["key"], h["score"]) for h in r_ref]
            if got != want:
                for (gk, gs), (wk, ws) in zip(got, want):
                    if gk == wk:
                        max_abs_diff = max(max_abs_diff, abs(gs - ws))
                mismatches.append({"query": qk, "config": cname,
                                   "got": got[:5], "want": want[:5]})
    res["search_cmps"] = n_cmp
    res["search_exact_ok"] = not mismatches
    res["search_mismatches"] = mismatches[:5]
    res["max_abs_score_diff"] = max_abs_diff

    # ---- 3) admin 冒烟 ----
    info = admin_inspect(root, list_generations=True)
    res["admin_ok"] = bool(info["n"] == n and len(info["generations"]) >= 1)
    res["generation"] = info["pointer"]["generation"]
    res["manifest_sha256"] = info["pointer"]["manifest_sha256"]
    res["provenance_kind"] = (info.get("provenance") or {}).get("kind")
    res["elapsed_s"] = round(time.time() - t0, 2)
    ok = (res["data_all_ok"] and res["search_exact_ok"] and res["admin_ok"])
    res["status"] = "PASS" if ok else "FAIL"
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stores", default=",".join(STORES))
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--report-out", default=OUT)
    args = ap.parse_args()
    names = [s.strip() for s in args.stores.split(",") if s.strip()]
    results = [check_store(name, args.queries) for name in names]
    report = {"time_unix": time.time(),
              "tool": "tools/store_v2_equivalence.py",
              "report": results}
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, allow_nan=False)
    n_pass = n_fail = n_skip = 0
    for r in results:
        if r["status"] == "PASS":
            n_pass += 1
            print(f"[PASS] {r['store']}: n={r['n_v2']} 数据逐位={r['emb_bitwise_ok']} "
                  f"检索对拍={r['search_cmps']} 组精确 | gen={r['generation'][:10]}… "
                  f"({r['elapsed_s']}s)")
        elif r["status"] == "SKIP":
            n_skip += 1
            print(f"[SKIP] {r['store']}: {r['reason']}")
        else:
            n_fail += 1
            print(f"[FAIL] {r['store']}: data_ok={r.get('data_all_ok')} "
                  f"search_ok={r.get('search_exact_ok')} admin_ok={r.get('admin_ok')} "
                  f"mismatch示例={r.get('search_mismatches', [])[:1]}")
    print(f"\n== 等价性对拍: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ==\n"
          f"报告 → {args.report_out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
