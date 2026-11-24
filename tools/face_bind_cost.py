# -*- coding: utf-8 -*-
"""人脸实时绑定 → 选择器池：真实代价量测。

要回答的问题
------------
把「高置信度人脸 → 身份簇」的绑定直接写进选择器池，选择器会变好还是变坏？
原来那条实验（8 段 79.3%）里人脸身份是用 GT 代替的，等于假设识别全对。
这里换成**真实人脸识别结果**，把误绑定的破坏也算进去：
    S0       仅历史库（不用池）
    S1_real  库 + 池，池由**真实人脸识别**写入（含错）
    S1_gt    库 + 池，池由**真值**写入（等于「识别全对」的上限）
    S2_real  仅真实池

口径
----
- 因果：绑定时刻 t_face = t0 + (frame - f_rep) / 25；只有 t_face < 查询 t0 的绑定能进池。
- 留一：查询轨自身一定从库与池里剔除。
- 分层：库与池都按机位（与生产一致）。
- 查询集：默认只用「所属机位×实体在库里有其它模板」的轨（否则留一后无候选）。
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp" / "backend"))

import selector as S  # noqa: E402

FPS = 25.0


def load_bindings(path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for s in d["suggestions"]:
        out.append({
            "key": s["key"], "cam": s["cam"], "tid": int(s["tid"]),
            "entity": s["target_entity"],      # 人脸认定的身份簇
            "ident": s["ident"],               # 人脸认定的 GIL 身份（仅展示）
            "score": float(s["score"]), "margin": float(s.get("margin") or 0.0),
            "frame": s.get("frame"), "truth": s.get("truth"),
            "track_entity": s.get("target_from"),
            "hit": bool(s.get("hit")),
        })
    return out


def bind_time(sel, b: dict) -> float:
    """绑定的因果时刻：人脸被观测到的那一帧对应的观测时间轴。"""
    i = sel.idx_of(b["cam"], b["tid"])
    if i is None:
        return None
    t = sel.meta[i]
    t0, t1 = float(t.get("t0") or 0.0), float(t.get("t1") or 0.0)
    f, fr = b.get("frame"), t.get("f_rep")
    if f is None or fr is None:
        return t1
    v = t0 + (float(f) - float(fr)) / FPS
    return min(max(v, t0), t1)


def build_pool(sel, binds: list[dict], tau: float, margin: float,
               use_truth: bool) -> tuple[dict, dict]:
    """按闸门生成池；返回 (pool, stats)。

    pool: {(scope_key, entity): [ {i, cam, tid, t_face, ...} ]}，与 selector 内部结构一致。

    ★ 槽位口径（两种必须分开，混用会得出相反结论）：
      use_truth=False —— 人脸说这条轨属于 target_entity，就写进 target_entity。
                         写错 = 污染别人的簇，这是**真实部署**的后果。
      use_truth=True  —— 假设人脸识别全对，这条轨必然属于它自己所在的簇，写进自身簇。
                         这是**上限口径**（等于 8 段实验里用 GT 代替人脸身份）。
    ★ 正确性用 `hit`（ident == truth），**不能**用 target_entity == truth —— 两者命名空间不同。
    """
    pool: dict = {}
    st = collections.Counter()
    for b in binds:
        i = sel.idx_of(b["cam"], b["tid"])
        if i is None:
            st["skip_no_track"] += 1
            continue
        if b["score"] < tau or b["margin"] < margin:
            st["skip_gate"] += 1
            continue
        ent = sel.ent_of[i] if use_truth else b["entity"]
        if not ent:
            st["skip_no_ent"] += 1
            continue
        st["ok" if b["hit"] else "wrong"] += 1
        if b["hit"] and not use_truth and ent != sel.ent_of[i]:
            st["hit_but_other_cluster"] += 1
        tf = bind_time(sel, b)
        slot = (b["cam"], ent)          # scope='cam' 与生产一致
        pool.setdefault(slot, []).append({
            "i": i, "cam": b["cam"], "tid": b["tid"], "src": "face_auto",
            "q": round(b["score"], 4), "at": tf, "note": b["ident"],
        })
    return pool, dict(st)


def pool_at(pool: dict, cam: str, t_now: float, exclude: set) -> dict:
    """把池裁剪成「此刻已知」的版本（因果）。"""
    out = {}
    for (scope, ent), items in pool.items():
        if scope != cam:                 # scope='cam'
            continue
        live = [it for it in items if it["i"] not in exclude
                and it["at"] is not None and it["at"] < t_now]
        if live:
            out[(scope, ent)] = live[-8:]  # FIFO 容量 8
    return out


def run(sel, pool: dict, queries: list[int], tau: float, scope: str = "cam",
        lib_allow: dict | None = None) -> dict:
    n = 0
    hit = collections.Counter()
    cov = collections.Counter()
    by_cam = {}
    for qi in queries:
        t = sel.meta[qi]
        cam = t["cam"]
        ent = sel.ent_of[qi]
        t0 = float(t.get("t0") or 0.0)
        sel.pool = pool_at(pool, cam, t0, {qi})
        try:
            r = sel.query(cam, int(t["tid"]), scope=scope, tau=tau,
                          lib_allow=(lib_allow or {}).get(cam))
        except Exception:
            continue
        if not r["candidates"]:
            continue
        n += 1
        st = by_cam.setdefault(cam, {"n": 0, "lib": 0, "both": 0})
        st["n"] += 1
        for k, tk in (("lib", "top_lib"), ("both", "top_both")):
            tp = r[tk]
            if tp and tp["accepted"]:
                cov[k] += 1
                if tp["entity"] == ent:
                    hit[k] += 1
                    st[k] += 1
    sel.pool = {}
    return {"n": n, "cov_lib": cov["lib"], "cov_both": cov["both"],
            "hit_lib": hit["lib"], "hit_both": hit["both"], "by_cam": by_cam}


def pack(r: dict, key: str) -> str:
    c = r["cov_lib"] if key == "lib" else r["cov_both"]
    h = r["hit_lib"] if key == "lib" else r["hit_both"]
    cp = 100.0 * c / r["n"] if r["n"] else 0.0
    hp = 100.0 * h / c if c else float("nan")
    return f"{h}/{c} ({hp:.1f}%) 覆盖 {cp:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.80, help="选择器判定阈值")
    ap.add_argument("--gate", type=float, default=0.50, help="人脸绑定分数闸门")
    ap.add_argument("--sweep", default="", help="逗号分隔的闸门扫描，如 0.35,0.45,0.55,0.65")
    ap.add_argument("--margin", type=float, default=0.05, help="人脸绑定分差闸门")
    ap.add_argument("--split", default="half", choices=["loo", "half", "all"])
    args = ap.parse_args()

    sel = S.init()
    if not sel.ready:
        print("选择器未就绪：", sel.err)
        return 2
    binds = load_bindings(S.REID / "face_suggest.json")
    print(f"绑定证据 {len(binds)} 条")

    # 查询集：与 selector.evaluate 同口径
    if args.split == "loo":
        queries = [i for i in range(len(sel.meta))
                   if len(sel.groups.get((sel.meta[i]["cam"], sel.ent_of[i]), [])) >= 2
                   and float(sel.meta[i].get("h_med") or 0) > 0]
    elif args.split == "half":
        queries = []
        for cam in sorted({t["cam"] for t in sel.meta}):
            rows = [i for i in range(len(sel.meta))
                    if sel.meta[i]["cam"] == cam and float(sel.meta[i].get("h_med") or 0) > 0]
            if len(rows) < 4:
                continue
            mid = sorted(float(sel.meta[i].get("t0") or 0) for i in rows)[len(rows) // 2]
            seen = {(cam, sel.ent_of[i]) for i in rows
                    if float(sel.meta[i].get("t0") or 0) < mid}
            queries += [i for i in rows
                        if float(sel.meta[i].get("t0") or 0) >= mid
                        and (cam, sel.ent_of[i]) in seen]
    else:
        queries = [i for i in range(len(sel.meta)) if float(sel.meta[i].get("h_med") or 0) > 0]
    print(f"split={args.split} → 查询 {len(queries)} 条")

    # 库侧约束：half 只允许前半段轨作模板
    lib_allow = None
    if args.split == "half":
        lib_allow = {}
        for cam in sorted({t["cam"] for t in sel.meta}):
            rows = [i for i in range(len(sel.meta))
                    if sel.meta[i]["cam"] == cam and float(sel.meta[i].get("h_med") or 0) > 0]
            if len(rows) < 4:
                continue
            mid = sorted(float(sel.meta[i].get("t0") or 0) for i in rows)[len(rows) // 2]
            lib_allow[cam] = {i for i in rows if float(sel.meta[i].get("t0") or 0) < mid}
        sel._lib_allow_patch = lib_allow

    variants = {}
    gates = ([args.gate] if not args.sweep
             else [float(x) for x in args.sweep.split(",") if x.strip()])
    pad = {}
    for tau_g in gates:
        variants[f"real@{tau_g:.2f}"] = build_pool(sel, binds, tau_g, args.margin, False)
        pad[f"real@{tau_g:.2f}"] = tau_g
    variants["S1_gt"] = build_pool(sel, binds, gates[0], args.margin, True)
    variants["S0"] = ({}, {})

    print(f"\n{'闸门':>6} {'池条目':>5} {'对':>3} {'错':>3} {'精度':>7}  "
          f"{'库+池 判对/覆盖':>18} {'覆盖率':>7}")
    rows = []
    for name in ["S0"] + [f"real@{g:.2f}" for g in gates] + ["S1_gt"]:
        pool, st = variants[name]
        r = run(sel, pool, queries, args.tau, lib_allow=lib_allow)
        npass = sum(len(v) for v in pool.values())
        ok, bad = st.get("ok", 0), st.get("wrong", 0)
        pr = 100.0 * ok / max(1, ok + bad)
        covp = 100.0 * r["cov_both"] / max(1, r["n"])
        hp = 100.0 * r["hit_both"] / max(1, r["cov_both"])
        rows.append((name, npass, ok, bad, pr, r, covp, hp))
        label = "—" if name in ("S0",) else ("GT" if name == "S1_gt" else name.split("@")[1])
        print(f"{label:>6} {npass:5} {ok:3} {bad:3} {pr:6.1f}%  "
              f"{r['hit_both']:>3}/{r['cov_both']:<3} ({hp:5.1f}%)  {covp:6.1f}%"
              + (f"   {st}" if st else ""))

    print(f"\n口径：split={args.split}，选择器 τ={args.tau}，人脸闸门 margin≥{args.margin}，"
          f"库/池按机位分层，查询留一，池严格因果（t_face < 查询 t0）。")
    print("★ 判对/覆盖 的分母是「库+池都给出候选且过 τ 的查询」，与 S0 同分母，可直接比。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
