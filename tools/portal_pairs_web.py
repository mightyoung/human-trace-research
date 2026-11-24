#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门户交接补口批：316/305/323 源跨机位对 → 待裁决载荷（s6b_pending.json 第 4 组）。

目的（docs/eval4-next-portal-report.md §3 验收缺口）：
  "判读集补齐 316/305/323 出口的交接对（当前 55 判几乎全为 343 侧）"——
  本批给用户判读多门机位源的交接对，直接服务"预测先验/relay 排序"的复验。

选择分四组（全局去重，优先级 G1>G2>G3>G4）：
  G1 先验改变排序：rank_old ≥ 5 且 rank_new ≤ 3（先验顶起——判读最直接验证先验方向）
  G2 竞争对：每 a 端 rank_new 第 1/2 名（二选一——测"正确接续 vs 误接"）
  G3 高 sim 被压制：sim ≥ 0.945 且 rank_new ≥ 5（先验压低——判"异人"验证压制正确）
  G4 relay 备书对：active/confirmed 边本体（多门源、未判——判"同人"直接背书 relay）

约束：未判读（labels.json v∈0/1/2）、a/b 端轨迹 n≥12（可渲染）、源∈{305,316,323}。
渲染：a 端取轨迹尾部 [-3,-10]、b 端取头部 [2,9]；外扩 45%、高 200px、q82；
      文件名 PRT_{cam}-{eid}_{a|b}{k}.jpg → webapp/frontend/public/s6b_thumbs/
输出：
  webapp/frontend/public/s6b_pending.json（合并保留旧 3 组 + 新组 portal_v2；幂等）
  output_crosscam/analysis/truth/v2portal/selection_v1.json（选择记录/口径）
用法：
  DET_VIEW=clean2 python tools/portal_pairs_web.py [--dry-run]
"""
import argparse
import collections
import glob
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

os.environ.setdefault("DET_VIEW", "clean2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from crosscam_relay import load_entities  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
PRIOR = os.path.join(ANA, "relay_v2", "scored_v2_prior.json")
LABELS_RM = os.path.join(ROOT, "output_crosscam", "tmp_remote", "labels_remote.json")
GIL = os.path.join(ANA, "gil", "store_v1", "gil.sqlite")
PUB = os.path.join(ROOT, "webapp", "frontend", "public")
S6B = os.path.join(PUB, "s6b_pending.json")
TH = os.path.join(PUB, "s6b_thumbs")
SYNCED = os.path.join(ROOT, "media-resource", "multi-cemra", "synced-10min")
OUTDIR = os.path.join(ANA, "truth", "v2portal")
CAMS = ["305", "316", "318", "320", "323", "343"]
NATIVE = {"316": (2560, 1440), "305": (2560, 1440), "318": (1920, 1080),
          "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}
MULTI = {"305", "316", "323"}
FPS = 25.0
PAD = 0.45
H_OUT = 200
CAP = {"G1": 12, "G2": 10, "G3": 10, "G4": 8}   # 目标 ~40 对


def parse_pid(pid):
    left, right = pid.split("__")
    la, lb = left.split("_"), right.split("_")
    return f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只选择不渲染/不写")
    args = ap.parse_args()

    recs = json.load(open(PRIOR, encoding="utf-8"))["candidates"]
    lb = json.load(open(LABELS_RM, encoding="utf-8")).get("pair", {})
    judged = {k for k, v in lb.items() if isinstance(v, dict) and v.get("v") in (0, 1, 2)}

    ents = {cam: load_entities(cam) for cam in CAMS}

    def ok_ent(key):
        cam, eid = key.split(":", 1)
        e = ents.get(cam, {}).get(eid)
        return e is not None and len(e[1]) >= 12

    # ---------------- 候选池 ----------------
    pool = []
    for r in recs:
        ca, cb = r["a"].split(":")[0], r["b"].split(":")[0]
        if ca not in MULTI or not r.get("portal_applied"):
            continue
        pid = f"PRT_{r['a'].replace(':', '_E_')}__{r['b'].replace(':', '_E_')}"
        if pid in judged:
            continue
        if not ok_ent(r["a"]) or not ok_ent(r["b"]):
            continue
        pool.append(dict(pid=pid, r=r))
    print(f"[pool] 多门源可渲染未判候选 n={len(pool)}")

    # ---------------- 分组选择 ----------------
    sel, seen, selmap = [], set(), {}

    def take(item, grp):
        if item["pid"] in seen:
            return False
        seen.add(item["pid"])
        item["grp"] = grp
        selmap[item["pid"]] = item
        sel.append(item)
        return True

    def pick_quota(cands, cap, key):
        """机位配额轮询：源内按 key 排序，跨源轮询取值（保证 305/316/323 均衡）。"""
        bys = collections.defaultdict(list)
        for c in cands:
            bys[c["r"]["a"].split(":")[0]].append(c)
        for s in bys:
            bys[s] = sorted(bys[s], key=key)
        out = []
        while len(out) < cap:
            added = False
            for s in sorted(bys):
                if bys[s] and len(out) < cap:
                    out.append(bys[s].pop(0))
                    added = True
            if not added:
                break
        return out

    # G1 先验改变排序（rank 增益 ≥ 2：先验把候选相对顶起）
    g1 = [x for x in pool
          if (x["r"]["rank_old"] or 99) - (x["r"]["rank_new"] or 99) >= 2]
    n1 = sum(take(x, "G1") for x in pick_quota(g1, CAP["G1"], lambda x: -x["r"]["score_prior"]))

    # G2 竞争对（每 a 端 top-1/top-2 各一条；a 端按机位轮询保证均衡）
    by_a = collections.defaultdict(list)
    for x in pool:
        by_a[x["r"]["a"]].append(x)
    a_order = sorted(by_a, key=lambda a: -max(x["r"]["score_prior"] for x in by_a[a]))
    a_by_src = collections.defaultdict(list)
    for a in a_order:
        a_by_src[a.split(":")[0]].append(a)
    a_pick = []
    while any(a_by_src.values()):
        for s in sorted(a_by_src):
            if a_by_src[s]:
                a_pick.append(a_by_src[s].pop(0))
    n2 = 0
    for a in a_pick:
        if n2 >= CAP["G2"]:
            break
        lst = sorted(by_a[a], key=lambda x: x["r"]["rank_new"])
        picks = [x for x in lst if x["r"]["rank_new"] in (1, 2) and x["pid"] not in seen][:2]
        if len(picks) < 2:
            continue
        for x in picks:
            take(x, "G2")
            n2 += 1

    # G3 高 sim 被压制
    g3 = [x for x in pool if x["r"]["sim"] >= 0.94 and (x["r"]["rank_new"] or 0) >= 4
          and x["pid"] not in seen]
    n3 = sum(take(x, "G3") for x in pick_quota(g3, CAP["G3"], lambda x: -x["r"]["sim"]))

    # G4 relay 备书对（active/confirmed 边）
    g4 = [x for x in pool if x["r"].get("status") in ("active", "confirmed")
          and x["pid"] not in seen]
    n4 = sum(take(x, "G4") for x in pick_quota(g4, CAP["G4"], lambda x: -(x["r"]["sim"] or 0)))

    print(f"[select] G1 先验改变 {n1} | G2 竞争对 {n2} | G3 高sim压制 {n3} | G4 备书 {n4}"
          f" | 合计 {len(sel)}")
    by_grp = collections.Counter(x["grp"] for x in sel)
    by_src = collections.Counter(x["r"]["a"].split(":")[0] for x in sel)
    print(f"[select] 组分布 {dict(by_grp)} | 源分布 {dict(by_src)}")

    if args.dry_run:
        for x in sel[:12]:
            r = x["r"]
            print(f"  [{x['grp']}] {x['pid']} sim={r['sim']} rank {r['rank_old']}->{r['rank_new']}"
                  f" w={r['w_portal']} status={r.get('status')}")
        return

    # ---------------- 渲染 ----------------
    os.makedirs(TH, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
    TMP = tempfile.mkdtemp(prefix="prt_frames_")
    vpath, fcache = {}, {}

    def video_of(cam):
        if cam not in vpath:
            g = sorted(glob.glob(os.path.join(SYNCED, f"*_{cam}", f"{cam}_*.mp4")))
            vpath[cam] = g[0] if g else None
        return vpath[cam]

    def frame_img(cam, f_idx):
        k = (cam, int(f_idx))
        if k not in fcache:
            vp = video_of(cam)
            if not vp:
                fcache[k] = None
                return None
            out = os.path.join(TMP, f"{cam}_{int(f_idx)}.jpg")
            subprocess.run([FF, "-hide_banner", "-loglevel", "error",
                            "-ss", f"{int(f_idx) / FPS:.3f}", "-i", vp,
                            "-frames:v", "1", "-q:v", "2", "-y", out],
                           capture_output=True)
            fcache[k] = out if os.path.exists(out) else None
        return fcache[k]

    # gid（当前库 gperson label；参考展示）
    import sqlite3
    gm = {}
    try:
        con = sqlite3.connect(GIL)
        gl = {r[0]: r[1] for r in con.execute("SELECT id,label FROM gperson")}
        oid2key = {r[0]: f"{r[1]}|{r[2]}" for r in
                   con.execute("SELECT id,cam,tid FROM obs WHERE tombstone=0")}
        for gid, nid in con.execute("SELECT gperson_id,node_id FROM membership WHERE node_kind='obs'"):
            k = oid2key.get(nid)
            if k:
                gm[k] = gl.get(gid, str(gid))
        # entity → 成员 obs keys（取任一成员）
        ent2k = {}
        for c, tid, eid in con.execute("SELECT cam,tid,entity_id FROM obs "
                                       "WHERE tombstone=0 AND entity_id IS NOT NULL"):
            ent2k.setdefault(eid, f"{c}|{tid}")
    except Exception as e:
        print("[warn] gperson 参考查询失败:", e)
        ent2k = {}
    ent_ids = {}
    try:
        for r in con.execute("SELECT id,eid FROM entity"):
            pass
    except Exception:
        pass
    # entity eid（'316-E012'）→ 主库 entity.id：
    eid2id = {}
    try:
        for r in con.execute("SELECT id,cam,eid FROM entity"):
            eid2id[r[2]] = r[0]
    except Exception:
        pass

    n_thumb, miss = 0, []

    def crop_ent(cam, eid, which, k):
        """which='a' 尾部帧 [-3,-10]；'b' 头部帧 [2,9]"""
        nonlocal n_thumb
        fn = f"PRT_{eid}_{which}{k}.jpg"
        p = os.path.join(TH, fn)
        if os.path.exists(p):
            return fn
        e = ents[cam].get(eid)
        if e is None:
            miss.append((eid, which, k, "no ent"))
            return None
        fr, u, v, h = e
        n = len(fr)
        idx = (n - 3 - 5 * k) if which == "a" else (2 + 5 * k)
        idx = max(0, min(idx, n - 1))
        fp = frame_img(cam, fr[idx])
        if not fp:
            miss.append((eid, which, k, "frame fail"))
            return None
        im = cv2.imread(fp)
        if im is None:
            miss.append((eid, which, k, "decode fail"))
            return None
        H, W = im.shape[:2]
        nw, nh = NATIVE[cam]
        kx, ky = W / nw, H / nh
        cx = u[idx] * nw * kx
        cyv = v[idx] * nh * ky
        hh = h[idx] * ky
        cy = cyv - hh / 2
        bw, bh = hh * 0.55, hh
        w = max(bw * (1 + 2 * PAD), 96)
        h2 = max(bh * (1 + 2 * PAD), 128)
        X1 = int(max(0, cx - w / 2)); X2 = int(min(W, cx + w / 2))
        Y1 = int(max(0, cy - h2 / 2)); Y2 = int(min(H, cy + h2 / 2))
        sub = im[Y1:Y2, X1:X2]
        if sub.size == 0:
            miss.append((eid, which, k, "empty"))
            return None
        sub = cv2.resize(sub, (max(1, int(sub.shape[1] * H_OUT / sub.shape[0])), H_OUT))
        cv2.imwrite(p, sub, [cv2.IMWRITE_JPEG_QUALITY, 82])
        n_thumb += 1
        return fn

    def gid_of(cam, eid):
        i = eid2id.get(eid)
        k = ent2k.get(i) if i is not None else None
        return gm.get(k) if k else None

    items = []
    for x in sel:
        r = x["r"]
        ca, eida = r["a"].split(":", 1)
        cb, eidb = r["b"].split(":", 1)
        sa = crop_ent(ca, eida, "a", 0)
        sa2 = crop_ent(ca, eida, "a", 1)
        sb = crop_ent(cb, eidb, "b", 0)
        sb2 = crop_ent(cb, eidb, "b", 1)
        side_a = [dict(f=f, gid=gid_of(ca, eida)) for f in (sa, sa2) if f]
        side_b = [dict(f=f, gid=gid_of(cb, eidb)) for f in (sb, sb2) if f]
        grp_note = {"G1": "先验改变排序（rank 增益≥2）", "G2": "同源竞争 top-2",
                    "G3": "高 cos 被先验压制", "G4": "relay active/confirmed 备书"}[x["grp"]]
        items.append(dict(
            id=x["pid"], kind="pair",
            a=f"{ca}_E_{eida}", b=f"{cb}_E_{eidb}", sa=eida, sb=eidb,
            dt=r.get("dt"), cos=r.get("sim"),
            relay_v2=(r.get("status") or "candidate"),
            w_portal=r.get("w_portal"), rank_new=r.get("rank_new"),
            machine={"G1": "prior_changed", "G2": "compete",
                     "G3": "prior_suppressed", "G4": "relay_backing"}[x["grp"]],
            note=(f"[{x['grp']}] {grp_note} · 排序 {r.get('rank_old')}→{r.get('rank_new')}"
                  f" · 预测门 {r.get('predicted_gate')}"
                  + ("（人端对齐已由先验压缩）" if False else "")),
            side_a=side_a, side_b=side_b,
        ))
    print(f"[render] 缩略图 {n_thumb} 张；缺失 {len(miss)}")
    for m in miss[:10]:
        print("   miss:", m)

    # ---------------- 合并载荷 ----------------
    s6b = json.load(open(S6B, encoding="utf-8"))
    newg = dict(key="portal_v2",
                title="④ 门户交接补口（316/305/323 源）",
                hint="判：这两段观测是同一人吗？（补口批：为多门机位先验/relay 排序验收补真值；"
                     "A=源（离开方）尾部帧、B=目标（进入方）头部帧）",
                count=len(items), items=items)
    s6b["groups"] = [g for g in s6b["groups"] if g.get("key") != "portal_v2"] + [newg]
    s6b["title"] = "待裁决批：S6-b 倒影 + 门户交接补口（316/305/323 源）"
    s6b["note"] = ("本页两批：①②③= S6-b 倒影清单（前缀 S6B_）；④= 门户交接补口"
                   "（前缀 PRT_）。判定均存入 labels.json 的 pair 命名空间，不影响既有 241 对统计。")
    json.dump(s6b, open(S6B, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {S6B}（groups={[g['key'] for g in s6b['groups']]}）")

    rec = dict(version="v2portal/1", note="门户交接补口批选择记录",
               pool_n=len(pool), selected_n=len(sel),
               cap=CAP, by_grp=dict(by_grp), by_src=dict(by_src),
               source={"prior": "analysis/relay_v2/scored_v2_prior.json",
                       "labels_snapshot": os.path.relpath(LABELS_RM, ROOT).replace("\\", "/")},
               items=[dict(pid=x["pid"], grp=x["grp"], a=x["r"]["a"], b=x["r"]["b"],
                           sim=x["r"]["sim"], rank_old=x["r"]["rank_old"],
                           rank_new=x["r"]["rank_new"], w_portal=x["r"]["w_portal"],
                           status=x["r"].get("status")) for x in sel])
    json.dump(rec, open(os.path.join(OUTDIR, "selection_v1.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[out] {OUTDIR}/selection_v1.json")
    # 本地 thumbs 清单
    print("\n上传清单：")
    print(f"  webapp/frontend/public/s6b_thumbs/PRT_*.jpg  ({n_thumb} 张)")
    print("  webapp/frontend/public/s6b_pending.json")


if __name__ == "__main__":
    main()
