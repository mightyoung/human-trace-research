#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L0 轨迹级活体/证据质量规则层（零模型，v0）

定位：对 P0 抓取产物做"非人面孔 / 证据质量"筛查，只产 flag 与证据卡，
不改 GIL、不改任何上游产物。威胁模型见 docs/face-liveness-survey.md：
本项目"活体"语义 = 证据质量控制（防海报/屏幕/静物/倒影等非人面孔入链）。

规则（阈值=经验初值，待 val 校准）：
- R1 static_unaccompanied：静态簇 + 多数时刻无伴检（无轨迹关联）
  → 海报/屏幕/静物嫌疑（强规则）
- R2 static_lowq：静态簇 + 全程无可用质量（弱规则，需人工审计）
- R3 assoc_gap：可用脸但无轨迹关联（证据完整性问题，非活体）
- R4 reflection_linked：obs 命中 S5/S6 倒影标注（空间复制，与活体互补）

输出：
- output_crosscam/analysis/face/liveness_rule_v0.json
- output_crosscam/analysis/face/debug_assoc/liveness/*.jpg 证据卡 + montage

用法：python tools/gil_face_liveness.py --tag p0c [--out-tag v0]
"""
import argparse
import json
import os
import sqlite3
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DIR = os.path.join(ROOT, "output_crosscam", "analysis", "face")
GIL_DB = os.path.join(ROOT, "output_crosscam", "analysis", "gil", "store_v1", "gil.sqlite")
OUT_DIR = os.path.join(FACE_DIR, "debug_assoc", "liveness")

VIDEOS = {
    "305": "media-resource/multi-cemra/synced-10min/02_办公区1_305/305_085500-090500.mp4",
    "316": "media-resource/multi-cemra/synced-10min/01_行政走廊通道_316/316_085500-090500.mp4",
    "318": "media-resource/multi-cemra/synced-10min/03_办公区2_318/318_085500-090500.mp4",
    "320": "media-resource/multi-cemra/synced-10min/04_办公区球机_320/320_085500-090500.mp4",
    "323": "media-resource/multi-cemra/synced-10min/05_茶水间_323/323_085500-090500.mp4",
    "343": "media-resource/multi-cemra/synced-10min/06_电梯间_343/343_085500-090500.mp4",
}

# ---- 规则阈值（经验初值，待 val 校准）----
R1 = dict(min_n=20, min_span=180.0, max_std=10.0, min_tn=0.35)
R2 = dict(min_n=15, min_span=120.0, max_std=16.0, max_qmax=0.70)
BIN_PX = 40          # 细网格
FPS = 25.0


def load_shots(tag):
    with open(os.path.join(FACE_DIR, f"face_shot_{tag}.json"), encoding="utf-8") as f:
        return json.load(f)["shots"]


def bins_of_shots(shots_cam):
    """40px 网格 → {bin_key: [shots]}"""
    bins = defaultdict(list)
    for s in shots_cam:
        fx = (s["face"][0] + s["face"][2]) / 2
        fy = (s["face"][1] + s["face"][3]) / 2
        bins[(int(fx // BIN_PX), int(fy // BIN_PX))].append(s)
    return bins


def group_adjacent(keys):
    """把通过阈值的 bin 按 8 邻域归组（只在通过者之间合并，避免链式传染）"""
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ks = set(keys)
    for k in keys:
        bx, by = k
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (bx + dx, by + dy)
                if nb in ks:
                    union(k, nb)
    groups = defaultdict(list)
    for k in keys:
        groups[find(k)].append(k)
    return list(groups.values())


def cluster_stats(cl):
    frames = sorted(s["frame"] for s in cl)
    span = (frames[-1] - frames[0]) / FPS
    cxs = np.array([(s["face"][0] + s["face"][2]) / 2 for s in cl])
    cys = np.array([(s["face"][1] + s["face"][3]) / 2 for s in cl])
    std_r = float(np.sqrt(cxs.var() + cys.var()))
    w_med = float(np.median([s["face"][2] - s["face"][0] for s in cl]))
    tn = sum(1 for s in cl if s.get("tid") is None) / len(cl)
    usable_rate = sum(1 for s in cl if s.get("usable")) / len(cl)
    q_max = max(s["q"] for s in cl)
    return dict(n=len(cl), span=round(span, 1), std_r=round(std_r, 1), w_med=round(w_med, 1),
                tn=round(tn, 3), usable=round(usable_rate, 3), q_max=round(q_max, 3),
                cx=round(float(cxs.mean()), 0), cy=round(float(cys.mean()), 0),
                f0=frames[0], f1=frames[-1])


def classify(cl, st):
    """返回 rule 或 None"""
    if st["span"] >= R1["min_span"] and st["std_r"] <= R1["max_std"] \
            and st["n"] >= R1["min_n"] and st["tn"] >= R1["min_tn"]:
        return "R1_static_unaccompanied"
    if st["span"] >= R2["min_span"] and st["std_r"] <= R2["max_std"] \
            and st["n"] >= R2["min_n"] and st["usable"] == 0.0 and st["q_max"] < R2["max_qmax"]:
        return "R2_static_lowq"
    return None


def video_path(cam):
    p = VIDEOS.get(cam)
    if not p:
        return None
    full = os.path.join(ROOT, p)
    return full if os.path.exists(full) else None


def make_cards(cam, cid, cl, sample_k=3):
    """证据卡：精确框放大 x4 + 上下文（±120px），返回文件名列表"""
    sel = sorted(cl, key=lambda s: s["frame"])
    idxs = sorted(set([0, len(sel) // 2, len(sel) - 1]))[:sample_k]
    vp = video_path(cam)
    cards = []
    if vp is None:
        return cards, []
    cap = cv2.VideoCapture(vp)
    samples = []
    for k, i in enumerate(idxs):
        s = sel[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame"])
        ok, img = cap.read()
        if not ok:
            continue
        f = s["face"]
        a, b, c, d = max(0, int(f[0])), max(0, int(f[1])), int(f[2]), int(f[3])
        crop = img[b:d, a:c]
        if crop.size == 0:
            continue
        m = 120
        a2, b2 = max(0, a - m), max(0, b - m)
        c2, d2 = min(img.shape[1], c + m), min(img.shape[0], d + m)
        ctx = img[b2:d2, a2:c2]
        # 拼成一张卡：上=zoom，下=ctx（统一高 150+150）
        z = cv2.resize(crop, (max(1, int(crop.shape[1] * 4)), max(1, int(crop.shape[0] * 4))),
                       interpolation=cv2.INTER_NEAREST)
        zh = 170
        z = cv2.resize(z, (int(z.shape[1] * zh / z.shape[0]), zh))
        ch = 170
        ctx = cv2.resize(ctx, (int(ctx.shape[1] * ch / ctx.shape[0]), ch))
        W = max(z.shape[1], ctx.shape[1])
        canvas = np.full((zh + ch + 24, W, 3), 245, np.uint8)
        canvas[0:zh, 0:z.shape[1]] = z
        canvas[zh + 24:zh + 24 + ch, 0:ctx.shape[1]] = ctx
        txt = f"{cam} c{cid} f{s['frame']} q={s['q']:.2f} tid={s.get('tid')}"
        cv2.putText(canvas, txt, (4, zh + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        fn = f"live_{cam}_c{cid}_{k}_f{s['frame']}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, fn), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cards.append(fn)
        samples.append(dict(frame=s["frame"], q=round(s["q"], 3), tid=s.get("tid"),
                            face=[round(float(v), 1) for v in f]))
    cap.release()
    return cards, samples


def make_montage(cluster_recs, out_name="liveness_clusters_montage.jpg", cols=4, cell_w=300):
    """把各簇第一张证据卡拼成 montage 便于总览审计"""
    imgs = []
    labels = []
    for rec in cluster_recs:
        if not rec["cards"]:
            continue
        p = os.path.join(OUT_DIR, rec["cards"][0])
        img = cv2.imread(p)
        if img is None:
            continue
        sc = cell_w / img.shape[1]
        img = cv2.resize(img, (cell_w, int(img.shape[0] * sc)))
        imgs.append(img)
        labels.append(f"{rec['cam']} c{rec['id']} {rec['rule'][:2]} n={rec['stats']['n']} "
                      f"std={rec['stats']['std_r']:.0f} tn={rec['stats']['tn']:.0%}")
    if not imgs:
        return None
    cell_h = max(i.shape[0] for i in imgs)
    rows = (len(imgs) + cols - 1) // cols
    canvas = np.full((rows * (cell_h + 22) + 10, cols * (cell_w + 8) + 8, 3), 248, np.uint8)
    for i, (img, lab) in enumerate(zip(imgs, labels)):
        r, c = divmod(i, cols)
        y0 = r * (cell_h + 22) + 10
        x0 = c * (cell_w + 8) + 8
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        cv2.putText(canvas, lab, (x0, y0 + img.shape[0] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (20, 20, 20), 1, cv2.LINE_AA)
    fn = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(fn, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p0c")
    ap.add_argument("--out-tag", default="v0")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    shots = load_shots(args.tag)
    print(f"[liveness] tag={args.tag} shots={len(shots)}", flush=True)

    # ---- R1/R2 静态簇（bin 级判定，再合并通过者）----
    cluster_recs = []
    for cam in sorted(set(s["cam"] for s in shots)):
        cam_shots = [s for s in shots if s["cam"] == cam]
        bins = bins_of_shots(cam_shots)
        passing = {}   # bin_key -> (rule, stats)
        for k, cl in bins.items():
            st = cluster_stats(cl)
            rule = classify(cl, st)
            if rule:
                passing[k] = (rule, st)
        groups = group_adjacent(list(passing.keys())) if passing else []
        cid = 0
        for grp in groups:
            # 代表 bin：R1 优先，其次 n 最大
            rep = max(grp, key=lambda k: (passing[k][0].startswith("R1"), passing[k][1]["n"]))
            rule, st = passing[rep]
            cl = []
            for k in grp:
                cl.extend(bins[k])
            cards, samples = make_cards(cam, cid, cl)
            st_merged = cluster_stats(cl)
            cluster_recs.append(dict(cam=cam, id=cid, rule=rule,
                                     stats=st, stats_merged=st_merged,
                                     members=[list(k) for k in grp],
                                     cards=cards, samples=samples))
            cid += 1
        print(f"  [{cam}] bins={len(bins)} passing={len(passing)} clusters={cid}", flush=True)

    # ---- R3 关联缺口（可用脸无轨迹）----
    assoc_gap = [dict(cam=s["cam"], frame=s["frame"], t=s["t"], q=round(s["q"], 3),
                      face=[round(float(v), 1) for v in s["face"]], obs_id=s.get("obs_id"))
                 for s in shots if s.get("usable") and s.get("tid") is None]
    gap_by_cam = defaultdict(int)
    for r in assoc_gap:
        gap_by_cam[r["cam"]] += 1

    # ---- R4 倒影交叉引用 ----
    # 注意：annotation 的 obs 级 target_id 是 "cam|tid" 字符串 → 经 obs 表映射到 obs.id
    refl_obs, mask_obs = set(), set()
    n_ann_skipped = 0
    if os.path.exists(GIL_DB):
        con = sqlite3.connect(GIL_DB)
        obs_map = defaultdict(list)
        for oid, cam, tid in con.execute("select id, cam, tid from obs"):
            obs_map[(str(cam), int(tid))].append(int(oid))
        for tgt, val in con.execute("select target_id, value from annotation where target_kind='obs'"):
            if val not in ("reflection", "wall_mirror", "mask_unexplained"):
                continue
            s_tgt = str(tgt)
            key = None
            if "|" in s_tgt:
                cam_s, _, tid_s = s_tgt.partition("|")
                try:
                    key = (cam_s, int(tid_s))
                except ValueError:
                    key = None
            if key is None:
                try:
                    oid = int(s_tgt)
                    (mask_obs if val == "mask_unexplained" else refl_obs).add(oid)
                    continue
                except ValueError:
                    n_ann_skipped += 1
                    continue
            for oid in obs_map.get(key, []):
                if val == "mask_unexplained":
                    mask_obs.add(oid)
                else:
                    refl_obs.add(oid)
        con.close()
    refl_linked = []
    for s in shots:
        oid = s.get("obs_id")
        if oid is not None and int(oid) in refl_obs:
            refl_linked.append(dict(cam=s["cam"], frame=s["frame"], tid=s.get("tid"),
                                    q=round(s["q"], 3), usable=bool(s.get("usable")),
                                    obs_id=int(oid), gperson_id=s.get("gperson_id")))
    n_refl_usable = sum(1 for r in refl_linked if r["usable"])

    # ---- 汇总 ----
    montage = make_montage(cluster_recs)
    out = dict(
        meta=dict(created=__import__("time").strftime("%Y-%m-%d %H:%M:%S"), tag=args.tag,
                  note="L0 零模型规则层：只产 flag 与证据卡，不改 GIL/上游；阈值经验初值待 val 校准"),
        params=dict(r1=R1, r2=R2, bin_px=BIN_PX),
        static_clusters=cluster_recs,
        assoc_gap=dict(n=len(assoc_gap), by_cam=dict(gap_by_cam), items=assoc_gap),
        reflection_linked=dict(n=len(refl_linked), n_usable=n_refl_usable,
                               note="obs 命中 S5/S6 倒影标注（仅存在时非空）", items=refl_linked),
        n_mask_unexplained_obs=len(mask_obs),
        montage=os.path.basename(montage) if montage else None,
    )
    out_path = os.path.join(FACE_DIR, f"liveness_rule_{args.out_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"[liveness] R1/R2 clusters: {len(cluster_recs)} "
          f"(R1={sum(1 for r in cluster_recs if r['rule'].startswith('R1'))}, "
          f"R2={sum(1 for r in cluster_recs if r['rule'].startswith('R2'))})", flush=True)
    print(f"[liveness] R3 assoc_gap usable: {len(assoc_gap)} by cam {dict(gap_by_cam)}", flush=True)
    print(f"[liveness] R4 reflection_linked: {len(refl_linked)} (usable {n_refl_usable})", flush=True)
    print(f"[liveness] -> {out_path}", flush=True)
    if montage:
        print(f"[liveness] montage -> {montage}", flush=True)


if __name__ == "__main__":
    main()
