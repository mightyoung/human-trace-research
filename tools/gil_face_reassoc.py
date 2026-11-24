#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重关联 v2（离线）：升级 face_shot_p0 的人脸↔轨迹关联，不重检测。

P0 v1 关联 = 「脸中心落原框内，取最小面积框」→ 305/318 关联率偏低（脸伸出人体框）。
v2 规则（与 gil_face_capture.associate_track 同源）：
  1) in_exact：中心落原框内 → 取最小面积（v1 旧规则）
  2) in_head ：中心落头部扩展框（侧 12% / 上 35% / 下 8%）→ 取「头锚点」距离最近
  3) nearest ：兜底：归一距离 ≤ 0.50 且歧义守卫（次近 ≥ 1.35× 最近 + 0.08）

输入：face_shot_{src}.json + cleanset_v2/{cam}_detections.jsonl + GIL sqlite
输出：face_shot_{out}.json + face_best_{out}.json + crops/ 缺失补齐
      + face_reassoc_audit_{out}.json + debug_assoc/ 抽样目检图

用法：python tools/gil_face_reassoc.py --src-tag p0 --out-tag p0b
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gil_face_capture import (  # noqa: E402
    associate_track, build_best_pools, load_tracks, load_gil_maps,
    VIDEOS, FACE_DIR, CROPS_DIR, OC, HEAD_PAD_X, HEAD_PAD_TOP, HEAD_PAD_BOT,
    NEAREST_D_MAX, NEAREST_RATIO)

DEBUG_DIR = os.path.join(FACE_DIR, "debug_assoc")


def crop_of(img, face, margin=(0.35, 0.45)):
    x1, y1, x2, y2 = face
    mw, mh = (x2 - x1) * margin[0], (y2 - y1) * margin[1]
    a = max(0, int(x1 - mw)); b = max(0, int(y1 - mh))
    c = min(img.shape[1], int(x2 + mw)); d = min(img.shape[0], int(y2 + mh))
    if c <= a or d <= b:
        return None
    return img[b:d, a:c].copy()


def iter_scanned(frames_needed, cam):
    """按需随机 seek 读帧（帧数少时比全片扫描快）。yield (frame, img)

    ★ 视频路径统一走 gil_face_capture.video_path，才能跟着 --pack 切到 35 分钟长卷。
    """
    from gil_face_capture import video_path
    vp = video_path(cam)
    cap = cv2.VideoCapture(vp)
    try:
        for fr in sorted(set(frames_needed)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
            ok, img = cap.read()
            if ok:
                yield fr, img
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-tag", default="p0")
    ap.add_argument("--out-tag", default="p0b")
    ap.add_argument("--no-crops", action="store_true", help="跳过缺失 crop 补齐与 ent/gp 复制")
    ap.add_argument("--debug-n", type=int, default=12)
    args = ap.parse_args()

    with open(os.path.join(FACE_DIR, f"face_shot_{args.src_tag}.json"), encoding="utf-8") as f:
        src = json.load(f)
    shots = src["shots"]
    cams = src["meta"]["cams"]
    print(f"loaded {len(shots)} shots from tag={args.src_tag}", flush=True)

    gil = load_gil_maps()
    stats = {}
    changed = []
    # v1 基线：usable 关联率
    v1 = defaultdict(lambda: [0, 0])
    for s in shots:
        if s.get("usable"):
            v1[s["cam"]][1] += 1
            if s.get("tid") is not None:
                v1[s["cam"]][0] += 1

    for cam in cams:
        tb = load_tracks(cam)
        n_faces = n_assoc = n_assoc_u = n_faces_u = 0
        reason_c = defaultdict(int)
        for s in shots:
            if s["cam"] != cam:
                continue
            n_faces += 1
            tracks = tb.get(s["frame"], [])
            tid2, reason = associate_track(s["face"], tracks)
            reason_c[reason] += 1
            old = s.get("tid")
            if tid2 is not None:
                n_assoc += 1
            if s.get("usable"):
                n_faces_u += 1
                if tid2 is not None:
                    n_assoc_u += 1
            if tid2 != old:
                changed.append(dict(cam=cam, frame=s["frame"], t=s.get("t"),
                                    face=s["face"], old_tid=old, new_tid=tid2,
                                    reason=reason, q=s.get("q"), usable=bool(s.get("usable"))))
            s["tid"] = tid2
            s["assoc_reason"] = reason
            g = gil.get((cam, tid2), {}) if tid2 is not None else {}
            s["obs_id"] = g.get("obs_id")
            s["entity_id"] = g.get("entity_id")
            s["gperson_id"] = g.get("gperson_id")
        a1u, a1n = v1[cam]
        stats[cam] = dict(
            faces=n_faces, assoc=n_assoc, assoc_rate=round(n_assoc / max(n_faces, 1), 4),
            faces_usable=n_faces_u, assoc_usable=n_assoc_u,
            assoc_rate_usable=round(n_assoc_u / max(n_faces_u, 1), 4),
            assoc_rate_usable_v1=round(a1u / max(a1n, 1), 4),
            reasons=dict(reason_c))
        print(f"[{cam}] faces={n_faces} assoc={n_assoc} ({stats[cam]['assoc_rate']}) "
              f"usable assoc {a1u}/{a1n} -> {n_assoc_u}/{n_faces_u} "
              f"({stats[cam]['assoc_rate_usable_v1']} -> {stats[cam]['assoc_rate_usable']})", flush=True)

    # ---- 质量池重建 ----
    best_tid, best_ent, best_gp, _ = build_best_pools(shots, cams)
    print(f"pools: tids={len(best_tid)} entities={len(best_ent)} gpersons={len(best_gp)}", flush=True)

    # ---- crop_map 重建 + 缺失补齐 ----
    crop_map = {}
    for fn in os.listdir(CROPS_DIR):
        if not fn.startswith("tid_"):
            continue
        # tid_{cam}_{tid}_f{frame}_q{q}.jpg
        parts = fn.split("_")
        if len(parts) >= 5:
            crop_map[f"{parts[1]}_{parts[2]}"] = fn
    n_added = 0
    if not args.no_crops:
        need = []
        for key, lst in best_tid.items():
            cam, tid_s = key.split("|")
            s0 = lst[0]
            if not (s0.get("usable") and s0.get("tid") is not None):
                continue
            cur = crop_map.get(f"{cam}_{tid_s}")
            if cur is not None and f"_f{s0['frame']}_" in cur:
                continue          # 现存 crop 已指向最新最优帧
            need.append((cam, tid_s, s0, cur))
        by_cam = defaultdict(list)
        for cam, tid_s, s0, cur in need:
            by_cam[cam].append((tid_s, s0, cur))
        for cam, items in by_cam.items():
            frames_needed = [s0["frame"] for _, s0, _ in items]
            img_by_fr = dict(iter_scanned(frames_needed, cam))
            for tid_s, s0, cur in items:
                img = img_by_fr.get(s0["frame"])
                if img is None:
                    continue
                crop = crop_of(img, s0["face"])
                if crop is None:
                    continue
                fn = f"tid_{cam}_{tid_s}_f{s0['frame']}_q{s0['q']:.2f}.jpg"
                if cv2.imwrite(os.path.join(CROPS_DIR, fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    if cur and cur != fn:
                        try:
                            os.remove(os.path.join(CROPS_DIR, cur))
                        except OSError:
                            pass
                    crop_map[f"{cam}_{tid_s}"] = fn
                    n_added += 1
        print(f"crops refreshed/added={n_added} (total tid crops={len(crop_map)})", flush=True)

    # ---- ent/gp 代表图复制 ----
    gp_label = {}
    import sqlite3
    con = sqlite3.connect(os.path.join(OC, "analysis", "gil", "store_v1", "gil.sqlite"))
    gp_label = {r[0]: r[1] for r in con.execute("select id, label from gperson")}
    con.close()
    n_ent = n_gp = 0
    if not args.no_crops:
        import shutil
        for k, lst in best_ent.items():
            s0 = lst[0]
            sr = crop_map.get(f'{s0["cam"]}_{s0["tid"]}') if s0.get("tid") is not None else None
            if sr:
                shutil.copy(os.path.join(CROPS_DIR, sr), os.path.join(CROPS_DIR, f"ent_{k}.jpg"))
                n_ent += 1
        for k, lst in best_gp.items():
            s0 = lst[0]
            sr = crop_map.get(f'{s0["cam"]}_{s0["tid"]}') if s0.get("tid") is not None else None
            if sr:
                shutil.copy(os.path.join(CROPS_DIR, sr), os.path.join(CROPS_DIR, f"gp_{gp_label.get(k, k)}.jpg"))
                n_gp += 1
        print(f"ent imgs={n_ent} gp imgs={n_gp}", flush=True)

    # ---- 审计 + debug 抽样 ----
    audit = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), from_tag=args.src_tag,
                 rules=dict(head_pad=[HEAD_PAD_X, HEAD_PAD_TOP, HEAD_PAD_BOT],
                            nearest_d=NEAREST_D_MAX, nearest_ratio=NEAREST_RATIO),
                 stats=stats, n_changed=len(changed), changed=changed)
    with open(os.path.join(FACE_DIR, f"face_reassoc_audit_{args.out_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=1)

    os.makedirs(DEBUG_DIR, exist_ok=True)
    cand = [c for c in changed if c["new_tid"] is not None and c["usable"]
            and c["reason"] in ("in_head", "nearest")]
    picks = []
    per_cam_n = defaultdict(int)
    for c in cand:                      # 各机位最多 2 张，保证目检覆盖面
        if per_cam_n[c["cam"]] < 2:
            per_cam_n[c["cam"]] += 1
            picks.append(c)
        if len(picks) >= args.debug_n:
            break
    if picks:
        by_cam = defaultdict(list)
        for c in picks:
            by_cam[c["cam"]].append(c)
        idx = 0
        for cam, items in by_cam.items():
            tb = load_tracks(cam)
            img_by_fr = dict(iter_scanned([c["frame"] for c in items], cam))
            for c in items:
                img = img_by_fr.get(c["frame"])
                if img is None:
                    continue
                canvas = img.copy()
                for tid, b in tb.get(c["frame"], []):
                    col = (60, 200, 60) if tid == c["new_tid"] else (128, 128, 128)
                    th = 3 if tid == c["new_tid"] else 1
                    cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, th)
                    if tid == c["new_tid"]:
                        cv2.putText(canvas, f"tid={tid}", (int(b[0]), max(20, int(b[1]) - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 220, 60), 2, cv2.LINE_AA)
                fx = c["face"]
                cv2.rectangle(canvas, (int(fx[0]), int(fx[1])), (int(fx[2]), int(fx[3])), (0, 0, 230), 2)
                info = f"{c['reason']} old={c['old_tid']} q={c['q']:.2f}"
                cv2.putText(canvas, info, (int(fx[0]), max(20, int(fx[1]) - 34)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2, cv2.LINE_AA)
                out = os.path.join(DEBUG_DIR, f"reassoc_{idx:02d}_{cam}_f{c['frame']}_{c['reason']}.jpg")
                cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
                idx += 1
        print(f"debug samples -> {DEBUG_DIR} ({idx})", flush=True)

    # ---- 输出 ----
    meta = dict(src["meta"])
    meta["reassoc"] = dict(from_tag=args.src_tag, out_tag=args.out_tag,
                           created=time.strftime("%Y-%m-%d %H:%M:%S"),
                           stats_v2=stats, n_changed=len(changed),
                           rules=audit["rules"])
    out = dict(meta=meta, stats=stats, shots=shots)
    with open(os.path.join(FACE_DIR, f"face_shot_{args.out_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    best = dict(meta=meta, per_tid=best_tid, per_entity=best_ent,
                per_gperson=best_gp, gp_label=gp_label, crop_map=crop_map)
    with open(os.path.join(FACE_DIR, f"face_best_{args.out_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=1, default=float)
    print(f"changed={len(changed)} | shots={len(shots)} tids={len(best_tid)} "
          f"entities={len(best_ent)} gpersons={len(best_gp)} -> tag {args.out_tag}", flush=True)


if __name__ == "__main__":
    main()
