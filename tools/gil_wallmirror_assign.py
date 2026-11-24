#!/usr/bin/env python
"""S6 / Q4-补：墙玻璃镜像的几何判据 —— 让「零掩膜」成为可能。

背景：
  S5 用水平面镜像（v 轴镜像）解决了**地板倒影**；而 316/323 的**墙玻璃镜像**
  （竖直面）此前只能靠"区域掩膜"剔除——掩膜被用户否定（逐帧删框打断轨迹 + 误删真人）。

物理关系（镜像面 = 竖直玻璃平面，图像中为一条 u=u_axis 的竖线）：
  u_R ≈ 2·u_axis − u_S   （u 轴反相）  ⇒ 逐帧锁步：Δu_R ≈ −Δu_S  ⇒ corr(Δu_R, Δu_S) ≈ −1
  v_R ≈ v_S               （v 同相）    ⇒ corr(Δv_R, Δv_S) ≈ +1
  h_R ≈ h_S               （尺度同量级；距离玻璃远近会让尺度有偏差）

★ 自监督：u_axis 未知，但对每条镜像对可估 u_axis = (u_R + u_S)/2；多条对的 u_axis
  若聚集在同一竖线上，那条竖线就是"玻璃墙"的位置（= 结构语义发现）。

用法：
  python tools/gil_wallmirror_assign.py --cams 316,323
  python tools/gil_wallmirror_assign.py --apply     # 另写 GIL annotation
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
from mirror_kinematics import load_tracks, pearson  # noqa: E402

CAMS = ["305", "316", "318", "320", "323", "343"]
DIMS = {"305": (2560, 1440), "316": (2560, 1440), "318": (1920, 1080),
        "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}

RULE = dict(n_min=5, du_corr_max=-0.50, dv_corr_min=0.30,
            dv_max=0.15, hr_lo=0.70, hr_hi=1.45, axis_tol=0.04)


def cooccur_pairs(tracks, n_min):
    fmap = collections.defaultdict(list)
    for tid, l in tracks.items():
        if len(l) < n_min:
            continue
        for f, b in l:
            fmap[f].append(tid)
    cnt = collections.Counter()
    for f, tids in fmap.items():
        tids = sorted(set(tids))
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                cnt[(tids[i], tids[j])] += 1
    return [k for k, n in cnt.items() if n >= n_min]


def wall_metrics(tracks, W, H, a, b):
    da = {f: x for f, x in tracks[a]}
    db = {f: x for f, x in tracks[b]}
    common = sorted(set(da) & set(db))
    if len(common) < RULE["n_min"]:
        return None
    ua, ub, va, vb, ha, hb = [], [], [], [], [], []
    for f in common:
        x, y = da[f], db[f]
        ua.append((x[0] + x[2]) / 2 / W); ub.append((y[0] + y[2]) / 2 / W)
        va.append((x[1] + x[3]) / 2 / H); vb.append((y[1] + y[3]) / 2 / H)
        ha.append((x[3] - x[1]) / H); hb.append((y[3] - y[1]) / H)
    ua, ub, va, vb, ha, hb = map(np.asarray, (ua, ub, va, vb, ha, hb))
    if np.median(ua) > np.median(ub):          # 谁在右（镜中）
        R, S, ur, us, vr, vs, hr, hs = a, b, ua, ub, va, vb, ha, hb
    else:
        R, S, ur, us, vr, vs, hr, hs = b, a, ub, ua, vb, va, hb, ha
    dua, dus, dva, dvs = [], [], [], []
    for i in range(1, len(common)):
        if common[i] - common[i - 1] > 10:
            continue
        dua.append(ur[i] - ur[i - 1]); dus.append(us[i] - us[i - 1])
        dva.append(vr[i] - vr[i - 1]); dvs.append(vs[i] - vs[i - 1])
    ax = float(np.median((ur + us) / 2))
    return dict(R=R, S=S, n=len(common), u_axis=round(ax, 4),
                du_corr=pearson(dua, dus), dv_corr=pearson(dva, dvs),
                dv=round(float(np.median(np.abs(vr - vs)) / (hs.mean() + 1e-9)), 4),
                hr=round(float(np.median(hr / (hs + 1e-9))), 3),
                t0=round(common[0] / 25.0, 1), t1=round(common[-1] / 25.0, 1))


def rough_pass(m):
    return (m and m["du_corr"] is not None and m["dv_corr"] is not None
            and m["du_corr"] <= RULE["du_corr_max"] and m["dv_corr"] >= RULE["dv_corr_min"]
            and m["dv"] <= RULE["dv_max"]
            and RULE["hr_lo"] <= m["hr"] <= RULE["hr_hi"])


def assign(edges):
    """唯一性 + 镜像不作源（切断"镜像的镜像"）。"""
    is_r = {e["R"] for e in edges}
    role, paired = {}, {}
    for e in sorted(edges, key=lambda x: x["du_corr"]):
        r, s = e["R"], e["S"]
        if r in paired or role.get(r) == "person":
            continue
        if role.get(s) == "mirror":
            continue
        paired[r] = s
        role[r] = "mirror"
        role.setdefault(s, "person")
    return role, paired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="316,323")
    ap.add_argument("--raw-dir", default="output_crosscam")
    ap.add_argument("--status-json", default="webapp/backend/data/cams.json")
    ap.add_argument("--out", default="output_crosscam/analysis/gil/wallmirror_assign_v1.json")
    ap.add_argument("--store", default="output_crosscam/analysis/gil/store_v1")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    cams = [c.strip() for c in a.cams.split(",") if c.strip()]

    st_path = os.path.join(ROOT, a.status_json)
    status = json.load(open(st_path, encoding="utf-8")).get("track_status", {}) if os.path.exists(st_path) else {}

    out = dict(version="wallmirror_assign_v1", rule=RULE, cams=cams)
    for cam in cams:
        W, H = DIMS[cam]
        tr = load_tracks(os.path.join(ROOT, a.raw_dir, f"{cam}_detections.jsonl"))
        pairs = cooccur_pairs(tr, RULE["n_min"])
        rough = [m for m in (wall_metrics(tr, W, H, x, y) for x, y in pairs) if rough_pass(m)]
        # ★ 自监督镜面位置：u_axis 主聚类（中位数 ±tol）
        axes = np.array([m["u_axis"] for m in rough]) if rough else np.array([])
        axis_main = float(np.median(axes)) if len(axes) else None
        kept = [m for m in rough if axis_main is not None and abs(m["u_axis"] - axis_main) <= RULE["axis_tol"]]
        role, paired = assign(kept)
        mirror = sorted(paired)
        print(f"[{cam}] 轨 {len(tr)}  共现对 {len(pairs)}  粗筛(镜像几何) {len(rough)}  "
              f"→ 自监督镜面 u_axis={axis_main if axis_main is None else round(axis_main,3)}  "
              f"主簇 {len(kept)}  指派镜像 {len(mirror)}")

        # 与掩膜（track_status）对比
        st = status.get(cam, {})
        masked = {int(t) for t, v in st.items() if v == "masked"}
        keep = {int(t) for t, v in st.items() if v == "keep"}
        refl = {int(t) for t, v in st.items() if v == "reflection"}
        confirmed = sorted(set(mirror) & masked)          # 掩膜剔的、几何确认
        unexplained = sorted(masked - set(mirror))        # 掩膜剔的、几何无法解释
        geom_only = sorted(set(mirror) - masked)          # 几何判倒影、掩膜漏网
        print(f"     掩膜剔 {len(masked)}：几何确认 {len(confirmed)}  "
              f"★无法解释（待复核） {len(unexplained)}  几何独有（掩膜漏网） {len(geom_only)}")
        print(f"     掩膜剔的里面含 track_status=keep 的？{len(masked & keep)}（信息性）")
        if unexplained:
            print(f"     待复核 tid（前 20）: {unexplained[:20]}")
        out[cam] = dict(n_tracks=len(tr), n_pairs=len(pairs), n_rough=len(rough),
                        axis_main=None if axis_main is None else round(axis_main, 4),
                        n_axis_cluster=len(kept), n_mirror=len(mirror),
                        mirror_rows=[dict(tid=r, source=paired[r],
                                          **{k: m[k] for k in ("u_axis", "du_corr", "dv_corr",
                                                               "dv", "hr", "n", "t0", "t1")})
                                     for r in mirror
                                     for m in [next(x for x in kept if x["R"] == r and x["S"] == paired[r])]],
                        mask_confirm=confirmed, mask_unexplained=unexplained,
                        geom_only=geom_only)

    outp = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {a.out}")

    if a.apply:
        from gil import GILStore
        st = GILStore(os.path.join(ROOT, a.store), create=False)
        n_ann, n_flag, n_missing = 0, 0, 0
        for cam in cams:
            mir = {x["tid"]: x for x in out[cam]["mirror_rows"]}
            for tid, x in mir.items():
                st.con.execute(
                    "INSERT OR REPLACE INTO annotation(target_kind,target_id,value,confidence,author,ts) "
                    "VALUES('obs',?,'wall_mirror',?,'wallmirror_assign_v1',datetime('now'))",
                    (f"{cam}|{tid}", f"u_axis={x['u_axis']:.3f}_src={x['source']}_ducorr={x['du_corr']}"))
                n_ann += 1
                row = st.con.execute("SELECT id,flags FROM obs WHERE cam=? AND tid=? AND tombstone=0",
                                     (cam, tid)).fetchone()
                if row:
                    f = json.loads(row[1]) if row[1] else {}
                    f["wall_mirror"] = 1.0
                    f["wall_mirror_src"] = int(x["source"])
                    f["wall_mirror_u_axis"] = x["u_axis"]
                    st.con.execute("UPDATE obs SET flags=? WHERE id=?",
                                   (json.dumps(f, ensure_ascii=False), int(row[0])))
                    n_flag += 1
                else:
                    n_missing += 1
            for tid in out[cam]["mask_unexplained"]:
                st.con.execute(
                    "INSERT OR REPLACE INTO annotation(target_kind,target_id,value,confidence,author,ts) "
                    "VALUES('obs',?,'mask_unexplained','low','wallmirror_assign_v1',datetime('now'))",
                    (f"{cam}|{tid}",))
                n_ann += 1
        st.con.commit()
        print(f"[gil] annotation {n_ann} 条（wall_mirror + mask_unexplained）；"
              f"obs.flags 更新 {n_flag} 条（{n_missing} 条判定在 clean2 不存在=被掩膜剔）")
        print("[gil] stats:", json.dumps(st.stats(), ensure_ascii=False))
        st.close()


if __name__ == "__main__":
    main()
