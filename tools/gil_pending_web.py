# -*- coding: utf-8 -*-
"""S6-b 待裁决数据包：把 A/B 分析出的「需要人判定」的项做成前端可直接判的载荷 + 缩略图。

三类：
  ① relay 冲突边（16）：倒影端重定向到源后，与身份层（gperson）冲突 → 判「是否同一人」
  ② 倒影搭桥合并（3）：倒影把互不相关的真人连成一簇 → 判「是否同一人（该拆/该留）」
  ③ 倒影实体（11）：端点实体的成员全是/含倒影 → 判「是倒影 / 是真人」

产物：
  webapp/frontend/public/s6b_pending.json    载荷（静态托管，免后端改动）
  webapp/frontend/public/s6b_thumbs/*.jpg    实体裁剪图（视频抽帧）

用法：python tools/gil_pending_web.py
"""
import collections
import json
import os
import sqlite3
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mirror_kinematics import load_tracks  # noqa: E402

GIL = os.path.join(ROOT, "output_crosscam/analysis/gil/store_v1/gil.sqlite")
AB = os.path.join(ROOT, "output_crosscam/analysis/gil/reflection_ab_v1.json")
MEDIA = os.path.join(ROOT, "webapp/backend/data/media")
# ★ 全分辨率源（原生 2560×1440 / 1920×1080）；webapp/backend/data/media/*.mp4 是 960×540 预览，
#   坐标空间不匹配 → 裁剪会越界。优先用 synced-10min 的全分辨率切片。
SYNCED = os.path.join(ROOT, "media-resource/multi-cemra/synced-10min")
NATIVE = {"305": (2560, 1440), "316": (2560, 1440), "318": (1920, 1080),
          "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}
PUB = os.path.join(ROOT, "webapp/frontend/public")
TH = os.path.join(PUB, "s6b_thumbs")
FPS = 25.0
PAD = 0.45          # 裁剪外扩比例
H_OUT = 200         # 输出高


def main():
    os.makedirs(TH, exist_ok=True)
    con = sqlite3.connect(GIL)
    con.row_factory = sqlite3.Row
    obs = {}
    for r in con.execute("SELECT id,cam,tid,t_first,t_last,entity_id,flags FROM obs WHERE tombstone=0"):
        obs[(r["cam"], r["tid"])] = dict(r)
    oid2key = {r["id"]: f"{r['cam']}|{r['tid']}" for r in
               con.execute("SELECT id,cam,tid FROM obs WHERE tombstone=0")}
    ent = {r["id"]: dict(r) for r in con.execute("SELECT id,cam,eid,t0,t1,n_obs FROM entity")}
    eid2id = {r["eid"]: r["id"] for r in con.execute("SELECT id,cam,eid FROM entity")}
    ent_mem = collections.defaultdict(list)
    for r in con.execute("SELECT cam,tid,entity_id FROM obs WHERE tombstone=0 AND entity_id IS NOT NULL"):
        ent_mem[r["entity_id"]].append(f"{r['cam']}|{r['tid']}")

    ab = json.load(open(AB, encoding="utf-8"))

    # ---------------- 缩略图 ----------------
    import subprocess
    import tempfile

    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
    TMP = tempfile.mkdtemp(prefix="s6b_frames_")
    tracks, vpath, frame_cache = {}, {}, {}

    def traj(cam):
        if cam not in tracks:
            tracks[cam] = load_tracks(os.path.join(ROOT, "output_crosscam", f"{cam}_detections.jsonl"))
        return tracks[cam]

    n_thumb, miss = 0, []

    def video_of(cam):
        """优先全分辨率 synced-10min 切片；找不到再退回 webapp 预览。"""
        if cam not in vpath:
            import glob
            g = sorted(glob.glob(os.path.join(SYNCED, f"*_{cam}", f"{cam}_*.mp4")))
            vpath[cam] = g[0] if g else os.path.join(MEDIA, f"{cam}.mp4")
        return vpath[cam]

    def frame_img(cam, frame_idx):
        """按帧号取全分辨率帧（ffmpeg 快速定位；同帧只解一次）。"""
        k = (cam, int(frame_idx))
        if k not in frame_cache:
            out = os.path.join(TMP, f"{cam}_{int(frame_idx)}.jpg")
            subprocess.run([FF, "-hide_banner", "-loglevel", "error",
                            "-ss", f"{int(frame_idx) / FPS:.3f}", "-i", video_of(cam),
                            "-frames:v", "1", "-q:v", "2", "-y", out],
                           capture_output=True)
            frame_cache[k] = out if os.path.exists(out) else None
        return frame_cache[k]

    def crop(cam, tid):
        """抽该 obs 中位帧的框，外扩裁剪 → 文件名 cam-tid.jpg；返回文件名或 None。"""
        nonlocal n_thumb
        fn = f"{cam}-{tid}.jpg"
        p = os.path.join(TH, fn)
        if os.path.exists(p):
            return fn
        tr = traj(cam).get(tid)
        if not tr:
            miss.append((cam, tid, "no traj"))
            return None
        fr, box = tr[len(tr) // 2]                  # 中位样本 = (frame, box)
        fp = frame_img(cam, fr)
        if not fp:
            miss.append((cam, tid, "frame fail"))
            return None
        im = cv2.imread(fp)
        if im is None:
            miss.append((cam, tid, "decode fail"))
            return None
        H, W = im.shape[:2]
        nw, nh = NATIVE[cam]
        kx, ky = W / nw, H / nh                     # 检测坐标 → 视频坐标
        x1, y1, x2, y2 = box[0] * kx, box[1] * ky, box[2] * kx, box[3] * ky
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # 保底：框太小（远/遮挡）时给一个最小窗
        w = max(bw * (1 + 2 * PAD), 96)
        h = max(bh * (1 + 2 * PAD), 128)
        X1 = int(max(0, cx - w / 2)); X2 = int(min(W, cx + w / 2))
        Y1 = int(max(0, cy - h / 2)); Y2 = int(min(H, cy + h / 2))
        sub = im[Y1:Y2, X1:X2]
        if sub.size == 0:
            miss.append((cam, tid, "empty crop"))
            return None
        sub = cv2.resize(sub, (max(1, int(sub.shape[1] * H_OUT / sub.shape[0])), H_OUT))
        cv2.imwrite(p, sub, [cv2.IMWRITE_JPEG_QUALITY, 82])
        n_thumb += 1
        return fn

    def oinfo(k):
        c, t = k.split("|")
        return obs.get((c, int(t)))

    def ent_thumb(cam, eid, limit=2):
        """实体代表图：优先取时长最长的成员 obs。"""
        i = eid2id.get(eid)
        ks = sorted(ent_mem.get(i, []),
                    key=lambda k: -(oinfo(k)["t_last"] - oinfo(k)["t_first"]) if oinfo(k) else 0)
        out = []
        for k in ks[:limit]:
            c, t = k.split("|")
            o = oinfo(k)
            f = crop(c, int(t))
            if f and o:
                out.append(dict(f=f, obs=k, t0=o["t_first"], t1=o["t_last"], gid=gm.get(k)))
        return out

    # obs -> gperson label（现状口径 A）
    gm = {}
    gl = {r["id"]: r["label"] for r in con.execute("SELECT id,label FROM gperson")}
    for r in con.execute("SELECT gperson_id,node_id FROM membership WHERE node_kind='obs'"):
        gm[oid2key.get(r["node_id"], "")] = gl.get(r["gperson_id"], str(r["gperson_id"]))

    groups = []

    # ---------------- ① relay 冲突边 ----------------
    items = []
    for row in ab["relay_contamination"]["rows"]:
        if row["verdict"] != "contradicted":
            continue
        ea, eb = row["edge"].split("->")
        ca, eida = ea.split(":"); cb, eidb = eb.split(":")
        ia, ib = ent.get(eid2id.get(eida, -1), {}), ent.get(eid2id.get(eidb, -1), {})
        pid = f"S6B_P_{ca}_E_{eida}__{cb}_E_{eidb}"
        mir = row["mirror_sides"]
        mside = list(mir.keys())[0] if mir else None
        items.append(dict(
            id=pid, kind="pair",
            a=f"{ca}_E_{eida}", b=f"{cb}_E_{eidb}", sa=eida, sb=eidb,
            dt=round((ib.get("t0", 0) or 0) - (ia.get("t1", 0) or 0), 1),
            cos=row["score"], relay=1, relay_v2=row["status"],
            machine="contradicted",
            side_a=ent_thumb(ca, eida), side_b=ent_thumb(cb, eidb),
            note=(f"倒影实体 {mir[mside]['endpoint']}（成员 {mir[mside]['hit']}）"
                  f"重定向到源后：邻居={row['g_neighbor']} vs 源={row['g_source']} → 身份层判不同人"
                  if mside else ""),
            mirror_side=(mside and mir[mside]["endpoint"]), mirror_hit=(mir[mside]["hit"] if mside else None),
        ))
    groups.append(dict(key="relay_conflict", title="① relay 冲突边（倒影端重定向后与身份层冲突）",
                       hint="判：这两段观测是同一人吗？（若同人 → relay 边对、身份层漏接；若不同人 → relay 边是假链）",
                       count=len(items), items=items))

    # ---------------- ② 桥接合并 ----------------
    items = []
    for i, s in enumerate(ab["spurious_merges"]):
        bc = s["b_clusters"]
        labs = list(bc.keys())
        if len(labs) < 2:
            continue
        big = max(labs, key=lambda k: len(bc[k]))
        other = [k for k in labs if k != big]
        pid = f"S6B_B_{s['a_cluster']}"
        def side(members):
            out = []
            for k in members[:3]:
                c, t = k.split("|")
                o = oinfo(k)
                f = crop(c, int(t))
                if f and o:
                    out.append(dict(f=f, obs=k, t0=o["t_first"], t1=o["t_last"], gid=gm.get(k)))
            return out
        items.append(dict(
            id=pid, kind="pair",
            a=f"来自 {s['a_cluster']} / {other[0]}", b=f"来自 {s['a_cluster']} / {big}",
            sa=other[0], sb=big, dt=None, cos=None, relay=0, relay_v2="",
            machine="bridge",
            side_a=side(bc[other[0]]), side_b=side(bc[big]),
            note=(f"现状（A 臂）把这两组并成一个人（{s['a_cluster']}，含倒影 {s['n_reflection']} 条）；"
                  f"倒影降权后拆开。判：该拆还是该留？"),
        ))
    groups.append(dict(key="bridge_merge", title="② 倒影「搭桥」造成的合并（3 处）",
                       hint="判：左右两组是同一人吗？（同人 → 该留；不同人 → 该拆）",
                       count=len(items), items=items))

    # ---------------- ③ 倒影实体 ----------------
    items = []
    for v in ab["reflection_entities"]:
        eid = v["entity"]
        cam, key = eid.split("-")[0], eid
        members = v["members"]
        # 源：从 flags 拿
        srcs = []
        for k in members:
            c, t = k.split("|")
            o = obs.get((c, int(t)))
            if not o:
                continue
            fl = json.loads(o["flags"]) if o["flags"] else {}
            st = fl.get("wall_mirror_src") if fl.get("wall_mirror") else fl.get("explained_by")
            if st is not None:
                f = crop(c, int(st))
                if f:
                    srcs.append(dict(f=f, obs=f"{c}|{st}", t0=obs.get((c, int(st)), {}).get("t_first"),
                                     t1=obs.get((c, int(st)), {}).get("t_last")))
        pid = f"S6B_R_{eid}"
        side_a = []
        for k in members:
            c, t = k.split("|")
            o = oinfo(k)
            f = crop(c, int(t))
            if f and o:
                side_a.append(dict(f=f, obs=k, t0=o["t_first"], t1=o["t_last"], gid=gm.get(k)))
        items.append(dict(
            id=pid, kind="mirror",
            entity=eid, members=members, n_reflection=v["n_reflection"],
            pure=len(members) == v["n_reflection"],
            side_a=side_a, side_b=srcs,
            note=("该实体成员全是倒影 → 纯倒影实体" if len(members) == v["n_reflection"]
                  else f"该实体 {len(members)} 个成员中 {v['n_reflection']} 个是倒影（混有真人）"),
        ))
    groups.append(dict(key="reflection_entity", title="③ 倒影实体（身份层的幽灵）",
                       hint="判：左图这条轨迹是玻璃镜像倒影，还是真人？（右图为其「源」参考，可能为空）",
                       count=len(items), items=items))

    # ---------------- ⑤ 人脸 must-link 与时间互斥冲突 ----------------
    # 来源：face_constraint_absorption_v1.json（人脸候选合并判「同人」的对，与 GIL「时间互斥」硬约束冲突）
    #   同机位时间重叠的两条轨迹物理上不可能是同一人 → 只能人看视频定夺「同人(人脸对/GIL分割了这人) vs 不同人(人脸误判)」。
    fca_p = os.path.join(ROOT, "output_crosscam/analysis/gil/face_constraint_absorption_v1.json")
    fml_p = os.path.join(ROOT, "output_crosscam/analysis/face/face_mustlink_v1.json")
    items = []
    if os.path.exists(fca_p):
        fca = json.load(open(fca_p, encoding="utf-8"))
        cos_of = {}
        if os.path.exists(fml_p):
            for c in json.load(open(fml_p, encoding="utf-8")).get("must_link", []):
                for e in c.get("conf_edges", []):
                    cos_of[tuple(sorted([e["a"], e["b"]]))] = e.get("cos")
        seen = set()
        for cl in fca.get("clusters", []):
            for x in cl.get("conflicts", []):
                a, b = x["a"], x["b"]
                kk = tuple(sorted([a, b]))
                if kk in seen:
                    continue
                seen.add(kk)
                ca, ta = a.split("|")
                cb, tb = b.split("|")
                pid = f"S6B_M_{ca}_{ta}__{cb}_{tb}"
                def one(k):
                    c, t = k.split("|")
                    o = oinfo(k)
                    f = crop(c, int(t))
                    return (dict(f=f, obs=k, t0=o["t_first"], t1=o["t_last"], gid=gm.get(k))
                            if f and o else None)
                sa, sb = one(a), one(b)
                items.append(dict(
                    id=pid, kind="pair",
                    a=f"{a}（{gm.get(a) or '?'}）", b=f"{b}（{gm.get(b) or '?'}）",
                    sa=a, sb=b, dt=None, cos=cos_of.get(kk), relay=0, relay_v2="",
                    machine="face_conflict",
                    side_a=[sa] if sa else [], side_b=[sb] if sb else [],
                    note=(f"人脸候选合并判这两条同机位轨迹为同一人（簇 {cl['cid']}，cos={cos_of.get(kk)}），"
                          f"但与 GIL 时间互斥冲突：{a} {x['a_span']} × {b} {x['b_span']} 时间重叠"
                          f" → 物理上不可能是同一人。判：同人（人脸对 → GIL 把这人拆成了两条轨迹）"
                          f"还是不同人（人脸误判）？"),
                ))
    groups.append(dict(key="face_conflict",
                       title="⑤ 人脸 must-link 与时间互斥冲突（同一人被拆 / 人脸误判）",
                       hint="判：左右两条同机位轨迹是同一个人吗？（同人 → 人脸对、GIL 把这人拆成了两条轨迹；不同人 → 人脸误判）",
                       count=len(items), items=items))

    out = dict(version="s6b_pending_v1", thumbs_base="/s6b_thumbs/",
               title="S6-b 待裁决：倒影降权 A/B 的分歧清单",
               note="本批不参与既有 241 对指标；判定结果与其一并存入 labels.json（pair 命名空间，"
                    "id 前缀 S6B_P_ / S6B_B_ / S6B_R_ / S6B_M_），对既有统计无影响。",
               rules="pair 类：1=同人 0=不是 2=看不清；mirror 类：1=是倒影 0=是真人 2=看不清。",
               groups=groups)
    p = os.path.join(PUB, "s6b_pending.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {p}  缩略图 {n_thumb} 张")
    for g in groups:
        print(f"  {g['key']:18s} {g['count']} 项")
    if miss:
        print(f"  ⚠ 缺图 {len(miss)}：{miss[:8]}")


if __name__ == "__main__":
    main()
