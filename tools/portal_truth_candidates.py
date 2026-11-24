# -*- coding: utf-8 -*-
"""短窗"出入口事件"候选 + 跨机位配对候选 —— 供人工审定，形成**独立真值**。

背景（见 docs/external-plan-review-2026-10-14.md §1.4）：
  本地**没有**独立身份真值，"更少误接/漏接"因此无法评价。公开基准（Market/Duke/
  WILDTRACK 等）不含本场景的玻璃反光、鱼眼、不重叠接力条件，不能替代。故先做**小而硬**
  的真值闭环：在 343（电梯间，进出楼层必经，超近景 h≈548px）与 316（走廊脊柱，通行必经）
  上枚举短窗**通行事件**，渲染事件帧缩略图，并给出跨机位配对候选——人工只需判定
  "这两个事件是否同一人"，即可直接得到 误接/漏接 的分子分母。

事件定义（实体级，实体图 = clean2 规范口径）：
  entry  该实体在本机位的首帧远离窗口起点（真的是"新进画面/新入门户"）
  exit   末帧远离窗口终点
  at_start / at_end  = 首/末帧贴在窗口边界，无法判断是否真实进出（需人工看录像或弃用）

时间：统一时间轴 t = T0[cam] + frame / 25（秒，相对 08:55:00；见 SCENE.md §2）

用法：
  DET_VIEW=clean2 python tools/portal_truth_candidates.py --cams 343 316
输出（output_crosscam/analysis/truth/）：
  events_<cam>.csv    单机位事件表
  events_all.csv      合并（按统一时间排序）
  thumbs/<event>.jpg  事件帧缩略图（画框 + 脚点标记）
  pair_candidates.csv 跨机位配对候选（时间门 + relay v5 支持 + v2 状态）
  README.md           人工标注规则（填空列说明）

口径约定（2026-10-14 固化，改动前必须对账 241 对底稿）：
  - 事件 id = f"{cam}_E_{eid}"（"E" 固定字面量；真值链路 build_truth_data.py 按 "_E_" 解析）；
  - 默认 --min-obs 0 / --topk 0 → 与 343×316 已判底稿（241 对 / 107 事件）逐位一致；
    收紧采样（如 min-obs=8 / topk=3）会产出不同的事件/配对集，仅在做新版底稿时显式使用。
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from det_view import local_det, describe, net_tracks_path      # noqa: E402

SRC = os.path.join(ROOT, "output_crosscam")
OUT = os.path.join(SRC, "analysis", "truth")
VIDEO = {c: os.path.join(ROOT, "media-resource", "multi-cemra", "synced-10min",
                         f"{d}", f"{c}_085500-090500.mp4")
         for c, d in [("316", "01_行政走廊通道_316"), ("343", "06_电梯间_343"),
                      ("305", "02_办公区1_305"), ("318", "03_办公区2_318"),
                      ("320", "04_办公区球机_320"), ("323", "05_茶水间_323")]}
RES = {"316": (2560, 1440), "305": (2560, 1440), "318": (1920, 1080),
       "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}
T0 = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}
FPS = 25.0
RELAY_V5 = os.path.join(SRC, "analysis", "relay_clean2", "relay_chains_cradio_v5.json")
RELAY_V2 = os.path.join(SRC, "analysis", "relay_v2", "edges_v2.json")
BORDER_MARGIN = 150          # 距窗口边界 <150 帧视为贴边（无法判定真实进出）
# 事件 id 格式 = f"{cam}_E_{eid}"（"E"=Event，固定字面量）。真值链路
# build_truth_data.py 按 "_E_" 解析 event_id —— 勿改为 IN/OUT 等变体；
# kind（entry/exit）由 kind 列区分，不进 id。
MIN_OBS_DEFAULT = 0          # 默认不过滤（与 241 对真值底稿口径一致）；过滤碎片用 --min-obs 8


def load_obs(cam):
    """tid -> sorted list[(frame, box, score)]（DET_VIEW 决定口径）"""
    from collections import defaultdict
    acc = defaultdict(list)
    with open(local_det(cam), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids") or []
            boxes = r.get("boxes") or []
            scores = r.get("scores") or []
            for i, t in enumerate(tids):
                if t is None or int(t) < 0 or i >= len(boxes):
                    continue
                acc[int(t)].append((int(r["frame"]), [float(x) for x in boxes[i]],
                                    float(scores[i]) if i < len(scores) else None))
    for k in acc:
        acc[k].sort()
    return acc


def box_at(obs, frame, tol=5):
    """取该 tid 在最接近 frame 的检测框（stride=5，容差 tol 帧）"""
    if not obs:
        return None
    best, bd = None, 1 << 30
    for fr, box, sc in obs:
        d = abs(fr - frame)
        if d < bd:
            best, bd = (fr, box, sc), d
        if fr > frame + tol:
            break
    return best if bd <= tol else None


def entities(cam):
    d = json.load(open(net_tracks_path(cam), encoding="utf-8"))
    return d["entities"]


def render(cam, entries, out_dir):
    """渲染事件缩略图；entries = [(event_id, frame, box)]"""
    import cv2
    import numpy as np
    vp = VIDEO[cam]
    if not os.path.exists(vp):
        print(f"    [skip] 无视频 {vp}")
        return {}
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        print(f"    [skip] 打不开 {vp}")
        return {}
    W, H = RES[cam]
    paths = {}
    for eid, frame, box in entries:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, img = cap.read()
        if not ok:
            continue
        if img.shape[1] != W or img.shape[0] != H:      # 防御：分辨率不一致则按比例缩放坐标
            sx, sy = img.shape[1] / W, img.shape[0] / H
        else:
            sx = sy = 1.0
        x1, y1, x2, y2 = [int(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(box)]
        mx = int(0.45 * (x2 - x1))
        my = int(0.25 * (y2 - y1))
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
        cx2, cy2 = min(img.shape[1], x2 + mx), min(img.shape[0], y2 + my)
        crop = img[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 2)
        foot = ((x1 + x2) // 2 - cx1, y2 - cy1)
        cv2.drawMarker(crop, foot, (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(crop, f"{eid} f={frame} t={T0[cam]+frame/FPS:7.2f}s",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if crop.shape[0] > 520:                          # 限高，便于拼图浏览
            k = 520 / crop.shape[0]
            crop = cv2.resize(crop, None, fx=k, fy=k)
        p = os.path.join(out_dir, f"{eid}.jpg")
        cv2.imwrite(p, crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
        paths[eid] = os.path.relpath(p, ROOT).replace("\\", "/")
    cap.release()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", nargs="+", default=["343", "316"])
    ap.add_argument("--window", nargs=2, type=int, default=[0, 15000],
                    help="帧窗口（默认 0 15000 = 10min @25fps）")
    ap.add_argument("--dt-max", type=float, default=60.0, help="跨机位配对时间门（秒）")
    ap.add_argument("--min-obs", type=int, default=MIN_OBS_DEFAULT,
                    help="实体观测帧数下限（低于此的碎片实体不进事件表，单独计数）")
    ap.add_argument("--topk", type=int, default=0,
                    help="每个 A 机位出画事件最多保留 k 个最近时间的候选（0=不限；"
                         "底稿口径=0，与 241 对一致；收紧采样才传 3）")
    ap.add_argument("--no-thumb", action="store_true")
    args = ap.parse_args()
    if os.environ.get("DET_VIEW") != "clean2":
        raise SystemExit("[中止] 请显式 DET_VIEW=clean2（真值必须在 clean2 口径上建立）")
    os.makedirs(OUT, exist_ok=True)
    th_dir = os.path.join(OUT, "thumbs")
    os.makedirs(th_dir, exist_ok=True)
    f0w, f1w = args.window
    print(describe())
    print(f"[truth] 窗口 frame {f0w}..{f1w}（t {f0w/FPS:.1f}..{f1w/FPS:.1f}s）")

    all_events = []
    for cam in args.cams:
        obs = load_obs(cam)
        rows, thumb_req = [], []
        n_frag = 0
        for e in entities(cam):
            tids = [int(t) for t in e["tracks"]]
            frs = []
            for t in tids:
                frs += [(fr, t) for fr, _, _ in obs.get(t, [])]
            frs = [x for x in sorted(frs) if f0w <= x[0] <= f1w]
            if not frs:
                continue
            if len(frs) < args.min_obs:      # 碎片实体：不进事件表（避免用 0.x 秒的抖动当"通行"）
                n_frag += 1
                continue
            for kind, (fr, tid) in (("entry", frs[0]), ("exit", frs[-1])):
                got = box_at(obs.get(tid, []), fr)
                if got is None:
                    continue
                gfr, box, sc = got
                u = (box[0] + box[2]) / 2 / RES[cam][0]
                v = box[3] / RES[cam][1]
                edge = "at_start" if gfr - f0w < BORDER_MARGIN else (
                    "at_end" if f1w - gfr < BORDER_MARGIN else "mid")
                ev = f"{cam}_E_{e['eid']}"
                rows.append(dict(
                    event_id=ev, cam=cam, kind=kind, edge=edge, entity=e["eid"],
                    tid=tid, frame=gfr, t=round(T0[cam] + gfr / FPS, 2),
                    u=round(u, 4), v=round(v, 4), box=box, score=sc,
                    eid_n_frag=e.get("n_frag"), eid_tracks=",".join(str(x) for x in tids),
                    dur_s=round((frs[-1][0] - frs[0][0]) / FPS, 2),
                    n_obs=len(frs), thumb="", human_person_id="", human_note=""))
                thumb_req.append((ev, gfr, box))
        if not args.no_thumb:
            paths = render(cam, thumb_req, th_dir)
            for r in rows:
                r["thumb"] = paths.get(r["event_id"], "")
        with open(os.path.join(OUT, f"events_{cam}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                               ["event_id", "cam", "kind"])
            w.writeheader()
            w.writerows(rows)
        print(f"  {cam}: 事件 {len(rows)}（entry {sum(1 for r in rows if r['kind']=='entry')}"
              f" / exit {sum(1 for r in rows if r['kind']=='exit')}）"
              f" 贴边 {sum(1 for r in rows if r['edge']!='mid')}"
              f" ｜碎片实体剔除 {n_frag}（n_obs<{args.min_obs}）")
        all_events += rows

    all_events.sort(key=lambda r: (r["t"], r["cam"]))
    with open(os.path.join(OUT, "events_all.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_events[0].keys()))
        w.writeheader()
        w.writerows(all_events)
    print(f"[out] events_all.csv（{len(all_events)} 条）")

    # ---- 跨机位配对候选（343 出 → 另一机位 入，时间门内）----
    try:
        rel = json.load(open(RELAY_V5, encoding="utf-8"))
        rel_set = {(e["a"], e["b"]) for e in rel["edges"]}
    except Exception:  # noqa: BLE001
        rel_set = set()
        print("[warn] relay v5 缺失，配对候选不含接力证据列")
    # Relay v2 状态（active=生产/晋升；audit=审计池；blocked=冲突显式化）
    try:
        v2d = json.load(open(RELAY_V2, encoding="utf-8"))
        v2_state = {(e["a"], e["b"]): ("active" if e.get("status") == "active" else "audit")
                    for e in v2d["edges"]}
        for g in v2d.get("conflict_groups", []):
            for x in g["blocked"]:
                v2_state[(g["a"], x["b"])] = "blocked"
    except Exception:  # noqa: BLE001
        v2_state = {}
        print("[warn] relay v2 缺失，配对候选不含 v2 状态列")
    a_cam, b_cam = args.cams[0], args.cams[-1]
    A = [r for r in all_events if r["cam"] == a_cam and r["kind"] == "exit"]
    B = [r for r in all_events if r["cam"] == b_cam and r["kind"] == "entry"]
    pairs = []
    for a in A:
        cand = []
        for b in B:
            dt = b["t"] - a["t"]
            if 0 <= dt <= args.dt_max:
                ea, eb = f"{a['cam']}:{a['entity']}", f"{b['cam']}:{b['entity']}"
                cand.append((dt, (ea, eb), b))
        cand.sort(key=lambda x: x[0])
        if args.topk:
            cand = cand[:args.topk]
        for dt, (ea, eb), b in cand:
            pairs.append(dict(pair_id=f"{a['event_id']}__{b['event_id']}",
                              cam_a=a["cam"], event_a=a["event_id"], t_a=a["t"],
                              cam_b=b["cam"], event_b=b["event_id"], t_b=b["t"],
                              dt=round(dt, 2),
                              in_relay_v5=int((ea, eb) in rel_set),
                              in_relay_v2=v2_state.get((ea, eb), ""),
                              thumb_a=a["thumb"], thumb_b=b["thumb"],
                              human_same_person="", human_confidence="", human_note=""))
    pairs.sort(key=lambda r: (r["cam_a"], r["t_a"], r["dt"]))
    with open(os.path.join(OUT, "pair_candidates.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()) if pairs else ["pair_id"])
        w.writeheader()
        w.writerows(pairs)
    n_rel = sum(p["in_relay_v5"] for p in pairs)
    n_v2 = sum(1 for p in pairs if p["in_relay_v2"])
    print(f"[out] pair_candidates.csv（{len(pairs)} 条候选，relay v5 支持 {n_rel} 条 / "
          f"v2 状态非空 {n_v2} 条）")

    # ---- 标注说明 ----
    readme = f"""# 短窗出入口事件真值（{a_cam} × {b_cam}，synced-10min 08:55:00–09:05:00）

生成时间：2026-10-14 ｜ 口径：DET_VIEW=clean2，实体图 `stitch_clean2`
生成脚本：`tools/portal_truth_candidates.py`（可复现）

## 1. 为什么需要这份表

本地没有独立身份真值 → "更少误接 / 更少漏接"不可评价。公开 ReID 基准不含本场景的
玻璃反光、鱼眼、不重叠接力条件，不能替代。本表把判定缩到**唯一一个人工动作**：
"这两个通行事件是不是同一个人"。

## 2. 文件

| 文件 | 内容 |
|---|---|
| `events_{a_cam}.csv` / `events_{b_cam}.csv` | 单机位通行事件（entry / exit） |
| `events_all.csv` | 合并、按统一时间排序 |
| `thumbs/<event_id>.jpg` | 事件帧缩略图（红框=检测框，绿十字=脚点，左上=事件id/帧/统一时刻） |
| `pair_candidates.csv` | 跨机位配对候选（时间门 0–{args.dt_max:g}s；含 relay v5 支持 + v2 状态） |

## 3. 列说明与要填的列

事件表关键列：`event_id` `cam` `kind`(entry/exit) `edge`(mid=真实进出；at_start/at_end=贴窗
口边界的截断，不能判为进出) `frame` `t`(统一时刻,秒,相对 08:55:00) `u,v`(脚点,归一化)
`entity` `dur_s` `n_obs` `thumb`。

**要填（人工）**：
- `events_*.csv` → `human_person_id`：同一人的事件给同一个编号（可跨机位、跨时间）；
  若压缩包/画面判断为反光伪影或误检 → 填 `FP`。
- `pair_candidates.csv` → `human_same_person`：`1`=同一人 / `0`=不是 / `?`=看不清；
  `human_confidence`：`high/mid/low`；`human_note` 自由备注。

## 4. 判定要点（本场景特有）

1. **脚点**用框底边中点（≈脚下），不要用框中心；316 近景 h≈211px、343 超近景 h≈548px。
2. **反光**：343/316 都有玻璃面，反光伪影与真人**垂直对齐**；若两个框垂直对齐且形状成镜像，
   优先怀疑反光（本项目已清洗，但边界样本仍需人工确认）。
3. `edge=at_start/at_end` 的事件**不要**当作真实进出（窗口截断）。
4. `dt` 是统一时间轴差值；316 的 T0=+4.0s、343 的 T0=+19.0s，读数本身有 ±0.1s 量级不稳。
5. `in_relay_v5=1` 只说明算法给出了这条边，**不是**真值——本表的目的正是检验它。
   `in_relay_v2`（active/audit/blocked）：Relay v2 改造后的边状态（active=生产集或已晋升 /
   audit=审计池待复核 / blocked=冲突被挤出）；空 = 不在 v2 任何池。同样**不是**真值。

## 5. 产出怎么用

- 由 `human_same_person` 得到跨机位配对的 **TP / FP / FN** → 直接算 误接率、漏接率；
- 配合 `in_relay_v5` 即可给出"固定误接下 relay v5 的召回"→ H1/H2 的第一版验收口径；
- 事件级 `human_person_id` 还允许在**无跨机位配对**时统计"该时段实际通行人数"（人数账本真值）。
"""
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(readme)
    print(f"[out] {os.path.join(OUT, 'README.md')}")


if __name__ == "__main__":
    main()
