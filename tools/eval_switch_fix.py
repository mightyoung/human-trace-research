#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用已标注的 gallery 串人点，自动比较新跟踪配置是否把"交界转移"断开了。

原理（不需要新的人工标注）：
  旧轨迹 X 在 id_switch_frame=F 处从 A 换成了 B（标注员给出）。
  取 X 在 F 之前 [F-W, F) 和之后 [F, F+W] 的旧框，逐帧到新结果里找对应新框：
     IoU >= 0.5 → 取该新框的 id；否则旧框被新框包含 >= 0.8（旧头框 → 新身体框）→ 取最小的那个宿主框 id。
  前段多数 id = pre，后段多数 id = post：
     pre != post → FIXED（新配置在串人点断开或换了人，正确）
     pre == post → PERSIST（串人仍在）
     任一侧找不到 → UNMATCHED（多为旧头框被去重且无宿主，或新配置漏检）
  对照组：标注为"无串人"的 gallery 成员（id_switch 为空），在轨迹 25/50/75% 处做同样检查，
     pre != post 记为 FALSE_SPLIT（把干净轨迹切断的代价；宁短勿脏下可接受一部分，但要量化）。

用法：
  python tools/eval_switch_fix.py \
     --gt <pkg>/gt/gt_gallery_members_FILLED.csv \
     --old "output_crosscam_local/1012_yolo8_dd/{seg}/{seg}_detections.jsonl" \
     --new v1_s10=output_crosscam_local/1012_yolo_v2/v1_s10/{seg}/{seg}_detections.jsonl \
     --new v2_s10_veto=output_crosscam_local/1012_yolo_v2/v2_s10_veto/{seg}/{seg}_detections.jsonl \
     --segs 323_train_510 --out-dir output_crosscam_local/switch_eval
  --old 必须是生成 gallery 台账（tracks_all.csv 里的 tid）的那份原始 jsonl；脚本会用台账帧数自检。
输出：events.csv（逐事件）、controls.csv、summary.md（每个配置一行）。
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(path):
    """-> frames: {frame: (ids array, boxes (N,4))}"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ids = r.get("track_ids") or []
            bx = r.get("boxes") or []
            keep = [i for i, t in enumerate(ids) if t is not None and int(t) >= 0 and i < len(bx)]
            out[int(r["frame"])] = (np.array([int(ids[i]) for i in keep], dtype=np.int64),
                                    np.array([bx[i] for i in keep], dtype=np.float64).reshape(-1, 4))
    return out


def track_frames(det, tid):
    res = []
    for fr, (ids, bx) in det.items():
        k = np.flatnonzero(ids == tid)
        if len(k):
            res.append((fr, bx[k[0]]))
    res.sort(key=lambda x: x[0])
    return res


def map_box(newdet, fr, box, iou_thr, ioa_thr):
    """返回 (new_tid, how) 或 (None, why)"""
    if fr not in newdet:
        return None, "no_frame"
    ids, bx = newdet[fr]
    if len(ids) == 0:
        return None, "empty"
    x1 = np.maximum(box[0], bx[:, 0]); y1 = np.maximum(box[1], bx[:, 1])
    x2 = np.minimum(box[2], bx[:, 2]); y2 = np.minimum(box[3], bx[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a0 = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    an = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
    iou = inter / np.maximum(a0 + an - inter, 1e-9)
    j = int(iou.argmax())
    if iou[j] >= iou_thr:
        return int(ids[j]), "iou"
    ioa = inter / a0
    cand = np.flatnonzero(ioa >= ioa_thr)
    if len(cand):
        j = cand[np.argmin(an[cand])]
        return int(ids[j]), "contain"
    return None, "miss"


def side_id(newdet, obs, iou_thr, ioa_thr):
    got = [map_box(newdet, fr, b, iou_thr, ioa_thr)[0] for fr, b in obs]
    got = [g for g in got if g is not None]
    if not got:
        return None, 0, 0
    c = Counter(got).most_common(1)[0]
    return c[0], c[1], len(got)


def parse_events(row, t0, t1):
    """id_switch_frame 标注有多种写法（单值 / 逗号多值 / 'f21770' / 填错列），统一抽取 [t0,t1] 内的整数帧号。"""
    txt = " ".join(row.get(k, "") or "" for k in ("id_switch_frame", "unk_reason", "note"))
    txt = re.sub(r"(?i)\b(fe)?male\d+|unk\w*|\[review\]", " ", txt)   # 去掉人名里的数字
    fr = sorted({int(x) for x in re.findall(r"\d+", txt) if t0 <= int(x) <= t1})
    return fr


def judge(newdet, obs, F, prev_F, next_F, W, guard, iou_thr, ioa_thr):
    pre = [(f, b) for f, b in obs if max(prev_F, F - W) <= f < F - guard]
    post = [(f, b) for f, b in obs if F + guard <= f <= min(next_F, F + W)]
    pid, pn, pt = side_id(newdet, pre, iou_thr, ioa_thr)
    qid, qn, qt = side_id(newdet, post, iou_thr, ioa_thr)
    if not pre or not post:
        out = "NO_WINDOW"
    elif pid is None or qid is None:
        out = "UNMATCHED"
    elif pid != qid:
        out = "SPLIT"
    else:
        out = "SAME"
    return out, pid, qid, len(pre), len(post), pt, qt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="gt_gallery_members_FILLED.csv")
    ap.add_argument("--ledger", default="", help="tracks_all.csv（默认与 --gt 同包 ledger/）")
    ap.add_argument("--old", required=True, help="旧 jsonl 路径模板，含 {seg}")
    ap.add_argument("--new", action="append", required=True, help="name=路径模板（可多次）")
    ap.add_argument("--segs", default="", help="逗号分隔；默认 GT 中出现的全部段")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--guard-sec", type=float, default=0.0, help="F 前后各留空的秒数（标注帧有误差时用 0.4）")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--ioa", type=float, default=0.8)
    ap.add_argument("--out-dir", default="output_crosscam_local/switch_eval")
    ap.add_argument("--allow-weak-ledger", action="store_true",
                    help="台账帧数一致率 <90% 时仍然继续（默认直接失败，防静默跑错文件）")
    ap.add_argument("--windows", default="",
                    help="switch_windows.json（M1-A 切点唯一来源）；给定时按 old_usable 分列统计")
    a = ap.parse_args()

    gt_path = Path(a.gt)
    ledger = Path(a.ledger) if a.ledger else gt_path.parent.parent / "ledger" / "tracks_all.csv"
    L = {r["track_id"]: r for r in csv.DictReader(open(ledger, encoding="utf-8-sig"))}
    G = list(csv.DictReader(open(gt_path, encoding="utf-8-sig")))
    segs = [s for s in a.segs.split(",") if s] or sorted({r["seg"] for r in G})
    W, guard = int(a.window_sec * a.fps), int(a.guard_sec * a.fps)
    news = [n.split("=", 1) for n in a.new]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # M1-A 切点表（可选）：只用来给事件打 old_usable 标记，不改判据
    winp = Path(a.windows) if a.windows else gt_path.parent / "switch_windows.json"
    USABLE = {}
    if winp.exists():
        USABLE = {x["track_id"]: bool(x.get("old_usable")) for x in json.load(open(winp, encoding="utf-8"))}
        print(f"[windows] {winp} 载入 {len(USABLE)} 条（old_usable=true {sum(USABLE.values())}）")
    else:
        print(f"[windows] 未找到 {winp}，跳过 old_usable 分列")

    ev_rows, ct_rows = [], []
    for seg in segs:
        op = Path(a.old.format(seg=seg))
        if not op.exists():
            print(f"[miss old] {op}")
            continue
        old = load_jsonl(op)
        newdets = {}
        for name, tpl in news:
            p = Path(tpl.format(seg=seg))
            if p.exists():
                newdets[name] = load_jsonl(p)
            else:
                print(f"[miss new] {name}: {p}")
        rows = [r for r in G if r["seg"] == seg and r["track_id"] in L]
        # 自检：旧 jsonl 是否就是台账那份
        chk = []
        for r in rows[:30]:
            tid = int(r["track_id"].split("#")[1])
            chk.append(len(track_frames(old, tid)) == int(L[r["track_id"]]["n_frames"]))
        rate = sum(chk) / max(1, len(chk))
        msg = f"[{seg}] 旧 jsonl 与台账帧数一致率 {rate:.0%}（<90% 说明 --old 指错了文件）"
        if rate < 0.9 and not a.allow_weak_ledger:
            raise SystemExit(
                msg + "\n台账由 output_crosscam_local/1012_yolo8_dd/{seg}/{seg}_detections.jsonl 生成；"
                "output_crosscam/1012_final 是另一套跑法（stride5 / 轨迹编号不同），不能用作 --old。\n"
                "确认要在这种情况下继续，加 --allow-weak-ledger。")
        print(msg)

        for r in rows:
            tid = int(r["track_id"].split("#")[1])
            obs = track_frames(old, tid)
            if not obs:
                continue
            t0, t1 = obs[0][0], obs[-1][0]
            sw = (r.get("id_switch") or "").strip()
            if sw == "1":
                evs = parse_events(r, t0, t1)
                if not evs:
                    ev_rows.append(dict(seg=seg, track_id=r["track_id"], box_type=r["box_type"], F="",
                                        k=0, config="*", outcome="UNPARSED"))
                    continue
                for k, F in enumerate(evs):
                    prev_F = evs[k - 1] if k else -10**9
                    next_F = evs[k + 1] - 1 if k + 1 < len(evs) else 10**9
                    for name, nd in newdets.items():
                        out, pid, qid, npre, npost, pt, qt = judge(nd, obs, F, prev_F, next_F, W, guard,
                                                                   a.iou, a.ioa)
                        ev_rows.append(dict(seg=seg, track_id=r["track_id"], box_type=r["box_type"], F=F, k=k,
                                            config=name, outcome=out, pre_id=pid, post_id=qid,
                                            n_pre=npre, n_post=npost, pre_mapped=pt, post_mapped=qt,
                                            at_track_end=int(F == t1),
                                            old_usable=int(USABLE.get(r["track_id"], -1))))
            elif sw == "" and (r.get("person_id") or "").strip() not in ("", "unk", "unknown"):
                if len(obs) < 8:
                    continue
                for q in (0.25, 0.5, 0.75):
                    F = obs[int(len(obs) * q)][0]
                    for name, nd in newdets.items():
                        out, pid, qid, npre, npost, pt, qt = judge(nd, obs, F, -10**9, 10**9, W, guard,
                                                                   a.iou, a.ioa)
                        ct_rows.append(dict(seg=seg, track_id=r["track_id"], box_type=r["box_type"], F=F, q=q,
                                            config=name, outcome=out, pre_id=pid, post_id=qid))
        # 每个配置的整体轨迹统计
        for name, nd in newdets.items():
            span = defaultdict(lambda: [10**12, -1])
            for fr, (ids, _) in nd.items():
                for t in ids:
                    s = span[int(t)]
                    s[0] = min(s[0], fr); s[1] = max(s[1], fr)
            d = sorted((e - s) / a.fps for s, e in span.values())
            print(f"[{seg}] {name}: tracks={len(span)} median_len={d[len(d)//2] if d else 0:.1f}s")

    def dump(rows, path):
        if not rows:
            return
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    dump(ev_rows, out_dir / "events.csv")
    dump(ct_rows, out_dir / "controls.csv")

    lines = ["| 配置 | 串人事件 | 断开 SPLIT | 仍串 SAME | 对不上 UNMATCHED | 断开率(可判) | 头框事件断开 | 身体框事件断开 "
             "| 对照点 | 误切 FALSE_SPLIT | 误切率(可判) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, _ in news:
        E = [r for r in ev_rows if r["config"] == name]
        C = [r for r in ct_rows if r["config"] == name]
        c = Counter(r["outcome"] for r in E)
        cc = Counter(r["outcome"] for r in C)
        judg = c["SPLIT"] + c["SAME"]
        cj = cc["SPLIT"] + cc["SAME"]

        def bt(t):
            x = [r for r in E if r["box_type"] == t and r["outcome"] in ("SPLIT", "SAME")]
            return f"{sum(r['outcome'] == 'SPLIT' for r in x)}/{len(x)}"
        lines.append(f"| {name} | {len(E)} | {c['SPLIT']} | {c['SAME']} | {c['UNMATCHED']} | "
                     f"{c['SPLIT'] / max(1, judg):.0%} | {bt('head')} | {bt('body')} | {len(C)} | "
                     f"{cc['SPLIT']} | {cc['SPLIT'] / max(1, cj):.0%} |")
    unp = sum(1 for r in ev_rows if r["outcome"] == "UNPARSED")
    if USABLE:
        lines += ["", "**old_usable=true 子集（M1-A 判为切点可定位的轨迹）**", "",
                  "| 配置 | 可判事件 | 断开 SPLIT | 仍串 SAME | 断开率 |", "|---|---|---|---|---|"]
        for name, _ in news:
            E = [r for r in ev_rows if r["config"] == name and r.get("old_usable") == 1
                 and r["outcome"] in ("SPLIT", "SAME")]
            sp = sum(1 for r in E if r["outcome"] == "SPLIT")
            lines.append(f"| {name} | {len(E)} | {sp} | {len(E) - sp} | {sp / max(1, len(E)):.0%} |")
    lines += ["", f"窗口 ±{a.window_sec}s，guard {a.guard_sec}s；未能解析帧号的串人标注 {unp} 条。",
              "判读：断开率越高越好；误切率是代价（宁短勿脏允许一定误切，但 >20% 会明显拉低 gallery 覆盖）。",
              "UNMATCHED 多数是旧头框在新配置里被去重且找不到宿主身体框，单独看 events.csv。"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
