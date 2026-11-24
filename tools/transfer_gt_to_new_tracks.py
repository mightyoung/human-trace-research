#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把旧跟踪器上的标注真值（gt_gallery_members / gt_query_gt）迁移到新跟踪器的新轨迹上。

为什么需要：换了跟踪器后轨迹编号（seg#tid）全部重排，而 257 条图库标注 + 229 条查询真值
都绑在旧 tid 上；不迁移的话 eval_precision.py 会把所有查询判成 U。

做法（逐帧几何对齐，不用任何模型）：
  新轨迹的每一帧，在旧结果同帧里找 IoU >= --iou 的旧框，取 IoU 最大者的 tid 投票；
  coverage = 投到票的帧数 / 新轨迹帧数；share = 得票最多的旧 tid 占比。
  coverage >= --min-coverage 且 share >= --min-share 才给 person_id，否则留空（与现场 GT 口径一致）。

同时输出「这次换跟踪器到底改了什么」的量化（不需要新标注，全部相对旧标注算）：
  - 丢失：旧标注轨迹没有被任何新轨迹认领
  - 碎片：一条旧标注轨迹的帧被 >=2 条新轨迹各占 >=20%（宁短勿脏的代价）
  - 仍串：一条新轨迹的票里 >=2 个旧 tid 且 person_id 不同（串人没修掉或修成了新串）

用法：
  python tools/transfer_gt_to_new_tracks.py \
    --old "output_crosscam_local/1012_yolo8_dd/{seg}/{seg}_detections.jsonl" \
    --new "output_crosscam_local/1012_yolo_v2/v2_s10/{seg}/{seg}_detections.jsonl" \
    --out-dir output_crosscam_local/gt_transfer/v2_s10
输出：
  <out>/gt_gallery_members_FILLED.csv、<out>/gt_query_gt_FILLED.csv（列与现场一致，另加诊断列）
  <out>/transfer_report.{json,md}
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEF_GT = ROOT / "deliverable/data-request-v2/gt"


def load_jsonl(path):
    """-> {frame: (tids (N,), boxes (N,4))}；跳过 id<0 与长度不齐的行。"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ids = r.get("track_ids") or []
            bx = r.get("boxes") or []
            n = min(len(ids), len(bx))
            keep = [i for i in range(n) if ids[i] is not None and int(ids[i]) >= 0]
            out[int(r["frame"])] = (np.array([int(ids[i]) for i in keep], dtype=np.int64),
                                    np.array([bx[i] for i in keep], dtype=np.float64).reshape(-1, 4))
    return out


def iou_vec(box, bx):
    x1 = np.maximum(box[0], bx[:, 0]); y1 = np.maximum(box[1], bx[:, 1])
    x2 = np.minimum(box[2], bx[:, 2]); y2 = np.minimum(box[3], bx[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a0 = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    an = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
    return inter / np.maximum(a0 + an - inter, 1e-9)


def track_index(det):
    """{frame: (tids, boxes)} -> {tid: [(frame, box), ...]}"""
    idx = collections.defaultdict(list)
    for fr, (ids, bx) in det.items():
        for k, t in enumerate(ids):
            idx[int(t)].append((fr, bx[k]))
    return idx


def transfer_seg(seg, old, new, old_person, old_box_type, iou_thr):
    """返回 (rows, pairs, old_assigned, stats)"""
    new_idx = track_index(new)
    rows, pairs = [], collections.defaultdict(int)
    old_assigned = collections.defaultdict(int)     # "seg#旧tid" -> 被认领的帧数
    cov_list = []
    for tid, obs in sorted(new_idx.items()):
        nkey = f"{seg}#{tid}"
        votes = collections.Counter()
        matched = 0
        for fr, box in obs:
            hit = old.get(fr)
            if hit is None or len(hit[0]) == 0:
                continue
            v = iou_vec(box, hit[1])
            j = int(v.argmax())
            if v[j] >= iou_thr:
                ot = f"{seg}#{int(hit[0][j])}"
                votes[ot] += 1
                pairs[(nkey, ot)] += 1
                old_assigned[ot] += 1
                matched += 1
        n = max(1, len(obs))
        cov = matched / n
        cov_list.append(cov)
        if votes:
            ot, cnt = votes.most_common(1)[0]
            share = cnt / max(1, sum(votes.values()))
            pid = old_person.get(ot, "")
            bt = old_box_type.get(ot, "")
        else:
            ot, share, pid, bt = "", 0.0, "", ""
        rows.append(dict(seg=seg, tid=tid, track_id=nkey, n_frames=len(obs),
                         src_tid=ot, src_person=pid, person_id=pid, box_type=bt,
                         coverage=round(cov, 3), share=round(share, 3), n_votes=sum(votes.values())))
    stats = dict(n_new=len(new_idx), n_old=len({int(t) for _, (ids, _) in old.items() for t in ids}),
                 cov_p10=round(float(np.percentile(cov_list, 10)), 3) if cov_list else 0,
                 cov_p50=round(float(np.percentile(cov_list, 50)), 3) if cov_list else 0,
                 cov_p90=round(float(np.percentile(cov_list, 90)), 3) if cov_list else 0)
    return rows, pairs, old_assigned, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-gallery", default=str(DEF_GT / "gt_gallery_members_FILLED.csv"))
    ap.add_argument("--gt-query", default=str(DEF_GT / "gt_query_gt_FILLED.csv"))
    ap.add_argument("--old", required=True, help="旧 jsonl 模板，须与台账同源（1012_yolo8_dd）")
    ap.add_argument("--new", required=True, help="新 jsonl 模板（v2 输出）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--segs", default="", help="逗号分隔；默认取两张 GT 的全部段")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--min-share", type=float, default=0.5)
    ap.add_argument("--frag-share", type=float, default=0.2, help="碎片/混杂判定的最小占比")
    a = ap.parse_args()

    GG = list(csv.DictReader(open(a.gt_gallery, encoding="utf-8-sig")))
    GQ = list(csv.DictReader(open(a.gt_query, encoding="utf-8-sig")))
    old_person = {r["track_id"]: (r["person_id"] or "").strip() for r in GG}
    old_box_type = {r["track_id"]: (r["box_type"] or "").strip() for r in GG}
    for r in GQ:
        pid = (r["person_id"] or "").strip()
        if pid:
            old_person[r["query_id"]] = pid
            old_box_type.setdefault(r["query_id"], (r["box_type"] or "").strip())

    segs = [s for s in a.segs.split(",") if s] or sorted(
        {r["seg"] for r in GG} | {r["seg"] for r in GQ})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    gal_rows, qry_rows = [], []
    per_seg, all_pairs, all_assigned = {}, {}, {}
    for seg in segs:
        op, np_ = Path(a.old.format(seg=seg)), Path(a.new.format(seg=seg))
        if not op.exists() or not np_.exists():
            print(f"[skip {seg}] old={op.exists()} new={np_.exists()}")
            continue
        old, new = load_jsonl(op), load_jsonl(np_)
        rows, pairs, assigned, st = transfer_seg(seg, old, new, old_person, old_box_type, a.iou)
        per_seg[seg] = st
        all_pairs.update({k: v for k, v in pairs.items()})
        for k, v in assigned.items():
            all_assigned[k] = all_assigned.get(k, 0) + v
        is_query = any(r["seg"] == seg for r in GQ) and not any(r["seg"] == seg for r in GG)
        for r in rows:
            if is_query:
                qry_rows.append(dict(query_id=r["track_id"], seg=seg, cam=seg.split("_")[0],
                                     n_frames=r["n_frames"], evaluable=1, person_id=r["person_id"],
                                     box_type=r["box_type"], src_tid=r["src_tid"],
                                     coverage=r["coverage"], share=r["share"], n_votes=r["n_votes"]))
            else:
                gal_rows.append(r)
        print(f"[{seg}] new_tracks={st['n_new']} old_tracks={st['n_old']} cov p50={st['cov_p50']}")

    # ---- 诊断：丢失 / 碎片 / 仍串 ----
    done = set(per_seg)
    labeled_old = {k for k, v in old_person.items() if v and k.split("#")[0] in done}
    lost = sorted(t for t in labeled_old if all_assigned.get(t, 0) == 0)

    new_by_old = collections.defaultdict(collections.Counter)
    for (nt, ot), c in all_pairs.items():
        new_by_old[ot][nt] += c
    frag = []
    for ot, cnt in new_by_old.items():
        tot = sum(cnt.values())
        big = [(nt, c) for nt, c in cnt.items() if c / max(1, tot) >= a.frag_share]
        if len(big) >= 2:
            frag.append((ot, len(big), sorted(big, key=lambda x: -x[1])[:3]))

    old_by_new = collections.defaultdict(collections.Counter)
    for (nt, ot), c in all_pairs.items():
        old_by_new[nt][ot] += c
    mixed = []
    for nt, cnt in old_by_new.items():
        tot = sum(cnt.values())
        big = [(ot, c) for ot, c in cnt.items() if c / max(1, tot) >= a.frag_share]
        pids = {old_person.get(ot, "") for ot, _ in big} - {""}
        if len(pids) >= 2:
            mixed.append((nt, sorted(pids), sorted(big, key=lambda x: -x[1])[:3]))

    ok = [r for r in gal_rows if r["coverage"] >= a.min_coverage and r["share"] >= a.min_share
          and r["person_id"]]
    lab = [r for r in gal_rows if r["person_id"]]
    q_ok = [r for r in qry_rows if r["coverage"] >= a.min_coverage and r["share"] >= a.min_share
            and r["person_id"]]

    # ---- 写出 ----
    def dump(rows, path, first):
        if not rows:
            Path(path).write_text("", encoding="utf-8")
            return
        keys = first + [k for k in rows[0] if k not in first]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    dump(sorted(gal_rows, key=lambda r: (r["seg"], r["tid"])), out / "gt_gallery_members_FILLED.csv",
         ["track_id", "person_id"])
    dump(sorted(qry_rows, key=lambda r: (r["seg"], int(r["query_id"].split("#")[1]))),
         out / "gt_query_gt_FILLED.csv", ["query_id", "person_id"])

    rep = dict(params=vars(a), per_seg=per_seg,
               gallery=dict(new_tracks=len(gal_rows), labeled=len(lab),
                            labeled_rate=round(len(lab) / max(1, len(gal_rows)), 3)),
               query=dict(new_tracks=len(qry_rows), labeled=len(q_ok)),
               labeled_old_total=len(labeled_old), lost=len(lost),
               fragmented_tracks=len(frag), still_mixed_tracks=len(mixed))
    (out / "transfer_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    allcov = [r["coverage"] for r in gal_rows + qry_rows]
    cov_line = ("p10/p50/p90 = %.3f / %.3f / %.3f" % (
        np.percentile(allcov, 10), np.percentile(allcov, 50), np.percentile(allcov, 90))) if allcov else "—"
    lines = [f"# 真值迁移报告（{a.new}）", "",
             f"- 旧标注轨迹 {len(labeled_old)} 条（图库+查询）｜新轨迹 图库 {len(gal_rows)} / 查询 {len(qry_rows)}",
             f"- 迁移成功（coverage≥{a.min_coverage} 且 share≥{a.min_share}）：图库 {len(ok)}/{len(gal_rows)}"
             f"（{len(ok) / max(1, len(gal_rows)):.0%}）｜查询 {len(q_ok)}/{len(qry_rows)}",
             f"- coverage {cov_line}",
             "", "## 换跟踪器改了什么", "",
             f"- **丢失**：{len(lost)}/{len(labeled_old)} 条旧标注轨迹没被任何新轨迹认领",
             f"- **碎片**：{len(frag)} 条旧轨迹的帧被 ≥2 条新轨迹各占 ≥{a.frag_share:.0%}",
             f"- **仍串**：{len(mixed)} 条新轨迹仍含 ≥2 个不同 person_id",
             "", "| 段 | 新轨迹 | 旧轨迹 | cov p10 | cov p50 | cov p90 |", "|---|---|---|---|---|---|"]
    for seg, st in per_seg.items():
        lines.append(f"| {seg} | {st['n_new']} | {st['n_old']} | {st['cov_p10']} | {st['cov_p50']} | {st['cov_p90']} |")
    if lost:
        lines += ["", "丢失样例：" + "、".join(lost[:10])]
    if frag:
        lines += ["", "碎片样例：" + "；".join(f"{o}→{n}条" for o, n, _ in frag[:8])]
    if mixed:
        lines += ["", "仍串样例：" + "；".join(f"新{t}={'/'.join(p)}" for t, p, _ in mixed[:8])]
    (out / "transfer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
