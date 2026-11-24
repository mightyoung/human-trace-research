"""v2 跟踪器归因臂的评估：只回答「并行重复轨迹 / 串人」这两件事有没有变好。

为什么不复用 diag_fragmentation：那个脚本是簇级的（要知道 query→cluster 的判决），
而归因臂只重跑了检测+跟踪、没重跑编码聚类。本脚本在**轨迹级**直接比，成本几分钟。

四个指标（同段、同 stride、同一份真值）：
  A 轨迹数 / 中位帧长
  B 同帧重复框：同一采样帧里两条轨迹的框 IoU ≥ thr 的对数（去重成轨迹对）
    —— 不需要真值，直接量「一个人被框成两个」
  C 标注迁移率：新轨迹靠同帧 IoU 投票继承旧轨迹的 person_id，报告能继承的旧轨覆盖率
  D 同身份重叠对：迁移后同一 person_id 的两条轨迹，帧级重叠 / 较短者 ≥ 0.5 的对数
    —— 这就是 305「77 对里 52 对帧级重叠」那个口径
  E 混人轨迹：一条轨迹继承到 ≥2 个身份（票数均 ≥2）的条数

用法：
  PY tools/eval_v2_attrib.py --seg 305_train_529 --old-root output_crosscam_local/1012_yolo8_dd \
      --new-root output_crosscam_local/1012_yolo_v2/attrib
  PY tools/eval_v2_attrib.py --seg ... --iou-pair 0.5        # 换重复框判据
输出：results/v2_attrib/<seg>.json + 控制台表格
"""
import argparse
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, "deliverable", "data-request-v2")


def load_tracks(path):
    """-> {tid: {frame: (x1,y1,x2,y2,conf)}}；帧号已按 jsonl 原值（采样帧序号）。"""
    tr = collections.defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        f = r["frame"]
        for t, b in zip(r["track_ids"], r["boxes"]):
            if len(b) >= 5:
                tr[int(t)][f] = tuple(float(x) for x in b[:5])
            else:
                tr[int(t)][f] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]), 1.0)
    return dict(tr)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def by_frame(tracks):
    """-> {frame: [(tid, box), ...]}"""
    out = collections.defaultdict(list)
    for t, d in tracks.items():
        for f, b in d.items():
            out[f].append((t, b))
    return out


def duplicate_pairs(tracks, thr):
    """同一帧内框 IoU ≥ thr 的轨迹对（去重），返回 (对集合, 涉及帧数, 总采样帧数)。"""
    frames = by_frame(tracks)
    pairs = collections.Counter()
    for f, items in frames.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if iou(items[i][1], items[j][1]) >= thr:
                    pairs[tuple(sorted((items[i][0], items[j][0])))] += 1
    return set(pairs), len(frames)


def load_gt():
    """seg -> {tid: person_id}，查询段与图库段都算（两份真值文件合并）。"""
    import csv
    out = {}
    q = os.path.join(DELIV, "gt", "gt_query_gt_FILLED.csv")
    for r in csv.DictReader(open(q, encoding="utf-8-sig")):
        pid = (r["person_id"] or "").strip()
        if pid and pid.upper() != "UNK":
            out.setdefault(r["seg"], {})[int(r["query_id"].split("#")[1])] = pid
    g = os.path.join(DELIV, "gt", "gt_gallery_members_FILLED.csv")
    for r in csv.DictReader(open(g, encoding="utf-8-sig")):
        pid = (r["person_id"] or "").strip()
        if pid and pid.upper() != "UNK":
            out.setdefault(r["seg"], {})[int(r["track_id"].split("#")[1])] = pid
    return out


def transfer_labels(old, new, gt_seg, min_iou=0.5, min_vote=1):
    """新轨迹按同帧 IoU 投票继承旧轨迹的 person_id。返回 (new_tid->pid, 统计)。"""
    old_frames = by_frame(old)
    out, stats = {}, collections.Counter()
    for nt, nd in new.items():
        votes = collections.Counter()
        hit = 0
        for f, nb in nd.items():
            for ot, ob in old_frames.get(f, ()):
                if ot in gt_seg and iou(nb, ob) >= min_iou:
                    votes[gt_seg[ot]] += 1
                    hit += 1
                    break
        if not votes:
            continue
        top, n = votes.most_common(1)[0]
        if n < min_vote:
            stats["low_vote_skip"] += 1
            continue
        out[nt] = top
        stats["transferred"] += 1
        # 混人：第二名身份票数 ≥2 且 ≥ 首名 20%，避免单帧擦碰就记混
        ranked = votes.most_common()
        if len(ranked) >= 2 and ranked[1][1] >= 2 and ranked[1][1] >= 0.2 * ranked[0][1]:
            stats["mixed"] += 1
    return out, stats


def overlapping_pairs(tracks, labels):
    """同身份、帧级重叠 / 较短者 ≥ 0.5 的轨迹对。"""
    by_pid = collections.defaultdict(list)
    for t, p in labels.items():
        if t in tracks:
            by_pid[p].append(t)
    pairs = []
    for p, ts in by_pid.items():
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = tracks[ts[i]], tracks[ts[j]]
                shared = len(set(a) & set(b))
                if not shared:
                    continue
                frac = shared / min(len(a), len(b))
                pairs.append((p, ts[i], ts[j], round(frac, 3), shared))
    return pairs


def summarize(name, tracks, labels, thr):
    dups, nframes = duplicate_pairs(tracks, thr)
    lens = sorted(len(d) for d in tracks.values())
    ov = overlapping_pairs(tracks, labels)
    return dict(name=name, tracks=len(tracks), frames=nframes,
                median_len=lens[len(lens) // 2] if lens else 0,
                dup_pairs=len(dups),
                labeled_tracks=len(labels),
                same_id_anyoverlap=len(ov),
                same_id_halfoverlap=sum(1 for x in ov if x[3] >= 0.5),
                same_id_overlap_detail=sorted(ov, key=lambda x: -x[3])[:12])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--old-root", default="output_crosscam_local/1012_yolo8_dd")
    ap.add_argument("--new-root", default="output_crosscam_local/1012_yolo_v2/attrib")
    ap.add_argument("--iou-pair", type=float, default=0.3)
    ap.add_argument("--min-iou-transfer", type=float, default=0.5)
    a = ap.parse_args()

    p_old = os.path.join(ROOT, a.old_root, a.seg, a.seg + "_detections.jsonl")
    p_new = os.path.join(ROOT, a.new_root, a.seg, a.seg + "_detections.jsonl")
    for p in (p_old, p_new):
        if not os.path.exists(p):
            raise SystemExit("缺少 %s" % p)

    gt = load_gt()
    gt_seg = gt.get(a.seg, {})
    old = load_tracks(p_old)
    new = load_tracks(p_new)

    # 旧轨直接用真值；新轨靠迁移
    old_lab = {t: gt_seg[t] for t in old if t in gt_seg}
    new_lab, tstats = transfer_labels(old, new, gt_seg, a.min_iou_transfer)

    s_old = summarize("old(1012_yolo8_dd)", old, old_lab, a.iou_pair)
    s_new = summarize("new(v2 attrib)", new, new_lab, a.iou_pair)

    print("段 %s｜重复框判据 IoU ≥ %.2f｜标注迁移 IoU ≥ %.2f"
          % (a.seg, a.iou_pair, a.min_iou_transfer))
    print("%-22s %7s %8s %9s %10s %10s %10s %8s" %
          ("", "轨迹", "中位帧长", "重复框对", "有标注轨", "同身份≥1帧", "同身份≥50%", "混人轨"))
    for s in (s_old, s_new):
        print("%-22s %7d %8d %9d %10d %10d %10d %8s" %
              (s["name"], s["tracks"], s["median_len"], s["dup_pairs"],
               s["labeled_tracks"], s["same_id_anyoverlap"], s["same_id_halfoverlap"],
               tstats.get("mixed", 0) if s is s_new else "-"))
    print("标注迁移：%d 条新轨继承身份（低票跳过 %d，多身份 %d）"
          % (tstats.get("transferred", 0), tstats.get("low_vote_skip", 0), tstats.get("mixed", 0)))
    print("旧轨有标注 %d 条 / 共 %d 条（%.1f%%）；新轨已标 %d / %d（%.1f%%）"
          % (len(old_lab), len(old), 100.0 * len(old_lab) / max(1, len(old)),
             len(new_lab), len(new), 100.0 * len(new_lab) / max(1, len(new))))

    os.makedirs(os.path.join(DELIV, "results", "v2_attrib"), exist_ok=True)
    out = os.path.join(DELIV, "results", "v2_attrib", a.seg + ".json")
    json.dump(dict(seg=a.seg, iou_pair=a.iou_pair, old=s_old, new=s_new,
                   transfer=tstats, old_gt_cover=len(old_lab), new_gt_cover=len(new_lab)),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ %s" % out)


if __name__ == "__main__":
    main()
