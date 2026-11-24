#!/usr/bin/env python
"""应用 P2 用户复核修正 + 生成六路净轨迹 + 扩展评估标签。

⚠️ 已降级为**legacy（仅生成标签）**——2026-10-14 起实体图由
   `tools/rebuild_clean2_entities.py` 在 clean2 口径上生成（analysis/stitch_clean2）。
   本脚本的 net_tracks 读的是**无掩膜** `analysis/track_cleaning/{cam}_cleaned.jsonl`，
   与生产/评测口径（dualclean）不一致，若重跑并覆盖会**回退** clean2 修复（见
   docs/clean2-rebuild-report.md §2.1）。因此默认**拒绝覆盖**规范实体图目录；
   确需复现旧链时显式设置 `P2_ALLOW_LEGACY=1`。
   `labels_v2.json`（评估对标签）仍由本脚本产出，与实体图口径无关，可继续使用。

修正（2026-10-12 用户复核）：
- 318: 撤销链接 (36,39), (46,59), (57,62)
- 318: 强制合并白衣人 21 段为一实体（q1 同意）
- 318: 远区深发组不做额外合并（q2 同意，维持现状）
- 320: 剔除非人轨迹 [8,9,13,16,17]（q3: 屏内画面误检）
- 320: 合并 [18,23]（此前经 318#120 跨机位确认同一人）

输出：
- stitch_{318,320}_groups_corrected.json  修正版链
- net_tracks_{cam}.json  六路净轨迹（**legacy 口径**；实为无掩膜，勿用于评测）
- labels_v2.json  扩展评估标签（同机位正/负 + 跨机位）
- user_review_2026-10-12.json  复核决策记录
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STITCH = ROOT / "output_crosscam/analysis/stitch"
CLEAN = ROOT / "output_crosscam/analysis/track_cleaning"
CANONICAL_NET_DIRS = {"stitch_clean2"}   # 禁止覆盖的规范实体图目录

if os.environ.get("P2_ALLOW_LEGACY") != "1" and STITCH.name in CANONICAL_NET_DIRS:
    raise SystemExit(
        "[中止] apply_p2_corrections 为 legacy（无掩膜口径），拒绝写入规范实体图目录 "
        f"{STITCH}。\n       实体图请用 tools/rebuild_clean2_entities.py；"
        "确需复现旧链请设 P2_ALLOW_LEGACY=1。")

BLACKLIST_318 = [(36, 39), (46, 59), (57, 62)]
WHITE_318 = [3, 42, 49, 50, 58, 70, 72, 74, 77, 81, 83, 85, 86, 91,
             102, 111, 123, 134, 144, 148, 149]
NONPERSON_320 = [8, 9, 13, 16, 17]
FORCE_320 = [(18, 23)]


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_track_spans(cam):
    """track_id -> (f0, f1, n)"""
    spans = {}
    with open(CLEAN / f"{cam}_cleaned.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            for i in range(len(r.get("boxes", []) or [])):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    tid = int(tids[i])
                    s = spans.get(tid)
                    fr = r["frame"]
                    if s is None:
                        spans[tid] = [fr, fr, 1]
                    else:
                        s[0] = min(s[0], fr)
                        s[1] = max(s[1], fr)
                        s[2] += 1
    return spans


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_entities(links, forced_groups, all_tracks):
    """links: [(a,b)]; forced_groups: [[tid,...],...]; all_tracks: iterable"""
    uf = UnionFind()
    for t in all_tracks:
        uf.find(t)
    for a, b in links:
        uf.union(a, b)
    for grp in forced_groups:
        for x in grp[1:]:
            uf.union(grp[0], x)
    groups = {}
    for t in all_tracks:
        groups.setdefault(uf.find(t), []).append(t)
    return sorted(groups.values(), key=lambda g: min(g))


def main():
    # ---------- 318 ----------
    g318 = load_json(STITCH / "stitch_318_groups.json")
    links318 = [(l["tid_a"], l["tid_b"]) for l in g318["links"]]
    links318_kept = [p for p in links318 if p not in BLACKLIST_318]

    # 新候选：撤销后重跑互选匹配涌现的替代链接
    import csv
    cands = []
    for r in csv.DictReader((STITCH / "stitch_318.csv").open(encoding="utf-8")):
        a, b = int(r["tid_a"]), int(r["tid_b"])
        try:
            cos, pe = float(r["cos"]), float(r["pe_adj"])
        except (TypeError, ValueError):
            continue
        if cos < 0.80 or pe > 0.08 or (a, b) in BLACKLIST_318:
            continue
        cands.append((a, b, cos, pe, cos - 2 * pe))
    cands.sort(key=lambda x: x[4], reverse=True)
    best_succ, best_pred = {}, {}
    for a, b, cos, pe, sc in cands:
        best_succ.setdefault(a, (a, b, cos, pe))
        best_pred.setdefault(b, (a, b, cos, pe))
    new_links = []
    for a, r in best_succ.items():
        b = r[1]
        if b in best_pred and best_pred[b][0] == a:
            if (a, b) not in links318_kept:
                new_links.append({"a": a, "b": b, "cos": round(r[2], 4),
                                  "pe_adj": round(r[3], 4)})

    spans318 = load_track_spans("318")
    ents318 = build_entities(links318_kept, [WHITE_318], sorted(spans318))
    ents318_out = []
    for eid_i, members in enumerate(ents318, 1):
        f0 = min(spans318[t][0] for t in members)
        f1 = max(spans318[t][1] for t in members)
        note = ""
        if sorted(members) == sorted(WHITE_318):
            note = "白衣人(工位) 21 段合并——用户确认 q1"
        ents318_out.append({
            "eid": f"318-E{eid_i:03d}", "tracks": sorted(members),
            "f0": f0, "f1": f1, "n_frag": len(members), "note": note,
        })

    corr318 = {
        "cam": "318", "source": "stitch_318_groups.json + user review 2026-10-12",
        "removed_links": BLACKLIST_318,
        "kept_links": [list(p) for p in links318_kept],
        "forced_merge": [{"name": "white_shirt", "tracks": WHITE_318}],
        "new_candidates_after_removal": new_links,
        "n_links_before": len(links318), "n_links_after": len(links318_kept),
    }
    (STITCH / "stitch_318_groups_corrected.json").write_text(
        json.dumps(corr318, ensure_ascii=False, indent=2), encoding="utf-8")

    net318 = {
        "cam": "318", "source": "P2 corrections 2026-10-12",
        "n_tracks": len(spans318), "n_entities": len(ents318_out),
        "non_person": [], "pending": {"new_candidate_links": new_links},
        "entities": ents318_out,
    }
    (STITCH / "net_tracks_318.json").write_text(
        json.dumps(net318, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 320 ----------
    g320 = load_json(STITCH / "stitch_320_groups.json")
    links320 = [(l["tid_a"], l["tid_b"]) for l in g320["links"]]
    links320_kept = [p for p in links320
                     if p[0] not in NONPERSON_320 and p[1] not in NONPERSON_320]
    spans320 = load_track_spans("320")
    alive320 = [t for t in sorted(spans320) if t not in NONPERSON_320]
    ents320 = build_entities(links320_kept, [list(x) for x in FORCE_320], alive320)
    ents320_out = []
    for eid_i, members in enumerate(ents320, 1):
        f0 = min(spans320[t][0] for t in members)
        f1 = max(spans320[t][1] for t in members)
        note = ""
        if sorted(members) == [18, 23]:
            note = "皮领女士：跨机位确认(318#120:18 与 120:23)——合并"
        ents320_out.append({
            "eid": f"320-E{eid_i:03d}", "tracks": sorted(members),
            "f0": f0, "f1": f1, "n_frag": len(members), "note": note,
        })

    corr320 = {
        "cam": "320", "source": "stitch_320_groups.json + user review 2026-10-12",
        "removed_non_person_tracks": NONPERSON_320,
        "removed_links": [list(p) for p in links320 if p not in links320_kept],
        "forced_merge": [{"name": "leather_collar_lady", "tracks": [18, 23]}],
        "n_links_before": len(links320), "n_links_after": len(links320_kept),
    }
    (STITCH / "stitch_320_groups_corrected.json").write_text(
        json.dumps(corr320, ensure_ascii=False, indent=2), encoding="utf-8")

    net320 = {
        "cam": "320", "source": "P2 corrections 2026-10-12",
        "n_tracks": len(spans320) - len(NONPERSON_320),
        "n_entities": len(ents320_out),
        "non_person": NONPERSON_320,
        "pending": {"t22_question": "22 是否并入 [18,23]（证据：b18.b22=0.725, b22.b23=0.729, a120.b22=0.566；映射位置吻合；需复核）"},
        "entities": ents320_out,
    }
    (STITCH / "net_tracks_320.json").write_text(
        json.dumps(net320, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 其他四路净轨迹（未额外复核，沿用链） ----------
    summary = {"318": {"tracks": len(spans318), "entities": len(ents318_out)},
               "320": {"tracks": len(spans320) - len(NONPERSON_320),
                       "entities": len(ents320_out)}}
    for cam in ["305", "316", "323", "343"]:
        g = load_json(STITCH / f"stitch_{cam}_groups.json")
        lks = [(l["tid_a"], l["tid_b"]) for l in g["links"]]
        sp = load_track_spans(cam)
        ents = build_entities(lks, [], sorted(sp))
        ents_out = []
        for eid_i, members in enumerate(ents, 1):
            ents_out.append({
                "eid": f"{cam}-E{eid_i:03d}", "tracks": sorted(members),
                "f0": min(sp[t][0] for t in members),
                "f1": max(sp[t][1] for t in members),
                "n_frag": len(members), "note": "",
            })
        (STITCH / f"net_tracks_{cam}.json").write_text(
            json.dumps({"cam": cam, "source": "P2 (auto, 未经用户复核)",
                        "n_tracks": len(sp), "n_entities": len(ents_out),
                        "non_person": [], "entities": ents_out},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        summary[cam] = {"tracks": len(sp), "entities": len(ents_out)}

    # ---------- 扩展评估标签 ----------
    # 318 同机位正样本：34 条保留链接 + 白衣人相邻段对
    white_sorted = sorted(WHITE_318, key=lambda t: spans318[t][0])
    white_adj = [(white_sorted[i], white_sorted[i + 1])
                 for i in range(len(white_sorted) - 1)]
    samecam_pos = (
        [{"cam": 318, "a": a, "b": b, "source": "kept-link"} for a, b in links318_kept]
        + [{"cam": 318, "a": a, "b": b, "source": "white-adjacent"} for a, b in white_adj]
        + [{"cam": 320, "a": 19, "b": 20, "source": "kept-link"},
           {"cam": 320, "a": 25, "b": 26, "source": "kept-link"},
           {"cam": 320, "a": 26, "b": 27, "source": "kept-link"},
           {"cam": 320, "a": 18, "b": 23, "source": "crosscam-confirmed"}]
    )
    samecam_neg = [{"cam": 318, "a": a, "b": b, "source": "rejected"}
                   for a, b in BLACKLIST_318]
    labels = {
        "date": "2026-10-12",
        "same_camera": {"positives": samecam_pos, "negatives": samecam_neg},
        "cross_camera": {
            "positives": [[1, 1], [14, 3], [69, 14], [120, 18], [120, 23],
                          [139, 24], [146, 28], [37, 11]],
            "negatives": [[14, 1], [131, 21], [61, 14], [50, 11], [132, 21],
                          [3, 1], [7, 1], [146, 21], [127, 19], [123, 19],
                          [57, 11], [131, 23], [124, 18], [131, 24], [3, 4],
                          [126, 18], [43, 11], [10, 1], [131, 28], [98, 26],
                          [142, 26], [53, 11], [123, 18], [111, 18]],
            "note": "318:a ↔ 320:b 跨机位；配对细节见 candidates_review.csv",
        },
        "pending": {"new_links_318": new_links,
                    "t22_320": "22 是否并入 [18,23]"},
        "non_person_320": NONPERSON_320,
    }
    (STITCH / "labels_v2.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 复核决策记录 ----------
    review = {
        "date": "2026-10-12", "reviewer": "user",
        "decisions": {
            "q1_white_merge": {"decision": "approved",
                               "tracks": WHITE_318, "cam": 318},
            "q2_far_group": {"decision": "approved_keep_separate"},
            "q3_320_first_four": {"decision": "not_person",
                                  "tracks": NONPERSON_320, "cam": 320,
                                  "evidence": "屏幕画面误检（28x34px 静止框）"},
            "q4_wrong_links": {"decision": "removed",
                               "links": [list(p) for p in BLACKLIST_318]},
        },
        "new_candidates_for_next_review": {
            "links_318": new_links,
            "t22_320": "22 与 [18,23] 关系待定",
        },
    }
    (STITCH / "user_review_2026-10-12.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 打印摘要 ----------
    print("=== 318 修正 ===")
    print(f"  撤销链接: {BLACKLIST_318}")
    print(f"  保留链接: {len(links318_kept)}/37")
    print(f"  新浮现候选: {new_links}")
    white_ent = [e for e in ents318_out if e['note'].startswith('白衣')]
    print(f"  白衣人合并: {white_ent[0]['tracks'] if white_ent else 'N/A'}")
    print(f"  轨迹 {len(spans318)} -> 实体 {len(ents318_out)}")
    print()
    print("=== 320 修正 ===")
    print(f"  剔除非人: {NONPERSON_320}")
    print(f"  保留链接: {links320_kept}")
    print(f"  轨迹 {len(spans320)} -> 有效 {len(spans320)-len(NONPERSON_320)} -> 实体 {len(ents320_out)}")
    print()
    print("=== 六路净轨迹摘要 ===")
    for cam, s in summary.items():
        print(f"  {cam}: {s['tracks']} 轨迹 -> {s['entities']} 实体")
    print()
    print("=== labels_v2 ===")
    print(f"  同机位: 正 {len(samecam_pos)} / 负 {len(samecam_neg)}")
    print(f"  跨机位: 正 8 / 负 24")


if __name__ == "__main__":
    main()
