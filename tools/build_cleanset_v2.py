# -*- coding: utf-8 -*-
"""cleanset v2 构建：按全量审计（reflection_gate/audit_full_2026-10-13.json）修正后的剔除清单。

v1 → v2 修正（恢复被误剔的真人轨）：
  - 318: 全部标记 FP → clean=raw（恢复 53/82/98/135）
  - 305: 标记 FP → clean=raw（恢复 12）
  - 316: #32/36 FP → 恢复 #32（其余 17 条 TP 保留）
  - 323: #193/167 FP → 恢复 #193（其余 32 条 TP 保留）
  - 343: 4/4 TP 不变；320 无标记
10min 运动学剔除 60 → 53 条（FP 7 条恢复）。

另产出 long35-316 v2（恢复 #32；38 条 TP 保留）。

用法: python tools/build_cleanset_v2.py
"""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "output_crosscam")
TC = os.path.join(SRC, "analysis", "track_cleaning", "track_cleaning.json")
OUT = os.path.join(SRC, "cleanset_v2")
MASKS = os.path.join(SRC, "glass_mask", "glass_masks.json")

CAMS = ["305", "316", "318", "320", "323", "343"]
MASKED = {"316": (2560, 1440), "323": (1920, 1080)}   # 掩膜机位（v1 规则）
FP_RESTORE = {"316": {32}, "323": {193}, "318": {53, 82, 98, 135}, "305": {12},
              "320": set(), "343": set()}             # 审计修正：恢复的 FP 轨

LONG35 = {
    "jsonl": os.path.join(SRC, "long35", "cleaning", "316_masked_cleaned.jsonl"),
    "raw": os.path.join(SRC, "long35", "316_long35_detections.jsonl"),
    "out": os.path.join(SRC, "long35", "cleaning", "316_masked_cleaned_v2.jsonl"),
    "restore": {32},
    "wh": (2560, 1440),
}


def md5(p, bs=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def apply_pipeline(raw_p, remove_tids, mask_cfg):
    """raw -> (可选掩膜) -> 删除 remove_tids 的框 -> 行列表（与 v1 序列等价：并集删除）"""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from glass_mask import pt_in_poly  # noqa: E402
    polys, W, H = [], 0, 0
    if mask_cfg is not None:
        m = json.load(open(MASKS, encoding="utf-8"))
        mc = m["cameras"][mask_cfg["cam"]]
        polys = mc["regions"] if mc["enabled"] else []
        W, H = mask_cfg["wh"]
    out, stats = [], {"rows": 0, "boxes_in": 0, "boxes_out": 0,
                      "removed_by_mask": 0, "removed_by_tid": 0,
                      "count_fixed": 0, "scores_synced": 0}
    # 逐框契约（必须同长度）：boxes / track_ids / scores；count 必须 == len(boxes)。
    # 修复：删除框时用同一组 keep 索引同步过滤 scores，并重算 count（评审缺陷 #1）。
    with open(raw_p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            stats["rows"] += 1
            boxes = r.get("boxes", []) or []
            tids = r.get("track_ids", []) or []
            scores = r.get("scores", []) or []
            kb, kt, ks = [], [], []
            for i, b in enumerate(boxes):
                stats["boxes_in"] += 1
                tid = tids[i] if i < len(tids) else None
                if tid is not None and int(tid) in remove_tids:
                    stats["removed_by_tid"] += 1
                    continue
                if polys:
                    foot = (((b[0] + b[2]) / 2) / W, b[3] / H)
                    if any(pt_in_poly(foot, poly) for poly in polys):
                        stats["removed_by_mask"] += 1
                        continue
                kb.append(b)
                kt.append(tid)
                if i < len(scores):
                    ks.append(scores[i])
                stats["boxes_out"] += 1
            r["boxes"], r["track_ids"] = kb, kt
            r["scores"] = ks
            if r.get("count") != len(kb):
                stats["count_fixed"] += 1
            r["count"] = len(kb)
            if len(r["scores"]) == len(kb):
                stats["scores_synced"] += 1
            out.append(json.dumps(r, ensure_ascii=False))
    stats["boxes_out"] = stats["boxes_in"] - stats["removed_by_mask"] - stats["removed_by_tid"]
    assert len(out) == stats["rows"]
    return out, stats


def main():
    os.makedirs(OUT, exist_ok=True)
    tc = json.load(open(TC, encoding="utf-8"))
    manifest = {
        "version": "cleanset_v2",
        "created": "2026-10-13",
        "based_on": "cleanset_v1 + reflection_gate/audit_full_2026-10-13.json 全量审计修正",
        "pipeline": tc["rule"],
        "cameras": {},
    }
    print(f"{'cam':>4} {'剔除v2':>6} {'恢复FP':>6} {'删框(掩膜/tid)':>14} {'产物 md5':<10}")
    for cam in CAMS:
        raw_p = os.path.join(SRC, f"{cam}_detections.jsonl")
        v1_remove = {int(t) for t in tc["summary"][cam]["flags"].keys()}
        restore = FP_RESTORE[cam]
        assert restore <= v1_remove, (cam, "FP 必须原属标记集")
        remove_v2 = v1_remove - restore
        mask_cfg = {"cam": cam, "wh": MASKED[cam]} if cam in MASKED else None
        lines, stats = apply_pipeline(raw_p, remove_v2, mask_cfg)
        out_p = os.path.join(OUT, f"{cam}_detections.jsonl")
        with open(out_p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        manifest["cameras"][cam] = {
            "removed_mirror_count_v1": len(v1_remove),
            "restored_fp": sorted(restore),
            "removed_mirror_count_v2": len(remove_v2),
            "mask_enabled": bool(mask_cfg),
            "removed_by_mask": stats["removed_by_mask"],
            "removed_by_tid": stats["removed_by_tid"],
            "md5_out": md5(out_p),
            "source_raw": f"output_crosscam/{cam}_detections.jsonl",
        }
        print(f"{cam:>4} {len(remove_v2):>6} {len(restore):>6} "
              f"{stats['removed_by_mask']:>6}/{stats['removed_by_tid']:<6} "
              f"{manifest['cameras'][cam]['md5_out'][:8]:<10}")

    tot_v1 = sum(c["removed_mirror_count_v1"] for c in manifest["cameras"].values())
    tot_v2 = sum(c["removed_mirror_count_v2"] for c in manifest["cameras"].values())
    manifest["totals"] = {"kinematic_removed_v1": tot_v1, "kinematic_removed_v2": tot_v2,
                          "restored": tot_v1 - tot_v2}
    json.dump(manifest, open(os.path.join(OUT, "clean_manifest_v2.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[cleanset_v2] 运动学剔除 {tot_v1} -> {tot_v2}（恢复 {tot_v1 - tot_v2} 条 FP 轨）")

    # ---- long35-316 v2：raw - mask - (长窗flags-{32})，等价于 v1 输出恢复 #32 ----
    tcL = json.load(open(os.path.join(SRC, "long35", "cleaning",
                                      "track_cleaning.json"), encoding="utf-8"))
    flagsL = {int(t) for t in tcL["summary"]["316"]["flags"].keys()}
    restoreL = LONG35["restore"]
    assert restoreL <= flagsL
    removeL = flagsL - restoreL
    lines, stats = apply_pipeline(LONG35["raw"], removeL, {"cam": "316", "wh": LONG35["wh"]})
    with open(LONG35["out"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[long35-316 v2] flags {len(flagsL)} -> {len(removeL)}（恢复 #32）；"
          f"mask删框 {stats['removed_by_mask']} / tid删框 {stats['removed_by_tid']} -> {LONG35['out']}")


if __name__ == "__main__":
    main()
