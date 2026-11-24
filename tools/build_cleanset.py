# -*- coding: utf-8 -*-
"""cleanset 构建：统一「清洗视图」数据集 + 可审计清单（R4.1 集成第一步）。

各机位档位（依据 docs/reflection-cleaning-integration-plan.md §三）：
  316 / 323 : dualclean（玻璃掩膜 + 运动学去倒影）← *_masked_cleaned.jsonl
  305 / 318 / 320 / 343 : clean（仅运动学去倒影）← *_cleaned.jsonl
  （343 掩膜为低置信草稿、待用户圈改确认；确认后升 dualclean）

产物：
  output_crosscam/cleanset/{cam}_detections.jsonl    统一命名（下游 det_view 直接换目录）
  output_crosscam/cleanset/clean_manifest.json       规则/版本/md5/删除清单（可审计、可断言）

用法：
  python tools/build_cleanset.py            # 构建 + 校验 + 清单
"""
import hashlib
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "output_crosscam")
TC = os.path.join(SRC, "analysis", "track_cleaning")
OUT = os.path.join(SRC, "cleanset")

CAMS = ["305", "316", "318", "320", "323", "343"]
GRADE = {"316": "dualclean", "323": "dualclean",
         "305": "clean", "318": "clean", "320": "clean", "343": "clean"}


def md5(p, bs=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_diff(a_path, b_path):
    """b 相对 a 的帧级删除统计：差异帧数 / 删除框数 / 涉及 tid 集合。"""
    n_diff_frames = n_removed = 0
    rem_tids = set()
    with open(a_path, encoding="utf-8") as fa, open(b_path, encoding="utf-8") as fb:
        for la, lb in zip(fa, fb):
            ra, rb = json.loads(la), json.loads(lb)
            ca, cb = len(ra.get("boxes") or []), len(rb.get("boxes") or [])
            if ca != cb:
                n_diff_frames += 1
                n_removed += ca - cb
                tb = set(rb.get("track_ids") or [])
                for t in (ra.get("track_ids") or []):
                    if t not in tb:
                        rem_tids.add(int(t))
    return n_diff_frames, n_removed, sorted(rem_tids)


def main():
    os.makedirs(OUT, exist_ok=True)
    tc = json.load(open(os.path.join(TC, "track_cleaning.json"), encoding="utf-8"))
    manifest = {
        "version": "cleanset_v1",
        "created": "2026-10-13",
        "pipeline": {
            "kinematic": {"script": "tools/track_cleaning.py", "rule": tc["rule"]},
            "glass_mask": {"script": "tools/glass_mask.py",
                           "config": "output_crosscam/glass_mask/glass_masks.json",
                           "note": "316/323 enabled；343 草稿未启用（待确认）"},
        },
        "cameras": {},
        "notes": tc.get("notes", []),
    }
    print(f"{'cam':>4} {'档位':<10} {'帧数':>6} {'轨(n>=4)':>9} {'删除':>5} "
          f"{'掩膜删框':>8} {'产物 md5':<10}")
    for cam in CAMS:
        src_raw = os.path.join(SRC, f"{cam}_detections.jsonl")
        kin = os.path.join(TC, f"{cam}_cleaned.jsonl")
        masked = os.path.join(TC, f"{cam}_masked_cleaned.jsonl")
        use = masked if GRADE[cam] == "dualclean" else kin
        assert os.path.exists(use), use

        out_p = os.path.join(OUT, f"{cam}_detections.jsonl")
        shutil.copyfile(use, out_p)

        sm = tc["summary"][cam]
        rec = {
            "grade": GRADE[cam],
            "source_raw": os.path.relpath(src_raw, ROOT).replace("\\", "/"),
            "source_used": os.path.relpath(use, ROOT).replace("\\", "/"),
            "md5_raw": md5(src_raw),
            "md5_out": md5(out_p),
            "frames": sum(1 for _ in open(out_p, encoding="utf-8")),
            "tracks_n_ge4": sm["tracks_total"],
            "removed_mirror_tracks": [int(t) for t in sm["flags"].keys()],
            "removed_mirror_count": sm["mirror_tracks"],
            "known_hit": sm["known_hit"],
        }
        if GRADE[cam] == "dualclean":
            df, nb, tids = count_diff(kin, use)
            rec["mask_removed_boxes"] = nb
            rec["mask_diff_frames"] = df
            rec["mask_touched_tids_count"] = len(tids)
        else:
            rec["mask_removed_boxes"] = 0
        manifest["cameras"][cam] = rec
        print(f"{cam:>4} {rec['grade']:<10} {rec['frames']:>6} {rec['tracks_n_ge4']:>9} "
              f"{sm['mirror_tracks']:>5} {rec['mask_removed_boxes']:>8} "
              f"{rec['md5_out'][:8]:<10}")

    mf = os.path.join(OUT, "clean_manifest.json")
    json.dump(manifest, open(mf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tot_tracks = sum(r["tracks_n_ge4"] for r in manifest["cameras"].values())
    tot_removed = sum(r["removed_mirror_count"] for r in manifest["cameras"].values())
    print(f"\n[cleanset] {OUT}")
    print(f"  总计 {tot_tracks} 轨（n>=4）| 运动学剔除 {tot_removed} 条 | "
          f"掩膜帧级剔除 316={manifest['cameras']['316']['mask_removed_boxes']} "
          f"323={manifest['cameras']['323']['mask_removed_boxes']} 框")
    print(f"  [manifest] {mf}")


if __name__ == "__main__":
    main()
