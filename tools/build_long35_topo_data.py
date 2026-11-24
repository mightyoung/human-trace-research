# -*- coding: utf-8 -*-
"""TOP-1 数据准备：构建 long35 拓扑检验数据目录（碎片级，修正口径 v2）。

产出 output_crosscam/analysis/topology/long35_data/：
  - {cam}_detections.jsonl（六路 raw 拷贝，供 load_foot 读取）
  - splits_v4.json（keep 集 = 各路清洗后存活轨迹；口径与 10min 一致：
      316=masked_cleaned_v2 / 323·343=运动学清洗 / 318·320=既有 cleaned / 305=raw）
仅本地读写，不改动任何既有产物。
"""
import json
import os
import shutil
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "output_crosscam", "long35")
CLEAN = os.path.join(SRC, "cleaning")
OUT = os.path.join(ROOT, "output_crosscam", "analysis", "topology", "long35_data")
CAMS = ["305", "316", "318", "320", "323", "343"]

CLEAN_FILES = {
    "316": "316_masked_cleaned_v2.jsonl",
    "318": "318_cleaned.jsonl",
    "320": "320_cleaned.jsonl",
    "323": "323_cleaned.jsonl",
    "343": "343_cleaned.jsonl",
    "305": None,   # raw（10min 口径：305 无可靠倒影证据，保持 raw）
}


def tids_in(path):
    s = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for t in d.get("track_ids", []):
                if t is not None and int(t) >= 0:
                    s.add(int(t))
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    keep = {}
    for cam in CAMS:
        raw = os.path.join(SRC, f"{cam}_long35_detections.jsonl")
        cf = CLEAN_FILES[cam]
        if cf:
            k = tids_in(os.path.join(CLEAN, cf))
            src_tag = cf
        else:
            k = tids_in(raw)
            src_tag = "raw"
        keep[cam] = sorted(k)
        shutil.copyfile(raw, os.path.join(OUT, f"{cam}_detections.jsonl"))
        print(f"[{cam}] keep={len(k)}  (source: {src_tag})")

    splits = {
        "note": "long35 keep sets（TOP-1 检验用；= 清洗后存活轨迹）",
        "train": {"tracks": {cam: keep[cam] for cam in CAMS}},
    }
    with open(os.path.join(OUT, "splits_v4.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False)

    # 与 10min 的重叠量化（316 代表）
    p10 = os.path.join(ROOT, "output_crosscam", "316_detections.jsonl")
    if os.path.exists(p10):
        n10 = sum(1 for _ in open(p10, encoding="utf-8"))
        same = 0
        with open(p10, encoding="utf-8") as f1, open(
                os.path.join(OUT, "316_detections.jsonl"), encoding="utf-8") as f2:
            for l1, l2 in zip(f1, f2):
                if l1 == l2:
                    same += 1
                else:
                    break
        print(f"[overlap] 10min-316 行数={n10}，与 long35 前缀逐行一致={same} "
              f"→ 重叠 {same * 5 / 25:.0f}s（占 10min 的 {same / n10:.0%}）")
    print("staged ->", OUT)


if __name__ == "__main__":
    main()
