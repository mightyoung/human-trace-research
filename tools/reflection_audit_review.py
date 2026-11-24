# -*- coding: utf-8 -*-
"""316/323/343 倒影标记全量审计 —— 证据图生成。

输入：analysis/reflection_gate/flags_scan.json（117 条扫描记录）
输出：analysis/reflection_gate/review/
  - single_{cam}_{t}_{s}.jpg   单对放大图（含中间时刻；移动对额外附首/尾时刻三联）
  - montage_{cam}_{idx}.jpg    2x2 拼图（批量初审用）

用法: python tools/reflection_audit_review.py [cam ...]  （默认 316 323 343）
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = os.path.join(ROOT, "output_crosscam", "analysis", "reflection_gate", "flags_scan.json")
OUTD = os.path.join(ROOT, "output_crosscam", "analysis", "reflection_gate", "review")

CAMS = {
    "316": {
        "tag": "316-10m",
        "det": "output_crosscam/316_detections.jsonl",
        "vid": "media-resource/multi-cemra/synced-10min/01_行政走廊通道_316/316_085500-090500.mp4",
    },
    "323": {
        "tag": "323-10m",
        "det": "output_crosscam/323_detections.jsonl",
        "vid": "media-resource/multi-cemra/synced-10min/05_茶水间_323/323_085500-090500.mp4",
    },
    "343": {
        "tag": "343-10m",
        "det": "output_crosscam/343_detections.jsonl",
        "vid": "media-resource/multi-cemra/synced-10min/06_电梯间_343/343_085500-090500.mp4",
    },
    "316L": {
        "tag": "316-L",
        "det": "output_crosscam/long35/316_long35_detections.jsonl",
        "vid": "media-resource/multi-cemra/long-35min/316_long35.mp4",
    },
}


def load_tracks(path, tids):
    acc = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for t, b in zip(r["track_ids"], r["boxes"]):
                if t in tids:
                    acc.setdefault(int(t), {})[int(r["frame"])] = b
    return acc


def render_panel(vid_cache, vid_path, tracks, t, s, frame, panel_w=1000):
    """在 frame 上画 t/s 两框并裁剪放大，输出 panel_w 宽的图；失败返回 None"""
    if frame not in tracks.get(t, {}) or frame not in tracks.get(s, {}):
        return None
    cap = vid_cache.get(vid_path)
    if cap is None:
        cap = cv2.VideoCapture(vid_path)
        vid_cache[vid_path] = cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, fr = cap.read()
    if not ok:
        return None
    boxes = [(t, tracks[t][frame], (0, 0, 255)), (s, tracks[s][frame], (255, 120, 0))]
    xs1 = min(b[0] for _, b, _ in boxes)
    ys1 = min(b[1] for _, b, _ in boxes)
    xs2 = max(b[2] for _, b, _ in boxes)
    ys2 = max(b[3] for _, b, _ in boxes)
    mx = (xs2 - xs1) * 0.75 + 50
    my = (ys2 - ys1) * 0.75 + 50
    x0 = int(max(0, xs1 - mx)); y0 = int(max(0, ys1 - my))
    x1 = int(xs2 + mx); y1 = int(ys2 + my)
    for tid, b, col in boxes:
        cv2.rectangle(fr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
        cv2.putText(fr, f"#{tid}", (int(b[0]), max(22, int(b[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    crop = fr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    sc = panel_w / crop.shape[1]
    z = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4)
    return z


def stamp(img, text, color=(0, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return img


def main():
    os.makedirs(OUTD, exist_ok=True)
    scan = json.load(open(SCAN, encoding="utf-8"))
    todo = [c for c in CAMS if len(sys.argv) <= 1 or c in sys.argv[1:]]
    made = []
    vids = {}
    for cam in todo:
        cfg = CAMS[cam]
        flags = [r for r in scan if r["tag"] == cfg["tag"]]
        flags.sort(key=lambda r: -r["n"])
        tids = set()
        for r in flags:
            tids.add(int(r["t"])); tids.add(int(r["s"]))
        tracks = load_tracks(os.path.join(ROOT, cfg["det"]), tids)
        print(f"== {cam}: {len(flags)} 条标记, {len(tracks)} 轨加载 ==")
        cam_made = []
        for r in flags:
            t, s = int(r["t"]), int(r["s"])
            common = sorted(set(tracks.get(t, {})) & set(tracks.get(s, {})))
            if not common:
                print(f"  {cam} #{t}->#{s} 无共同帧，跳过")
                continue
            mid = common[len(common) // 2]
            # 单图：中间时刻（长对附加首/尾）
            z = render_panel(vids, os.path.join(ROOT, cfg["vid"]), tracks, t, s, mid)
            if z is None:
                print(f"  {cam} #{t}->#{s} 渲染失败")
                continue
            stat = (f"{cam} #{t}->#{s}  n={r['n']} perr={r['perr']:.3f} "
                    f"vR={r['vR']:.2f} vS={r['vS']:.2f} lum={r['lum_ratio']:.2f} f={mid}")
            stamp(z, stat)
            cv2.imwrite(os.path.join(OUTD, f"single_{cam}_{t}_{s}.jpg"), z)
            cam_made.append((cam, t, s, stat, z))
        # 拼图（每 cam 内部独立计数）
        idx = 0
        for i in range(0, len(cam_made), 4):
            group = cam_made[i:i + 4]
            h = max(g[4].shape[0] for g in group)
            w = max(g[4].shape[1] for g in group)
            canvas = np.zeros((h * 2 + 10, w * 2 + 10, 3), dtype=np.uint8)
            for j, g in enumerate(group):
                rr, cc = divmod(j, 2)
                canvas[rr * (h + 10):rr * (h + 10) + g[4].shape[0],
                       cc * (w + 10):cc * (w + 10) + g[4].shape[1]] = g[4]
            idx += 1
            p = os.path.join(OUTD, f"montage_{cam}_{idx:02d}.jpg")
            cv2.imwrite(p, canvas)
            print(f"  [montage] {p} ({len(group)} 格)")
        made += cam_made
    for cap in vids.values():
        cap.release()
    print(f"[done] 单图 {len(made)} 张 -> {OUTD}")


if __name__ == "__main__":
    main()
