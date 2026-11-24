"""WILDTRACK · 阶段 A：把每一帧每个视角里每个人的框裁成 jpg，落盘缓存。

为什么拆开：整条链路（裁剪 → OSNet 特征 → AUC）在本机 CPU 上要 10 分钟以上，
超过单次命令时限，进程会被回收。裁剪结果与 meta 一旦落盘，特征阶段可断点续跑。

产物：
  tmp/wt/crops_wt/{view}_{stem}_{pid}.jpg   160×320 便于目视抽查
  tmp/wt/crops_wt/meta.json                 [{k, stem, pid, file}]
"""
import json
from pathlib import Path

import cv2

ROOT = Path("tmp/wt/data/Wildtrack_dataset")
ANN = ROOT / "annotations_positions"
IMG = ROOT / "Image_subsets"
OUT = Path("tmp/wt/crops_wt")


def load_ann(path):
    """返回 [(personID, {view: bbox})]，只保留至少一个视角可见的。"""
    arr = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for rec in arr if isinstance(arr, list) else []:
        pid = rec.get("personID")
        if pid is None:
            continue
        vis = {}
        for v in rec.get("views") or []:
            k = v.get("viewNum")
            if not isinstance(k, int) or not (0 <= k < 7):
                continue
            x1, y1, x2, y2 = (v.get("xmin", -1), v.get("ymin", -1),
                              v.get("xmax", -1), v.get("ymax", -1))
            if min(x1, y1, x2, y2) < 0 or x2 - x1 < 10 or y2 - y1 < 20:
                continue
            vis[k] = (x1, y1, x2, y2)
        if vis:
            out.append((pid, vis))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    png = {}
    for p in IMG.rglob("*.png"):
        k = int(p.parent.name[1:]) - 1
        png.setdefault(p.stem, {})[k] = p
    frames = sorted(png)
    print(f"有图像的帧 {len(frames)}", flush=True)

    meta = []
    for n, stem in enumerate(frames):
        rec = load_ann(ANN / f"{stem}.json")
        for pid, vis in rec:
            for k, box in vis.items():
                p = png[stem].get(k)
                if p is None:
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                H, W = img.shape[:2]
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                px, py = int(w * 0.08), int(h * 0.08)
                a1, b1 = max(0, x1 - px), max(0, y1 - py)
                a2, b2 = min(W, x2 + px), min(H, y2 + py)
                if a2 - a1 < 10 or b2 - b1 < 20:
                    continue
                name = f"v{k+1}_{stem}_{pid}.jpg"
                # ★ 存 160×320 给目视抽查；特征阶段直接读这个文件，保证两段同源
                cv2.imwrite(str(OUT / name),
                            cv2.resize(img[b1:b2, a1:a2], (160, 320)))
                meta.append(dict(k=k, stem=stem, pid=int(pid), file=name))
        if n % 4 == 0:
            print(f"  帧 {n+1}/{len(frames)} 累计 {len(meta)}", flush=True)

    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"完成：{len(meta)} 个 crop -> {OUT}")


if __name__ == "__main__":
    main()
