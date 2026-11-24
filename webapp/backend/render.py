# -*- coding: utf-8 -*-
"""服务端渲染：用 supervision 把轨迹/检测烘焙到视频帧上（可视化模块的服务端能力）。

- render_frame(cam, frame, ...) -> JPEG bytes   （单帧叠加）
- export_video(cam, f0, f1, ...) -> 后台渲染标注 mp4（H.264，直接输出网页可播）
"""
from __future__ import annotations
import json
import os
import subprocess
import threading
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import supervision as sv
    HAS_SV = True
except Exception:  # pragma: no cover
    sv = None
    HAS_SV = False

DATA = Path(__file__).resolve().parent / "data"
MEDIA_ROOT = Path(os.environ.get("WEBAPP_MEDIA_ROOT", str(DATA / "media")))
# 原生 1440p 素材（不在 data/ 下，约 1.9 GB / 6 路）：裁剪核对用它，
# 用 960×540 代理裁出来的图放大后只剩色块，判不了身份（实测 200px 人 → 75px）。
NATIVE_ROOT = Path(os.environ.get("WEBAPP_NATIVE_ROOT",
                                  str(DATA.parents[2] / "media-resource" / "multi-cemra" / "long-35min")))
EXPORT_DIR = DATA / "exports"

_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6", "#d37295",
]

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _track_color(tid: int) -> str:
    return _PALETTE[int(tid) % len(_PALETTE)]


def _load_tracks(cam: str) -> dict:
    p = DATA / "tracks" / f"{cam}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _load_status(cam: str) -> dict:
    p = DATA / "cams.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return (d.get("track_status", {}) or {}).get(cam, {})


# ================= 身份口径臂（S6-b 三档：A 现状 / B 倒影降权 / H 人工裁决版）=================
_ARM_FILES = {"A": "gil_owner_A.json", "B": "gil_owner_deduprefl.json", "H": "gil_owner_human.json"}
_arm_cache: dict = {}


def _arm_path(fname: str):
    """载荷位置：优先 frontend/public（开发），退回 frontend/dist（部署后）。"""
    base = Path(__file__).resolve().parent.parent / "frontend"
    for sub in ("public", "dist"):
        p = base / sub / fname
        if p.exists():
            return p
    return None


def load_arm(arm: str | None):
    """读身份口径载荷 → (tid2g, owner_idx)；arm 为空/载荷缺失时返回 (None, None)。"""
    if not arm:
        return None, None
    if arm in _arm_cache:
        return _arm_cache[arm]
    fname = _ARM_FILES.get(arm)
    p = _arm_path(fname) if fname else None
    if p is None:
        _arm_cache[arm] = (None, None)
        return None, None
    d = json.loads(p.read_text(encoding="utf-8"))
    tid2g = d.get("tid2g") or {}
    idx: dict = {}
    for r in d.get("owners") or []:
        idx.setdefault(int(r[0]), []).append((float(r[2]), float(r[3]), r[1]))
    for k in idx:
        idx[k].sort()
    _arm_cache[arm] = (tid2g, idx)
    return _arm_cache[arm]


def _cam_delay(cam: str) -> float:
    """机位本地时间 → 会话全局时间的偏移（与前端 delayOf 同口径）。"""
    p = DATA / "cams.json"
    if not p.exists():
        return 0.0
    d = json.loads(p.read_text(encoding="utf-8"))
    minoff = float((d.get("session") or {}).get("min_offset", 0.0))
    for c in d.get("cameras") or []:
        if c.get("id") == cam:
            return float(c.get("offset", 0.0)) - minoff
    return 0.0


def _apply_arm(cam: str, dets: list, fk: int, arm_data, fps: float = 25.0):
    """按身份口径去重：同一全局人只保留责任机位那一个框 → (过滤后 dets, {tid: gid})。"""
    tid2g, idx = arm_data or (None, None)
    if not tid2g:
        return dets, {}
    t = fk / float(fps) + _cam_delay(cam)
    out, gmap = [], {}
    for tid, p in dets:
        gid = tid2g.get(f"{cam}:{int(tid)}")
        if gid is None:
            out.append((tid, p))                    # 无归属：保留（不丢信息）
            continue
        oc = None
        for t0, t1, cm in idx.get(int(gid), []):
            if t0 <= t <= t1:
                oc = cm
                break
        if oc is None or oc == cam:                 # 被本机位负责，或此刻无 owner 记录
            out.append((tid, p))
            gmap[int(tid)] = int(gid)
    return out, gmap


def _arm_labels(tid, gmap: dict) -> list:
    return [f"g{gmap[int(x)]:04d}" if int(x) in gmap else f"#{int(x)}" for x in tid]


def _load_masks(cam: str) -> list:
    p = DATA / "masks.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    c = (d.get("cameras", {}) or {}).get(cam, {}) or {}
    return c.get("regions", []) if c.get("enabled") else []


def open_video(cam: str, root: Path | None = None):
    if cv2 is None:
        raise RuntimeError("cv2 不可用")
    root = root or MEDIA_ROOT
    # 代理目录里是 305.mp4，原生素材盘里是 305_long35.mp4 —— 同一函数按顺序试
    for name in (f"{cam}.mp4", f"{cam}_long35.mp4", f"{cam}.mkv", f"{cam}.avi"):
        p = root / name
        if p.exists():
            return cv2.VideoCapture(str(p)), str(p)
    raise FileNotFoundError(f"找不到 {cam} 的网页视频（{root}/{cam}.mp4）")


def read_frame(cap, frame: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    if not ok:
        return None
    return img


def _index_frames(tracks: dict):
    idx: dict[int, list] = {}
    for tid, pts in tracks.items():
        for p in pts:
            idx.setdefault(p[0], []).append((int(tid), p))
    return idx


_FRAME_CACHE: dict[str, dict] = {}


def frame_dets(cam: str, frame: int):
    t = _load_tracks(cam)
    stride = t.get("stride", 5)
    if cam not in _FRAME_CACHE:
        _FRAME_CACHE[cam] = _index_frames(t.get("tracks", {}))
    idx = _FRAME_CACHE[cam]
    fk = int(round(frame / stride) * stride)
    return fk, idx.get(fk, [])


def render_frame(cam: str, frame: int, view: str = "all", trail: int = 0,
                 show_masks: bool = True, jpeg_quality: int = 85,
                 arm: str | None = None) -> bytes:
    if cv2 is None:
        raise RuntimeError("cv2 不可用")
    status = _load_status(cam)
    t = _load_tracks(cam)
    W, H = t.get("w", 0), t.get("h", 0)
    fps = float(t.get("fps", 25.0) or 25.0)
    fk, dets_raw = frame_dets(cam, frame)

    cap, _ = open_video(cam)
    try:
        img = read_frame(cap, fk)
    finally:
        cap.release()
    if img is None:
        raise RuntimeError(f"读帧失败 cam={cam} f={frame}")

    # 代理视频可能被缩小，检测坐标是原始分辨率 → 按实际帧尺寸缩放
    sx = img.shape[1] / float(W) if W else 1.0
    sy = img.shape[0] / float(H) if H else 1.0

    dets_raw = _filter_view(dets_raw, status, view)
    gmap: dict = {}
    if arm:                                    # ★ 身份口径：同一人只画责任机位那一个框
        dets_raw, gmap = _apply_arm(cam, dets_raw, fk, load_arm(arm), fps)
    if dets_raw and HAS_SV:
        xyxy = np.array([[d[1][1] * sx, d[1][2] * sy, d[1][3] * sx, d[1][4] * sy]
                         for d in dets_raw], dtype=np.float32)
        conf = np.array([d[1][5] if d[1][5] is not None else 0.5 for d in dets_raw], dtype=np.float32)
        tid = np.array([d[0] for d in dets_raw], dtype=int)
        classes = np.zeros(len(xyxy), dtype=int)
        dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=classes, tracker_id=tid)
        colors = [sv.Color.from_hex(_track_color(x)) for x in tid]
        palette = sv.ColorPalette(colors=colors)
        box_ann = sv.BoxAnnotator(color=palette, color_lookup=sv.ColorLookup.INDEX, thickness=2)
        txt = _arm_labels(tid, gmap)
        label_ann = sv.LabelAnnotator(color=palette, color_lookup=sv.ColorLookup.INDEX,
                                      text_scale=0.6, text_thickness=1)
        img = box_ann.annotate(scene=img, detections=dets)
        img = label_ann.annotate(scene=img, detections=dets, labels=txt)
    elif dets_raw:
        for tid, p in dets_raw:
            _draw_box(img, p, (f"g{gmap[int(tid)]:04d}" if int(tid) in gmap else f"#{tid}"), sx, sy)

    if trail > 0:
        _draw_trails(img, cam, fk, status, view, trail, sx, sy,
                     arm_data=(load_arm(arm) if arm else None))

    if show_masks:
        for poly in _load_masks(cam):
            pts = np.array([[int(u * img.shape[1]), int(v * img.shape[0])] for u, v in poly], dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (60, 60, 220))
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.polylines(img, [pts], True, (40, 40, 200), 2)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def crop_thumb(cam: str, tid: int, frame: int | None = None, pad: float = 0.18,
               width: int = 192, jpeg_quality: int = 82, root: Path | None = None,
               mark: bool = False, scene: bool = False) -> bytes:
    """裁剪某条轨迹的人物缩略图（ReID/聚类工作台用）。

    frame=None 时取轨迹中点帧；返回 JPEG bytes。
    root 传 NATIVE_ROOT 时从 1440p 原生裁，宽输出不再是放大糊图。
    mark=True 在图上画出目标框 —— 外扩取「带环境」的宽画面时必须画：
      画面里往往站着别人，不标框就只能靠猜谁是被检索的那个人。
    scene=True 走「场景窗口」：固定 16:9、以人为中心，而不是按框的倍数外扩。
      ★ 按倍数外扩在细长的人形框上会退化成一根竖条（实测 849px 高的人 → 960×1382 竖图），
        既不像环境也没法看；场景窗口才稳定。
    """
    if cv2 is None:
        raise RuntimeError("cv2 不可用")
    t = _load_tracks(cam)
    pts = (t.get("tracks") or {}).get(str(tid)) or []
    if not pts:
        raise FileNotFoundError(f"轨迹不存在 {cam}#{tid}")
    W, H = t.get("w", 0), t.get("h", 0)
    stride = t.get("stride", 5) or 5
    if frame is None:
        frame = pts[len(pts) // 2][0]
    fk = int(round(frame / stride) * stride)
    best = min(pts, key=lambda p: abs(p[0] - fk))
    x1, y1, x2, y2 = best[1], best[2], best[3], best[4]

    cap, _ = open_video(cam, root)
    try:
        img = read_frame(cap, fk)
    finally:
        cap.release()
    if img is None:
        raise RuntimeError(f"读帧失败 {cam}@{fk}")
    sx, sy = img.shape[1] / float(W or 1), img.shape[0] / float(H or 1)
    bw, bh = (x2 - x1) * sx, (y2 - y1) * sy
    cx, cy = (x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy
    if scene:
        # 16:9 场景窗口：高度至少 600 px（远景小人也有环境），按人高 1.35 倍给下限
        ih, iw = img.shape[0], img.shape[1]
        ch = min(float(ih), max(bh * 1.35, 600.0))
        cw = min(float(iw), ch * 16.0 / 9.0)
        ch = min(ch, cw * 9.0 / 16.0)          # 撞到左右边界时按宽度反算，保持 16:9
        ax1 = int(max(0, min(iw - cw, cx - cw / 2)))
        ay1 = int(max(0, min(ih - ch, cy - ch / 2)))
        ax2, ay2 = int(ax1 + cw), int(ay1 + ch)
    else:
        hw, hh = bw * (0.5 + pad), bh * (0.5 + pad)
        ax1 = max(0, int(cx - hw)); ay1 = max(0, int(cy - hh))
        ax2 = min(img.shape[1], int(cx + hw)); ay2 = min(img.shape[0], int(cy + hh))
    if ax2 - ax1 < 4 or ay2 - ay1 < 4:
        raise RuntimeError("裁剪区域为空")
    crop = img[ay1:ay2, ax1:ax2]
    scale = width / crop.shape[1]
    crop = cv2.resize(crop, (width, max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    if mark:
        # 框换算到裁剪区域坐标，再乘缩放比；外扩宽画面里没有这个框就认不出是哪个人
        k = width / max(1.0, float(ax2 - ax1))
        bx1, by1 = int((x1 * sx - ax1) * k), int((y1 * sy - ay1) * k)
        bx2, by2 = int((x2 * sx - ax1) * k), int((y2 * sy - ay1) * k)
        th = max(2, width // 220)
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 200, 255), th)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def crop_point(cam: str, u: float, v: float, h: float, frame: int,
               width: int = 192, pad: float = 0.10, aspect: float = 0.45,
               jpeg_quality: int = 84):
    """按「点 + 高」裁人物图：u=框中心 x，v=框**底边**（脚点）y，h=框高。

    ★ 为什么不复用 crop_thumb：ReID 库（store_meta.json）里的 tid 与渲染用的
      `tracks/{cam}.json` **不同源** —— 实测同名 tid 的框高中位差 20–68%，
      按 tid 去查框会裁到别人。库里已经存了该轨自己的 (u, v, h_med, f_rep)，
      直接用它裁，就不会串人。
    ★ 优先读原生 1440p（NATIVE_ROOT），读不到再退回 960×540 代理并做坐标缩放。

    返回 JPEG bytes。
    """
    if cv2 is None:
        raise RuntimeError("cv2 不可用")
    h = float(h or 0.0)
    if h <= 1.0:
        raise ValueError(f"{cam} 的框高无效：{h}")
    cap = None
    img = None
    used_native = False
    try:
        cap, _ = open_video(cam, NATIVE_ROOT)
        used_native = True
    except Exception:
        cap, _ = open_video(cam, MEDIA_ROOT)
    try:
        img = read_frame(cap, int(frame))
    finally:
        cap.release()
    if img is None:
        raise RuntimeError(f"读帧失败 {cam}@{frame}")

    sx = sy = 1.0
    if not used_native:
        # 代理是 960×540，坐标来自原生分辨率 → 按轨表里的原生 W/H 换算
        try:
            t = _load_tracks(cam)
            nw, nh = float(t.get("w") or 0), float(t.get("h") or 0)
            if nw > 0 and nh > 0:
                sx, sy = img.shape[1] / nw, img.shape[0] / nh
        except Exception:
            pass
    H, W = img.shape[:2]
    bh = h * (1.0 + pad) * sy
    bw = max(4.0, h * aspect * sx)
    cx, cy = float(u) * sx, float(v) * sy - bh / 2.0      # v 是脚点 → 往上推半个框高
    x1 = int(max(0, min(W - 2, cx - bw / 2)))
    x2 = int(max(x1 + 2, min(W, cx + bw / 2)))
    y1 = int(max(0, min(H - 2, cy - bh / 2)))
    y2 = int(max(y1 + 2, min(H, cy + bh / 2)))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError("裁剪区域为空")
    scale = width / float(crop.shape[1])
    crop = cv2.resize(crop, (width, max(1, int(crop.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def crop_box(cam: str, box, frame: int, width: int = 192, pad: float = 0.12,
             aspect: float = 0.5, jpeg_quality: int = 84):
    """按一个已知框裁人物图（框是该帧的真实检测框，坐标与视频同分辨率）。

    box = (x1, y1, x2, y2)；pad 上下外扩比例；aspect = 裁出宽度 / 高度。
    ★ 不直接沿用框的宽高比：坐姿/桌面场景的检测框常是方的（实测 1:1），
      照框裁会得到一张「桌子加一个小人」的方图，缩略图里认不出人。
      固定按高度给宽度，人物才占满画面。
    """
    if cv2 is None:
        raise RuntimeError("cv2 不可用")
    x1, y1, x2, y2 = [float(x) for x in box]
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("框无效")
    cap = None
    img = None
    used_native = False
    try:
        cap, _ = open_video(cam, NATIVE_ROOT)
        used_native = True
    except Exception:
        cap, _ = open_video(cam, MEDIA_ROOT)
    try:
        img = read_frame(cap, int(frame))
    finally:
        cap.release()
    if img is None:
        raise RuntimeError(f"读帧失败 {cam}@{frame}")
    H, W = img.shape[:2]
    sx = sy = 1.0
    if not used_native:
        try:
            t = _load_tracks(cam)
            nw, nh = float(t.get("w") or 0), float(t.get("h") or 0)
            if nw > 0 and nh > 0:
                sx, sy = W / nw, H / nh
        except Exception:
            pass
    bw, bh = (x2 - x1) * sx, (y2 - y1) * sy
    cx = (x1 + x2) / 2 * sx
    cy = (y1 + y2) / 2 * sy
    bh = bh * (1.0 + pad)
    bw = max(4.0, bh * aspect)
    ax1 = int(max(0, min(W - 2, cx - bw / 2)))
    ax2 = int(max(ax1 + 2, min(W, cx + bw / 2)))
    ay1 = int(max(0, min(H - 2, cy - bh / 2)))
    ay2 = int(max(ay1 + 2, min(H, cy + bh / 2)))
    crop = img[ay1:ay2, ax1:ax2]
    if crop.size == 0:
        raise RuntimeError("裁剪区域为空")
    scale = width / float(crop.shape[1])
    crop = cv2.resize(crop, (width, max(1, int(crop.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def _filter_view(dets, status, view="all"):
    out = []
    for tid, p in dets:
        s = status.get(str(tid), "keep")
        if view == "all":
            out.append((tid, p))
        elif view == "clean" and s == "keep":
            out.append((tid, p))
        elif view == "reflection" and s == "reflection":
            out.append((tid, p))
        elif view == "masked" and s == "masked":
            out.append((tid, p))
    return out


def _draw_box(img, p, label, sx=1.0, sy=1.0):
    x1, y1, x2, y2 = [int(v) for v in (p[1] * sx, p[2] * sy, p[3] * sx, p[4] * sy)]
    cv2.rectangle(img, (x1, y1), (x2, y2), (180, 140, 60), 2)
    cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 140, 60), 1, cv2.LINE_AA)


def _draw_trails(img, cam, fk, status, view, trail, sx=1.0, sy=1.0, arm_data=None):
    t = _load_tracks(cam)
    idx = _FRAME_CACHE.get(cam) or _index_frames(t.get("tracks", {}))
    at = list(idx.get(fk, []))
    if arm_data:                                   # 轨迹线也按身份口径去重，避免与框不一致
        at, _ = _apply_arm(cam, at, fk, arm_data, float(t.get("fps", 25.0) or 25.0))
    active = {d[0] for d in _filter_view(at, status, view)}
    for tid in active:
        pts = t.get("tracks", {}).get(str(tid), [])
        tail = [p for p in pts if fk - trail * (t.get("stride", 5)) <= p[0] <= fk]
        if len(tail) < 2:
            continue
        px = [(int((p[1] + p[3]) / 2 * sx), int(p[4] * sy)) for p in tail]
        for i in range(1, len(px)):
            cv2.line(img, px[i - 1], px[i], (60, 200, 240), 2, cv2.LINE_AA)


def export_video(cam: str, f0: int, f1: int, view: str = "all", trail: int = 12,
                 fps: int = 25, name: str | None = None, arm: str | None = None):
    """后台把 [f0,f1] 区间用 supervision 叠加后写成 H.264 mp4。

    arm='A'|'B'|'H' 时按身份口径去重（同一人只画责任机位那一个框，标签换 g####）。
    """
    job_id = name or f"{cam}_{f0}_{f1}_{view}" + (f"_{arm}" if arm else "")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORT_DIR / f"{job_id}.mp4"

    def worker():
        try:
            status = _load_status(cam)
            t = _load_tracks(cam)
            W, H = t.get("w", 1920), t.get("h", 1080)
            cap, _ = open_video(cam)
            stride = t.get("stride", 5)
            probe = read_frame(cap, f0)
            if probe is None:
                raise RuntimeError("无法读取起始帧")
            ih, iw = probe.shape[:2]
            sx, sy = iw / float(W), ih / float(H)
            ff = subprocess.Popen(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{iw}x{ih}",
                 "-r", str(fps), "-i", "pipe:0",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
                stdin=subprocess.PIPE)
            total = max(1, (f1 - f0) // stride)
            done = 0
            for f in range(f0, f1 + 1, stride):
                img = read_frame(cap, f)
                if img is None:
                    break
                _, dets = frame_dets(cam, f)
                dets = _filter_view(dets, status, view)
                gmap: dict = {}
                if arm:
                    dets, gmap = _apply_arm(cam, dets, f, load_arm(arm), float(fps))
                if dets and HAS_SV:
                    xyxy = np.array([[d[1][1] * sx, d[1][2] * sy, d[1][3] * sx, d[1][4] * sy]
                                     for d in dets], dtype=np.float32)
                    conf = np.array([d[1][5] if d[1][5] is not None else 0.5 for d in dets], dtype=np.float32)
                    tid = np.array([d[0] for d in dets], dtype=int)
                    dd = sv.Detections(xyxy=xyxy, confidence=conf, class_id=np.zeros(len(xyxy), int), tracker_id=tid)
                    pal = sv.ColorPalette(colors=[sv.Color.from_hex(_track_color(x)) for x in tid])
                    img = sv.BoxAnnotator(color=pal, color_lookup=sv.ColorLookup.INDEX, thickness=2).annotate(img, dd)
                    img = sv.LabelAnnotator(color=pal, color_lookup=sv.ColorLookup.INDEX,
                                            text_scale=0.6, text_thickness=1).annotate(
                        img, dd, labels=_arm_labels(tid, gmap))
                if trail > 0:
                    _draw_trails(img, cam, f, status, view, trail, sx, sy,
                                 arm_data=(load_arm(arm) if arm else None))
                ff.stdin.write(img.tobytes())
                done += 1
                with _lock:
                    _jobs[job_id]["progress"] = round(done / total, 3)
            cap.release()
            ff.stdin.close()
            ff.wait()
            with _lock:
                _jobs[job_id].update({"status": "done", "path": str(out_path), "progress": 1.0})
        except Exception as e:  # pragma: no cover
            with _lock:
                _jobs[job_id].update({"status": "error", "error": str(e)})

    with _lock:
        _jobs[job_id] = {"status": "running", "progress": 0.0, "cam": cam, "f0": f0,
                         "f1": f1, "view": view, "arm": arm}
    threading.Thread(target=worker, daemon=True).start()
    return job_id, str(out_path)


def job_status(job_id: str) -> dict:
    with _lock:
        return dict(_jobs.get(job_id, {"status": "unknown"}))
