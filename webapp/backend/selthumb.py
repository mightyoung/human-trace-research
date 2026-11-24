"""身份选择器的轨迹缩略图（按几何匹配到渲染轨，再用真实框裁）。

为什么不是「按 tid 查 tracks」：
  库（store_meta.json）与渲染轨 `data/tracks/{cam}.json` **轨号不同源** ——
  实测 305 只有 3/106 的 tid 相同；直接按 tid 查框会裁到别人。

为什么不只用库里的 (u,v,h_med)：
  u/v 记的是**最后一帧**的框底，h_med 却是整条轨的**中位高**；人在轨内走近/走远时
  两者不自洽（远景轨尤其明显，实测裁出来人贴在画面边缘）。

这里的做法（一次性算好、落盘缓存）：
  1. 用「时刻 + 位置」把库轨匹配到渲染轨：在 t_last 附近取有检测的帧，
     找框中心离 (u,v) 最近、且时间跨度有交集的那条 —— 实测 97–100% 命中，
     归一化距离中位 0.000（两条流水线对同一个人的框是同一组坐标，只是编号不同）；
  2. 匹配上就选该轨**框最大且不贴边**的那一帧，用真实框裁；
  3. 匹配不上（<3%）退回库内 (u,v,h_med,f_rep)。

产物：
  data/reid/thumbs/{cam}_{tid}.jpg        宽 448
  data/reid/thumb_track_map.json          库 tid → 渲染 tid 的对照（复核用）
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

import render

REID = Path(__file__).resolve().parent / "data" / "reid"
TRACKS = Path(__file__).resolve().parent / "data" / "tracks"
THUMB_DIR = REID / "thumbs"
MAP_PATH = REID / "thumb_track_map.json"
W = 448          # 缓存宽度：页面用 96–128，点开放大时按原尺寸显示，448 够清楚
_lock = threading.Lock()
_tracks_cache: dict[str, dict] = {}
_index_cache: dict[str, tuple[dict, dict]] = {}
_map_cache: dict[str, dict] | None = None
_resize_cache: dict[tuple[str, int], bytes] = {}

# 库里的 t 与帧的关系：t = offset + frame/fps（OSD 标定值，见 tools/global_reid_store.py）
STORE_OFF = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}


def path_of(cam: str, tid: int) -> Path:
    return THUMB_DIR / f"{cam}_{int(tid)}.jpg"


def _tracks(cam: str) -> dict:
    if cam not in _tracks_cache:
        _tracks_cache[cam] = json.loads((TRACKS / f"{cam}.json").read_text(encoding="utf-8"))
    return _tracks_cache[cam]


def _index(cam: str):
    """(frame -> [(tid, box)]) + (tid -> (f0, f1))，建一次复用。"""
    if cam in _index_cache:
        return _index_cache[cam]
    tj = _tracks(cam)
    frame_idx: dict[int, list] = {}
    spans: dict[int, tuple[int, int]] = {}
    for tid, pts in (tj.get("tracks") or {}).items():
        if not pts:
            continue
        t = int(tid)
        spans[t] = (pts[0][0], pts[-1][0])
        for p in pts:
            frame_idx.setdefault(int(p[0]), []).append((t, (p[1], p[2], p[3], p[4])))
    _index_cache[cam] = (frame_idx, spans)
    return _index_cache[cam]


def _load_map() -> dict:
    global _map_cache
    if _map_cache is None:
        try:
            _map_cache = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        except Exception:
            _map_cache = {}
    return _map_cache


def _save_map() -> None:
    if _map_cache is None:
        return
    with _lock:
        MAP_PATH.write_text(json.dumps(_map_cache, ensure_ascii=False), encoding="utf-8")


def _pick_frame(pts, lo=None, hi=None):
    """在渲染轨里挑一帧：框最大、不贴边，且**优先落在库轨自己的时间窗内**。

    限定时间窗是为了让缩略图反映这条库轨当时的样貌 —— 渲染轨可能比库轨长得多
    （同一人坐一整天），跨窗取帧会拿到几小时后的画面。
    """
    if not pts:
        return None
    pool = pts
    if lo is not None and hi is not None:
        win = [p for p in pts if lo <= p[0] <= hi]
        if len(win) >= 1:
            pool = win
    cand = []
    for p in pool:
        x1, y1, x2, y2 = p[1], p[2], p[3], p[4]
        h = y2 - y1
        if h <= 6 or x1 <= 2 or y1 <= 2:
            continue
        cand.append((h, p))
    if not cand:
        cand = [(p[4] - p[2], p) for p in pool if p[4] - p[2] > 6]
    if not cand:
        return None
    cand.sort(key=lambda x: -x[0])
    # 取接近最大但不极端的一档（第 10% 位置），避开半身出画/被遮挡的极端帧
    k = min(len(cand) - 1, max(0, int(len(cand) * 0.10)))
    p = cand[k][1]
    return (p[1], p[2], p[3], p[4]), int(p[0])


def _window(sel, cam: str, i: int):
    """库轨的 [t0, t1] → 该机位视频的帧区间（略放宽 1.5 s）。"""
    t = sel.meta[int(i)]
    fps = float(_tracks(cam).get("fps") or 25.0)
    off = STORE_OFF.get(cam, 0.0)
    lo = int(round((float(t.get("t0") or 0.0) - off) * fps)) - int(1.5 * fps)
    hi = int(round((float(t.get("t1") or 0.0) - off) * fps)) + int(1.5 * fps)
    return lo, hi


def match_track(sel, cam: str, tid: int):
    """库轨 → (rtid, (box, frame))；匹配不上返回 None。"""
    i = sel.idx_of(cam, tid)
    if i is None:
        return None
    key = f"{cam}:{int(tid)}"
    hist = _tracks(cam).get("tracks") or {}
    lo, hi = _window(sel, cam, i)
    m = _load_map()
    if key in m:
        rtid = int(m[key])
        got = _pick_frame(hist.get(str(rtid)) or [], lo, hi)
        return (rtid, got) if got else None

    t = sel.meta[int(i)]
    u, v = float(t.get("u") or 0.0), float(t.get("v") or 0.0)
    hm = max(20.0, float(t.get("h_med") or 0.0))
    tj = _tracks(cam)
    fps = float(tj.get("fps") or 25.0)
    stride = int(tj.get("stride") or 5)
    cap = int(tj.get("nframes") or 0)
    frame_idx, spans = _index(cam)
    off = STORE_OFF.get(cam, 0.0)

    t0, t1 = float(t.get("t0") or 0.0), float(t.get("t1") or 0.0)
    f_last = int(round((t1 - off) * fps))
    if cap:
        f_last = max(0, min(cap - 1, f_last))
    f_lo = int(round((t0 - off) * fps)) - int(3 * fps)
    f_hi = int(round((t1 - off) * fps)) + int(3 * fps)

    best = None
    for df in range(0, 11):
        for f in (f_last + df * stride, f_last - df * stride):
            fk = int(round(f / stride) * stride)
            for rtid, box in frame_idx.get(fk, ()):
                s0, s1 = spans[rtid]
                if s0 > f_hi or s1 < f_lo:      # 时间无交集 → 不可能是同一人
                    continue
                x1, y1, x2, y2 = box
                d = ((((x1 + x2) / 2) - u) ** 2 + (y2 - v) ** 2) ** 0.5 / hm
                if best is None or d < best[0]:
                    best = (d, int(rtid))
        if best and best[0] < 0.10:
            break

    if not best or best[0] > 0.6:
        return None
    rtid = best[1]
    got = _pick_frame(hist.get(str(rtid)) or [], lo, hi)
    if not got:
        return None
    m[key] = rtid
    _save_map()
    return rtid, got


def ensure(sel, cam: str, tid: int, force: bool = False) -> Path:
    """返回缩略图路径；缺失则现裁一张落盘。sel = selector.Selector 实例。"""
    p = path_of(cam, tid)
    if p.exists() and not force and p.stat().st_size > 512:
        return p
    i = sel.idx_of(cam, tid)
    if i is None:
        raise KeyError(f"{cam}#{tid} 不在 ReID 库")
    t = sel.meta[int(i)]

    got = match_track(sel, cam, tid)
    if got:
        _rtid, (box, frame) = got
        buf = render.crop_box(cam, box, frame, width=W, pad=0.10)
    else:
        buf = render.crop_point(cam, t.get("u") or 0.0, t.get("v") or 0.0,
                                t.get("h_med") or 0.0, int(t.get("f_rep") or 0),
                                width=W)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        p.write_bytes(buf)
    return p


def resize_bytes(p: Path, width: int) -> bytes:
    """按请求宽度返回 JPEG；width>=缓存宽度时原样返回。

    ★ 页面一次会拉几百张（查询轨条 216 个），缩放到 96 宽后要解码+重编码；
      加一层进程内缓存，避免每次滚动都重算。
    """
    if width >= W or render.cv2 is None:
        return p.read_bytes()
    key = (str(p), int(width))
    hit = _resize_cache.get(key)
    if hit is not None:
        return hit
    raw = p.read_bytes()
    arr = render.cv2.imdecode(np.frombuffer(raw, np.uint8), render.cv2.IMREAD_COLOR)
    if arr is None:
        return raw
    scale = width / float(arr.shape[1])
    arr = render.cv2.resize(arr, (width, max(1, int(arr.shape[0] * scale))),
                            interpolation=render.cv2.INTER_AREA)
    ok, buf = render.cv2.imencode(".jpg", arr, [int(render.cv2.IMWRITE_JPEG_QUALITY), 84])
    out = buf.tobytes() if ok else raw
    if len(_resize_cache) > 4000:
        _resize_cache.clear()
    _resize_cache[key] = out
    return out
