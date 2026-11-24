# -*- coding: utf-8 -*-
"""数据视图开关：raw（原始）| clean（清洗集）| clean2（修正清洗集）—— 清洗数据与实体图加载的**唯一入口**。

清洗集（cleanset）由 `tools/build_cleanset.py` 离线生成一次：
  output_crosscam/cleanset/{cam}_detections.jsonl
  316/323 = dualclean（玻璃掩膜 + 运动学去倒影）；305/318/320/343 = clean（仅运动学）
  清单（规则/版本/md5/删除轨迹）见 output_crosscam/cleanset/clean_manifest.json

开关（环境变量）：
  DET_VIEW = raw | clean | clean2   （默认 raw，向后兼容）
    clean  = cleanset/（v1，含未同步字段 + 未恢复误删）
    clean2 = cleanset_v2/（修正版：逐框字段同步 + 恢复 7 条误删真人）
  DET_DATA = 远程数据根目录（默认 /share/data4/yhh/data/human-count）

用法（本地脚本）：
  from det_view import local_det, describe
  print(describe())
  tracks = load_tracks(local_det(cam))

用法（远程脚本，det_view.py 需与脚本同目录或 sys.path 可达）：
  from det_view import remote_det
  tracks = load_tracks(remote_det(cam, DATA))    # DET_VIEW=clean → inference-out-clean/

注意：视图是"数据口径"的一部分——实验记录必须写明 DET_VIEW（数据纪律 §三线数据）。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = "/share/data4/yhh/data/human-count"
VALID = ("raw", "clean", "clean2")

# 视图 → (本地子目录, 远程子目录)；clean2 = 修正版清洗集（字段同步 + 误删恢复，见 build_cleanset_v2）
VIEW_DIRS = {
    "raw":    ("", "inference-out"),
    "clean":  ("cleanset", "inference-out-clean"),
    "clean2": ("cleanset_v2", "inference-out-clean2"),
}

MANIFEST_PATH = os.path.join(ROOT, "output_crosscam", "cleanset", "clean_manifest.json")
MANIFEST_PATH_V2 = os.path.join(ROOT, "output_crosscam", "cleanset_v2", "clean_manifest_v2.json")


def current_view():
    v = (os.environ.get("DET_VIEW", "raw") or "raw").strip().lower()
    if v not in VALID:
        raise ValueError(f"DET_VIEW={v!r} 无效（应为 raw|clean）")
    return v


def _data_root() -> str:
    """检测/清洗集根目录。默认 output_crosscam；换片段（如 long35）用 DET_ROOT 指过去。

    ★ 与 DET_VIEW 正交：DET_ROOT 说"哪一段数据"，DET_VIEW 说"什么清洗口径"。
    """
    return os.environ.get("DET_ROOT") or os.path.join(ROOT, "output_crosscam")


def local_det(cam, view=None):
    """本地检测源：raw → <root>/；clean → cleanset/；clean2 → cleanset_v2/"""
    view = view or current_view()
    root = _data_root()
    sub = VIEW_DIRS[view][0]
    if not sub:
        return os.path.join(root, f"{cam}_detections.jsonl")
    return os.path.join(root, sub, f"{cam}_detections.jsonl")


def remote_det(cam, data_root=None, view=None):
    """远程检测源：raw → inference-out/；clean → inference-out-clean/；clean2 → inference-out-clean2/"""
    view = view or current_view()
    data_root = data_root or os.environ.get("DET_DATA", DEFAULT_DATA)
    return f"{data_root}/{VIEW_DIRS[view][1]}/{cam}_detections.jsonl"


def manifest():
    """读清洗集清单：优先本地 cleanset(_v2)/，其次远程对应目录。"""
    v = current_view()
    local = MANIFEST_PATH_V2 if v == "clean2" else MANIFEST_PATH
    remote = os.path.join(os.environ.get("DET_DATA", DEFAULT_DATA),
                          VIEW_DIRS[v][1], "clean_manifest_v2.json" if v == "clean2"
                          else "clean_manifest.json")
    for p in (local, remote):
        if os.path.exists(p):
            try:
                return json.load(open(p, encoding="utf-8"))
            except Exception:
                pass
    return {}


def removed_tids(cam):
    """该机位被运动学清洗剔除的轨迹 id 集合（审计/断言用；raw 视图返回空集）。"""
    if current_view() == "raw":
        return set()
    rec = manifest().get("cameras", {}).get(cam, {})
    return set(rec.get("removed_mirror_tracks", []))


def describe():
    """一行视图描述（脚本启动时打印，防跑错口径）。"""
    v = current_view()
    if v == "raw":
        return "[data-view] DET_VIEW=raw（原始检测，含倒影）"
    m = manifest()
    grades = {c: r.get("grade", "?") for c, r in m.get("cameras", {}).items()}
    gs = "/".join(f"{c}:{g}" for c, g in grades.items())
    return f"[data-view] DET_VIEW={v}（清洗集 {m.get('version', '?')}；{gs}）"


# ---------------------------------------------------------------------------
# 实体图（net_tracks）目录解析 —— 与检测视图**必须配套**，否则出现"口径分裂"
#   历史缺陷（详见 docs/clean2-rebuild-report.md）：实体图用无掩膜口径构建，
#   而生产/评测用掩膜后（dualclean）口径 → 492 实体中 88 个引用 clean 不存在的轨迹。
#   2026-10-14 起 `stitch_clean2` 为**规范实体图**（411 实体、悬空 0）。
# 开关：
#   NET_DIR = 目录名（相对 output_crosscam/analysis）或绝对路径；不设则用规范目录
# ---------------------------------------------------------------------------
CANONICAL_NET_DIR = "stitch_clean2"
LEGACY_NET_DIR = "stitch"
# 检测视图 → 与之配套的实体图目录（措辞：实体图必须在同一口径上重建）
VIEW_NET_DIRS = {
    "clean2": CANONICAL_NET_DIR,
    "clean":  LEGACY_NET_DIR,
    "raw":    LEGACY_NET_DIR,
}


def net_dir(cam=None):
    """实体图目录。默认 = 规范目录 stitch_clean2（2026-10-14 起的干净口径）。
    可用 NET_DIR 覆盖（相对名 → output_crosscam/analysis/<名>；绝对路径原样使用）。"""
    env = os.environ.get("NET_DIR")
    if env:
        return env if os.path.isabs(env) else os.path.join(
            ROOT, "output_crosscam", "analysis", env)
    return os.path.join(ROOT, "output_crosscam", "analysis", CANONICAL_NET_DIR)


def net_tracks_path(cam, d=None):
    """某机位的 net_tracks_{cam}.json 路径。"""
    return os.path.join(d or net_dir(), f"net_tracks_{cam}.json")


def net_consistency():
    """(ok, msg)：检测视图与实体图目录是否配套。不配套时返回 False + 说明。"""
    v = current_view()
    want = VIEW_NET_DIRS.get(v)
    got = os.path.basename(net_dir().rstrip("/\\"))
    if want and got != want:
        return False, (f"[net-view] 口径不配套：DET_VIEW={v} 期望实体图 {want}，"
                       f"实际 {got}（用 NET_DIR 或 DET_VIEW 对齐）")
    return True, f"[net-view] 实体图 {got} 与 DET_VIEW={v} 配套"

