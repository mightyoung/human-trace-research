#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批次 A 验收器材（A2 / A4 / A5 合并；A1/A3 各有独立 selftest）。

用法（需 venv python：numpy/cv2/supervision）：
  python tools/batch_a_check.py freeze    # A2：冻结集复算（sha1/counts/覆盖数字 vs 产物）
  python tools/batch_a_check.py thr       # A4：validate_track_thresholds 组合矩阵（合法/非法）
  python tools/batch_a_check.py track     # A4：ByteTrack det_thresh 解耦（构造→覆写实证）
  python tools/batch_a_check.py embed     # A5：prep=v2 函数级冒烟（合成帧+mock cap+公式等价）
  python tools/batch_a_check.py all

只读被测工件与数据；写临时文件用系统 TEMP 并在 finally 清理。
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import traceback
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
RV2 = os.path.join(ANA, "relay_v2")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

_NP = [0]
_NF = [0]


def chk(name, cond, detail=""):
    if cond:
        _NP[0] += 1
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        _NF[0] += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
    return bool(cond)


# ---------------- A2：冻结集复算 ----------------
def check_freeze():
    print("== A2 freeze 复算 ==")
    fz = json.load(open(os.path.join(RV2, "frozen_eval_v1.json"), encoding="utf-8"))
    fair = json.load(open(os.path.join(RV2, "frozen_fair_v1.json"), encoding="utf-8"))

    rows = sorted([[p["a"], p["b"], p["v"]] for p in fz["pairs"]])
    sha = hashlib.sha1(json.dumps(rows, ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    chk("sha1_16 复算一致", sha == fz["sha1_16"], f"{sha}")
    chk("n 一致", len(fz["pairs"]) == fz["n"], f"n={fz['n']}")
    c = Counter(p["label"] for p in fz["pairs"])
    chk("counts 一致", {k: c.get(k, 0) for k in ("same", "diff", "unclear")} == fz["counts"],
        f"{dict(c)}")
    chk("字段完整", all({"a", "b", "label", "v", "src", "key"} <= set(p) for p in fz["pairs"]))

    g = json.load(open(os.path.join(RV2, "edges_v2_gated_human.json"), encoding="utf-8"))
    act = {(e["a"], e["b"]) for e in g["edges"] if e.get("status") == "active"}
    pool = {(e["a"], e["b"]) for e in g["edges"]
            if e.get("status") in ("active", "audit", "gate", "confirmed")}
    pos = {(p["a"], p["b"]) for p in fz["pairs"] if p["label"] == "same"}
    neg = {(p["a"], p["b"]) for p in fz["pairs"] if p["label"] == "diff"}

    s = json.load(open(os.path.join(RV2, "edge_states.json"), encoding="utf-8"))
    soft = set()
    for k, st in s["states"].items():
        if any(e.get("src") in ("human_pass", "human_fail") for e in st.get("evidence", [])):
            a, b = k.split("->")
            soft.add((a, b))
    adj = json.load(open(os.path.join(ANA, "gil", "human_adjudication_v2.json"), encoding="utf-8"))
    hard = set()
    for it in adj["items"]:
        if it.get("action") in ("confirm", "reject") and it.get("edge"):
            a, b = it["edge"].split("->")
            hard.add((a, b))
    assisted = soft | hard

    cov = fair["coverage"]
    chk("pos∩active 复算", len(pos & act) == cov["active"]["current"]["n"],
        f"{len(pos & act)}")
    chk("fair(active) 复算", len(pos & (act - assisted)) == cov["active"]["fair"]["n"],
        f"{len(pos & (act - assisted))}")
    chk("pos∩pool 复算", len(pos & pool) == cov["production_pool"]["current"]["n"],
        f"{len(pos & pool)}")
    chk("fair(pool) 复算", len(pos & (pool - assisted)) == cov["production_pool"]["fair"]["n"],
        f"{len(pos & (pool - assisted))}")
    chk("一致性 neg∩active==0", len(neg & act) == 0, f"{len(neg & act)}")


# ---------------- A4：阈值矩阵 ----------------
def _extract_func(path, name):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            ns = {"np": np}
            exec(seg, ns)
            return ns[name]
    raise RuntimeError(f"{name} 未在 {path} 中找到")


def check_thr():
    print("== A4 阈值矩阵（AST 提取 validate_track_thresholds，避免重依赖）==")
    fn = _extract_func(os.path.join(TOOLS, "infer_supervision.py"), "validate_track_thresholds")
    legal = [
        ((0.15, 0.25, 0.30), "标准组合"),
        ((0.11, 0.11, 0.11), "三值相等（>0.1）"),
        ((0.2, 0.5, 0.9), "宽区间"),
        ((0.1 + 1e-9, 0.25, 0.30), "floor 紧贴 0.1 边界"),
        ((0.3, 0.3, 0.3), "全 0.3"),
    ]
    illegal = [
        ((0.10, 0.25, 0.30), "floor=0.1（低分池下界冲突）"),
        ((0.26, 0.25, 0.30), "high<floor"),
        ((0.15, 0.31, 0.30), "new<high"),
        ((0.15, 0.25, 1.01), ">1"),
        ((float("nan"), 0.25, 0.30), "NaN"),
        (("0.15", 0.25, 0.30), "字符串"),
        ((0.15, -0.1, 0.30), "负值"),
        ((0.15, 0.25, 0), "0 超界"),
    ]
    ok = 0
    for args, tag in legal:
        try:
            fn(*args)
            ok += 1
        except Exception as e:
            print(f"    合法组合误报: {args} {tag} → {e}")
    chk("合法组合全部通过", ok == len(legal), f"{ok}/{len(legal)}")
    ok = 0
    for args, tag in illegal:
        try:
            fn(*args)
            print(f"    非法组合漏报: {args} {tag}")
        except ValueError:
            ok += 1
        except Exception as e:
            print(f"    异常类型不符: {args} {tag} → {type(e).__name__}")
    chk("非法组合全部 ValueError", ok == len(illegal), f"{ok}/{len(illegal)}")


# ---------------- A4：tracker 解耦 ----------------
def check_track():
    print("== A4 ByteTrack det_thresh 解耦 ==")
    try:
        import supervision as sv
    except Exception as e:
        chk("supervision 可导入", False, f"{e}")
        return
    tr = sv.ByteTrack(track_activation_threshold=0.25)
    chk("版本留痕", True, f"supervision {getattr(sv, '__version__', '?')}")
    if not chk("hasattr(tracker,'det_thresh')", hasattr(tr, "det_thresh")):
        return
    d0 = float(tr.det_thresh)
    chk("默认耦合值 0.25+0.1=0.35", abs(d0 - 0.35) < 1e-9, f"det_thresh={d0}")
    tr.det_thresh = 0.30
    chk("覆写解耦生效 0.30", abs(float(tr.det_thresh) - 0.30) < 1e-9,
        f"det_thresh={float(tr.det_thresh)}")


# ---------------- A5：embed prep 冒烟 ----------------
class _FakeCap:
    """合成视频流：64 帧 640x480，矩形块随帧右移（确定性）。"""

    def __init__(self, n=64, w=640, h=480):
        import cv2
        self._cv2 = cv2
        self.n, self.w, self.h = n, w, h
        self.cur = 0

    def get(self, prop):
        cv2 = self._cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.h)
        return 0.0

    def set(self, prop, val):
        if prop == self._cv2.CAP_PROP_POS_FRAMES:
            self.cur = int(val)
            return True
        return False

    def read(self):
        import numpy as np
        cv2 = self._cv2
        f = self.cur
        if f >= self.n:
            return False, None
        img = np.full((self.h, self.w, 3), 60, np.uint8)
        cx = 120 + f * 3
        cv2.rectangle(img, (cx, 150), (cx + 80, 400), (200, 180, 160), -1)
        cv2.circle(img, (cx + 40, 190), 22, (120, 140, 170), -1)  # 头部（纹理）
        img = cv2.GaussianBlur(img, (3, 3), 0)
        self.cur = f + 1
        return True, img


def _load_embed_server():
    path = os.path.join(TOOLS, "embed_server.py")
    spec = importlib.util.spec_from_file_location("embed_server_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_embed():
    print("== A5 prep=v2 函数级冒烟 ==")
    try:
        mod = _load_embed_server()
        chk("embed_server 导入", True)
    except Exception as e:
        chk("embed_server 导入", False, f"{type(e).__name__}: {e}")
        return

    frames = list(range(0, 64, 2))
    tf3 = [(f, [120 + f * 3, 150, 200 + f * 3, 400], 0.9 - f * 0.001) for f in frames]
    tf2 = [(f, b) for f, b, _ in tf3]
    fb = {f: [[120 + f * 3, 150, 200 + f * 3, 400]] for f in frames}

    # legacy 抽样（基线形态）
    cap = _FakeCap()
    try:
        cl = mod._sample_crops(cap, tf2, 8)
        chk("legacy 采样非空", len(cl) >= 4, f"n={len(cl)}")
        chk("legacy crop 224²", all(c.shape == (224, 224, 3) for c in cl))
    except Exception as e:
        chk("legacy 采样", False, f"{type(e).__name__}: {e}")

    # v2 采样
    cap = _FakeCap()
    try:
        crops, ws = mod._sample_crops_v2(cap, tf3, fb, samples=8, min_gap=10)
        chk("v2 采样非空", len(crops) >= 4, f"n={len(crops)}")
        chk("v2 crop 224²", all(c.shape == (224, 224, 3) for c in crops))
        chk("v2 权重 >0 有限", len(ws) == len(crops) and all(np.isfinite(w) and w > 0 for w in ws),
            f"w=[{min(ws):.3f},{max(ws):.3f}]" if ws else "空")
        crops2, ws2 = mod._sample_crops_v2(cap, tf3, fb, samples=8, min_gap=10)
        same = (len(crops) == len(crops2) and np.array_equal(np.asarray(crops), np.asarray(crops2))
                and np.allclose(ws, ws2))
        chk("v2 同输入两次逐位一致（确定性）", same)
    except Exception as e:
        chk("v2 采样", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[:300]}")

    # _load_tracks_v2 解析
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(tmp, "w", encoding="utf-8") as f:
            for i, fr in enumerate(frames[:6]):
                f.write(json.dumps({"frame": fr,
                                    "boxes": [[10 + i, 10, 60 + i, 110], [100, 20, 140, 120]],
                                    "track_ids": [1, 2],
                                    "scores": [0.9 - i * 0.01, 0.8]}) + "\n")
        t, fb2 = mod._load_tracks_v2(tmp)
        chk("tracks 解析", set(t.keys()) == {1, 2} and len(t[1]) == 6, f"keys={sorted(t.keys())}")
        chk("frame_boxes 解析", len(fb2.get(frames[0], [])) == 2)
        chk("分数三元组保留", abs(t[1][0][2] - 0.9) < 1e-9)
    except Exception as e:
        chk("_load_tracks_v2", False, f"{type(e).__name__}: {e}")
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

    # 聚合公式等价性：加权和（实现）≡ 加权平均（显式归一），L2 归一后逐位一致
    rng = np.random.default_rng(7)
    E = rng.normal(size=(6, 16)).astype(np.float32)
    w = rng.uniform(0.05, 1.0, 6).astype(np.float32)[:, None]
    m1 = (E * w).sum(axis=0)
    m1 = m1 / (np.linalg.norm(m1) + 1e-9)
    m2 = (E * w).sum(axis=0) / max(float(w.sum()), 1e-9)
    m2 = m2 / (np.linalg.norm(m2) + 1e-9)
    chk("加权和≡加权平均（L2 归一后）", np.allclose(m1, m2, atol=1e-6),
        f"max|Δ|={np.abs(m1 - m2).max():.2e}")


def main():
    ap = argparse.ArgumentParser(description="批次 A 验收器材（A2/A4/A5）")
    ap.add_argument("cmd", choices=["freeze", "thr", "track", "embed", "all"])
    args = ap.parse_args()
    if args.cmd in ("freeze", "all"):
        check_freeze()
    if args.cmd in ("thr", "all"):
        check_thr()
    if args.cmd in ("track", "all"):
        check_track()
    if args.cmd in ("embed", "all"):
        check_embed()
    print(f"\n=== 结果：{_NP[0]}/{_NP[0] + _NF[0]} 通过，{_NF[0]} 失败 ===")
    sys.exit(1 if _NF[0] else 0)


if __name__ == "__main__":
    main()
