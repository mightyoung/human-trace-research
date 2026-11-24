#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SCRFD 封装安全修复 · 验收器材（2026-10-16）

背景：gpt-v2 研究包（docs/commit-015/gpt-v2-...）的 overlay/scrfd_dnn.py 修复了
旧版封装的两个缺陷；本机用真实 ONNX 实测复现后，已同步修复三个文件：
  tools/scrfd_dnn.py（cv2.dnn 版）、tools/gil_face_capture.py::Scrfd（onnxruntime 版）、
  tools/_scrfd_cv2dnn_probe.py::detect（探针）。
旧版缺陷：
  1) 竖幅输入（比约 1.05:1 更竖）越界崩溃（900×450 实测 broadcast 错误）；
  2) 16:9 输入（本数据集 6 路全部 2560×1440 / 1920×1080）x 坐标系统性 +2.27%。
版本来源：旧版备份 tools/*.bak_pre_scrfd_fix（修复前 md5 留档于运行报告）。

用法：
  python tools/scrfd_fix_check.py all       # 全部组
  python tools/scrfd_fix_check.py synth     # 合成网络几何不变式（不依赖真实权重）
  python tools/scrfd_fix_check.py real      # 真实 ONNX 冒烟（竖幅/16:9/素材）
  python tools/scrfd_fix_check.py legacy    # 旧版对照（复现崩溃 + x 偏差量化）
  python tools/scrfd_fix_check.py cross     # cv2 版 vs onnxruntime 版交叉校验
输出：控制台 PASS/FAIL/SKIP 汇总 + JSON 报告
  output_crosscam/analysis/face/scrfd_fix_check_report.json
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
MODEL = os.path.join(ROOT, "models", "face", "scrfd_2.5g_bnkps.onnx")
V318 = os.path.join(ROOT, "media-resource", "multi-cemra", "synced-10min",
                    "03_办公区2_318", "318_085500-090500.mp4")
BUS = os.path.join(ROOT, "testdata", "bus.jpg")
REPORT = os.path.join(ROOT, "output_crosscam", "analysis", "face",
                      "scrfd_fix_check_report.json")


def _load(path, name):
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def _contract(name, out, img_hw, expect_n=None):
    """输出契约：N×15、finite、x1<x2、y1<y2、界内。返回 detail。"""
    h, w = img_hw
    assert out.ndim == 2 and out.shape[1] == 15, f"{name}: shape {out.shape}"
    assert np.isfinite(out).all(), f"{name}: non-finite"
    if len(out):
        assert (out[:, 2] > out[:, 0]).all() and (out[:, 3] > out[:, 1]).all(), f"{name}: degenerate box"
        assert (out[:, [0, 2]] >= 0).all() and (out[:, [0, 2]] <= w).all(), f"{name}: x out of image"
        assert (out[:, [1, 3]] >= 0).all() and (out[:, [1, 3]] <= h).all(), f"{name}: y out of image"
        assert (out[:, 14] >= 0).all() and (out[:, 14] <= 1).all(), f"{name}: score range"
    if expect_n is not None:
        assert len(out) == expect_n, f"{name}: expect {expect_n} boxes got {len(out)}"
    return {"n": int(len(out))}


class FakeSCRFD:
    """合成 9 输出网络：把指定网格位置拉高分（不依赖真实权重，只验证几何）。"""

    def __init__(self, hits=()):
        # hits: (stride, row, col, anchor, score, box_dist)
        self.hits = list(hits)
        self.blob = None

    def getUnconnectedOutLayersNames(self):
        return [f"o{i}" for i in range(9)]

    def setInput(self, blob):
        self.blob = blob

    def forward(self, names):
        out = []
        for stride in (8, 16, 32):
            g = 640 // stride
            n = g * g * 2
            s = np.zeros((1, n, 1), np.float32)
            b = np.zeros((1, n, 4), np.float32)
            k = np.zeros((1, n, 10), np.float32)
            for (st, row, col, anchor, score, dist) in self.hits:
                if st != stride:
                    continue
                ix = (row * g + col) * 2 + anchor
                s[0, ix, 0] = score
                b[0, ix, :] = dist
            out += [s, b, k]
        return out


def run_synth(results):
    m = _load(os.path.join(TOOLS, "scrfd_dnn.py"), "scrfd_dnn_check")

    def s1():
        net = FakeSCRFD(hits=[(8, 20, 20, 0, 0.9, 2.0)])
        out = m.ScrfdDNN(net=net).detect(np.zeros((900, 450, 3), np.uint8))
        assert net.blob.shape == (1, 3, 640, 640), net.blob.shape
        _contract("S1", out, (900, 450), expect_n=1)
        exp = np.array([144, 144, 176, 176]) * 900 / 640
        assert np.allclose(out[0, :4], exp, atol=1e-3), (out[0, :4], exp)
        return {"box": out[0, :4].round(3).tolist()}

    def s2():
        net = FakeSCRFD(hits=[(8, 20, 20, 0, 0.9, 2.0)])
        out = m.ScrfdDNN(net=net).detect(np.zeros((333, 1000, 3), np.uint8))
        _contract("S2", out, (333, 1000), expect_n=1)
        exp = [144 / (640 / 1000), 144 / (213 / 333), 176 / (640 / 1000), 176 / (213 / 333)]
        assert np.allclose(out[0, :4], exp, atol=1e-2), (out[0, :4], exp)
        assert abs(out[0, 14] - 0.9) < 1e-6
        return {"box": [round(float(x), 3) for x in out[0, :4]]}

    def s3():
        net = FakeSCRFD(hits=[(8, 30, 10, 0, 0.9, 2.0),   # 中心 y=240 -> padding（nh=100）
                              (8, 5, 10, 0, 0.9, 1.0)])   # 中心 y=40  -> 有效
        out = m.ScrfdDNN(net=net).detect(np.zeros((300, 1920, 3), np.uint8))
        _contract("S3", out, (300, 1920), expect_n=1)
        exp = [216.0, 96.0, 264.0, 144.0]
        assert np.allclose(out[0, :4], exp, atol=1e-2), (out[0, :4], exp)
        return {"kept": out[0, :4].round(3).tolist(), "filtered_padding_center": True}

    def s4():
        class Bad(FakeSCRFD):
            def forward(self, names):
                return [np.zeros((1, 3, 7))]
        err = None
        try:
            m.ScrfdDNN(net=Bad()).detect(np.zeros((200, 200, 3), np.uint8))
        except ValueError as e:
            err = str(e)
        assert err, "bad output layout must raise ValueError"
        err2 = None
        try:
            m.ScrfdDNN(net=FakeSCRFD()).detect(np.zeros((0, 200, 3), np.uint8))
        except ValueError as e:
            err2 = str(e)
        assert err2, "empty image must raise"
        return {"errors": [err, err2]}

    def s5():
        class Empty(FakeSCRFD):
            def forward(self, names):
                return [a * 0 for a in super().forward(names)]
        out = m.ScrfdDNN(net=Empty()).detect(np.zeros((200, 200, 3), np.uint8))
        assert out.shape == (0, 15), out.shape
        return {"shape": list(out.shape)}

    for name, fn in [("S1 竖幅 900x450 几何与契约", s1), ("S2 横图 1000x333 x/y 分别还原", s2),
                     ("S3 padding 中心过滤", s3), ("S4 未知输出布局显式报错", s4),
                     ("S5 无检测返回 (0,15)", s5)]:
        try:
            results.append({"group": "synth", "name": name, "status": "PASS", "detail": fn()})
        except Exception as e:
            results.append({"group": "synth", "name": name, "status": "FAIL",
                            "detail": f"{type(e).__name__}: {e}"})


def _grab318(frame_idx=300):
    if not os.path.isfile(V318):
        return None
    import cv2
    cap = cv2.VideoCapture(V318)
    for _ in range(frame_idx):
        cap.grab()
    ok, fr = cap.retrieve()
    cap.release()
    return fr if ok else None


def run_real(results):
    import cv2
    m = _load(os.path.join(TOOLS, "scrfd_dnn.py"), "scrfd_dnn_real")
    if not os.path.isfile(MODEL):
        results.append({"group": "real", "name": "R0 模型存在", "status": "SKIP",
                        "detail": f"missing {MODEL}"})
        return
    det = m.ScrfdDNN(MODEL, thresh=0.35, nms=0.4)

    def r1():
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (900, 450, 3), dtype=np.uint8)
        out = det.detect(img)
        return _contract("R1", out, (900, 450))

    def r2():
        rng = np.random.default_rng(7)
        img = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
        out = det.detect(img)
        return _contract("R2", out, (1080, 1920))

    def r3():
        assert os.path.isfile(BUS), "bus.jpg missing"
        img = cv2.imread(BUS)
        out = det.detect(img)
        d = _contract("R3", out, img.shape[:2])
        assert d["n"] >= 1, "bus.jpg expects >=1 face"
        return d

    def r4():
        fr = _grab318()
        if fr is None:
            return None  # -> SKIP
        out = det.detect(fr)
        return _contract("R4", out, fr.shape[:2])

    for name, fn in [("R1 竖幅 900x450（旧版崩溃输入）", r1), ("R2 1080p 16:9 不崩溃", r2),
                     ("R3 真实素材 bus.jpg 检出", r3), ("R4 真实视频帧 318#300", r4)]:
        try:
            d = fn()
            if d is None:
                results.append({"group": "real", "name": name, "status": "SKIP", "detail": "素材缺失"})
            else:
                results.append({"group": "real", "name": name, "status": "PASS", "detail": d})
        except Exception as e:
            results.append({"group": "real", "name": name, "status": "FAIL",
                            "detail": f"{type(e).__name__}: {e}"})


def run_legacy(results):
    import cv2
    bak = os.path.join(TOOLS, "scrfd_dnn.py.bak_pre_scrfd_fix")
    if not (os.path.isfile(bak) and os.path.isfile(MODEL)):
        results.append({"group": "legacy", "name": "L0 旧版备份", "status": "SKIP",
                        "detail": "备份或模型缺失"})
        return
    old = _load(bak, "scrfd_dnn_legacy")
    new = _load(os.path.join(TOOLS, "scrfd_dnn.py"), "scrfd_dnn_new2")

    def l1():
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (900, 450, 3), dtype=np.uint8)
        err = None
        try:
            old.ScrfdDNN(MODEL, thresh=0.35).detect(img)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        assert err, "旧版应在此输入上崩溃（复现缺陷）"
        new.ScrfdDNN(MODEL, thresh=0.35).detect(img)  # 新版不崩
        return {"old_crash": err, "new_ok": True}

    def l2():
        fr = _grab318()
        if fr is None:
            return None
        o = old.ScrfdDNN(MODEL, thresh=0.35, nms=0.4).detect(fr)
        n = new.ScrfdDNN(MODEL, thresh=0.35, nms=0.4).detect(fr)
        if len(o) == 0 or len(n) == 0:
            return None
        j = int(np.argmin(np.abs(n[:, 14] - o[0, 14])))
        rel = float((n[j, 0] - o[0, 0]) / o[0, 0])
        # 理论 -2.27%（2560/1920 的 16:9 均为该值）；允许像素量化容差
        assert -0.035 < rel < -0.010, f"x 偏差 {rel:.4f} 不在预期区间"
        return {"old_x1": round(float(o[0, 0]), 1), "new_x1": round(float(n[j, 0]), 1),
                "relative_diff": round(rel, 4)}

    for name, fn in [("L1 旧版竖幅崩溃复现", l1), ("L2 旧版 16:9 x 偏差量化（318 帧）", l2)]:
        try:
            d = fn()
            if d is None:
                results.append({"group": "legacy", "name": name, "status": "SKIP", "detail": "素材/检出不足"})
            else:
                results.append({"group": "legacy", "name": name, "status": "PASS", "detail": d})
        except Exception as e:
            results.append({"group": "legacy", "name": name, "status": "FAIL",
                            "detail": f"{type(e).__name__}: {e}"})


def run_cross(results):
    import cv2
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        results.append({"group": "cross", "name": "C0 onnxruntime 可用", "status": "SKIP",
                        "detail": "未安装 onnxruntime"})
        return
    if not os.path.isfile(MODEL) or not os.path.isfile(BUS):
        results.append({"group": "cross", "name": "C0 依赖素材", "status": "SKIP", "detail": "模型或素材缺失"})
        return
    cv2v = _load(os.path.join(TOOLS, "scrfd_dnn.py"), "scrfd_dnn_x")
    sys.path.insert(0, TOOLS)
    capv = _load(os.path.join(TOOLS, "gil_face_capture.py"), "gil_face_capture_x")
    img = cv2.imread(BUS)
    try:
        a = cv2v.ScrfdDNN(MODEL, thresh=0.35).detect(img)
        b = capv.Scrfd(MODEL, thresh=0.35).detect(img)
        assert len(a) == len(b), f"数量不一致 cv2={len(a)} ort={len(b)}"
        if len(a):
            j = np.argsort(a[:, 14])[::-1]
            k = np.argsort(b[:, 14])[::-1]
            dmax = float(np.abs(a[j, :14] - b[k, :14]).max())
            assert dmax < 3.0, f"双后端最大坐标差 {dmax}"
        else:
            dmax = 0.0
        results.append({"group": "cross", "name": "C1 cv2 版 vs onnxruntime 版一致（bus.jpg）",
                        "status": "PASS", "detail": {"n_cv2": len(a), "n_ort": len(b), "max_coord_diff": round(dmax, 3)}})
    except Exception as e:
        results.append({"group": "cross", "name": "C1 cv2 版 vs onnxruntime 版一致（bus.jpg）",
                        "status": "FAIL", "detail": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("group", nargs="?", default="all",
                    choices=["all", "synth", "real", "legacy", "cross"])
    ap.add_argument("--out", default=REPORT)
    a = ap.parse_args()
    results = []
    if a.group in ("all", "synth"):
        run_synth(results)
    if a.group in ("all", "real"):
        run_real(results)
    if a.group in ("all", "legacy"):
        run_legacy(results)
    if a.group in ("all", "cross"):
        run_cross(results)
    n_pass = sum(r["status"] == "PASS" for r in results)
    n_fail = sum(r["status"] == "FAIL" for r in results)
    n_skip = sum(r["status"] == "SKIP" for r in results)
    for r in results:
        print(f'[{r["status"]}] {r["group"]:6s} {r["name"]}')
        if r["status"] != "PASS":
            print(f'        {r["detail"]}')
    print(f"\n== {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ==")
    report = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "group": a.group,
        "model": MODEL,
        "model_md5": _md5(MODEL) if os.path.isfile(MODEL) else None,
        "legacy_md5": _md5(os.path.join(TOOLS, "scrfd_dnn.py.bak_pre_scrfd_fix"))
        if os.path.isfile(os.path.join(TOOLS, "scrfd_dnn.py.bak_pre_scrfd_fix")) else None,
        "fixed_md5": _md5(os.path.join(TOOLS, "scrfd_dnn.py")),
        "results": results,
        "summary": {"pass": n_pass, "fail": n_fail, "skip": n_skip},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("report ->", a.out)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
