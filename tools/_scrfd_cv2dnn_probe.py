# -*- coding: utf-8 -*-
"""探针：用 cv2.dnn 跑 SCRFD（绕开 onnxruntime 依赖）。在远程 rfdetr 环境运行。"""
import sys
import time

import cv2
import numpy as np

MODEL = "/home/r4900g3/yhh/human-count/models/face/scrfd_2.5g_bnkps.onnx"
VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/home/r4900g3/yhh/human-count/crosscam/in/318.mp4"


def build(thresh=0.5, nms=0.4):
    net = cv2.dnn.readNetFromONNX(MODEL)
    return net, thresh, nms


def detect(net, img, thresh=0.5, nms=0.4):
    """★ 2026-10-16 修复（同 tools/scrfd_dnn.py）：旧版竖幅越界崩溃 + 16:9 输入
    x 坐标偏 +2.27%（统一除 det_scale 所致）。现改为长短边联合 fit、x/y 分别
    还原、padding 中心过滤、输出契约显式校验。历史数据按旧坐标生成不重跑。"""
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 \
            or img.shape[2] != 3 or min(img.shape[:2]) < 1:
        raise ValueError('expected nonempty uint8 BGR HWC image')
    h, w = img.shape[:2]
    scale = min(640.0 / w, 640.0 / h)
    nw = max(1, min(640, round(w * scale))); nh = max(1, min(640, round(h * scale)))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((640, 640, 3), np.uint8); canvas[:nh, :nw] = im
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (640, 640),
                                 (127.5, 127.5, 127.5), swapRB=True)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    grouped = {}
    for raw in outs:
        a = np.asarray(raw)
        if a.ndim == 3 and a.shape[0] == 1: a = a[0]
        if a.ndim != 2 or a.shape[1] not in (1, 4, 10) or not np.isfinite(a).all():
            raise ValueError(f'unsupported/nonfinite SCRFD output {a.shape}')
        key = (a.shape[0], a.shape[1])
        if key in grouped: raise ValueError('duplicate SCRFD output shape')
        grouped[key] = a.astype(np.float32)
    expected = {(2 * (640 // s) ** 2, c) for s in (8, 16, 32) for c in (1, 4, 10)}
    if set(grouped) != expected: raise ValueError('SCRFD export must provide 3 score/box/5KPS scales')
    all_b, all_k, all_s = [], [], []
    for stride in (8, 16, 32):
        g = 640 // stride; n = 2 * g * g
        s = grouped[n, 1].reshape(-1); b = grouped[n, 4]; k = grouped[n, 10]
        yy, xx = np.mgrid[:g, :g]
        centers = np.repeat(np.column_stack((xx.reshape(-1), yy.reshape(-1))) * stride, 2, axis=0).astype(np.float32)
        pos = np.flatnonzero(s > thresh)
        if pos.size == 0: continue
        d = b[pos] * stride; c = centers[pos]
        cb = np.column_stack((c[:, 0] - d[:, 0], c[:, 1] - d[:, 1],
                              c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]))
        ck = k[pos].reshape(-1, 5, 2) * stride + c[:, None, :]
        real = (c[:, 0] < nw) & (c[:, 1] < nh)
        all_b.append(cb[real]); all_k.append(ck[real].reshape(-1, 10)); all_s.append(s[pos][real])
    if not all_b:
        return np.zeros((0, 15), np.float32)
    B = np.concatenate(all_b); K = np.concatenate(all_k); S = np.concatenate(all_s)
    B[:, [0, 2]] /= (nw / w); B[:, [1, 3]] /= (nh / h)
    K[:, 0::2] /= (nw / w); K[:, 1::2] /= (nh / h)
    B[:, [0, 2]] = np.clip(B[:, [0, 2]], 0, w); B[:, [1, 3]] = np.clip(B[:, [1, 3]], 0, h)
    keep = (B[:, 2] > B[:, 0]) & (B[:, 3] > B[:, 1])
    B, K, S = B[keep], K[keep], S[keep]
    if len(B) == 0:
        return np.zeros((0, 15), np.float32)
    xywh = np.stack([B[:, 0], B[:, 1], B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]], 1)
    idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), S.tolist(), thresh, nms)).reshape(-1)
    if len(idx) == 0:
        return np.zeros((0, 15), np.float32)
    out = np.zeros((len(idx), 15), np.float32)
    for j, i in enumerate(idx):
        out[j, :4] = B[i]; out[j, 4:14] = K[i]; out[j, 14] = S[i]
    return out


def main():
    net, th, nm = build()
    print("[dnn] model loaded, out layers:", net.getUnconnectedOutLayersNames(), flush=True)
    cap = cv2.VideoCapture(VIDEO)
    n = 0; nf = 0; t0 = time.time()
    while n < 200:
        ok, fr = cap.read()
        if not ok:
            break
        if n % 5 == 0:
            d = detect(net, fr, th, nm)
            nf += len(d)
        n += 1
    cap.release()
    dt = time.time() - t0
    print(f"[dnn] frames={n} face_boxes={nf} elapsed={dt:.1f}s "
          f"({1000*dt/max(1,n//5):.1f} ms/frame)", flush=True)


if __name__ == "__main__":
    main()
