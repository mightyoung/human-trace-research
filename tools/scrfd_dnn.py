#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SCRFD-2.5G 人脸检测器（cv2.dnn 后端，免 onnxruntime）。

动机：检测头 infer_supervision.py 跑在远程 rfdetr 环境（无 onnxruntime），
但人脸抓拍要做成「与人体检测共用同一解码帧」的并行支路 —— 于是用 cv2.dnn
（已随 opencv 提供）加载同一份 SCRFD ONNX，输出与原 onnxruntime 版**完全同构**：

    detect(img) -> np.ndarray [N, 15]
        [x1, y1, x2, y2, kp0x, kp0y, ..., kp4x, kp4y, score]，坐标为原图（BGR 输入）

与 gil_face_capture.Scrfd 的差异仅在推理调用（net.forward vs sess.run），
前处理（letterbox 640 / 127.5,128.0 归一 / BGR→RGB）与后处理（三尺度解码 +
NMS）逐行对齐，便于两侧交叉校验。

★ 2026-10-16 安全修复（来源：gpt-v2 研究包 overlay/scrfd_dnn.py；本机实测复现 + 验证）：
  1) 竖幅越界崩溃：旧版固定宽 640、高按比例取 32 倍数 —— 输入比约 1.05:1 更竖
     时 new_h>640，`canvas[:new_h] = im` 直接抛 broadcast 错误（900×450 实测
     复现 `could not broadcast (1280,640,3) into (640,640,3)`）。现改为长短边
     联合 fit：scale=min(640/w, 640/h)，任意长宽比不再越界。
  2) x 坐标系统性偏差：旧版统一除以 det_scale=new_h/h；对 16:9 输入（本数据集
     6 路均为 2560×1440 / 1920×1080）产生 x 方向 +2.27% 偏差（实测 318 帧上
     x1/x2 偏 12.5–14.2px）。现按取整后真实比例分别还原（sx=nw/w, sy=nh/h）。
  3) padding 假脸：中心落在人工填充区的 proposal 不再输出（旧版会把它们 clip
     到图像边缘形成贴边假脸）。
  4) 输出契约：仅接受三尺度（8/16/32）× 2 anchor 的 score/bbox/5KPS 共 9 输出
     （[N,C] 或 [1,N,C]），未知布局**显式报错**，不静默跳过。

  ★ 坐标体系变更声明：修复后同一人脸框 x 坐标比旧版左移约 2.27%（16:9 输入）。
    历史产物（P0 人脸池等）按旧坐标生成 —— **不重跑、不混用**；新运行按新坐标。

用法：
    from scrfd_dnn import ScrfdDNN
    det = ScrfdDNN("models/face/scrfd_2.5g_bnkps.onnx", thresh=0.5, nms=0.4)
    faces = det.detect(frame_bgr)
"""
from __future__ import annotations
import os
import cv2
import numpy as np

DEFAULT_MODEL=os.path.join('models','face','scrfd_2.5g_bnkps.onnx')


def prepare_image(img,size=640):
    if not isinstance(img,np.ndarray) or img.dtype!=np.uint8 or img.ndim!=3 or img.shape[2]!=3 or min(img.shape[:2])<1:
        raise ValueError('expected nonempty uint8 BGR HWC image')
    h,w=img.shape[:2]; scale=min(size/w,size/h)
    nw=max(1,min(size,round(w*scale))); nh=max(1,min(size,round(h*scale)))
    resized=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR)
    canvas=np.zeros((size,size,3),np.uint8); canvas[:nh,:nw]=resized
    # Keep actual x/y scales to account for rounding, rather than dividing all
    # coordinates by height scale. Top-left letterbox follows original export.
    return canvas,(nw/w,nh/h),(nw,nh)


class ScrfdDNN:
    """SCRFD-2.5G (cv2.dnn)。letterbox 640，输出 [x1,y1,x2,y2,kp10,score]（原图坐标）"""
    STRIDES=(8,16,32)
    def __init__(self,model=DEFAULT_MODEL,thresh=.5,nms=.4,threads=0,*,net=None):
        if not 0<=thresh<=1 or not 0<nms<=1: raise ValueError('invalid threshold')
        if net is None:
            if not os.path.isfile(model): raise FileNotFoundError(model)
            net=cv2.dnn.readNetFromONNX(model)
        self.net=net
        if threads>0: cv2.setNumThreads(threads)
        self.out_names=net.getUnconnectedOutLayersNames()
        self.thresh=float(thresh); self.nms=float(nms)

    def detect(self,img):
        h,w=img.shape[:2] if isinstance(img,np.ndarray) and img.ndim>=2 else (0,0)
        canvas,(sx,sy),(nw,nh)=prepare_image(img)
        blob=cv2.dnn.blobFromImage(canvas,1/128.,(640,640),(127.5,127.5,127.5),swapRB=True)
        self.net.setInput(blob); outs=self.net.forward(self.out_names)
        grouped={}
        for raw in outs:
            a=np.asarray(raw)
            if a.ndim==3 and a.shape[0]==1: a=a[0]
            if a.ndim!=2 or a.shape[1] not in (1,4,10) or not np.isfinite(a).all():
                raise ValueError(f'unsupported/nonfinite SCRFD output {a.shape}')
            key=(a.shape[0],a.shape[1])
            if key in grouped: raise ValueError('duplicate SCRFD output shape')
            grouped[key]=a.astype(np.float32)
        expected={(2*(640//s)**2,c) for s in self.STRIDES for c in (1,4,10)}
        if set(grouped)!=expected: raise ValueError('SCRFD export must provide 3 score/box/5KPS scales')
        boxes=[];points=[];scores=[]
        for stride in self.STRIDES:
            g=640//stride;n=2*g*g
            sc=grouped[n,1].reshape(-1);bb=grouped[n,4];kp=grouped[n,10]
            yy,xx=np.mgrid[:g,:g]
            centers=np.repeat(np.column_stack((xx.reshape(-1),yy.reshape(-1)))*stride,2,axis=0).astype(np.float32)
            pos=np.flatnonzero(sc>self.thresh)
            if pos.size==0: continue
            dist=bb[pos]*stride;center=centers[pos]
            b=np.column_stack((center[:,0]-dist[:,0],center[:,1]-dist[:,1],center[:,0]+dist[:,2],center[:,1]+dist[:,3]))
            p=kp[pos].reshape(-1,5,2)*stride+center[:,None,:]
            # Ignore proposals centered in artificial padding; do not clip them
            # into apparent faces at the edge of the source image.
            real=(center[:,0]<nw)&(center[:,1]<nh)
            boxes.append(b[real]);points.append(p[real]);scores.append(sc[pos][real])
        if not boxes or sum(len(b) for b in boxes)==0: return np.zeros((0,15),np.float32)
        b=np.concatenate(boxes);p=np.concatenate(points);sc=np.concatenate(scores)
        b[:,[0,2]]/=sx; b[:,[1,3]]/=sy; p[:,:,0]/=sx; p[:,:,1]/=sy
        b[:,[0,2]]=np.clip(b[:,[0,2]],0,w); b[:,[1,3]]=np.clip(b[:,[1,3]],0,h)
        valid=(b[:,2]>b[:,0])&(b[:,3]>b[:,1])
        b,p,sc=b[valid],p[valid],sc[valid]
        if len(b)==0: return np.zeros((0,15),np.float32)
        xywh=np.column_stack((b[:,:2],b[:,2:]-b[:,:2]))
        ix=np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(),sc.tolist(),self.thresh,self.nms),dtype=int).reshape(-1)
        if not len(ix): return np.zeros((0,15),np.float32)
        return np.column_stack((b[ix],p[ix].reshape(-1,10),sc[ix])).astype(np.float32)


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    d = ScrfdDNN(m)
    print("loaded", m, "out layers:", d.out_names)
