#!/usr/bin/env python3
"""human-count 远程推理 + supervision 可视化。

在 r4900g3 服务器上运行，使用 2x NVIDIA L20 之一跑 GroundingDINO 检测人，
并用 supervision 的 annotator 家族画框/标签/统计，输出标注视频与首帧图。

用法:
    python3 infer_supervision.py \
        --video in/clip01.mp4 \
        --model  /share/data4/yhh/models/dino-model \
        --prompt "person" \
        --out-dir output \
        --device cuda:0 \
        --max-frames 0        # 0 = 全部
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import supervision as sv

# Containment-NMS：解决同一目标被多尺度重复框（人头/上半身/全身）
from containment_nms import containment_nms

# ★ 跟踪层（2026-11-10）：包含匹配 + 轨迹融合。缺失时不影响默认 byte 模式。
try:
    from containment_bytetrack import ContainmentByteTrack, fuse_contained_tracks
    _HAS_CONT = True
    _CONT_ERR = None
except Exception as _e:  # noqa: BLE001
    ContainmentByteTrack = None
    fuse_contained_tracks = None
    _HAS_CONT = False
    _CONT_ERR = _e


class FrameAnnotator:
    """把「轨迹/框/标签/三口径统计」的绘制集中到一处。

    两处调用必须共用同一份实现（项目纪律：选择与绘制逻辑只能有一份），
    否则 `--tracker cont_fuse` 的第二遍重绘会和在线绘制不一致：
      · 在线：`byte` / `cont` 模式，边检测边画
      · 重绘：`cont_fuse` 模式，融合改号后再解码一遍补画
    """

    def __init__(self, w: int, h: int, default_label: str = "person"):
        self.w, self.h = int(w), int(h)
        self.default_label = default_label
        self.box = sv.BoxAnnotator(thickness=2)
        self.label = sv.LabelAnnotator(text_scale=0.6, text_thickness=1, text_padding=6)
        self.trace = sv.TraceAnnotator(thickness=2, trace_length=45,
                                       color_lookup=sv.ColorLookup.TRACK)

    def draw(self, frame, boxes, scores, tids, labels,
             tracked_count: int, deduped_count: int, raw_count: int):
        det = sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            confidence=np.asarray(scores, dtype=np.float32).reshape(-1),
            class_id=np.zeros(len(boxes), dtype=int),
            tracker_id=(np.asarray(tids, dtype=int).reshape(-1) if len(tids) else None),
        )
        txts = []
        for i in range(len(det)):
            tid = int(tids[i]) if i < len(tids) else -1
            sc = float(det.confidence[i])
            lb = labels[i] if i < len(labels) else self.default_label
            txts.append(f"#{tid} {lb} {sc:.2f}" if tid >= 0 else f"{lb} {sc:.2f}")

        ann = frame.copy()
        ann = self.trace.annotate(scene=ann, detections=det)
        ann = self.box.annotate(scene=ann, detections=det)
        ann = self.label.annotate(scene=ann, detections=det, labels=txts)

        # 人数统计：放右侧留白区（底部有摄像头自带 OSD，左上角有时间戳）
        txt1 = f"tracked: {tracked_count}"
        txt2 = f"deduped: {deduped_count}"
        txt3 = f"raw: {raw_count}"
        (t1w, t1h), _ = cv2.getTextSize(txt1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        (t2w, t2h), _ = cv2.getTextSize(txt2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        (t3w, t3h), _ = cv2.getTextSize(txt3, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bw = max(t1w, t2w, t3w) + 20
        bx = self.w - bw - 10
        by = int(self.h * 0.28)
        box_h = t1h + t2h + t3h + 38
        cv2.rectangle(ann, (bx, by), (self.w - 10, by + box_h), (0, 0, 0), -1)
        cv2.putText(ann, txt1, (bx + 10, by + t1h + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(ann, txt2, (bx + 10, by + t1h + t2h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 80), 2, cv2.LINE_AA)
        cv2.putText(ann, txt3, (bx + 10, by + t1h + t2h + t3h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        return ann


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--imgsz", type=int, default=800, help="shortest edge for processor")
    ap.add_argument("--lost-buffer", type=int, default=90,
                    help="ByteTrack lost_track_buffer；max_time_lost=fps/30*buffer，需 > stride")
    ap.add_argument("--track-activation", type=float, default=0.10,
                    help="ByteTrack track_activation_threshold（内部 +0.1 作为 det_thresh）")
    # --- ★ v2（评审 F7）：三阈值显式解耦，默认全关 → 行为逐字节不变 ---
    #   旧口径：检测截断在 box_threshold=0.30，且 activation=0.10 → 二级低分池
    #     (0.1, 0.10) 恒为空集（ByteTrack "低分框挽救命"机制完全未生效）。
    #   v2 口径（仅当显式给 --track-floor 时启用）：
    #     · 检测输出到 track_floor（如 0.15，须 >0.1）
    #     · s > track_high：一级关联（可续轨、可起轨候选）
    #     · track_floor < s ≤ track_high：二级低分池 —— 只续接已有轨迹、不起新轨
    #     · 起轨门槛 = new_track（默认 0.30，与旧口径实际起轨门槛一致）；构造后显式覆写
    #       tracker.det_thresh，防止 supervision "activation+0.1" 把起轨门槛隐式抬高
    ap.add_argument("--track-floor", type=float, default=None,
                    help="v2：送入跟踪器的最低检测分（如 0.15，须 >0.1）；不给则沿用旧口径")
    ap.add_argument("--track-high", type=float, default=0.25,
                    help="v2：一级关联阈值（默认 0.25；仅 --track-floor 生效时使用）")
    ap.add_argument("--new-track", type=float, default=0.30,
                    help="v2：起轨阈值（默认 0.30；仅 --track-floor 生效时使用，显式覆写 det_thresh）")
    ap.add_argument("--log-suppressed", action="store_true", default=False,
                    help="v2：在 jsonl 记录被 Containment-NMS 删除的框（审计'身后的人被误删'风险）")
    # --- 第一层：Containment-NMS 去重 ---
    ap.add_argument("--dedup", dest="dedup", action="store_true", default=True,
                    help="开启 Containment-NMS 去重（默认开）")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false",
                    help="关闭去重，保留原始多尺度框")
    ap.add_argument("--containment-threshold", type=float, default=0.85,
                    help="小框落在某大框内的面积占比超过此值即判为重复。"
                         "范围(0,1]；1.0=只有完全包含才去重，越小越激进。"
                         "关闭去重请用 --no-dedup")
    ap.add_argument("--dedup-keep", choices=["larger", "higher"], default="larger",
                    help="重复冲突时保留谁：larger=面积更大的框（推荐）；higher=分数更高的框")
    ap.add_argument("--dedup-policy", choices=["larger", "higher", "hybrid"], default="larger",
                    help="C 方案（2026-11-10）接入点：hybrid = 大框先占位，但小框分数"
                         "≥大框×--hy-score-k 且小框高度≥--hy-min-h 时让小框占位。"
                         "6 路验证：305 换号 83→72，其余 5 路零负收益")
    ap.add_argument("--hy-score-k", type=float, default=0.8,
                    help="hybrid：小框分数 / 大框分数 的下限比（仅 --dedup-policy hybrid）")
    ap.add_argument("--hy-min-h", type=float, default=200.0,
                    help="hybrid 只对「小框高度≥此值」生效（仅 --dedup-policy hybrid）。"
                         "必须 >0：远景机位让头/上半身顶掉全身框会让跟踪更抖")
    ap.add_argument("--no-track", dest="track", action="store_false", default=True,
                    help="关闭 ByteTrack，只画原始检测")
    # --- ★ 跟踪层（2026-11-10）：包含匹配 + 轨迹融合 ---
    ap.add_argument("--tracker", choices=["byte", "cont", "cont_fuse"], default="byte",
                    help="byte=原 ByteTrack（默认，行为逐字节不变）；"
                         "cont=ByteTrack + 包含匹配第二级（在线，无滞后，改号问题修一半）；"
                         "cont_fuse=cont + 轨迹级融合（需要全片，故先检测存框、融合后再解码"
                         "一遍重绘视频，解码代价 x2）")
    ap.add_argument("--cont-thresh", type=float, default=0.85,
                    help="包含匹配：检测框落在轨迹框内的面积占比下限（仅 --tracker cont*）")
    ap.add_argument("--cont-center", type=float, default=0.5,
                    help="包含匹配：**x 向**中心距/较大框宽 上限。只算 x——框形态突变时高度"
                         "本来就会变，算二维会把正确案例拒掉（305 f4325 实测 dy=104）")
    ap.add_argument("--cont-y1", type=float, default=0.15,
                    help="包含匹配：|Δ头顶 y1| / 较大框高 上限。区分「同一人的头+表面框」"
                         "（Δy1≈4px）与「一前一后被遮挡的两人」（Δy1≈133px）")
    ap.add_argument("--fuse-min-obs", type=int, default=5,
                    help="轨迹融合：参与融合的最少观测帧数（防 2 帧幽灵段当胶水）")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="从第几个原始帧开始读（默认 0；仅用于分段测试/续跑）")
    ap.add_argument("--save-video", action="store_true", default=True)
    ap.add_argument("--no-save-video", dest="save_video", action="store_false")
    # --- ★ 并行支路：人脸检测（与人体检测共用同一解码帧，CPU/cv2.dnn） ---
    ap.add_argument("--face", action="store_true", default=False,
                    help="开启人脸支路：在同一帧上跑 SCRFD-2.5G（cv2.dnn，免 onnxruntime），"
                         "写 {stem}_faces.jsonl。人体检测输出不受影响（逐字节不变）")
    ap.add_argument("--face-model", default="models/face/scrfd_2.5g_bnkps.onnx",
                    help="SCRFD ONNX 路径（相对 cwd）")
    ap.add_argument("--face-thresh", type=float, default=0.5)
    ap.add_argument("--face-nms", type=float, default=0.4)
    ap.add_argument("--face-stride", type=int, default=0,
                    help="人脸检测步长；0 = 与 --stride 相同（推荐，即每个已处理帧都检脸）")
    ap.add_argument("--face-out", default="",
                    help="人脸 jsonl 输出路径；空 = {out-dir}/{stem}_faces.jsonl")
    return ap


def validate_track_thresholds(floor, high, new):
    """三阈值组合校验（ByteTrack 语义，纯函数便于单测）。

    低分池 = (0.1, high) 供"只续接、不起轨"；起轨门槛 = new 独立设置。
    约束：0.1 < floor <= high <= new <= 1。
    floor 必须 > 0.1（ByteTrack 低分池固定下界）；new 应 >= high（防起轨被隐式抬高）。
    """
    for name, x in (("track-floor", floor), ("track-high", high), ("new-track", new)):
        if not isinstance(x, (int, float)) or not np.isfinite(x) or not 0 < x <= 1:
            raise ValueError(f"{name} 须为 (0,1] 内有限数：{x!r}")
    if not (0.1 < floor <= high <= new):
        raise ValueError(
            f"须满足 0.1 < floor({floor}) <= high({high}) <= new({new})；"
            f"floor 必须 >0.1（ByteTrack 低分池固定下界），new 应 >= high（防隐式抬高起轨门槛）")


def main():
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem

    device = args.device if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(idx)
        print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} "
              f"device={device} name={torch.cuda.get_device_name(idx)}", flush=True)
    else:
        print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
              f"-> device={device}", flush=True)

    print(f"[model] loading {args.model} ...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model = model.to(device).eval()
    print(f"[model] loaded in {time.time()-t0:.1f}s", flush=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[video] {args.video}  {w}x{h} @ {fps:.2f}fps  frames={total}", flush=True)

    # supervision：框 + 标签 + 轨迹 + 人数统计（在线绘制与第二遍重绘共用）
    ann_kit = FrameAnnotator(w, h, args.prompt)
    # ByteTrack：给每个检测分配稳定 track_id，抑制逐帧人数抖动。
    # 注意 frame_rate 传【原始视频帧率】—— tracker.frame_id 随原视频帧递增，
    # 而 max_time_lost = int(frame_rate/30 * lost_track_buffer)，
    # 必须让 max_time_lost > stride，否则每两帧之间轨迹就被判定丢失、ID 全部作废。
    # 例：fps=25, stride=5, lost_buffer=90 → max_time_lost=75 帧 = 15 次检测间隔，足够抗遮挡。
    # ★ v2 跟踪阈值解析（默认 = 旧口径；仅显式 --track-floor 时启用 v2 两段式）
    v2_thr = args.track_floor is not None
    if v2_thr:
        try:
            validate_track_thresholds(args.track_floor, args.track_high, args.new_track)
        except ValueError as e:
            sys.exit(f"[track] 阈值参数非法: {e}")
        det_input_thr = args.track_floor
        track_activation = args.track_high
    else:
        det_input_thr = args.box_threshold
        track_activation = args.track_activation

    track_kwargs = dict(
        # 注意：track_activation_threshold 内部会 +0.1 作为 det_thresh，
        # 若直接传 box_threshold，则 0.30~0.40 分段的真实目标全部无法起轨。
        # 故单独给一个更低的激活阈值，让低分目标也能建立 trajectory。
        track_activation_threshold=track_activation,
        lost_track_buffer=int(args.lost_buffer),
        minimum_matching_threshold=0.8,
        frame_rate=float(fps),
        minimum_consecutive_frames=1,
    )
    if args.tracker == "byte":
        tracker = sv.ByteTrack(**track_kwargs)
    else:
        if not _HAS_CONT:
            sys.exit(f"[track] 无法加载 containment_bytetrack（需要同目录该文件）: {_CONT_ERR}")
        # ★ 必须继承真实 ByteTrack 类；supervision>=0.28 的 sv.ByteTrack 是弃用代理层，
        #   由 containment_bytetrack._resolve_bytetrack() 处理。
        tracker = ContainmentByteTrack(
            cont_thresh=args.cont_thresh,
            center_ratio=args.cont_center,
            y1_tol=args.cont_y1,
            **track_kwargs,
        )
    if v2_thr:
        # ★ 显式解耦：起轨门槛 = new_track，不被 "activation+0.1" 隐式抬高
        # （G 包评审：Changing activation must not implicitly raise the new-track threshold）
        if hasattr(tracker, "det_thresh"):
            tracker.det_thresh = float(args.new_track)
        else:
            print("[track] 警告：当前 supervision 无 det_thresh 属性，解耦未生效（请核对版本）",
                  flush=True)
    print(f"[track] ByteTrack fps={fps} stride={args.stride} "
          f"activation={track_activation}"
          + (f" (det_thresh={tracker.det_thresh})" if v2_thr
             else f" (det_thresh={args.track_activation+0.1})")
          + f" lost_buffer={args.lost_buffer} -> max_time_lost="
          f"{int(float(fps)/30.0*args.lost_buffer)} frames"
          + (f" [v2 floor={args.track_floor} high={args.track_high} new={args.new_track}]"
             if v2_thr else ""), flush=True)
    if args.tracker != "byte":
        print(f"[track] 包含匹配开启 cont_thresh={args.cont_thresh} "
              f"center(x)={args.cont_center} y1={args.cont_y1}"
              + ("  → 轨迹融合需要全片：先检测存框，融合后第二遍解码重绘视频"
                 if args.tracker == "cont_fuse" else ""), flush=True)

    # ★ cont_fuse：融合判定依赖全片轨迹，不能边跑边画 —— 第一遍只检测存框，
    #   融合改号后再解码第二遍补画（视频与 jsonl 里的 id 才一致）。
    need_two_pass = (args.tracker == "cont_fuse")
    records = [] if need_two_pass else None

    vpath = None
    writer = None
    if args.save_video and not need_two_pass:
        vpath = out_dir / f"{stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vpath), fourcc, fps, (w, h))
        print(f"[out] video -> {vpath}", flush=True)
    elif args.save_video and need_two_pass:
        vpath = out_dir / f"{stem}_annotated.mp4"
        print(f"[out] video -> {vpath}（融合后第二遍重绘）", flush=True)

    jsonl_path = out_dir / f"{stem}_detections.jsonl"
    jf = jsonl_path.open("w", encoding="utf-8")

    # --- ★ 并行人脸支路初始化（失败不影响人体检测） ---
    face_det = None
    face_jf = None
    face_path = None
    face_ms, face_err = [], 0
    n_face_boxes = 0
    face_stride = args.face_stride or args.stride
    if args.face:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from scrfd_dnn import ScrfdDNN  # noqa: E402
            face_det = ScrfdDNN(args.face_model, thresh=args.face_thresh, nms=args.face_nms)
            face_path = Path(args.face_out) if args.face_out else (out_dir / f"{stem}_faces.jsonl")
            face_jf = face_path.open("w", encoding="utf-8")
            print(f"[face] SCRFD(cv2.dnn) {args.face_model} -> {face_path} "
                  f"stride={face_stride}", flush=True)
        except Exception as e:  # noqa: BLE001
            face_det = None
            print(f"[face] 初始化失败，跳过人脸支路（人体检测照常）: {e}", flush=True)

    frame_idx = 0
    if args.start_frame > 0:
        # 仅用于分段测试/续跑：跳读会以后续顺序读为代价，跳过的帧不计入 frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame))
        frame_idx = int(args.start_frame)
        print(f"[video] 从第 {args.start_frame} 帧开始读", flush=True)
    processed = 0
    t_start = time.time()
    first_frame_saved = False
    latencies = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and processed >= args.max_frames:
            break

        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        t1 = time.time()
        inputs = processor(images=pil, text=args.prompt + ".",
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=det_input_thr,   # v2: track_floor；v1: box_threshold（默认不变）
            text_threshold=args.text_threshold,
            target_sizes=[pil.size[::-1]],
        )[0]
        if device != "cpu":
            torch.cuda.synchronize()
        dt = time.time() - t1
        latencies.append(dt)

        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        # transformers 5.x: `labels` 为整数 id，文本在 `text_labels`；旧版则相反
        raw_labels = results.get("text_labels")
        if raw_labels is None:
            raw_labels = results.get("labels")
        if isinstance(raw_labels, torch.Tensor):
            raw_labels = raw_labels.detach().cpu().tolist()
        labels = [str(l) for l in (raw_labels if raw_labels is not None else [])]
        if len(labels) != len(boxes):
            labels = [args.prompt] * len(boxes)

        raw_count = len(boxes)

        # --- 第一层去重：Containment-NMS ---
        # 同一目标常被多尺度响应（头/上半身/全身），其嵌套对的 IoU 仅 0.2~0.65，
        # 逃过常规 NMS；改用"包含度"判据，保留面积更大的框（更完整的全身框）。
        suppressed = []
        if args.dedup and args.containment_threshold > 0 and raw_count > 1:
            _pre_boxes, _pre_scores = boxes, scores
            if args.dedup_policy == "hybrid":
                # C 方案：大框分数不够硬时让小框占位（尺寸条件化，避开远景机位）
                keep_idx = containment_nms(
                    boxes, scores,
                    containment_threshold=args.containment_threshold,
                    keep="larger",
                    hy_score_k=args.hy_score_k,
                    hy_min_h=args.hy_min_h,
                )
            else:
                keep_idx = containment_nms(
                    boxes, scores,
                    containment_threshold=args.containment_threshold,
                    keep=args.dedup_policy,
                )
            if args.log_suppressed:
                # ★ 审计：记录被删除的框（供"身后的人被误删"事后核查）
                _k = set(int(i) for i in keep_idx)
                suppressed = [[round(float(v), 1) for v in _pre_boxes[i]]
                              + [round(float(_pre_scores[i]), 4)]
                              for i in range(len(_pre_boxes)) if i not in _k]
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            labels = [labels[i] for i in keep_idx]
        deduped_count = len(boxes)

        det = sv.Detections(
            xyxy=boxes.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
        )

        # --- ByteTrack 跟踪 ---
        det = tracker.update_with_detections(det)
        tracked_count = len(det)
        track_ids = det.tracker_id if det.tracker_id is not None else np.array([])

        # 每帧结果（= jsonl 一行）。cont_fuse 模式先囤起来：融合要全片才能判定，
        # 改号后统一写出，视频也留到第二遍重绘，保证「视频里的 id == jsonl 里的 id」。
        jsonl_row = {
            "frame": frame_idx,
            "count": tracked_count,
            "deduped_count": deduped_count,
            "raw_count": raw_count,
            **({"suppressed": suppressed} if args.log_suppressed else {}),
            **({"track_floor": args.track_floor} if v2_thr else {}),
            "removed": raw_count - deduped_count,
            "track_ids": [int(t) for t in track_ids.tolist()] if len(track_ids) else [],
            "latency_ms": round(dt * 1000, 1),
            "boxes": [[round(float(v), 1) for v in b] for b in det.xyxy],
            "scores": [round(float(s), 4) for s in det.confidence],
            "labels": labels,
        }

        if need_two_pass:
            records.append({
                "row": jsonl_row,
                "boxes": det.xyxy.copy(),
                "scores": det.confidence.copy(),
                "tids": jsonl_row["track_ids"],
                "labels": labels,
            })
        else:
            ann = ann_kit.draw(frame, det.xyxy, det.confidence,
                               jsonl_row["track_ids"], labels,
                               tracked_count, deduped_count, raw_count)
            if writer is not None:
                writer.write(ann)

            if not first_frame_saved:
                fp = out_dir / f"{stem}_first_frame.jpg"
                cv2.imwrite(str(fp), ann)
                first_frame_saved = True
                print(f"[out] first frame -> {fp}  "
                      f"(tracked={tracked_count} deduped={deduped_count} raw={raw_count})",
                      flush=True)

            jf.write(json.dumps(jsonl_row, ensure_ascii=False) + "\n")

        # --- ★ 并行人脸支路：同一帧，人体检测(GPU)完成后在人脸(CPU)上并行推进 ---
        if face_det is not None and frame_idx % face_stride == 0:
            tf = time.time()
            try:
                fdet = face_det.detect(frame)
                face_jf.write(json.dumps({
                    "frame": frame_idx, "w": w, "h": h,
                    "n": int(len(fdet)),
                    # 坐标 1 位小数；score 保 4 位（避免量化到阈值边界附近丢分辨率）
                    "faces": [[round(float(v), 1) for v in row[:14]] + [round(float(row[14]), 4)]
                              for row in fdet],
                }, ensure_ascii=False) + "\n")
                face_ms.append((time.time() - tf) * 1000.0)
                n_face_boxes += int(len(fdet))
            except Exception:  # noqa: BLE001
                face_err += 1

        processed += 1
        frame_idx += 1
        if processed % 10 == 0:
            el = time.time() - t_start
            print(f"[prog] processed={processed} fps={processed/el:.2f} "
                  f"last_latency={dt*1000:.0f}ms", flush=True)

    cap.release()
    if writer is not None:
        writer.release()
    jf.close()
    if face_jf is not None:
        face_jf.close()

    # ---- ★ cont_fuse 收尾：轨迹融合 → 改号写 jsonl → 第二遍解码重绘视频 ----
    tid_map = {}
    if need_two_pass:
        t_fuse = time.time()
        tid_map = fuse_contained_tracks(
            [(r["row"]["frame"], r["tids"], r["boxes"]) for r in records],
            cont_thresh=args.cont_thresh,
            center_ratio=args.cont_center,
            y1_tol=args.cont_y1,
            min_obs_short=args.fuse_min_obs,
        ) if records else {}
        n_changed = 0
        with jsonl_path.open("w", encoding="utf-8") as jf2:
            for r in records:
                old = r["row"]["track_ids"]
                new = [int(tid_map.get(t, t)) for t in old]
                if new != old:
                    n_changed += 1
                r["row"]["track_ids"] = new
                r["tids"] = new
                jf2.write(json.dumps(r["row"], ensure_ascii=False) + "\n")
        mp_path = out_dir / f"{stem}_tid_map.json"
        with mp_path.open("w", encoding="utf-8") as mf:
            json.dump({
                "video": args.video,
                "merged": {str(k): int(v) for k, v in tid_map.items()},
                "n_merged": len(tid_map),
                "frames_reid": n_changed,
                "params": {"cont_thresh": args.cont_thresh,
                           "center_ratio_x": args.cont_center,
                           "y1_tol": args.cont_y1,
                           "min_obs": args.fuse_min_obs},
            }, mf, ensure_ascii=False, indent=1)
        print(f"[fuse] 轨迹融合 {len(tid_map)} 条 → 改号 {n_changed} 帧；映射 -> {mp_path}",
              flush=True)
        print(f"[fuse] 融合耗时 {time.time() - t_fuse:.1f}s", flush=True)

        if vpath is not None and records:
            t_r = time.time()
            cap2 = cv2.VideoCapture(args.video)
            wr = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            by_frame = {r["row"]["frame"]: r for r in records}
            last = max(by_frame)
            idx = 0
            while idx <= last:
                ok, frame = cap2.read()
                if not ok:
                    break
                r = by_frame.get(idx)
                if r is not None:
                    ann = ann_kit.draw(frame, r["boxes"], r["scores"],
                                       r["tids"], r["labels"],
                                       r["row"]["count"], r["row"]["deduped_count"],
                                       r["row"]["raw_count"])
                    wr.write(ann)
                    if not first_frame_saved:
                        fp = out_dir / f"{stem}_first_frame.jpg"
                        cv2.imwrite(str(fp), ann)
                        first_frame_saved = True
                        print(f"[out] first frame -> {fp}  "
                              f"(tracked={r['row']['count']} "
                              f"deduped={r['row']['deduped_count']} "
                              f"raw={r['row']['raw_count']})", flush=True)
                idx += 1
            cap2.release()
            wr.release()
            print(f"[out] video(重绘) -> {vpath}  耗时 {time.time() - t_r:.1f}s", flush=True)

    el = time.time() - t_start
    summary = {
        "video": args.video,
        "resolution": [w, h],
        "fps": round(fps, 3),
        "frames_total": total,
        "frames_processed": processed,
        "stride": args.stride,
        "device": device,
        "model": args.model,
        "prompt": args.prompt,
        "box_threshold": args.box_threshold,
        **({"track_floor": args.track_floor, "track_high": args.track_high,
            "new_track": args.new_track, "tracker_mode": "v2_two_stage"}
           if v2_thr else {}),
        "text_threshold": args.text_threshold,
        "elapsed_sec": round(el, 2),
        "throughput_fps": round(processed / el, 3) if el > 0 else None,
        "latency_ms_mean": round(float(np.mean(latencies)) * 1000, 1) if latencies else None,
        "latency_ms_p50": round(float(np.percentile(latencies, 50)) * 1000, 1) if latencies else None,
        "latency_ms_p90": round(float(np.percentile(latencies, 90)) * 1000, 1) if latencies else None,
        "tracking": ({"byte": "ByteTrack", "cont": "ByteTrack+containment",
                      "cont_fuse": "ByteTrack+containment+fuse"}[args.tracker]
                     if args.track else "none"),
        **({"cont_thresh": args.cont_thresh, "cont_center_x": args.cont_center,
            "cont_y1": args.cont_y1, "fuse_min_obs": args.fuse_min_obs,
            "n_merged": len(tid_map)} if args.tracker != "byte" else {}),
        "dedup": ("containment_nms" if (args.dedup and args.containment_threshold > 0)
                  else "none"),
        "containment_threshold": args.containment_threshold if args.dedup else None,
        "dedup_keep": args.dedup_keep if args.dedup else None,
        "dedup_policy": (args.dedup_policy if args.dedup else None),
        **({"hy_score_k": args.hy_score_k, "hy_min_h": args.hy_min_h}
           if (args.dedup and args.dedup_policy == "hybrid") else {}),
        "count_mean": None,
        # ★ 并行人脸支路
        "face_branch": bool(face_det is not None),
        "face_model": args.face_model if face_det is not None else None,
        "face_stride": face_stride if face_det is not None else None,
        "faces_jsonl": str(face_path) if face_path else None,
        "face_boxes_total": n_face_boxes if face_det is not None else None,
        "face_ms_mean": round(float(np.mean(face_ms)), 2) if face_ms else None,
        "face_ms_p50": round(float(np.percentile(face_ms, 50)), 2) if face_ms else None,
        "face_errors": face_err if face_det is not None else None,
    }
    # 从 jsonl 汇总人数（tracked / deduped / raw 三口径）
    counts, dedups, raws, all_ids = [], [], [], set()
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            counts.append(r["count"])
            dedups.append(r.get("deduped_count", r["count"]))
            raws.append(r.get("raw_count", r["count"]))
            all_ids.update(r.get("track_ids", []))
    if counts:
        summary["count_mean"] = round(float(np.mean(counts)), 2)
        summary["count_max"] = int(max(counts))
        summary["count_min"] = int(min(counts))
        summary["count_std"] = round(float(np.std(counts)), 3)
        summary["deduped_count_mean"] = round(float(np.mean(dedups)), 2)
        summary["raw_count_mean"] = round(float(np.mean(raws)), 2)
        summary["boxes_removed_total"] = int(sum(raws) - sum(dedups))
        summary["boxes_removed_pct"] = (
            round(100.0 * (1 - sum(dedups) / sum(raws)), 2) if sum(raws) else 0.0
        )
        summary["unique_track_ids"] = len(all_ids)

    sp = out_dir / f"{stem}_summary.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[out] summary -> {sp}", flush=True)


if __name__ == "__main__":
    main()
