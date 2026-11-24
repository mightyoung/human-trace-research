# -*- coding: utf-8 -*-
"""ByteTrack + 「包含匹配」第二级 —— 解决同一目标因框形态突变而换号。

问题（305 f4435 实测）
----------------------
`#133` 的框是「一个人的头 + 他面前的 AOC 显示器」（h286，其中约 210px 是显示器），
`#129` 是同一个头的纯头部框（h75）。两者包含度 1.000、但 IoU 只有 0.236。
ByteTrack 第一级用 `fuse_score` 揉分数，边界情况下判不成同一目标 → 新建轨迹 → 换号。

做法
----
只覆盖 `update_with_tensors`（`update_with_detections` 继承父类，靠多态生效），
原样保留上游状态机，**仅在第一级 `linear_assignment` 之后插入一段包含匹配**：
未匹配的 (track, detection) 对，若 `包含度 >= cont_thresh` 且
`中心距 <= center_ratio × 较大框宽` 且 `|Δy1| <= y1_tol × 较大框高`，则配对成功。

参数退化：`cont_thresh >= 1` 时行为与标准 ByteTrack 完全一致（可做 A/B）。

版本兼容
--------
supervision 0.28+ 换了实现，本文件同时兼容：
  - 0.24：`STrack(tlwh, score, cls, min_consecutive)`、`multi_predict(pool)`、
          `re_activate(det, fid, new_id=False)`、`track.end_frame`、tensors 带 class_id
  - 0.29：`STrack(tlwh, score, min_consecutive, shared_kalman, id_counter, ext_id_counter)`、
          `multi_predict(pool, shared_kalman)`、`re_activate(det, fid)`、`track.frame_id`、
          tensors 只有 5 列（xyxy + score，无 class_id）
判别方式 = 运行时检查 `STrack.__init__` 是否带 `shared_kalman` 参数。

注意
----
0.29 的 `update_with_detections` 末尾会用 `IoU < 0.5` 把 tracker 输出重新配回**输入检测**，
配不上的检测 `tracker_id = -1` 后被丢弃。而「整身框 ↔ 头肩框」的 IoU 只有 0.2-0.4，
正是本类要救的那批 —— 走 `update_with_detections` 会在输出层被整批滤掉。
因此调用方应直接用 `update_with_tensors` + `STrack.tlbr/external_track_id`（与 ultralytics
`_update` 的口径一致：输出轨迹的框，不是原始检测）。
"""
from __future__ import annotations

import inspect
from typing import List

import numpy as np
import supervision as sv
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.core import (
    STrack,
    TrackState,
    joint_tracks,
    remove_duplicate_tracks,
    sub_tracks,
)

# 0.29 的 STrack 增加了 shared_kalman / 两个 IdCounter 参数；0.24 是 (tlwh, score, cls, min_conf)
_ST_NEW = "shared_kalman" in inspect.signature(STrack.__init__).parameters


def _resolve_bytetrack():
    """取真正的 ByteTrack 类。

    supervision >= 0.28 把 `sv.ByteTrack` / `core.ByteTrack` 换成了
    `deprecate.proxy._DeprecatedProxy`（弃用中间层），直接继承会报
    `_DeprecatedProxy.__init__() takes 3 positional arguments but 4 were given`。
    真实类藏在 proxy 的 `_DeprecatedProxy__config.obj`。
    """
    from supervision.tracker.byte_tracker import core

    cls = getattr(core, "ByteTrack", None)
    if cls is None:
        cls = sv.ByteTrack
    if type(cls).__name__ == "_DeprecatedProxy":
        cfg = cls.__dict__.get("_DeprecatedProxy__config")
        real = getattr(cfg, "obj", None)
        if isinstance(real, type):
            return real
        return type(cls())
    return cls


_BaseByteTrack = _resolve_bytetrack()


def containment_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:(N,4) b:(M,4) xyxy → (N,M)，max(交集/area_a, 交集/area_b)。

    对称口径：大框包小框、小框被大框包都算命中（与 containment_nms 保持一致）。
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 1e-9, None) * np.clip(a[:, 3] - a[:, 1], 1e-9, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 1e-9, None) * np.clip(b[:, 3] - b[:, 1], 1e-9, None)
    return np.maximum(inter / area_a[:, None], inter / area_b[None, :])


def center_dist_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """**x 方向**中心距 / 较大框宽 → (N,M)。

    只算 x，不算二维距离：目标框形态突变时高度本来就会变
    （纯头 h75 vs 头+表面 h286），中心 y 必然拉开 —— 305 f4325 实测 dy=104，
    二维 ratio 0.683 会把正确案例整个拒掉。y 方向交给 y1 对齐约束单独判断。
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ca = (a[:, 0] + a[:, 2]) / 2
    cb = (b[:, 0] + b[:, 2]) / 2
    dx = np.abs(ca[:, None] - cb[None, :])
    wa = np.clip(a[:, 2] - a[:, 0], 1e-9, None)
    wb = np.clip(b[:, 2] - b[:, 0], 1e-9, None)
    return dx / np.maximum(wa[:, None], wb[None, :])


class ContainmentByteTrack(_BaseByteTrack):
    """标准 ByteTrack + 包含匹配第二级（仅改 Step 2 关联）。"""

    def __init__(self, *args, cont_thresh: float = 0.85, center_ratio: float = 0.5,
                 y1_tol: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.cont_thresh = cont_thresh
        self.center_ratio = center_ratio
        self.y1_tol = y1_tol
        self.n_cont_match = 0   # 诊断：包含匹配救回的次数

    # ---------------------------------------------------------------- 版本适配
    def _mk(self, tlbr, score, cls=0.0):
        if _ST_NEW:
            return STrack(STrack.tlbr_to_tlwh(tlbr), score,
                          self.minimum_consecutive_frames, self.shared_kalman,
                          self.internal_id_counter, self.external_id_counter)
        return STrack(STrack.tlbr_to_tlwh(tlbr), score, cls,
                      self.minimum_consecutive_frames)

    def _multi_predict(self, pool):
        if _ST_NEW:
            STrack.multi_predict(pool, self.shared_kalman)
        else:
            STrack.multi_predict(pool)

    def _reactivate(self, track, det, fid):
        if _ST_NEW:
            track.re_activate(det, fid)
        else:
            track.re_activate(det, fid, new_id=False)

    @staticmethod
    def _set_state(track, state):
        """0.24 用 mark_lost()/mark_removed()，0.29 直接赋 state。"""
        if state == TrackState.Lost:
            if hasattr(track, "mark_lost"):
                track.mark_lost()
                return
        elif hasattr(track, "mark_removed"):
            track.mark_removed()
            return
        track.state = state

    # ---------------------------------------------------------------- 主逻辑
    def update_with_tensors(self, tensors: np.ndarray) -> List[STrack]:
        """照抄上游 `update_with_tensors`，仅在第一级关联后插入包含匹配。"""
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # ★ 0.29 的 tensors 只有 5 列（xyxy + score），0.24 是 6 列（带 class_id）
        if tensors.ndim == 2 and tensors.shape[1] > 5:
            class_ids = tensors[:, 5]
        else:
            class_ids = np.zeros(len(tensors), dtype=np.float64)
        scores = tensors[:, 4]
        bboxes = tensors[:, :4]

        remain_inds = scores > self.track_activation_threshold
        inds_low = scores > 0.1
        inds_high = scores < self.track_activation_threshold

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = class_ids[remain_inds]
        cls_second = class_ids[inds_second]

        detections = [self._mk(tlbr, s, c)
                      for (tlbr, s, c) in zip(dets, scores_keep, cls_keep)]

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]

        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_tracks(tracked_stracks, self.lost_tracks)
        # Predict the current location with KF
        self._multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)

        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.minimum_matching_threshold
        )

        # ★★ 包含匹配（本项目新增，上游没有）：第一级 IoU 失败后，用「包含度 + 中心距 + 头顶对齐」
        # 再配一次。动机 —— 同一目标的框形态突变时 IoU 会崩：
        #   305 f4435：`#133`（一个人的头 + AOC 显示器，h286）→ `#129`（同一个头的纯头框，h75）
        #   包含度 1.000、中心距 0.06×W，但 IoU 只有 0.236；`fuse_score` 之后仍可能
        #   超出门限被判为新目标 ⇒ 换号。
        # 约束三条，缺一不可：
        #   a) 包含度 >= cont_thresh（默认 0.85）
        #   b) x 向中心距 <= center_ratio × 较大框宽（默认 0.5）—— 防左右错配
        #   c) **头顶对齐** |y1a - y1b| <= y1_tol × 较大框高（默认 0.15）—— 防上下错配
        #      (c) 是区分两类「包含」的关键：
        #        正确（同一人的「头+表面」框 vs 纯头框）：y1 都在头顶，差 4px
        #        错误（一前一后两人）：y1 差 133px（305 f4393 实测误合并 #93/#76）
        if self.cont_thresh < 1.0 and len(u_track) > 0 and len(u_detection) > 0:
            # linear_assignment 返回的 matches 可能是 ndarray，先转 list 才能追加
            matches = [tuple(int(v) for v in m) for m in matches]
            _sub_trk = np.asarray([strack_pool[i].tlbr for i in u_track], dtype=np.float64)
            _sub_det = np.asarray([detections[i].tlbr for i in u_detection], dtype=np.float64)
            _cont = containment_batch(_sub_det, _sub_trk)
            _ratio = center_dist_ratio(_sub_det, _sub_trk)
            _hdet = (_sub_det[:, 3] - _sub_det[:, 1])[:, None]
            _htrk = (_sub_trk[:, 3] - _sub_trk[:, 1])[None, :]
            _dy1 = np.abs(_sub_det[:, 1][:, None] - _sub_trk[:, 1][None, :])
            _ok = ((_cont >= self.cont_thresh)
                   & (_ratio <= self.center_ratio)
                   & (_dy1 <= self.y1_tol * np.maximum(_hdet, _htrk)))
            if _ok.any():
                # cost 行=检测、列=轨迹；不合格的对代价置 1.0（必被拒）
                _m2, _urow, _ucol = matching.linear_assignment(
                    np.where(_ok, 1.0 - _cont, 1.0), 1.0 - self.cont_thresh
                )
                for _r, _c in _m2:
                    matches.append((int(u_track[int(_c)]), int(u_detection[int(_r)])))
                    self.n_cont_match += 1
                _urow = np.asarray(_urow, dtype=int).reshape(-1)
                _ucol = np.asarray(_ucol, dtype=int).reshape(-1)
                u_track = [u_track[i] for i in _ucol]          # 列 = 轨迹
                u_detection = [u_detection[i] for i in _urow]  # 行 = 检测

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                self._reactivate(track, det, self.frame_id)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        detections_second = [self._mk(tlbr, s, c)
                             for (tlbr, s, c) in zip(dets_second, scores_second, cls_second)]
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                self._reactivate(track, det, self.frame_id)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                self._set_state(track, TrackState.Lost)
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)

        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            self._set_state(track, TrackState.Removed)
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_tracks:
            # 0.29 用 track.frame_id，0.24 用 track.end_frame
            _last = getattr(track, "frame_id", None)
            if _last is None:
                _last = track.end_frame
            if self.frame_id - _last > self.max_time_lost:
                self._set_state(track, TrackState.Removed)
                removed_stracks.append(track)

        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_starcks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks = removed_stracks
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        output_stracks = [track for track in self.tracked_tracks if track.is_activated]

        return output_stracks


def track_ext_id(track) -> int:
    """取对外轨迹号（0.29 为 external_track_id，0.24 为 track_id）。"""
    v = getattr(track, "external_track_id", None)
    return int(v if v is not None else track.track_id)


def _median_box(boxes) -> np.ndarray:
    return np.median(np.asarray(boxes, dtype=np.float64).reshape(-1, 4), axis=0)


def fuse_contained_tracks(
    out_frames,
    cont_thresh: float = 0.85,
    center_ratio: float = 0.5,
    y1_tol: float = 0.15,
    max_gap_frames: int = 250,
    min_obs_short: int = 5,
):
    """轨迹级融合：把「被包含 + 头顶对齐 + 时间不共现」的短轨并入主轨。

    为什么在匹配阶段解决不了
    ------------------------
    `#129`（纯头 h75）与 `#133`（头+显示器 h286）虽然包含度 1.000，但**各自都能匹配
    上自己的检测**：`#129` 的轨迹在 `lost_track_buffer`（75 个处理帧）内没被移除，
    检测一来就被它自己接走，轮不到包含匹配。所以只能在轨迹生成后做一次融合。

    ★ 适用范围：本条要求「两轨从不共现」。**同一帧内并存的整身框 + 头肩框永远无法
      被本函数融合**（它们是共现的），那类重复框必须在检测层解决
      （见 tmp/full/track_contain.py 的掩膜去重）。

    判据（四条全满足，宁可漏合并不误合并）
    --------------------------------------
    1. 两轨**从不共现**（无同帧）—— 共现即两人
    2. 时间区间重叠或间隔 ≤ max_gap_frames
    3. 中位框：被包含者 `containment >= cont_thresh`
    4. **头顶对齐** `|y1a - y1b| <= y1_tol × 较大框高` 且 x 中心距 <= center_ratio × 较大框宽

    返回 `{drop_tid: keep_tid}`（保留观测多的那条）。
    """
    seq = {}
    for fr, tids, boxes in out_frames:
        for t, b in zip(tids, boxes):
            seq.setdefault(int(t), []).append((int(fr), [float(v) for v in b]))
    if not seq:
        return {}

    tids = sorted(seq)
    covis = {}
    for fr, tl, _ in out_frames:
        if len(tl) < 2:
            continue
        for i in tl:
            covis.setdefault(int(i), set()).update(int(x) for x in tl if x != i)

    med = {t: _median_box([b for _, b in seq[t]]) for t in tids}
    f0 = {t: seq[t][0][0] for t in tids}
    f1 = {t: seq[t][-1][0] for t in tids}

    pairs = []
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            a, b = tids[i], tids[j]
            if len(seq[a]) < min_obs_short or len(seq[b]) < min_obs_short:
                continue
            if b in covis.get(a, ()):          # 共现 → 两人
                continue
            if f1[a] < f0[b] - max_gap_frames or f1[b] < f0[a] - max_gap_frames:
                continue
            ba, bb = med[a], med[b]
            c_ab = containment_batch(ba[None], bb[None])[0, 0]   # a 在 b 内
            c_ba = containment_batch(bb[None], ba[None])[0, 0]   # b 在 a 内
            if c_ab >= cont_thresh:
                inner, outer, c = a, b, c_ab
            elif c_ba >= cont_thresh:
                inner, outer, c = b, a, c_ba
            else:
                continue
            bi, bo = med[inner], med[outer]
            hmax = max(bi[3] - bi[1], bo[3] - bo[1])
            if abs(bi[1] - bo[1]) > y1_tol * hmax:
                continue
            wmax = max(bi[2] - bi[0], bo[2] - bo[0])
            if abs((bi[0] + bi[2]) / 2 - (bo[0] + bo[2]) / 2) > center_ratio * wmax:
                continue
            # 保留观测多的；观测相同时保留 outer
            keep, drop = (outer, inner) if len(seq[outer]) >= len(seq[inner]) else (inner, outer)
            pairs.append((drop, keep, c, len(seq[drop]), len(seq[keep])))

    mapping = {}
    for drop, keep, c, nd, nk in sorted(pairs, key=lambda x: -x[2]):
        if drop in mapping:
            continue
        root = keep
        seen = set()
        while root in mapping and root not in seen:
            seen.add(root)
            root = mapping[root]
        mapping[drop] = root
    return {k: v for k, v in mapping.items() if k != v}
