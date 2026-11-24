#!/usr/bin/env python3
"""Historical Gallery Construction & Orthogonal Ablation Evaluator (E1).

Implements the E1 experimental protocol in Section III-A/B & Section IV-B/C of
`main-ieee-r5.tex`:
  - Holds the exact historical image pool (t <= T) and frozen OSNet encoder
    constant while comparing:
      1. Clustering unit: `per_image` crops vs `track_8mean` (8-frame uniform
         tracklet average, matching the earlier D1 comparator).
      2. Camera conditioning (Eq. 3): `cam_agnostic` (lambda_cam = 0),
         `cam_offset` (finite lambda_cam > 0 + transit gate), and
         `cam_intra_only` (lambda_cam = +inf, strictly intra-camera clusters).
      3. Sparse face association (Eq. 5): `face_off` vs `face_on` (partial map
         pi: C -> J with support/dispersion gates and conflicting-face veto).
  - Evaluates resulting partitions against independent identity labels on:
      * All labeled historical crop pairs, plus an optional curated cross-camera
        reference pair set (`--eval-pairs-csv`, e.g., D1's 34+80 pairs).
      * Pairwise false links (FP), missed links (FN), BCubed Precision/Recall/F1.
      * Auditable clean identities covered, mixed-identity clusters, and true
        cross-camera person-camera pairs linked.
      * Deterministic operator correction cost (manual splits to purify mixed
        groups + manual merges to unify fragmented clean groups) logged
        separately from parameter calibration and evaluation annotation effort.

Performance: Uses vectorized O(N^2) NumPy Lance-Williams average-linkage with
connected-component block partitioning so 10,000+ historical crops run in seconds.
Input compatibility: Accepts the exact same `observations.csv` + `embeddings.npz`
format as `evaluate_e2_replay.py`.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from dataclasses import dataclass
import json
import math
import pathlib
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np


# ==============================================================================
# 1. Configuration (Shared Symbols with Equations 2--5 of main-ieee-r5.tex)
# ==============================================================================

@dataclass(frozen=True)
class E1Config:
    """Frozen parameters for the E1 historical gallery ablation."""

    cutoff_time: float = 3600.0
    # Eq. (2): Crop quality & visibility gate
    eta_det: float = 0.60
    h_min: float = 48.0
    gamma_occ: float = 0.35
    beta_edge: float = 4.0

    # Sampling alignment mode:
    # - `shared_8frame`: selects up to 8 uniform gated crops per tracklet and uses
    #   those EXACT crops for both `per_image` and `track_8mean` (strict E1 parity).
    # - `native_stride`: uses temporal stride for `per_image` and 8 uniform frames
    #   for `track_8mean`.
    sampling_mode: str = "shared_8frame"
    frames_per_track: int = 8
    subsample_stride_sec: float = 0.50

    # Eq. (3): Body clustering parameters
    eps_body_per_image: float = 0.28
    eps_body_track_mean: float = 0.28
    lambda_cam_offset: float = 0.18
    default_transit_sec: float = 2.0

    # Eq. (5): Sparse face clustering & cluster-wide propagation
    eta_face: float = 0.70
    q_min: float = 0.55
    gamma_in: float = 0.85
    alpha_head: float = 0.38
    tau_fclust: float = 0.72
    s_min_face_cluster: int = 1
    tau_flink: float = 0.78
    n_min_prop: int = 1
    rho_max: float = 0.42

    # Evaluation & human-effort accounting
    clean_group_purity: float = 1.00
    calibration_trials_logged: int = 0
    calibration_labeled_pairs_used: int = 0
    manual_eval_annotation_minutes: Optional[float] = None
    random_seed: int = 42


# ==============================================================================
# 2. Data Structures
# ==============================================================================

@dataclass
class HistCrop:
    """Single historical person crop observation (t <= T)."""

    obs_id: str
    track_id: str
    camera_id: str
    timestamp: float
    bbox_h: float
    det_score: float
    max_concurrent_iou: float
    border_margin_px: float
    body_emb: np.ndarray

    has_raw_face: bool = False
    face_conf: float = 0.0
    face_quality: float = 0.0
    face_iof: float = 0.0
    face_rel_y: float = 0.0
    face_overlaps_other_body: bool = False
    face_emb: Optional[np.ndarray] = None

    gt_person_id: Optional[str] = None
    outfit_id: Optional[str] = None
    posture_label: Optional[str] = None


@dataclass
class LabeledPair:
    """Optional external evaluation pair (e.g., D1's 34 same + 80 diff track pairs)."""

    pair_id: str
    unit_kind: str   # track | obs
    id_a: str
    id_b: str
    is_same_person: bool


# ==============================================================================
# 3. Gating, Uniform 8-Frame Selection & Fast Vectorized Lance-Williams Clustering
# ==============================================================================

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Zero-norm embedding encountered.")
    return (vec / norm).astype(np.float64)


def passes_crop_gate(obs: HistCrop, cfg: E1Config) -> bool:
    """Equation (2): visibility, border, height, and detector score gates."""
    return (
        obs.det_score >= cfg.eta_det
        and obs.bbox_h >= cfg.h_min
        and obs.max_concurrent_iou < cfg.gamma_occ
        and obs.border_margin_px >= cfg.beta_edge
    )


def has_unambiguous_paired_face(obs: HistCrop, cfg: E1Config) -> bool:
    """Section III-B (M2): unambiguous same-frame face-to-body geometric pairing."""
    return (
        obs.has_raw_face
        and obs.face_emb is not None
        and obs.face_conf >= cfg.eta_face
        and obs.face_quality >= cfg.q_min
        and obs.face_iof >= cfg.gamma_in
        and 0.0 <= obs.face_rel_y <= cfg.alpha_head
        and not obs.face_overlaps_other_body
    )


def select_uniform_k_per_tracklet(
    raw_crops: Sequence[HistCrop],
    cfg: E1Config,
) -> List[HistCrop]:
    """Selects up to `frames_per_track` (default 8) uniformly spaced gated crops per track."""
    by_track: Dict[str, List[HistCrop]] = {}
    for c in raw_crops:
        if c.timestamp <= cfg.cutoff_time and passes_crop_gate(c, cfg):
            by_track.setdefault(c.track_id, []).append(c)

    selected: List[HistCrop] = []
    k = cfg.frames_per_track
    for track_id in sorted(by_track.keys()):
        items = sorted(by_track[track_id], key=lambda x: (x.timestamp, x.obs_id))
        if len(items) <= k:
            selected.extend(items)
        else:
            raw_idx = np.linspace(0, len(items) - 1, num=k)
            chosen_idx = sorted({int(round(val)) for val in raw_idx})
            selected.extend(items[i] for i in chosen_idx)

    selected.sort(key=lambda x: (x.timestamp, x.obs_id))
    return selected


def select_stride_per_tracklet(
    raw_crops: Sequence[HistCrop],
    cfg: E1Config,
) -> List[HistCrop]:
    """Alternative native stride subsampling (highest det_score per Delta-t window)."""
    by_track: Dict[str, List[HistCrop]] = {}
    for c in raw_crops:
        if c.timestamp <= cfg.cutoff_time and passes_crop_gate(c, cfg):
            by_track.setdefault(c.track_id, []).append(c)

    selected: List[HistCrop] = []
    for track_id in sorted(by_track.keys()):
        items = sorted(by_track[track_id], key=lambda x: (x.timestamp, x.obs_id))
        t0 = items[0].timestamp
        buckets: Dict[int, List[HistCrop]] = {}
        for item in items:
            b = int(math.floor((item.timestamp - t0) / max(cfg.subsample_stride_sec, 1e-6) + 1e-9))
            buckets.setdefault(b, []).append(item)
        for b in sorted(buckets.keys()):
            best = max(
                buckets[b],
                key=lambda x: (1 if has_unambiguous_paired_face(x, cfg) else 0, x.det_score, -x.timestamp),
            )
            selected.append(best)

    selected.sort(key=lambda x: (x.timestamp, x.obs_id))
    return selected


def _lance_williams_block(dist_block: np.ndarray, threshold: float) -> List[List[int]]:
    """Vectorized O(M^2) UPGMA (average-linkage) clustering supporting +inf blocks."""
    m = dist_block.shape[0]
    if m == 0:
        return []
    if m == 1:
        return [[0]]

    d = dist_block.astype(np.float64, copy=True)
    np.fill_diagonal(d, np.inf)
    sizes = np.ones(m, dtype=np.float64)
    members: Dict[int, List[int]] = {i: [i] for i in range(m)}
    alive = np.ones(m, dtype=bool)

    for _ in range(m - 1):
        flat_idx = int(np.argmin(d))
        best_val = d.flat[flat_idx]
        if not np.isfinite(best_val) or best_val > threshold:
            break
        ia, ib = divmod(flat_idx, m)
        if ia > ib:
            ia, ib = ib, ia

        sa, sb = sizes[ia], sizes[ib]
        s_new = sa + sb
        # Lance-Williams UPGMA update; if either distance is +inf, merged distance is +inf
        new_row = (sa * d[ia, :] + sb * d[ib, :]) / s_new
        new_row[~alive] = np.inf
        new_row[ia] = np.inf
        new_row[ib] = np.inf

        d[ia, :] = new_row
        d[:, ia] = new_row
        d[ib, :] = np.inf
        d[:, ib] = np.inf

        sizes[ia] = s_new
        alive[ib] = False
        members[ia].extend(members[ib])
        del members[ib]

    return [sorted(members[k]) for k in sorted(members.keys())]


def average_linkage_clustering(dist_matrix: np.ndarray, threshold: float) -> List[List[int]]:
    """Connected-component pre-partitioning + vectorized O(M^2) UPGMA clustering."""
    n = dist_matrix.shape[0]
    if n == 0:
        return []

    # Connected components under finite candidate edges <= threshold
    adj = np.isfinite(dist_matrix) & (dist_matrix <= threshold)
    visited = np.zeros(n, dtype=bool)
    all_clusters: List[List[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        comp: List[int] = []
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            neighbors = np.flatnonzero(adj[u] & ~visited)
            if len(neighbors) > 0:
                visited[neighbors] = True
                stack.extend(neighbors.tolist())
        comp.sort()
        if len(comp) == 1:
            all_clusters.append(comp)
        else:
            sub_d = dist_matrix[np.ix_(comp, comp)]
            for local_c in _lance_williams_block(sub_d, threshold):
                all_clusters.append([comp[idx] for idx in local_c])

    all_clusters.sort(key=lambda c: c[0])
    return all_clusters


# ==============================================================================
# 4. Two Clustering Units x Three Camera Modes x Two Face Modes
# ==============================================================================

@dataclass
class ClusteringNode:
    """Unified node abstraction: either 1 crop (`per_image`) or 1 track (`track_8mean`)."""

    node_id: str
    track_id: str
    camera_id: str
    t_start: float
    t_end: float
    body_emb: np.ndarray
    member_crop_indices: List[int]


def build_nodes_from_selected_crops(
    selected_crops: Sequence[HistCrop],
    unit: str,
) -> List[ClusteringNode]:
    """Builds either per-image nodes or 8-frame track-averaged nodes on identical crops."""
    assert unit in ("per_image", "track_8mean")
    track_intervals: Dict[str, Tuple[float, float]] = {}
    for c in selected_crops:
        lo, hi = track_intervals.get(c.track_id, (c.timestamp, c.timestamp))
        track_intervals[c.track_id] = (min(lo, c.timestamp), max(hi, c.timestamp))

    if unit == "per_image":
        return [
            ClusteringNode(
                node_id=c.obs_id,
                track_id=c.track_id,
                camera_id=c.camera_id,
                t_start=track_intervals[c.track_id][0],
                t_end=track_intervals[c.track_id][1],
                body_emb=c.body_emb,
                member_crop_indices=[idx],
            )
            for idx, c in enumerate(selected_crops)
        ]

    by_track: Dict[str, List[int]] = {}
    for idx, c in enumerate(selected_crops):
        by_track.setdefault(c.track_id, []).append(idx)

    nodes: List[ClusteringNode] = []
    for track_id in sorted(by_track.keys()):
        idxs = by_track[track_id]
        mean_emb = l2_normalize(np.mean(np.stack([selected_crops[i].body_emb for i in idxs], axis=0), axis=0))
        lo, hi = track_intervals[track_id]
        nodes.append(
            ClusteringNode(
                node_id=track_id,
                track_id=track_id,
                camera_id=selected_crops[idxs[0]].camera_id,
                t_start=lo,
                t_end=hi,
                body_emb=mean_emb,
                member_crop_indices=idxs,
            )
        )
    return nodes


def cluster_body_nodes(
    nodes: Sequence[ClusteringNode],
    camera_mode: str,
    eps_body: float,
    cfg: E1Config,
) -> List[List[int]]:
    """Equation (3): runs vectorized camera-conditioned UPGMA clustering."""
    assert camera_mode in ("cam_agnostic", "cam_offset", "cam_intra_only")
    n = len(nodes)
    if n == 0:
        return []

    embs = np.stack([nd.body_emb for nd in nodes], axis=0)
    cos_dist = np.clip(1.0 - embs @ embs.T, 0.0, 2.0)

    cams = np.array([nd.camera_id for nd in nodes])
    trks = np.array([nd.track_id for nd in nodes])
    t_starts = np.array([nd.t_start for nd in nodes], dtype=np.float64)
    t_ends = np.array([nd.t_end for nd in nodes], dtype=np.float64)

    same_cam = cams[:, None] == cams[None, :]
    same_trk = trks[:, None] == trks[None, :]
    disjoint_time = (t_ends[:, None] < t_starts[None, :]) | (t_ends[None, :] < t_starts[:, None])

    tilde_d = np.full((n, n), np.inf, dtype=np.float64)
    if camera_mode == "cam_agnostic":
        tilde_d[:, :] = cos_dist
    elif camera_mode == "cam_intra_only":
        intra_valid = same_cam & (same_trk | disjoint_time)
        tilde_d[intra_valid] = cos_dist[intra_valid]
    else:  # `cam_offset`
        intra_valid = same_cam & (same_trk | disjoint_time)
        tilde_d[intra_valid] = cos_dist[intra_valid]
        gap = np.maximum(0.0, np.maximum(t_starts[:, None] - t_ends[None, :], t_starts[None, :] - t_ends[:, None]))
        cross_valid = (~same_cam) & (gap >= cfg.default_transit_sec)
        tilde_d[cross_valid] = cos_dist[cross_valid] + cfg.lambda_cam_offset

    np.fill_diagonal(tilde_d, 0.0)
    node_clusters = average_linkage_clustering(tilde_d, eps_body)
    crop_clusters: List[List[int]] = []
    for nc in node_clusters:
        merged_crops: List[int] = []
        for nd_idx in nc:
            merged_crops.extend(nodes[nd_idx].member_crop_indices)
        crop_clusters.append(sorted(merged_crops))
    return crop_clusters


def cluster_historical_faces(
    selected_crops: Sequence[HistCrop],
    cfg: E1Config,
) -> List[np.ndarray]:
    paired = [c for c in selected_crops if has_unambiguous_paired_face(c, cfg)]
    if not paired:
        return []
    f_embs = np.stack([c.face_emb for c in paired], axis=0)
    f_dist = np.clip(1.0 - f_embs @ f_embs.T, 0.0, 2.0)
    raw_clusters = average_linkage_clustering(f_dist, 1.0 - cfg.tau_fclust)
    prototypes: List[np.ndarray] = []
    for fc in raw_clusters:
        if len(fc) >= cfg.s_min_face_cluster:
            w_sum = sum(paired[m].face_quality * paired[m].face_emb for m in fc)
            prototypes.append(l2_normalize(w_sum))
    return prototypes


@dataclass
class PartitionOutcome:
    groups: List[List[int]]
    n_initial_body_clusters: int
    n_face_identities: int
    n_face_linked_body_clusters: int
    n_vetoed_conflict_clusters: int
    n_spared_conflict_crops: int


def apply_face_association_to_clusters(
    selected_crops: Sequence[HistCrop],
    body_clusters: Sequence[Sequence[int]],
    face_prototypes: Sequence[np.ndarray],
    face_on: bool,
    cfg: E1Config,
) -> PartitionOutcome:
    """Equation (5): merges body clusters sharing an anonymous face ID; vetoes conflicts."""
    if not face_on or not face_prototypes:
        return PartitionOutcome(
            groups=[list(c) for c in body_clusters],
            n_initial_body_clusters=len(body_clusters),
            n_face_identities=len(face_prototypes),
            n_face_linked_body_clusters=0,
            n_vetoed_conflict_clusters=0,
            n_spared_conflict_crops=0,
        )

    by_face_j: Dict[int, List[int]] = {}
    unlinked_groups: List[List[int]] = []
    n_linked = 0
    n_vetoed = 0
    n_spared = 0

    for c_members in body_clusters:
        c_embs = np.stack([selected_crops[i].body_emb for i in c_members], axis=0)
        c_mean = l2_normalize(np.mean(c_embs, axis=0))
        rho_ck = float(np.max(1.0 - c_embs @ c_mean))

        support_by_j: Dict[int, List[int]] = {}
        for i_crop in c_members:
            obs = selected_crops[i_crop]
            if has_unambiguous_paired_face(obs, cfg):
                sims = [float(np.dot(obs.face_emb, p)) for p in face_prototypes]
                best_j = int(np.argmax(sims))
                if sims[best_j] >= cfg.tau_flink:
                    support_by_j.setdefault(best_j, []).append(i_crop)

        supported_js = [j for j, idxs in support_by_j.items() if len(idxs) >= 1]
        if len(supported_js) == 1:
            j_star = supported_js[0]
            if len(support_by_j[j_star]) >= cfg.n_min_prop and rho_ck <= cfg.rho_max:
                n_linked += 1
                by_face_j.setdefault(j_star, []).extend(c_members)
            else:
                unlinked_groups.append(list(c_members))
        elif len(supported_js) >= 2:
            n_vetoed += 1
            face_crops = {i_c for idxs in support_by_j.values() for i_c in idxs}
            remainder = [i_c for i_c in c_members if i_c not in face_crops]
            n_spared += len(remainder)
            for j_conf, idxs in support_by_j.items():
                by_face_j.setdefault(j_conf, []).extend(idxs)
            if remainder:
                unlinked_groups.append(remainder)
        else:
            unlinked_groups.append(list(c_members))

    final_groups: List[List[int]] = [sorted(by_face_j[j]) for j in sorted(by_face_j.keys())]
    final_groups.extend(sorted(g) for g in unlinked_groups if g)
    return PartitionOutcome(
        groups=final_groups,
        n_initial_body_clusters=len(body_clusters),
        n_face_identities=len(face_prototypes),
        n_face_linked_body_clusters=n_linked,
        n_vetoed_conflict_clusters=n_vetoed,
        n_spared_conflict_crops=n_spared,
    )


# ==============================================================================
# 5. Vectorized Partition Quality, Coverage & Human-Effort Metrics
# ==============================================================================

def evaluate_partition(
    condition_name: str,
    unit: str,
    camera_mode: str,
    face_mode: str,
    selected_crops: Sequence[HistCrop],
    outcome: PartitionOutcome,
    cfg: E1Config,
    curated_pairs: Optional[Sequence[LabeledPair]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Vectorized computation of pairwise errors, BCubed, coverage & repair ops."""
    crop_to_group = np.zeros(len(selected_crops), dtype=np.int64)
    for g_idx, members in enumerate(outcome.groups):
        for m in members:
            crop_to_group[m] = g_idx

    labeled_indices = [i for i, c in enumerate(selected_crops) if c.gt_person_id is not None]
    n_lab_total = len(labeled_indices)
    if n_lab_total >= 2:
        lab_pids = np.array([selected_crops[i].gt_person_id for i in labeled_indices])
        lab_cams = np.array([selected_crops[i].camera_id for i in labeled_indices])
        lab_grps = crop_to_group[labeled_indices]

        triu_r, triu_c = np.triu_indices(n_lab_total, k=1)
        same_gt = lab_pids[triu_r] == lab_pids[triu_c]
        same_pred = lab_grps[triu_r] == lab_grps[triu_c]
        diff_cam = lab_cams[triu_r] != lab_cams[triu_c]

        tp = int(np.sum(same_gt & same_pred))
        fn = int(np.sum(same_gt & ~same_pred))
        fp = int(np.sum(~same_gt & same_pred))
        xcam_same_total = int(np.sum(same_gt & diff_cam))
        xcam_same_tp = int(np.sum(same_gt & diff_cam & same_pred))
        xcam_diff_fp = int(np.sum(~same_gt & diff_cam & same_pred))
    else:
        tp = fn = fp = xcam_same_total = xcam_same_tp = xcam_diff_fp = 0

    gt_counts: Dict[str, int] = {}
    for ia in labeled_indices:
        pid = selected_crops[ia].gt_person_id
        assert pid is not None
        gt_counts[pid] = gt_counts.get(pid, 0) + 1

    group_gt_breakdown: Dict[int, Dict[str, int]] = {}
    for g_idx, members in enumerate(outcome.groups):
        g_map: Dict[str, int] = {}
        for m in members:
            pid = selected_crops[m].gt_person_id
            if pid is not None:
                g_map[pid] = g_map.get(pid, 0) + 1
        group_gt_breakdown[g_idx] = g_map

    bcubed_precs: List[float] = []
    bcubed_recs: List[float] = []
    for ia in labeled_indices:
        pid = selected_crops[ia].gt_person_id
        assert pid is not None
        g_idx = int(crop_to_group[ia])
        g_map = group_gt_breakdown[g_idx]
        same_in_g = g_map[pid]
        bcubed_precs.append(same_in_g / sum(g_map.values()))
        bcubed_recs.append(same_in_g / gt_counts[pid])

    b_prec = float(np.mean(bcubed_precs)) if bcubed_precs else 0.0
    b_rec = float(np.mean(bcubed_recs)) if bcubed_recs else 0.0
    b_f1 = (2.0 * b_prec * b_rec / (b_prec + b_rec)) if (b_prec + b_rec) > 0 else 0.0

    all_gt_persons = sorted(gt_counts.keys())
    true_person_cam_pairs: Set[Tuple[str, str, str]] = set()
    person_cams_all: Dict[str, Set[str]] = {}
    for ia in labeled_indices:
        pid = selected_crops[ia].gt_person_id
        assert pid is not None
        person_cams_all.setdefault(pid, set()).add(selected_crops[ia].camera_id)
    for pid, cams in person_cams_all.items():
        sc = sorted(cams)
        for u in range(len(sc)):
            for v in range(u + 1, len(sc)):
                true_person_cam_pairs.add((pid, sc[u], sc[v]))

    n_clean_groups = 0
    n_mixed_groups = 0
    clean_groups_per_person: Dict[str, int] = {pid: 0 for pid in all_gt_persons}
    linked_person_cam_pairs: Set[Tuple[str, str, str]] = set()
    max_cams_in_single_clean_group: Dict[str, int] = {pid: 0 for pid in all_gt_persons}

    manual_splits_needed = 0
    for g_idx, members in enumerate(outcome.groups):
        g_map = group_gt_breakdown[g_idx]
        n_lab = sum(g_map.values())
        if n_lab == 0:
            continue
        top_pid, top_cnt = max(g_map.items(), key=lambda kv: kv[1])
        purity = top_cnt / n_lab
        if len(g_map) == 1 or purity >= cfg.clean_group_purity:
            n_clean_groups += 1
            clean_groups_per_person[top_pid] += 1
            cams_in_g = sorted({
                selected_crops[m].camera_id
                for m in members
                if selected_crops[m].gt_person_id == top_pid
            })
            max_cams_in_single_clean_group[top_pid] = max(
                max_cams_in_single_clean_group[top_pid], len(cams_in_g)
            )
            for u in range(len(cams_in_g)):
                for v in range(u + 1, len(cams_in_g)):
                    linked_person_cam_pairs.add((top_pid, cams_in_g[u], cams_in_g[v]))
        else:
            n_mixed_groups += 1
            manual_splits_needed += len(g_map) - 1

    auditable_identities = sum(1 for pid in all_gt_persons if clean_groups_per_person[pid] >= 1)
    fragments_after_splits: Dict[str, int] = dict(clean_groups_per_person)
    for g_idx in range(len(outcome.groups)):
        g_map = group_gt_breakdown[g_idx]
        if len(g_map) > 1 and (max(g_map.values()) / sum(g_map.values()) < cfg.clean_group_purity):
            for pid in g_map:
                fragments_after_splits[pid] = fragments_after_splits.get(pid, 0) + 1
    manual_merges_needed = sum(max(0, cnt - 1) for cnt in fragments_after_splits.values())

    curated_same_hit = curated_same_tot = curated_diff_fp = curated_diff_tot = 0
    if curated_pairs:
        obs_map = {c.obs_id: int(crop_to_group[i]) for i, c in enumerate(selected_crops)}
        trk_groups: Dict[str, Set[int]] = {}
        for i, c in enumerate(selected_crops):
            trk_groups.setdefault(c.track_id, set()).add(int(crop_to_group[i]))

        for pr in curated_pairs:
            if pr.unit_kind == "obs":
                if pr.id_a not in obs_map or pr.id_b not in obs_map:
                    continue
                is_linked = obs_map[pr.id_a] == obs_map[pr.id_b]
            else:
                if pr.id_a not in trk_groups or pr.id_b not in trk_groups:
                    continue
                is_linked = bool(trk_groups[pr.id_a] & trk_groups[pr.id_b])

            if pr.is_same_person:
                curated_same_tot += 1
                if is_linked:
                    curated_same_hit += 1
            else:
                curated_diff_tot += 1
                if is_linked:
                    curated_diff_fp += 1

    summary_row = {
        "condition": condition_name,
        "unit": unit,
        "camera_mode": camera_mode,
        "face_mode": face_mode,
        "n_selected_crops": len(selected_crops),
        "n_initial_body_clusters": outcome.n_initial_body_clusters,
        "n_final_groups": len(outcome.groups),
        "n_clean_groups": n_clean_groups,
        "n_mixed_groups": n_mixed_groups,
        "auditable_identities_covered": f"{auditable_identities}/{len(all_gt_persons)}",
        "person_cam_pairs_linked": f"{len(linked_person_cam_pairs)}/{len(true_person_cam_pairs)}",
        "mean_max_cams_per_person": round(float(np.mean(list(max_cams_in_single_clean_group.values()))) if all_gt_persons else 0.0, 3),
        "pairwise_false_links_fp": fp,
        "pairwise_missed_links_fn": fn,
        "pairwise_same_hits_tp": tp,
        "xcam_same_hits": f"{xcam_same_tp}/{xcam_same_total}",
        "xcam_false_links_fp": xcam_diff_fp,
        "bcubed_precision": round(b_prec, 4),
        "bcubed_recall": round(b_rec, 4),
        "bcubed_f1": round(b_f1, 4),
        "curated_same_hits": f"{curated_same_hit}/{curated_same_tot}" if curated_pairs else "na",
        "curated_diff_false_links": f"{curated_diff_fp}/{curated_diff_tot}" if curated_pairs else "na",
        "vetoed_conflict_clusters": outcome.n_vetoed_conflict_clusters,
        "spared_conflict_crops": outcome.n_spared_conflict_crops,
    }

    effort_row = {
        "condition": condition_name,
        "manual_splits_to_purify_mixed": manual_splits_needed,
        "manual_merges_to_unify_person": manual_merges_needed,
        "total_gallery_repair_ops": manual_splits_needed + manual_merges_needed,
        "multi_cam_groups_for_audit": sum(
            1 for members in outcome.groups
            if len({selected_crops[m].camera_id for m in members}) >= 2
        ),
        "calibration_trials_logged": cfg.calibration_trials_logged,
        "calibration_labeled_pairs_used": cfg.calibration_labeled_pairs_used,
        "independent_eval_labeled_crops": n_lab_total,
        "independent_eval_gt_persons": len(all_gt_persons),
        "independent_eval_annotation_minutes": (
            cfg.manual_eval_annotation_minutes if cfg.manual_eval_annotation_minutes is not None else "unreported"
        ),
    }
    return summary_row, effort_row


def render_latex_e1_table(
    summaries: Sequence[Dict[str, object]],
    efforts: Sequence[Dict[str, object]],
) -> str:
    eff_map = {str(e["condition"]): e for e in efforts}
    core_cam_modes = {"cam_agnostic", "cam_intra_only"}
    rows_tex: List[str] = []

    cam_label = {
        "cam_agnostic": "Agnostic ($\\lambda_{\\mathrm{cam}}=0$)",
        "cam_intra_only": "Intra-cam ($\\lambda_{\\mathrm{cam}}=\\infty$)",
    }
    unit_label = {
        "track_8mean": "8-frame track",
        "per_image": "Per-image",
    }

    for s in summaries:
        if s["camera_mode"] not in core_cam_modes:
            continue
        cond = str(s["condition"])
        ef = eff_map[cond]
        u_str = unit_label[str(s["unit"])]
        c_str = cam_label[str(s["camera_mode"])]
        f_str = "On" if s["face_mode"] == "face_on" else "Off"
        rows_tex.append(
            f"{u_str} & {c_str} & {f_str} & {s['n_final_groups']} ({s['n_mixed_groups']}) & "
            f"{s['pairwise_false_links_fp']} / {s['pairwise_missed_links_fn']} & "
            f"{s['auditable_identities_covered']} & {s['person_cam_pairs_linked']} & "
            f"{ef['total_gallery_repair_ops']} \\\\"
        )

    return "\n".join([
        r"\begin{table*}[t]",
        r"\caption{Historical gallery ablation (E1) on identical selected images and frozen OSNet features. Groups reports total (mixed) partitions; Repair ops counts manual splits plus merges to recover pure ground-truth identities.}",
        r"\label{tab:e1_historical}",
        r"\centering",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lllccccc@{}}",
        r"\toprule",
        r"Unit & Camera rule & Face $\pi$ & Groups (mixed) & False / Missed pairs & Clean IDs & Cam pairs & Repair ops\\",
        r"\midrule",
        *rows_tex,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])


# ==============================================================================
# 6. CSV Loading & Realistic D2-Calibrated Synthetic Verification (`--selftest`)
# ==============================================================================

def write_csv(path: pathlib.Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_hist_crops_from_csv_and_npz(
    csv_path: pathlib.Path,
    npz_path: pathlib.Path,
    cutoff_time: float,
) -> List[HistCrop]:
    arrays = np.load(npz_path)
    body_embs = arrays["body_embs"]
    face_embs = arrays["face_embs"] if "face_embs" in arrays else None

    crops: List[HistCrop] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["timestamp"])
            if ts > cutoff_time:
                continue
            b_idx = int(row["body_emb_idx"])
            f_idx = int(row["face_emb_idx"]) if row.get("face_emb_idx", "") not in ("", "-1") else -1
            f_vec = l2_normalize(face_embs[f_idx]) if (face_embs is not None and f_idx >= 0) else None
            crops.append(
                HistCrop(
                    obs_id=row["obs_id"],
                    track_id=row["track_id"],
                    camera_id=row["camera_id"],
                    timestamp=ts,
                    bbox_h=float(row["bbox_h"]),
                    det_score=float(row["det_score"]),
                    max_concurrent_iou=float(row["max_concurrent_iou"]),
                    border_margin_px=float(row["border_margin_px"]),
                    body_emb=l2_normalize(body_embs[b_idx]),
                    has_raw_face=row.get("has_raw_face", "0") in ("1", "true", "True", "yes"),
                    face_conf=float(row.get("face_conf") or 0.0),
                    face_quality=float(row.get("face_quality") or 0.0),
                    face_iof=float(row.get("face_iof") or 0.0),
                    face_rel_y=float(row.get("face_rel_y") or 0.0),
                    face_overlaps_other_body=row.get("face_overlaps_other_body", "0") in ("1", "true", "True", "yes"),
                    face_emb=f_vec,
                    gt_person_id=row.get("gt_person_id") or None,
                    outfit_id=row.get("outfit_id") or None,
                    posture_label=row.get("posture_label") or None,
                )
            )
    return crops


def load_curated_pairs_csv(path: pathlib.Path) -> List[LabeledPair]:
    pairs: List[LabeledPair] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append(
                LabeledPair(
                    pair_id=row["pair_id"],
                    unit_kind=row.get("unit_kind", "track"),
                    id_a=row["id_a"],
                    id_b=row["id_b"],
                    is_same_person=row["is_same_person"] in ("1", "true", "True", "yes"),
                )
            )
    return pairs


def generate_synthetic_e1_capture(cfg: E1Config) -> Tuple[List[HistCrop], List[LabeledPair]]:
    """Generates a realistic historical capture exercising all 3 E1 mechanisms:
      1. Cross-camera dark-coat impostor pairs (false merges under `cam_agnostic`,
         eliminated by `cam_intra_only`).
      2. Standing-vs-sitting similarity drop (~0.56, matching D2 Section V-B) plus
         one ByteTrack ID-switch tracklet mixing P_08 and P_09 (irreversibly mixed
         under `track_8mean`, cleanly separated by `per_image` + Eq. 5 conflict veto).
      3. Sparse faces (7–8 of 10 people) linking pure cross-camera clusters via pi.
    """
    rng = np.random.default_rng(cfg.random_seed)
    d_body, d_face = 64, 32
    n_people = 10
    crops: List[HistCrop] = []
    curated_pairs: List[LabeledPair] = []

    shared_dark_coat = l2_normalize(rng.normal(size=d_body))
    seated_desk_mode = l2_normalize(rng.normal(size=d_body))
    face_protos = {f"P_{p:02d}": l2_normalize(rng.normal(size=d_face)) for p in range(n_people)}
    id_vecs = {f"P_{p:02d}": l2_normalize(rng.normal(size=d_body)) for p in range(n_people)}

    stand_vecs_a: Dict[str, np.ndarray] = {}
    sit_vecs_b: Dict[str, np.ndarray] = {}

    for p in range(n_people):
        pid = f"P_{p:02d}"
        # P_00 & P_01 wear nearly identical coats in Cam A vs Cam B -> cross-cam impostor!
        coat = shared_dark_coat if p in (0, 1) else l2_normalize(rng.normal(size=d_body))
        stand_vecs_a[pid] = l2_normalize(0.42 * id_vecs[pid] + 0.90 * coat)
        sit_vecs_b[pid] = l2_normalize(0.45 * id_vecs[pid] + 0.45 * coat + 0.77 * seated_desk_mode)

    for p in range(n_people):
        pid = f"P_{p:02d}"
        # Cam A standing tracklet (8 frames)
        trk_a = f"trk_A_{pid}"
        for f_idx in range(8):
            has_f = (p < 8 and f_idx == 0)
            crops.append(
                HistCrop(
                    obs_id=f"{trk_a}_f{f_idx}",
                    track_id=trk_a,
                    camera_id="camA",
                    timestamp=50.0 + p * 35.0 + f_idx * 0.5,
                    bbox_h=240.0 + rng.uniform(-5, 5),
                    det_score=0.93,
                    max_concurrent_iou=0.04,
                    border_margin_px=16.0,
                    body_emb=l2_normalize(stand_vecs_a[pid] + 0.03 * rng.normal(size=d_body)),
                    has_raw_face=has_f,
                    face_conf=0.94 if has_f else 0.0,
                    face_quality=0.85 if has_f else 0.0,
                    face_iof=0.96 if has_f else 0.0,
                    face_rel_y=0.18 if has_f else 0.0,
                    face_emb=l2_normalize(face_protos[pid] + 0.02 * rng.normal(size=d_face)) if has_f else None,
                    gt_person_id=pid,
                    outfit_id="coat1",
                    posture_label="standing",
                )
            )

        # Cam B tracklet: for P_08, simulate a ByteTrack ID-switch mid-tracklet to P_09!
        trk_b = f"trk_B_{pid}"
        for f_idx in range(8):
            actual_pid = "P_09" if (pid == "P_08" and f_idx >= 4) else pid
            has_f = (p < 7 and f_idx == 2) or (pid == "P_08" and f_idx in (1, 6))
            crops.append(
                HistCrop(
                    obs_id=f"{trk_b}_f{f_idx}",
                    track_id=trk_b,
                    camera_id="camB",
                    timestamp=500.0 + p * 35.0 + f_idx * 0.5,
                    bbox_h=84.0 + rng.uniform(-4, 4),
                    det_score=0.91,
                    max_concurrent_iou=0.05,
                    border_margin_px=15.0,
                    body_emb=l2_normalize(sit_vecs_b[actual_pid] + 0.03 * rng.normal(size=d_body)),
                    has_raw_face=has_f,
                    face_conf=0.92 if has_f else 0.0,
                    face_quality=0.83 if has_f else 0.0,
                    face_iof=0.95 if has_f else 0.0,
                    face_rel_y=0.20 if has_f else 0.0,
                    face_emb=l2_normalize(face_protos[actual_pid] + 0.02 * rng.normal(size=d_face)) if has_f else None,
                    gt_person_id=actual_pid,
                    outfit_id="coat1",
                    posture_label="sitting",
                )
            )

        if pid not in ("P_08", "P_09"):
            curated_pairs.append(LabeledPair(f"same_{pid}", "track", trk_a, trk_b, True))

    curated_pairs.append(LabeledPair("impostor_0_1_camAB", "track", "trk_A_P_00", "trk_B_P_01", False))
    curated_pairs.append(LabeledPair("impostor_1_0_camAB", "track", "trk_A_P_01", "trk_B_P_00", False))
    return crops, curated_pairs


def run_e1_suite(
    raw_crops: Sequence[HistCrop],
    cfg: E1Config,
    out_dir: pathlib.Path,
    curated_pairs: Optional[Sequence[LabeledPair]] = None,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_crops = select_uniform_k_per_tracklet(raw_crops, cfg)
    stride_crops = (
        select_stride_per_tracklet(raw_crops, cfg)
        if cfg.sampling_mode == "native_stride"
        else shared_crops
    )

    summaries: List[Dict[str, object]] = []
    efforts: List[Dict[str, object]] = []
    group_membership_rows: List[Dict[str, object]] = []

    units = ["track_8mean", "per_image"]
    cam_modes = ["cam_agnostic", "cam_offset", "cam_intra_only"]
    face_modes = ["face_off", "face_on"]

    for unit in units:
        active_crops = stride_crops if (unit == "per_image" and cfg.sampling_mode == "native_stride") else shared_crops
        face_protos = cluster_historical_faces(active_crops, cfg)
        nodes = build_nodes_from_selected_crops(active_crops, unit)
        eps = cfg.eps_body_per_image if unit == "per_image" else cfg.eps_body_track_mean

        for cam_mode in cam_modes:
            body_clusters = cluster_body_nodes(nodes, cam_mode, eps, cfg)
            for face_mode in face_modes:
                cond_name = f"{unit}__{cam_mode}__{face_mode}"
                outcome = apply_face_association_to_clusters(
                    active_crops,
                    body_clusters,
                    face_protos,
                    face_on=(face_mode == "face_on"),
                    cfg=cfg,
                )
                s_row, e_row = evaluate_partition(
                    cond_name, unit, cam_mode, face_mode, active_crops, outcome, cfg, curated_pairs
                )
                summaries.append(s_row)
                efforts.append(e_row)

                for g_idx, members in enumerate(outcome.groups):
                    for m in members:
                        c = active_crops[m]
                        group_membership_rows.append({
                            "condition": cond_name,
                            "group_id": f"G_{g_idx:04d}",
                            "obs_id": c.obs_id,
                            "track_id": c.track_id,
                            "camera_id": c.camera_id,
                            "timestamp": c.timestamp,
                            "gt_person_id": c.gt_person_id or "",
                            "posture_label": c.posture_label or "",
                        })

    write_csv(out_dir / "summary_e1_ablation.csv", summaries)
    write_csv(out_dir / "ledger_e1_human_effort.csv", efforts)
    write_csv(out_dir / "ledger_e1_partitions.csv", group_membership_rows)

    latex_table = render_latex_e1_table(summaries, efforts)
    (out_dir / "table_e1_populated.tex").write_text(latex_table + "\n", encoding="utf-8")

    manifest = {
        "n_raw_historical_detections": len(raw_crops),
        "n_selected_shared_crops": len(shared_crops),
        "sampling_mode": cfg.sampling_mode,
        "config": dataclasses.asdict(cfg),
    }
    (out_dir / "e1_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "summaries": summaries, "efforts": efforts, "latex": latex_table}


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 Historical Gallery Orthogonal Ablation Evaluator")
    parser.add_argument("--observations-csv", type=pathlib.Path, help="Historical/site observations CSV")
    parser.add_argument("--embeddings-npz", type=pathlib.Path, help="NPZ containing body_embs and face_embs")
    parser.add_argument("--eval-pairs-csv", type=pathlib.Path, help="Optional D1-style curated reference pairs CSV")
    parser.add_argument("--config-json", type=pathlib.Path, help="Optional JSON overriding E1Config parameters")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True, help="Output directory for E1 CSVs and LaTeX")
    parser.add_argument("--selftest", action="store_true", help="Run built-in synthetic multi-camera E1 verification")
    args = parser.parse_args()

    cfg_dict = json.loads(args.config_json.read_text(encoding="utf-8")) if args.config_json else {}
    cfg = E1Config(**cfg_dict)

    if args.selftest:
        crops, curated = generate_synthetic_e1_capture(cfg)
    else:
        if not args.observations_csv or not args.embeddings_npz:
            parser.error("Provide --observations-csv and --embeddings-npz, or pass --selftest.")
        crops = load_hist_crops_from_csv_and_npz(args.observations_csv, args.embeddings_npz, cfg.cutoff_time)
        curated = load_curated_pairs_csv(args.eval_pairs_csv) if args.eval_pairs_csv else None

    res = run_e1_suite(crops, cfg, args.out_dir, curated)
    print("=== E1 RUN MANIFEST ===")
    print(json.dumps(res["manifest"], indent=2))
    print("\n=== GENERATED LATEX E1 TABLE ===")
    print(res["latex"])


if __name__ == "__main__":
    main()