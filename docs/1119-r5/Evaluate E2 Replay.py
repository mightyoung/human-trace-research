#!/usr/bin/env python3
"""Chronological Forward-Replay & Paired-Ledger Evaluator for Table I (E2).

Implements the exact mathematical formulation in Section III & Section IV of
`main-ieee-r5.tex` (Equations 2--10):
  - Eq. (2)-(3): Per-image crop gating, temporal subsampling, and camera-conditioned
    average-linkage body clustering (`body_only` vs `face_linked`).
  - Eq. (4)-(5): Historical face clustering up to cutoff T, anchored face prototypes,
    cluster-wide identity propagation with support/dispersion gates, and conflict veto.
  - Eq. (6)-(9): Strictly causal pre-write query scoring with normalized
    height-weighted cosine similarity and open-set margin refusal.
  - Eq. (10): Face-authorized template writes (one max-quality write per tracklet,
    with same-tracklet contradiction rollback), near-duplicate suppression, protected
    historical core + redundancy eviction, new-identity staging, and unresolved
    write-opportunity logging.
  - Section IV-B/C: Identical eligible faceless query pool isolated from writing
    tracklets, enrollment-fixed gallery-to-GT mapping, separate out-of-gallery and
    non-person denominators, paired recovery/loss ledgers, E3 stratum breakdowns,
    person-level cluster bootstrap CIs, and ready-to-paste LaTeX Table I output.

Dependencies: Python 3.9+ standard library and NumPy only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


# ==============================================================================
# 1. Hyperparameter Configuration (Aligned with Equations 2--10)
# ==============================================================================

@dataclass(frozen=True)
class ReplayConfig:
    """Frozen protocol parameters for historical enrollment and forward replay."""

    cutoff_time: float = 3600.0
    # Eq. (2): Crop selection & within-tracklet subsampling
    eta_det: float = 0.60
    h_min: float = 48.0
    gamma_occ: float = 0.35
    beta_edge: float = 4.0
    subsample_stride_sec: float = 0.50

    # Eq. (3): Camera-conditioned body clustering
    intra_camera_only: bool = True
    lambda_cam: float = 0.25
    default_transit_sec: float = 2.0
    eps_body: float = 0.28
    max_hist_medoids_per_cluster: int = 2

    # Eq. (5): Face pairing, clustering & cluster-wide propagation
    eta_face: float = 0.70
    q_min: float = 0.55
    gamma_in: float = 0.85
    alpha_head: float = 0.38
    tau_fclust: float = 0.72
    s_min_face_cluster: int = 1
    tau_flink: float = 0.78
    n_min_prop: int = 1
    rho_max: float = 0.35

    # Eq. (7)-(9): Scale-conditioned body scoring & refusal
    # `similarity_temp_beta = 0.0` implements exact linear Eq. (8); setting > 0
    # softly upweights the matching outfit among equal-height templates.
    sigma_height: float = 0.60
    similarity_temp_beta: float = 0.0
    tau_body: float = 0.80
    delta_body: float = 0.00
    tau_geff_body: float = 0.80

    # Eq. (10): Face-authorized writes, budget R_j, refresh & new identities
    tau_face_write: float = 0.80
    delta_face_write: float = 0.02
    template_budget_m: int = 8
    max_protected_hist: int = 4
    sigma_dup: float = 0.15
    tau_dup: float = 0.92
    tau_fenroll: float = 0.88
    face_horizon_sec: Optional[float] = None
    tau_new_id: float = 0.55
    n_new_corroborate: int = 2

    # Section IV-B/C: Query-write isolation & GT mapping
    near_dup_isolation_sec: float = 15.0
    gt_purity_threshold: float = 0.80
    bootstrap_iters: int = 2000
    random_seed: int = 42


# ==============================================================================
# 2. Data Structures
# ==============================================================================

@dataclass
class Observation:
    """Single person crop observation from historical or replay video."""

    obs_id: str
    track_id: str
    camera_id: str
    timestamp: float
    bbox_h: float
    det_score: float
    max_concurrent_iou: float
    border_margin_px: float
    body_emb: np.ndarray  # L2-normalized (D_b,)

    # Optional face observation detected in the same frame
    has_raw_face: bool = False
    face_conf: float = 0.0
    face_quality: float = 0.0
    face_iof: float = 0.0
    face_rel_y: float = 0.0
    face_overlaps_other_body: bool = False
    face_emb: Optional[np.ndarray] = None  # L2-normalized (D_f,)

    # Independent evaluation annotations (never accessed by clustering/matching!)
    query_id: Optional[str] = None
    population_type: str = "in_gallery"  # in_gallery | absent_from_gallery | non_person
    gt_person_id: Optional[str] = None
    outfit_id: Optional[str] = None
    posture_label: Optional[str] = None  # standing | sitting | unknown


@dataclass
class BodyTemplate:
    """Stored body appearance template u_r inside G_j(t)."""

    template_id: str
    obs_id: str
    track_id: str
    camera_id: str
    timestamp: float
    bbox_h: float
    body_emb: np.ndarray
    is_protected_hist: bool
    write_face_quality: float = 0.0
    source_cluster_id: Optional[int] = None
    gt_person_id_audit: Optional[str] = None


@dataclass
class FaceExemplar:
    """Stored face reference vector inside V_j(t)."""

    obs_id: str
    track_id: str
    timestamp: float
    quality: float
    face_emb: np.ndarray


@dataclass
class GalleryIdentity:
    """Anonymous site identity j in J(t)."""

    anon_id: str
    created_at: float
    body_templates: List[BodyTemplate] = field(default_factory=list)
    face_exemplars: List[FaceExemplar] = field(default_factory=list)
    anchored_prototype: Optional[np.ndarray] = None
    eval_status: str = "unresolved"  # clean | mixed | unresolved | runtime_new
    eval_gt_person_id: Optional[str] = None


@dataclass
class WriteAuditRecord:
    """Audit log entry for every replay face observation opportunity."""

    condition: str
    track_id: str
    obs_id: str
    camera_id: str
    timestamp: float
    gt_person_id: Optional[str]
    face_quality: float
    top1_anon_id: Optional[str]
    top1_face_sim: float
    face_margin: float
    outcome: str  # written | near_dup_refreshed | unresolved_band | staged_new | spawned_new | revoked_contradiction
    written_anon_id: Optional[str]
    mapped_gt_person_id: Optional[str]
    is_attribution_correct: Optional[bool]


@dataclass
class QueryDecisionRecord:
    """Single-query decision record under a specific historical + replay condition."""

    condition: str
    historical_build: str
    replay_rule: str
    query_id: str
    track_id: str
    camera_id: str
    timestamp: float
    population_type: str
    gt_person_id: str
    bbox_h: float
    is_cross_camera: str
    is_clothes_changed: str
    posture_label: str
    predicted_anon_id: str
    mapped_gt_person_id: str
    top1_score: float
    top2_score: float
    score_margin: float
    decision: str   # assign | refuse
    verdict: str    # correct | wrong_id | mixed_group_error | refused | false_accept_absent | false_accept_nonperson | correct_refusal


# ==============================================================================
# 3. Core Mathematical Helpers (Equations 2--10)
# ==============================================================================

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Encountered zero-norm embedding vector.")
    return (vec / norm).astype(np.float64)


def passes_crop_gate(obs: Observation, cfg: ReplayConfig) -> bool:
    """Equation (2): quality, height, occlusion, and frame-border gates."""
    return (
        obs.det_score >= cfg.eta_det
        and obs.bbox_h >= cfg.h_min
        and obs.max_concurrent_iou < cfg.gamma_occ
        and obs.border_margin_px >= cfg.beta_edge
    )


def has_unambiguous_paired_face(obs: Observation, cfg: ReplayConfig) -> bool:
    """Section III-B (M2): geometric & quality face-to-body pairing rule."""
    return (
        obs.has_raw_face
        and obs.face_emb is not None
        and obs.face_conf >= cfg.eta_face
        and obs.face_quality >= cfg.q_min
        and obs.face_iof >= cfg.gamma_in
        and 0.0 <= obs.face_rel_y <= cfg.alpha_head
        and not obs.face_overlaps_other_body
    )


def subsample_tracklets(
    observations: Sequence[Observation],
    cfg: ReplayConfig,
) -> List[Observation]:
    """Within each single-camera tracklet, keep highest-confidence crop per stride window."""
    by_track: Dict[str, List[Observation]] = {}
    for obs in observations:
        if passes_crop_gate(obs, cfg):
            by_track.setdefault(obs.track_id, []).append(obs)

    selected: List[Observation] = []
    for track_id in sorted(by_track.keys()):
        items = sorted(by_track[track_id], key=lambda x: x.timestamp)
        if cfg.subsample_stride_sec <= 0:
            selected.extend(items)
            continue
        t0 = items[0].timestamp
        buckets: Dict[int, List[Observation]] = {}
        for item in items:
            b_idx = int(math.floor((item.timestamp - t0) / cfg.subsample_stride_sec + 1e-9))
            buckets.setdefault(b_idx, []).append(item)
        for b_idx in sorted(buckets.keys()):
            best = max(
                buckets[b_idx],
                key=lambda x: (1 if has_unambiguous_paired_face(x, cfg) else 0, x.det_score, -x.timestamp),
            )
            selected.append(best)

    selected.sort(key=lambda x: (x.timestamp, x.obs_id))
    return selected


def average_linkage_clustering(dist_matrix: np.ndarray, threshold: float) -> List[List[int]]:
    """Deterministic agglomerative average-linkage clustering with +inf support."""
    n = dist_matrix.shape[0]
    if n == 0:
        return []
    clusters: Dict[int, List[int]] = {i: [i] for i in range(n)}
    active = sorted(clusters.keys())

    def cluster_dist(ca: List[int], cb: List[int]) -> float:
        sub = dist_matrix[np.ix_(ca, cb)]
        if np.any(np.isinf(sub)):
            return float("inf")
        return float(np.mean(sub))

    while len(active) > 1:
        best_pair: Optional[Tuple[int, int]] = None
        best_d = float("inf")
        for idx_a in range(len(active)):
            ka = active[idx_a]
            for idx_b in range(idx_a + 1, len(active)):
                kb = active[idx_b]
                d = cluster_dist(clusters[ka], clusters[kb])
                if d < best_d:
                    best_d = d
                    best_pair = (ka, kb)
        if best_pair is None or best_d > threshold:
            break
        ka, kb = best_pair
        clusters[ka] = clusters[ka] + clusters[kb]
        del clusters[kb]
        active.remove(kb)

    return [sorted(clusters[k]) for k in sorted(clusters.keys())]


def build_camera_conditioned_body_clusters(
    hist_crops: Sequence[Observation],
    cfg: ReplayConfig,
    transit_lookup: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[List[int]]:
    """Equation (3): camera-conditioned dissimilarity matrix + average-linkage."""
    n = len(hist_crops)
    if n == 0:
        return []

    track_intervals: Dict[str, Tuple[float, float]] = {}
    for obs in hist_crops:
        lo, hi = track_intervals.get(obs.track_id, (obs.timestamp, obs.timestamp))
        track_intervals[obs.track_id] = (min(lo, obs.timestamp), max(hi, obs.timestamp))

    embs = np.stack([c.body_emb for c in hist_crops], axis=0)
    cos_dist = np.clip(1.0 - embs @ embs.T, 0.0, 2.0)
    tilde_d = np.full((n, n), float("inf"), dtype=np.float64)
    np.fill_diagonal(tilde_d, 0.0)

    for a in range(n):
        ca = hist_crops[a]
        lo_a, hi_a = track_intervals[ca.track_id]
        for b in range(a + 1, n):
            cb = hist_crops[b]
            lo_b, hi_b = track_intervals[cb.track_id]
            if ca.camera_id == cb.camera_id:
                same_track = ca.track_id == cb.track_id
                disjoint_time = hi_a < lo_b or hi_b < lo_a
                if same_track or disjoint_time:
                    tilde_d[a, b] = tilde_d[b, a] = cos_dist[a, b]
            else:
                if not cfg.intra_camera_only:
                    t_min = (
                        transit_lookup.get((ca.camera_id, cb.camera_id), cfg.default_transit_sec)
                        if transit_lookup
                        else cfg.default_transit_sec
                    )
                    if abs(ca.timestamp - cb.timestamp) >= t_min:
                        tilde_d[a, b] = tilde_d[b, a] = cos_dist[a, b] + cfg.lambda_cam

    return average_linkage_clustering(tilde_d, cfg.eps_body)


def select_scale_stratified_medoids(
    cluster_indices: Sequence[int],
    hist_crops: Sequence[Observation],
    max_medoids: int,
) -> List[int]:
    """Selects up to max_medoids representative indices preserving height diversity."""
    if len(cluster_indices) <= max_medoids:
        return list(cluster_indices)
    sorted_by_h = sorted(cluster_indices, key=lambda idx: hist_crops[idx].bbox_h)
    if max_medoids == 1:
        embs = np.stack([hist_crops[i].body_emb for i in cluster_indices], axis=0)
        best_local = int(np.argmax(np.sum(embs @ embs.T, axis=1)))
        return [cluster_indices[best_local]]

    chosen: List[int] = []
    bins = np.array_split(np.array(sorted_by_h, dtype=int), max_medoids)
    for b in bins:
        if len(b) == 0:
            continue
        b_list = b.tolist()
        embs = np.stack([hist_crops[i].body_emb for i in b_list], axis=0)
        best_local = int(np.argmax(np.sum(embs @ embs.T, axis=1)))
        chosen.append(b_list[best_local])
    return chosen


def height_weight(h_q: float, h_r: float, sigma: float) -> float:
    """Equation (7): w(h_q, h_r) = exp(-|log h_q - log h_r| / sigma)."""
    return math.exp(-abs(math.log(max(h_q, 1.0)) - math.log(max(h_r, 1.0))) / sigma)


def score_identity_proposed(
    query_emb: np.ndarray,
    query_h: float,
    templates: Sequence[BodyTemplate],
    sigma: float,
    temp_beta: float = 0.0,
) -> float:
    """Equation (8): normalized height-weighted cosine score S(x_q, j; t^-).

    When `temp_beta == 0.0` (default), computes exact linear Eq. (8).
    """
    if not templates:
        return -1.0
    sims = np.array([float(np.dot(query_emb, tpl.body_emb)) for tpl in templates], dtype=np.float64)
    h_weights = np.array([height_weight(query_h, tpl.bbox_h, sigma) for tpl in templates], dtype=np.float64)
    if temp_beta > 0.0:
        shifted = temp_beta * (sims - np.max(sims))
        weights = h_weights * np.exp(shifted)
    else:
        weights = h_weights
    den = float(np.sum(weights))
    return float(np.sum(weights * sims) / den) if den > 1e-12 else -1.0


def score_identity_max(
    query_emb: np.ndarray,
    templates: Sequence[BodyTemplate],
) -> float:
    """GEFF baseline (Table I Row 2): per-identity maximum cosine similarity."""
    if not templates:
        return -1.0
    return max(float(np.dot(query_emb, tpl.body_emb)) for tpl in templates)


# ==============================================================================
# 4. Historical Gallery Construction (t <= T): Body-Only vs Face-Linked
# ==============================================================================

@dataclass
class HistoricalBuildResult:
    build_name: str  # body_only | face_linked
    identities: Dict[str, GalleryIdentity]
    num_body_clusters: int
    num_face_clusters: int
    num_linked_clusters: int
    num_vetoed_conflict_clusters: int
    num_spared_conflict_crops: int
    enrolled_gt_persons: set
    enrolled_person_cameras: Dict[str, set]
    enrolled_person_outfits: Dict[str, set]


def construct_historical_gallery(
    hist_crops: Sequence[Observation],
    body_clusters: Sequence[Sequence[int]],
    build_name: str,
    cfg: ReplayConfig,
) -> HistoricalBuildResult:
    """Constructs either `body_only` or `face_linked` historical gallery at t = T."""
    assert build_name in ("body_only", "face_linked")

    paired_face_records: List[Tuple[int, Observation]] = [
        (idx, hist_crops[idx])
        for idx in range(len(hist_crops))
        if has_unambiguous_paired_face(hist_crops[idx], cfg)
    ]

    face_clusters_j: List[List[int]] = []
    face_prototypes: List[np.ndarray] = []
    if paired_face_records:
        f_embs = np.stack([rec[1].face_emb for rec in paired_face_records], axis=0)
        f_dist = np.clip(1.0 - f_embs @ f_embs.T, 0.0, 2.0)
        raw_f_clusters = average_linkage_clustering(f_dist, 1.0 - cfg.tau_fclust)
        for fc in raw_f_clusters:
            if len(fc) >= cfg.s_min_face_cluster:
                face_clusters_j.append(fc)
                weighted = sum(
                    paired_face_records[m][1].face_quality * paired_face_records[m][1].face_emb
                    for m in fc
                )
                face_prototypes.append(l2_normalize(weighted))

    identities: Dict[str, GalleryIdentity] = {}
    num_linked = 0
    num_vetoed = 0
    num_spared = 0

    if build_name == "body_only":
        for k_idx, c_members in enumerate(body_clusters):
            anon_id = f"B_{k_idx:04d}"
            medoid_indices = select_scale_stratified_medoids(
                c_members, hist_crops, min(cfg.max_hist_medoids_per_cluster, cfg.max_protected_hist)
            )
            templates = [
                BodyTemplate(
                    template_id=f"{anon_id}_hist_{r}",
                    obs_id=hist_crops[m_i].obs_id,
                    track_id=hist_crops[m_i].track_id,
                    camera_id=hist_crops[m_i].camera_id,
                    timestamp=hist_crops[m_i].timestamp,
                    bbox_h=hist_crops[m_i].bbox_h,
                    body_emb=hist_crops[m_i].body_emb,
                    is_protected_hist=True,
                    source_cluster_id=k_idx,
                    gt_person_id_audit=hist_crops[m_i].gt_person_id,
                )
                for r, m_i in enumerate(medoid_indices)
            ]
            f_exemplars = [
                FaceExemplar(
                    obs_id=hist_crops[m_i].obs_id,
                    track_id=hist_crops[m_i].track_id,
                    timestamp=hist_crops[m_i].timestamp,
                    quality=hist_crops[m_i].face_quality,
                    face_emb=hist_crops[m_i].face_emb,
                )
                for m_i in c_members
                if has_unambiguous_paired_face(hist_crops[m_i], cfg)
            ]
            proto = l2_normalize(sum(fe.quality * fe.face_emb for fe in f_exemplars)) if f_exemplars else None
            identities[anon_id] = GalleryIdentity(
                anon_id=anon_id,
                created_at=cfg.cutoff_time,
                body_templates=templates,
                face_exemplars=f_exemplars,
                anchored_prototype=proto,
            )
    else:
        for j_idx, fc in enumerate(face_clusters_j):
            anon_id = f"J_{j_idx:04d}"
            f_exemplars = [
                FaceExemplar(
                    obs_id=paired_face_records[m][1].obs_id,
                    track_id=paired_face_records[m][1].track_id,
                    timestamp=paired_face_records[m][1].timestamp,
                    quality=paired_face_records[m][1].face_quality,
                    face_emb=paired_face_records[m][1].face_emb,
                )
                for m in fc
            ]
            identities[anon_id] = GalleryIdentity(
                anon_id=anon_id,
                created_at=cfg.cutoff_time,
                body_templates=[],
                face_exemplars=f_exemplars,
                anchored_prototype=face_prototypes[j_idx],
            )

        for k_idx, c_members in enumerate(body_clusters):
            c_embs = np.stack([hist_crops[i].body_emb for i in c_members], axis=0)
            c_mean = l2_normalize(np.mean(c_embs, axis=0))
            rho_ck = float(np.max(1.0 - c_embs @ c_mean))

            support_by_j: Dict[int, List[int]] = {}
            for i_crop in c_members:
                obs = hist_crops[i_crop]
                if has_unambiguous_paired_face(obs, cfg) and face_prototypes:
                    sims = [float(np.dot(obs.face_emb, p)) for p in face_prototypes]
                    best_j = int(np.argmax(sims))
                    if sims[best_j] >= cfg.tau_flink:
                        support_by_j.setdefault(best_j, []).append(i_crop)

            supported_js = [j for j, idxs in support_by_j.items() if len(idxs) >= 1]
            if len(supported_js) == 1:
                j_star = supported_js[0]
                if len(support_by_j[j_star]) >= cfg.n_min_prop and rho_ck <= cfg.rho_max:
                    num_linked += 1
                    anon_id = f"J_{j_star:04d}"
                    medoid_indices = select_scale_stratified_medoids(
                        c_members, hist_crops, cfg.max_hist_medoids_per_cluster
                    )
                    for m_i in medoid_indices:
                        identities[anon_id].body_templates.append(
                            BodyTemplate(
                                template_id=f"{anon_id}_c{k_idx}_{m_i}",
                                obs_id=hist_crops[m_i].obs_id,
                                track_id=hist_crops[m_i].track_id,
                                camera_id=hist_crops[m_i].camera_id,
                                timestamp=hist_crops[m_i].timestamp,
                                bbox_h=hist_crops[m_i].bbox_h,
                                body_emb=hist_crops[m_i].body_emb,
                                is_protected_hist=True,
                                source_cluster_id=k_idx,
                                gt_person_id_audit=hist_crops[m_i].gt_person_id,
                            )
                        )
                    continue

            if len(supported_js) >= 2:
                num_vetoed += 1
                face_supported_crops = {i_c for idxs in support_by_j.values() for i_c in idxs}
                num_spared += len(c_members) - len(face_supported_crops)
                for j_conf, idxs in support_by_j.items():
                    anon_id = f"J_{j_conf:04d}"
                    for m_i in idxs[:1]:
                        identities[anon_id].body_templates.append(
                            BodyTemplate(
                                template_id=f"{anon_id}_split_c{k_idx}_{m_i}",
                                obs_id=hist_crops[m_i].obs_id,
                                track_id=hist_crops[m_i].track_id,
                                camera_id=hist_crops[m_i].camera_id,
                                timestamp=hist_crops[m_i].timestamp,
                                bbox_h=hist_crops[m_i].bbox_h,
                                body_emb=hist_crops[m_i].body_emb,
                                is_protected_hist=True,
                                source_cluster_id=k_idx,
                                gt_person_id_audit=hist_crops[m_i].gt_person_id,
                            )
                        )
            else:
                anon_id = f"U_{k_idx:04d}"
                medoid_indices = select_scale_stratified_medoids(
                    c_members, hist_crops, min(cfg.max_hist_medoids_per_cluster, cfg.max_protected_hist)
                )
                identities[anon_id] = GalleryIdentity(
                    anon_id=anon_id,
                    created_at=cfg.cutoff_time,
                    body_templates=[
                        BodyTemplate(
                            template_id=f"{anon_id}_hist_{r}",
                            obs_id=hist_crops[m_i].obs_id,
                            track_id=hist_crops[m_i].track_id,
                            camera_id=hist_crops[m_i].camera_id,
                            timestamp=hist_crops[m_i].timestamp,
                            bbox_h=hist_crops[m_i].bbox_h,
                            body_emb=hist_crops[m_i].body_emb,
                            is_protected_hist=True,
                            source_cluster_id=k_idx,
                            gt_person_id_audit=hist_crops[m_i].gt_person_id,
                        )
                        for r, m_i in enumerate(medoid_indices)
                    ],
                )

        for gid, ident in list(identities.items()):
            if not ident.body_templates and not ident.face_exemplars:
                del identities[gid]
                continue
            if len(ident.body_templates) > cfg.max_protected_hist:
                ident.body_templates.sort(key=lambda t: t.bbox_h)
                bins = np.array_split(np.arange(len(ident.body_templates)), cfg.max_protected_hist)
                ident.body_templates = [ident.body_templates[int(b[0])] for b in bins if len(b) > 0]

    # Fix anonymous-label -> reference GT mapping on enrollment portion (Section IV-C)
    enrolled_gt_persons: set = set()
    enrolled_person_cameras: Dict[str, set] = {}
    enrolled_person_outfits: Dict[str, set] = {}
    for obs in hist_crops:
        if obs.gt_person_id:
            enrolled_gt_persons.add(obs.gt_person_id)
            enrolled_person_cameras.setdefault(obs.gt_person_id, set()).add(obs.camera_id)
            if obs.outfit_id:
                enrolled_person_outfits.setdefault(obs.gt_person_id, set()).add(obs.outfit_id)

    for ident in identities.values():
        gt_votes: Dict[str, int] = {}
        total_labeled = 0
        for tpl in ident.body_templates:
            if tpl.gt_person_id_audit:
                gt_votes[tpl.gt_person_id_audit] = gt_votes.get(tpl.gt_person_id_audit, 0) + 1
                total_labeled += 1
        if total_labeled == 0:
            ident.eval_status = "unresolved"
            ident.eval_gt_person_id = None
        else:
            top_gt, top_count = max(gt_votes.items(), key=lambda kv: kv[1])
            if top_count / total_labeled >= cfg.gt_purity_threshold:
                ident.eval_status = "clean"
                ident.eval_gt_person_id = top_gt
            else:
                ident.eval_status = "mixed"
                ident.eval_gt_person_id = None

    return HistoricalBuildResult(
        build_name=build_name,
        identities=identities,
        num_body_clusters=len(body_clusters),
        num_face_clusters=len(face_clusters_j),
        num_linked_clusters=num_linked,
        num_vetoed_conflict_clusters=num_vetoed,
        num_spared_conflict_crops=num_spared,
        enrolled_gt_persons=enrolled_gt_persons,
        enrolled_person_cameras=enrolled_person_cameras,
        enrolled_person_outfits=enrolled_person_outfits,
    )


# ==============================================================================
# 5. Query-Write Separation & Forward Chronological Replay (Equations 6--10)
# ==============================================================================

def partition_replay_streams(
    replay_observations: Sequence[Observation],
    cfg: ReplayConfig,
) -> Tuple[List[Observation], List[Observation]]:
    """Separates write-opportunity observations from independent later faceless queries.

    Retains chronological within-tracklet face observations so same-tracklet
    contradiction rollback (M3) can revoke an earlier lower-quality erroneous write,
    while capping active admitted templates to at most 1 per tracklet.
    """
    by_track: Dict[str, List[Observation]] = {}
    for obs in replay_observations:
        if obs.timestamp > cfg.cutoff_time:
            by_track.setdefault(obs.track_id, []).append(obs)

    write_candidates: List[Observation] = []
    write_track_ids: set = set()
    write_intervals: List[Tuple[str, Optional[str], float, float]] = []

    for track_id, items in sorted(by_track.items()):
        paired_faces = sorted(
            [x for x in items if has_unambiguous_paired_face(x, cfg) and passes_crop_gate(x, cfg)],
            key=lambda x: (x.timestamp, x.obs_id),
        )
        if paired_faces:
            # Keep initial highest-quality face per stride or progressive higher-quality faces
            best_q_so_far = -1.0
            for pf in paired_faces:
                if pf.face_quality > best_q_so_far:
                    write_candidates.append(pf)
                    best_q_so_far = pf.face_quality
            write_track_ids.add(track_id)
            t_lo = min(x.timestamp for x in items)
            t_hi = max(x.timestamp for x in items)
            write_intervals.append((paired_faces[0].camera_id, paired_faces[0].gt_person_id, t_lo, t_hi))

    eligible_queries: List[Observation] = []
    for track_id, items in sorted(by_track.items()):
        if track_id in write_track_ids:
            continue
        if any(has_unambiguous_paired_face(x, cfg) for x in items):
            continue
        gated = [x for x in items if passes_crop_gate(x, cfg) and x.query_id is not None]
        if not gated:
            continue
        q_obs = max(gated, key=lambda x: (x.det_score, -x.timestamp))

        is_near_dup_of_write = False
        for w_cam, w_gt, w_lo, w_hi in write_intervals:
            if q_obs.camera_id == w_cam and q_obs.gt_person_id is not None and q_obs.gt_person_id == w_gt:
                if (w_lo - cfg.near_dup_isolation_sec) <= q_obs.timestamp <= (w_hi + cfg.near_dup_isolation_sec):
                    is_near_dup_of_write = True
                    break
        if not is_near_dup_of_write:
            eligible_queries.append(q_obs)

    write_candidates.sort(key=lambda x: (x.timestamp, x.obs_id))
    eligible_queries.sort(key=lambda x: (x.timestamp, x.query_id or x.obs_id))
    return eligible_queries, write_candidates


def apply_template_write_proposed(
    ident: GalleryIdentity,
    obs: Observation,
    cfg: ReplayConfig,
) -> str:
    """Operator R_j in Equation (6) & M3: per-tracklet cap, near-dup refresh, eviction."""
    # At most one active write per single-camera tracklet (replace if higher quality)
    for idx, tpl in enumerate(ident.body_templates):
        if not tpl.is_protected_hist and tpl.track_id == obs.track_id:
            if obs.face_quality >= tpl.write_face_quality:
                ident.body_templates[idx] = BodyTemplate(
                    template_id=f"{ident.anon_id}_w_{obs.obs_id}",
                    obs_id=obs.obs_id,
                    track_id=obs.track_id,
                    camera_id=obs.camera_id,
                    timestamp=obs.timestamp,
                    bbox_h=obs.bbox_h,
                    body_emb=obs.body_emb,
                    is_protected_hist=False,
                    write_face_quality=obs.face_quality,
                    gt_person_id_audit=obs.gt_person_id,
                )
            return "near_dup_refreshed"

    for tpl in ident.body_templates:
        if (
            tpl.camera_id == obs.camera_id
            and abs(math.log(obs.bbox_h) - math.log(tpl.bbox_h)) <= cfg.sigma_dup
            and float(np.dot(obs.body_emb, tpl.body_emb)) >= cfg.tau_dup
        ):
            tpl.timestamp = obs.timestamp
            return "near_dup_refreshed"

    new_tpl = BodyTemplate(
        template_id=f"{ident.anon_id}_w_{obs.obs_id}",
        obs_id=obs.obs_id,
        track_id=obs.track_id,
        camera_id=obs.camera_id,
        timestamp=obs.timestamp,
        bbox_h=obs.bbox_h,
        body_emb=obs.body_emb,
        is_protected_hist=False,
        write_face_quality=obs.face_quality,
        gt_person_id_audit=obs.gt_person_id,
    )

    if len(ident.body_templates) < cfg.template_budget_m:
        ident.body_templates.append(new_tpl)
        return "written"

    dyn_indices = [idx for idx, t in enumerate(ident.body_templates) if not t.is_protected_hist]
    if not dyn_indices:
        return "budget_full_protected"

    best_evict_idx = dyn_indices[0]
    best_redundancy = -float("inf")
    for idx in dyn_indices:
        cand = ident.body_templates[idx]
        red = max(
            (
                height_weight(cand.bbox_h, other.bbox_h, cfg.sigma_height)
                * float(np.dot(cand.body_emb, other.body_emb))
                for o_idx, other in enumerate(ident.body_templates)
                if o_idx != idx
            ),
            default=-1.0,
        )
        if red > best_redundancy:
            best_redundancy = red
            best_evict_idx = idx

    ident.body_templates[best_evict_idx] = new_tpl
    return "written"


def apply_template_write_geff(
    ident: GalleryIdentity,
    obs: Observation,
    cfg: ReplayConfig,
) -> str:
    """GEFF-style simple append under identical template capacity M (1 write/tracklet)."""
    for idx, tpl in enumerate(ident.body_templates):
        if not tpl.is_protected_hist and tpl.track_id == obs.track_id:
            ident.body_templates[idx] = BodyTemplate(
                template_id=f"{ident.anon_id}_geff_{obs.obs_id}",
                obs_id=obs.obs_id,
                track_id=obs.track_id,
                camera_id=obs.camera_id,
                timestamp=obs.timestamp,
                bbox_h=obs.bbox_h,
                body_emb=obs.body_emb,
                is_protected_hist=False,
                write_face_quality=obs.face_quality,
                gt_person_id_audit=obs.gt_person_id,
            )
            return "near_dup_refreshed"

    new_tpl = BodyTemplate(
        template_id=f"{ident.anon_id}_geff_{obs.obs_id}",
        obs_id=obs.obs_id,
        track_id=obs.track_id,
        camera_id=obs.camera_id,
        timestamp=obs.timestamp,
        bbox_h=obs.bbox_h,
        body_emb=obs.body_emb,
        is_protected_hist=False,
        write_face_quality=obs.face_quality,
        gt_person_id_audit=obs.gt_person_id,
    )
    if len(ident.body_templates) < cfg.template_budget_m:
        ident.body_templates.append(new_tpl)
        return "written"

    dyn_indices = [idx for idx, t in enumerate(ident.body_templates) if not t.is_protected_hist]
    if not dyn_indices:
        return "budget_full_protected"
    oldest_dyn = min(dyn_indices, key=lambda i: ident.body_templates[i].timestamp)
    ident.body_templates[oldest_dyn] = new_tpl
    return "written"


def compute_face_similarity(
    face_emb: np.ndarray,
    ident: GalleryIdentity,
    now_ts: float,
    cfg: ReplayConfig,
) -> float:
    sims: List[float] = []
    if ident.anchored_prototype is not None:
        sims.append(float(np.dot(face_emb, ident.anchored_prototype)))
    for ex in ident.face_exemplars:
        if cfg.face_horizon_sec is None or (now_ts - ex.timestamp) <= cfg.face_horizon_sec:
            sims.append(float(np.dot(face_emb, ex.face_emb)))
    return max(sims) if sims else -1.0


def run_forward_replay_condition(
    hist_build: HistoricalBuildResult,
    replay_rule: str,
    eligible_queries: Sequence[Observation],
    write_candidates: Sequence[Observation],
    cfg: ReplayConfig,
) -> Tuple[List[QueryDecisionRecord], List[WriteAuditRecord]]:
    """Executes one cell of Table I (or complementary ablation) in strict time order."""
    assert replay_rule in (
        "frozen",
        "geff_max",
        "proposed",
        "proposed_max_sel",
        "oracle_proposed",
    )
    condition_name = f"{hist_build.build_name}__{replay_rule}"
    identities: Dict[str, GalleryIdentity] = copy.deepcopy(hist_build.identities)

    events: List[Tuple[float, int, str, Observation]] = []
    for q in eligible_queries:
        events.append((q.timestamp, 0, "query", q))
    if replay_rule != "frozen":
        for w in write_candidates:
            events.append((w.timestamp, 1, "write", w))
    events.sort(key=lambda ev: (ev[0], ev[1], ev[3].obs_id))

    staging_faces: List[Observation] = []
    active_track_assignment: Dict[str, Tuple[str, float]] = {}  # track_id -> (anon_id, face_q)
    new_id_counter = 0
    query_records: List[QueryDecisionRecord] = []
    write_records: List[WriteAuditRecord] = []

    use_height_selector = replay_rule in ("frozen", "proposed", "oracle_proposed")
    tau_accept = cfg.tau_body if use_height_selector else cfg.tau_geff_body

    for ts, _, ev_kind, obs in events:
        if ev_kind == "query":
            scored_candidates: List[Tuple[float, str]] = []
            for anon_id, ident in identities.items():
                if not ident.body_templates:
                    continue
                sc = (
                    score_identity_proposed(
                        obs.body_emb,
                        obs.bbox_h,
                        ident.body_templates,
                        cfg.sigma_height,
                        cfg.similarity_temp_beta,
                    )
                    if use_height_selector
                    else score_identity_max(obs.body_emb, ident.body_templates)
                )
                scored_candidates.append((sc, anon_id))

            scored_candidates.sort(key=lambda x: (-x[0], x[1]))
            s1, j1 = scored_candidates[0] if len(scored_candidates) >= 1 else (-1.0, "")
            s2, _ = scored_candidates[1] if len(scored_candidates) >= 2 else (-1.0, "")
            margin = s1 - s2 if len(scored_candidates) >= 2 else s1 - (-1.0)

            if j1 and s1 >= tau_accept and margin >= cfg.delta_body:
                decision = "assign"
                pred_anon = j1
                ident_hit = identities[j1]
                mapped_gt = ident_hit.eval_gt_person_id or f"<{ident_hit.eval_status}>"
            else:
                decision = "refuse"
                pred_anon = "REFUSED"
                ident_hit = None
                mapped_gt = "REFUSED"

            if obs.population_type == "in_gallery":
                if decision == "refuse":
                    verdict = "refused"
                elif ident_hit is not None and ident_hit.eval_status == "mixed":
                    verdict = "mixed_group_error"
                elif ident_hit is not None and ident_hit.eval_gt_person_id == obs.gt_person_id:
                    verdict = "correct"
                else:
                    verdict = "wrong_id"
            elif obs.population_type == "absent_from_gallery":
                verdict = "correct_refusal" if decision == "refuse" else "false_accept_absent"
            elif obs.population_type == "non_person":
                verdict = "correct_refusal" if decision == "refuse" else "false_accept_nonperson"
            else:
                raise ValueError(f"Unknown population_type: {obs.population_type}")

            gt_pid = obs.gt_person_id or "NON_PERSON"
            seen_cams = hist_build.enrolled_person_cameras.get(gt_pid, set())
            seen_outfits = hist_build.enrolled_person_outfits.get(gt_pid, set())
            is_cross_cam = (
                "yes" if (obs.population_type == "in_gallery" and obs.camera_id not in seen_cams)
                else ("no" if obs.population_type == "in_gallery" else "na")
            )
            is_cloth_chg = (
                "yes"
                if (obs.population_type == "in_gallery" and obs.outfit_id and seen_outfits and obs.outfit_id not in seen_outfits)
                else ("no" if (obs.population_type == "in_gallery" and obs.outfit_id and seen_outfits) else "unknown")
            )

            query_records.append(
                QueryDecisionRecord(
                    condition=condition_name,
                    historical_build=hist_build.build_name,
                    replay_rule=replay_rule,
                    query_id=obs.query_id or obs.obs_id,
                    track_id=obs.track_id,
                    camera_id=obs.camera_id,
                    timestamp=obs.timestamp,
                    population_type=obs.population_type,
                    gt_person_id=gt_pid,
                    bbox_h=obs.bbox_h,
                    is_cross_camera=is_cross_cam,
                    is_clothes_changed=is_cloth_chg,
                    posture_label=obs.posture_label or "unknown",
                    predicted_anon_id=pred_anon,
                    mapped_gt_person_id=mapped_gt,
                    top1_score=round(s1, 6),
                    top2_score=round(s2, 6),
                    score_margin=round(margin, 6),
                    decision=decision,
                    verdict=verdict,
                )
            )

        elif ev_kind == "write":
            if replay_rule == "oracle_proposed":
                matching_anons = [
                    aid for aid, id_obj in identities.items()
                    if id_obj.eval_status == "clean" and id_obj.eval_gt_person_id == obs.gt_person_id
                ]
                if matching_anons:
                    target_aid = sorted(matching_anons)[0]
                    w_status = apply_template_write_proposed(identities[target_aid], obs, cfg)
                    write_records.append(
                        WriteAuditRecord(
                            condition=condition_name,
                            track_id=obs.track_id,
                            obs_id=obs.obs_id,
                            camera_id=obs.camera_id,
                            timestamp=obs.timestamp,
                            gt_person_id=obs.gt_person_id,
                            face_quality=obs.face_quality,
                            top1_anon_id=target_aid,
                            top1_face_sim=1.0,
                            face_margin=1.0,
                            outcome=w_status,
                            written_anon_id=target_aid,
                            mapped_gt_person_id=identities[target_aid].eval_gt_person_id,
                            is_attribution_correct=True,
                        )
                    )
                continue

            face_ranking: List[Tuple[float, str]] = []
            for anon_id, ident in identities.items():
                f_sim = compute_face_similarity(obs.face_emb, ident, ts, cfg)
                if f_sim > -0.5:
                    face_ranking.append((f_sim, anon_id))
            face_ranking.sort(key=lambda x: (-x[0], x[1]))

            f_s1, f_j1 = face_ranking[0] if len(face_ranking) >= 1 else (-1.0, "")
            f_s2, _ = face_ranking[1] if len(face_ranking) >= 2 else (-1.0, "")
            f_margin = f_s1 - f_s2 if len(face_ranking) >= 2 else f_s1

            if f_j1 and f_s1 >= cfg.tau_face_write and f_margin >= cfg.delta_face_write:
                # Same-tracklet contradiction rollback (M3): if an earlier lower-quality
                # face on this tracklet wrote to j_prev != f_j1, revoke that write!
                if replay_rule in ("proposed", "proposed_max_sel") and obs.track_id in active_track_assignment:
                    prev_aid, prev_q = active_track_assignment[obs.track_id]
                    if prev_aid != f_j1 and obs.face_quality > prev_q and prev_aid in identities:
                        identities[prev_aid].body_templates = [
                            t for t in identities[prev_aid].body_templates
                            if t.is_protected_hist or t.track_id != obs.track_id
                        ]
                        identities[prev_aid].face_exemplars = [
                            fe for fe in identities[prev_aid].face_exemplars
                            if fe.track_id != obs.track_id
                        ]

                target_ident = identities[f_j1]
                if replay_rule == "geff_max":
                    w_status = apply_template_write_geff(target_ident, obs, cfg)
                else:
                    w_status = apply_template_write_proposed(target_ident, obs, cfg)
                    active_track_assignment[obs.track_id] = (f_j1, obs.face_quality)
                    if f_s1 >= cfg.tau_fenroll:
                        target_ident.face_exemplars.append(
                            FaceExemplar(
                                obs_id=obs.obs_id,
                                track_id=obs.track_id,
                                timestamp=obs.timestamp,
                                quality=obs.face_quality,
                                face_emb=obs.face_emb,
                            )
                        )

                is_corr = (
                    (target_ident.eval_status == "clean" and target_ident.eval_gt_person_id == obs.gt_person_id)
                    if obs.gt_person_id is not None
                    else False
                )
                write_records.append(
                    WriteAuditRecord(
                        condition=condition_name,
                        track_id=obs.track_id,
                        obs_id=obs.obs_id,
                        camera_id=obs.camera_id,
                        timestamp=obs.timestamp,
                        gt_person_id=obs.gt_person_id,
                        face_quality=round(obs.face_quality, 4),
                        top1_anon_id=f_j1,
                        top1_face_sim=round(f_s1, 6),
                        face_margin=round(f_margin, 6),
                        outcome=w_status,
                        written_anon_id=f_j1,
                        mapped_gt_person_id=target_ident.eval_gt_person_id,
                        is_attribution_correct=is_corr,
                    )
                )
            elif replay_rule in ("proposed", "proposed_max_sel") and f_s1 < cfg.tau_new_id:
                corroborated = [
                    st for st in staging_faces
                    if float(np.dot(obs.face_emb, st.face_emb)) >= cfg.tau_face_write
                ]
                if len(corroborated) + 1 >= cfg.n_new_corroborate:
                    new_id_counter += 1
                    new_aid = f"N_{new_id_counter:04d}"
                    group_obs = corroborated + [obs]
                    staging_faces = [st for st in staging_faces if st not in corroborated]
                    proto = l2_normalize(sum(x.face_quality * x.face_emb for x in group_obs))
                    new_ident = GalleryIdentity(
                        anon_id=new_aid,
                        created_at=obs.timestamp,
                        body_templates=[],
                        face_exemplars=[
                            FaceExemplar(x.obs_id, x.track_id, x.timestamp, x.face_quality, x.face_emb)
                            for x in group_obs
                        ],
                        anchored_prototype=proto,
                        eval_status="runtime_new",
                        eval_gt_person_id=None,
                    )
                    apply_template_write_proposed(new_ident, obs, cfg)
                    identities[new_aid] = new_ident
                    outcome_label = "spawned_new"
                    written_aid = new_aid
                else:
                    staging_faces.append(obs)
                    outcome_label = "staged_new"
                    written_aid = None

                write_records.append(
                    WriteAuditRecord(
                        condition=condition_name,
                        track_id=obs.track_id,
                        obs_id=obs.obs_id,
                        camera_id=obs.camera_id,
                        timestamp=obs.timestamp,
                        gt_person_id=obs.gt_person_id,
                        face_quality=round(obs.face_quality, 4),
                        top1_anon_id=f_j1 or None,
                        top1_face_sim=round(f_s1, 6),
                        face_margin=round(f_margin, 6),
                        outcome=outcome_label,
                        written_anon_id=written_aid,
                        mapped_gt_person_id=None,
                        is_attribution_correct=None,
                    )
                )
            else:
                write_records.append(
                    WriteAuditRecord(
                        condition=condition_name,
                        track_id=obs.track_id,
                        obs_id=obs.obs_id,
                        camera_id=obs.camera_id,
                        timestamp=obs.timestamp,
                        gt_person_id=obs.gt_person_id,
                        face_quality=round(obs.face_quality, 4),
                        top1_anon_id=f_j1 or None,
                        top1_face_sim=round(f_s1, 6),
                        face_margin=round(f_margin, 6),
                        outcome="unresolved_band",
                        written_anon_id=None,
                        mapped_gt_person_id=None,
                        is_attribution_correct=None,
                    )
                )

    return query_records, write_records


# ==============================================================================
# 6. Paired Recovery/Loss Audit & Person-Level Cluster Bootstrap
# ==============================================================================

def person_cluster_bootstrap_net_gain(
    records_a: Sequence[QueryDecisionRecord],
    records_b: Sequence[QueryDecisionRecord],
    n_iters: int,
    seed: int,
) -> Tuple[float, float]:
    """Person-level cluster bootstrap 95% CI for net change in correct answers (B - A)."""
    map_a = {r.query_id: r for r in records_a if r.population_type == "in_gallery"}
    map_b = {r.query_id: r for r in records_b if r.population_type == "in_gallery"}
    common_qids = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not common_qids:
        return (0.0, 0.0)

    person_deltas: Dict[str, int] = {}
    for qid in common_qids:
        pid = map_a[qid].gt_person_id
        ca = 1 if map_a[qid].verdict == "correct" else 0
        cb = 1 if map_b[qid].verdict == "correct" else 0
        person_deltas[pid] = person_deltas.get(pid, 0) + (cb - ca)

    pids = sorted(person_deltas.keys())
    deltas = np.array([person_deltas[p] for p in pids], dtype=np.float64)
    if len(deltas) <= 1 or n_iters <= 0:
        total = float(np.sum(deltas))
        return (total, total)

    rng = np.random.default_rng(seed)
    idx_samples = rng.integers(0, len(deltas), size=(n_iters, len(deltas)))
    boot_totals = np.sum(deltas[idx_samples], axis=1)
    lo, hi = np.percentile(boot_totals, [2.5, 97.5])
    return (round(float(lo), 2), round(float(hi), 2))


def build_paired_comparison_rows(
    cond_a: str,
    records_a: Sequence[QueryDecisionRecord],
    cond_b: str,
    records_b: Sequence[QueryDecisionRecord],
    cfg: ReplayConfig,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Computes full paired recovery/loss decomposition between two conditions."""
    map_a = {r.query_id: r for r in records_a if r.population_type == "in_gallery"}
    map_b = {r.query_id: r for r in records_b if r.population_type == "in_gallery"}
    common_qids = sorted(set(map_a.keys()) & set(map_b.keys()))

    newly_correct = 0
    ceased_correct = 0
    refused_to_correct = 0
    wrong_to_correct = 0
    correct_to_refused = 0
    correct_to_wrong = 0
    per_query_diffs: List[Dict[str, object]] = []

    for qid in common_qids:
        ra, rb = map_a[qid], map_b[qid]
        ca = ra.verdict == "correct"
        cb = rb.verdict == "correct"

        if not ca and cb:
            newly_correct += 1
            if ra.verdict == "refused":
                refused_to_correct += 1
                shift = "recovered_from_refusal"
            else:
                wrong_to_correct += 1
                shift = "recovered_from_wrong"
        elif ca and not cb:
            ceased_correct += 1
            if rb.verdict == "refused":
                correct_to_refused += 1
                shift = "lost_to_refusal"
            else:
                correct_to_wrong += 1
                shift = "lost_to_wrong"
        elif ca and cb:
            shift = "stayed_correct"
        else:
            shift = f"stayed_unsolved_{ra.verdict}_to_{rb.verdict}"

        per_query_diffs.append({
            "comparison_pair": f"{cond_a}_vs_{cond_b}",
            "query_id": qid,
            "gt_person_id": ra.gt_person_id,
            "camera_id": ra.camera_id,
            "timestamp": ra.timestamp,
            "is_cross_camera": ra.is_cross_camera,
            "is_clothes_changed": ra.is_clothes_changed,
            "posture_label": ra.posture_label,
            "verdict_a": ra.verdict,
            "score_a": ra.top1_score,
            "verdict_b": rb.verdict,
            "score_b": rb.top1_score,
            "paired_shift": shift,
        })

    ci_lo, ci_hi = person_cluster_bootstrap_net_gain(
        records_a, records_b, cfg.bootstrap_iters, cfg.random_seed
    )
    summary = {
        "condition_a": cond_a,
        "condition_b": cond_b,
        "n_answerable_queries": len(common_qids),
        "n_unique_persons": len({map_a[q].gt_person_id for q in common_qids}),
        "correct_a": sum(1 for q in common_qids if map_a[q].verdict == "correct"),
        "correct_b": sum(1 for q in common_qids if map_b[q].verdict == "correct"),
        "newly_correct": newly_correct,
        "ceased_correct": ceased_correct,
        "net_change": newly_correct - ceased_correct,
        "refused_to_correct": refused_to_correct,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_refused": correct_to_refused,
        "correct_to_wrong": correct_to_wrong,
        "person_bootstrap_ci95_lo": ci_lo,
        "person_bootstrap_ci95_hi": ci_hi,
    }
    return summary, per_query_diffs


def summarize_condition_cell(
    condition: str,
    records: Sequence[QueryDecisionRecord],
    writes: Sequence[WriteAuditRecord],
) -> Dict[str, object]:
    in_gal = [r for r in records if r.population_type == "in_gallery"]
    absent = [r for r in records if r.population_type == "absent_from_gallery"]
    non_p = [r for r in records if r.population_type == "non_person"]

    n_corr = sum(1 for r in in_gal if r.verdict == "correct")
    n_wrong = sum(1 for r in in_gal if r.verdict in ("wrong_id", "mixed_group_error"))
    n_ref = sum(1 for r in in_gal if r.verdict == "refused")

    n_write_opps = len(writes)
    n_written = sum(1 for w in writes if w.outcome == "written")
    n_written_correct = sum(1 for w in writes if w.outcome == "written" and w.is_attribution_correct is True)

    return {
        "condition": condition,
        "in_gallery_total": len(in_gal),
        "correct": n_corr,
        "wrong": n_wrong,
        "refused": n_ref,
        "absent_total": len(absent),
        "absent_false_accept": sum(1 for r in absent if r.verdict == "false_accept_absent"),
        "non_person_total": len(non_p),
        "non_person_false_accept": sum(1 for r in non_p if r.verdict == "false_accept_nonperson"),
        "write_opportunities": n_write_opps,
        "writes_admitted": n_written,
        "writes_correctly_attributed": n_written_correct,
    }


def summarize_e3_strata(records: Sequence[QueryDecisionRecord]) -> List[Dict[str, object]]:
    in_gal = [r for r in records if r.population_type == "in_gallery"]
    if not in_gal:
        return []
    cond = in_gal[0].condition
    strata_defs = [
        ("cross_camera", lambda r: r.is_cross_camera),
        ("clothes_changed", lambda r: r.is_clothes_changed),
        ("posture", lambda r: r.posture_label),
    ]
    rows: List[Dict[str, object]] = []
    for axis_name, getter in strata_defs:
        levels = sorted({getter(r) for r in in_gal})
        for lv in levels:
            sub = [r for r in in_gal if getter(r) == lv]
            rows.append({
                "condition": cond,
                "stratum_axis": axis_name,
                "stratum_value": lv,
                "n_persons": len({r.gt_person_id for r in sub}),
                "n_queries": len(sub),
                "correct": sum(1 for r in sub if r.verdict == "correct"),
                "wrong": sum(1 for r in sub if r.verdict in ("wrong_id", "mixed_group_error")),
                "refused": sum(1 for r in sub if r.verdict == "refused"),
            })
    return rows


def render_latex_table1(cell_summaries: Dict[str, Dict[str, object]]) -> str:
    def fmt_cell(build: str, rule: str) -> str:
        c = cell_summaries[f"{build}__{rule}"]
        return f"{c['correct']} / {c['wrong']} / {c['refused']}"

    n_q = cell_summaries["body_only__frozen"]["in_gallery_total"]
    return "\n".join([
        r"\begin{table}[t]",
        rf"\caption{{Forward replay outcomes on $N={n_q}$ identical later faceless tracks (correct / wrong / refused).}}",
        r"\label{tab:replay}",
        r"\centering",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Historical body clusters}\\",
        r"\cmidrule(lr){2-3}",
        r"Operation during replay & Body only & Face-linked\\",
        r"\midrule",
        rf"Frozen gallery & {fmt_cell('body_only', 'frozen')} & {fmt_cell('face_linked', 'frozen')}\\",
        rf"Face append and maximum score & {fmt_cell('body_only', 'geff_max')} & {fmt_cell('face_linked', 'geff_max')}\\",
        rf"Proposed update and selector & {fmt_cell('body_only', 'proposed')} & {fmt_cell('face_linked', 'proposed')}\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


# ==============================================================================
# 7. CSV I/O & Built-In Synthetic End-to-End Verification (`--selftest`)
# ==============================================================================

def write_csv(path: pathlib.Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_observations_from_csv_and_npz(
    csv_path: pathlib.Path,
    npz_path: pathlib.Path,
) -> List[Observation]:
    arrays = np.load(npz_path)
    body_embs = arrays["body_embs"]
    face_embs = arrays["face_embs"] if "face_embs" in arrays else None

    observations: List[Observation] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            b_idx = int(row["body_emb_idx"])
            f_idx = int(row["face_emb_idx"]) if row.get("face_emb_idx", "") not in ("", "-1") else -1
            f_vec = l2_normalize(face_embs[f_idx]) if (face_embs is not None and f_idx >= 0) else None
            observations.append(
                Observation(
                    obs_id=row["obs_id"],
                    track_id=row["track_id"],
                    camera_id=row["camera_id"],
                    timestamp=float(row["timestamp"]),
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
                    query_id=row.get("query_id") or None,
                    population_type=row.get("population_type") or "in_gallery",
                    gt_person_id=row.get("gt_person_id") or None,
                    outfit_id=row.get("outfit_id") or None,
                    posture_label=row.get("posture_label") or None,
                )
            )
    return observations


def generate_synthetic_selftest_dataset(cfg: ReplayConfig) -> List[Observation]:
    """Generates a multi-camera posture + coat-change capture exercising all gates."""
    rng = np.random.default_rng(cfg.random_seed)
    d_body, d_face = 64, 32
    n_enrolled_people = 12
    observations: List[Observation] = []

    person_face_proto: Dict[str, np.ndarray] = {}
    person_coat1_standing: Dict[str, np.ndarray] = {}
    person_coat1_seated: Dict[str, np.ndarray] = {}
    person_coat2_standing: Dict[str, np.ndarray] = {}

    for p in range(n_enrolled_people):
        pid = f"P_{p:02d}"
        person_face_proto[pid] = l2_normalize(rng.normal(size=d_face))
        base_id_vec = l2_normalize(rng.normal(size=d_body))
        coat1_vec = l2_normalize(0.82 * base_id_vec + 0.57 * l2_normalize(rng.normal(size=d_body)))
        seated_shift = l2_normalize(rng.normal(size=d_body))
        coat2_vec = l2_normalize(0.68 * base_id_vec + 0.73 * l2_normalize(rng.normal(size=d_body)))

        person_coat1_standing[pid] = coat1_vec
        person_coat1_seated[pid] = l2_normalize(0.65 * coat1_vec + 0.76 * seated_shift)
        person_coat2_standing[pid] = coat2_vec

        # Historical Cam A (standing, height ~245)
        for k in range(2):
            has_f = (p < 10 and k == 0)
            observations.append(
                Observation(
                    obs_id=f"hist_A_{pid}_{k}",
                    track_id=f"trk_hist_A_{pid}",
                    camera_id="camA",
                    timestamp=100.0 + p * 20.0 + k * 1.0,
                    bbox_h=245.0 + rng.uniform(-6, 6),
                    det_score=0.93,
                    max_concurrent_iou=0.04,
                    border_margin_px=16.0,
                    body_emb=l2_normalize(person_coat1_standing[pid] + 0.03 * rng.normal(size=d_body)),
                    has_raw_face=has_f,
                    face_conf=0.94 if has_f else 0.0,
                    face_quality=0.85 if has_f else 0.0,
                    face_iof=0.96 if has_f else 0.0,
                    face_rel_y=0.18 if has_f else 0.0,
                    face_emb=l2_normalize(person_face_proto[pid] + 0.02 * rng.normal(size=d_face)) if has_f else None,
                    gt_person_id=pid,
                    outfit_id="coat1",
                    posture_label="standing",
                )
            )

        # Historical Cam B (seated, height ~72)
        for k in range(2):
            has_f = (p < 8 and k == 0)
            observations.append(
                Observation(
                    obs_id=f"hist_B_{pid}_{k}",
                    track_id=f"trk_hist_B_{pid}",
                    camera_id="camB",
                    timestamp=600.0 + p * 20.0 + k * 1.0,
                    bbox_h=72.0 + rng.uniform(-4, 4),
                    det_score=0.91,
                    max_concurrent_iou=0.04,
                    border_margin_px=14.0,
                    body_emb=l2_normalize(person_coat1_seated[pid] + 0.03 * rng.normal(size=d_body)),
                    has_raw_face=has_f,
                    face_conf=0.92 if has_f else 0.0,
                    face_quality=0.83 if has_f else 0.0,
                    face_iof=0.95 if has_f else 0.0,
                    face_rel_y=0.21 if has_f else 0.0,
                    face_emb=l2_normalize(person_face_proto[pid] + 0.02 * rng.normal(size=d_face)) if has_f else None,
                    gt_person_id=pid,
                    outfit_id="coat1",
                    posture_label="sitting",
                )
            )

    # Replay t_1 > T: Face-confirmed observations in Coat 2 (mid-distance corridor cam, h ~145)
    for p in range(8):
        pid = f"P_{p:02d}"
        observations.append(
            Observation(
                obs_id=f"replay_write_{pid}",
                track_id=f"trk_write_{pid}",
                camera_id="camA",
                timestamp=cfg.cutoff_time + 120.0 + p * 30.0,
                bbox_h=145.0,
                det_score=0.94,
                max_concurrent_iou=0.02,
                border_margin_px=20.0,
                body_emb=l2_normalize(person_coat2_standing[pid] + 0.02 * rng.normal(size=d_body)),
                has_raw_face=True,
                face_conf=0.95,
                face_quality=0.88,
                face_iof=0.98,
                face_rel_y=0.16,
                face_emb=l2_normalize(person_face_proto[pid] + 0.02 * rng.normal(size=d_face)),
                gt_person_id=pid,
                outfit_id="coat2",
                posture_label="standing",
            )
        )

    # Replay t_2 > t_1: Independent later faceless tracks
    q_counter = 0
    for p in range(n_enrolled_people):
        pid = f"P_{p:02d}"
        q_counter += 1
        wore_coat2 = p < 8
        emb_q1 = person_coat2_standing[pid] if wore_coat2 else person_coat1_standing[pid]
        observations.append(
            Observation(
                obs_id=f"q_obs_{q_counter:03d}",
                track_id=f"trk_q_{q_counter:03d}",
                camera_id="camC",
                timestamp=cfg.cutoff_time + 900.0 + q_counter * 15.0,
                bbox_h=142.0 if wore_coat2 else 242.0,
                det_score=0.92,
                max_concurrent_iou=0.03,
                border_margin_px=18.0,
                body_emb=l2_normalize(emb_q1 + 0.03 * rng.normal(size=d_body)),
                has_raw_face=False,
                query_id=f"Q_{q_counter:03d}",
                population_type="in_gallery",
                gt_person_id=pid,
                outfit_id="coat2" if wore_coat2 else "coat1",
                posture_label="standing",
            )
        )
        q_counter += 1
        observations.append(
            Observation(
                obs_id=f"q_obs_{q_counter:03d}",
                track_id=f"trk_q_{q_counter:03d}",
                camera_id="camB",
                timestamp=cfg.cutoff_time + 1500.0 + q_counter * 15.0,
                bbox_h=74.0,
                det_score=0.90,
                max_concurrent_iou=0.05,
                border_margin_px=15.0,
                body_emb=l2_normalize(person_coat1_seated[pid] + 0.03 * rng.normal(size=d_body)),
                has_raw_face=False,
                query_id=f"Q_{q_counter:03d}",
                population_type="in_gallery",
                gt_person_id=pid,
                outfit_id="coat1",
                posture_label="sitting",
            )
        )

    for k in range(3):
        q_counter += 1
        observations.append(
            Observation(
                obs_id=f"q_absent_{k}",
                track_id=f"trk_absent_{k}",
                camera_id="camC",
                timestamp=cfg.cutoff_time + 2200.0 + k * 20.0,
                bbox_h=210.0,
                det_score=0.88,
                max_concurrent_iou=0.0,
                border_margin_px=15.0,
                body_emb=l2_normalize(rng.normal(size=d_body)),
                query_id=f"Q_{q_counter:03d}",
                population_type="absent_from_gallery",
                gt_person_id=f"ABSENT_{k}",
            )
        )
    for k in range(2):
        q_counter += 1
        observations.append(
            Observation(
                obs_id=f"q_nonperson_{k}",
                track_id=f"trk_nonperson_{k}",
                camera_id="camA",
                timestamp=cfg.cutoff_time + 2400.0 + k * 20.0,
                bbox_h=150.0,
                det_score=0.82,
                max_concurrent_iou=0.0,
                border_margin_px=12.0,
                body_emb=l2_normalize(rng.normal(size=d_body)),
                query_id=f"Q_{q_counter:03d}",
                population_type="non_person",
                gt_person_id=None,
            )
        )

    return observations


def run_full_e2_pipeline(
    all_observations: Sequence[Observation],
    cfg: ReplayConfig,
    out_dir: pathlib.Path,
) -> Dict[str, object]:
    """Runs the complete 2x3 Table I matrix + complementary selector/oracle ablations."""
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_raw = [x for x in all_observations if x.timestamp <= cfg.cutoff_time]
    replay_raw = [x for x in all_observations if x.timestamp > cfg.cutoff_time]

    hist_crops = subsample_tracklets(hist_raw, cfg)
    body_clusters = build_camera_conditioned_body_clusters(hist_crops, cfg)

    builds = {
        "body_only": construct_historical_gallery(hist_crops, body_clusters, "body_only", cfg),
        "face_linked": construct_historical_gallery(hist_crops, body_clusters, "face_linked", cfg),
    }

    eligible_queries, write_candidates = partition_replay_streams(replay_raw, cfg)

    replay_rules = ["frozen", "geff_max", "proposed", "proposed_max_sel", "oracle_proposed"]
    all_query_rows: List[Dict[str, object]] = []
    all_write_rows: List[Dict[str, object]] = []
    all_strata_rows: List[Dict[str, object]] = []
    cell_summaries: Dict[str, Dict[str, object]] = {}
    records_by_cond: Dict[str, List[QueryDecisionRecord]] = {}

    for b_name in ("body_only", "face_linked"):
        for r_rule in replay_rules:
            q_recs, w_recs = run_forward_replay_condition(
                builds[b_name], r_rule, eligible_queries, write_candidates, cfg
            )
            cond_key = f"{b_name}__{r_rule}"
            records_by_cond[cond_key] = q_recs
            cell_summaries[cond_key] = summarize_condition_cell(cond_key, q_recs, w_recs)
            all_query_rows.extend(dataclasses.asdict(r) for r in q_recs)
            all_write_rows.extend(dataclasses.asdict(w) for w in w_recs)
            all_strata_rows.extend(summarize_e3_strata(q_recs))

    key_pairs = [
        ("face_linked__frozen", "face_linked__proposed"),
        ("body_only__frozen", "body_only__proposed"),
        ("face_linked__geff_max", "face_linked__proposed"),
        ("body_only__frozen", "face_linked__frozen"),
        ("body_only__proposed", "face_linked__proposed"),
        ("face_linked__proposed_max_sel", "face_linked__proposed"),
        ("face_linked__proposed", "face_linked__oracle_proposed"),
    ]

    paired_summaries: List[Dict[str, object]] = []
    paired_query_diffs: List[Dict[str, object]] = []
    for ca, cb in key_pairs:
        s_row, diff_rows = build_paired_comparison_rows(
            ca, records_by_cond[ca], cb, records_by_cond[cb], cfg
        )
        paired_summaries.append(s_row)
        paired_query_diffs.extend(diff_rows)

    write_csv(out_dir / "table1_e2_cells.csv", list(cell_summaries.values()))
    write_csv(out_dir / "ledger_e2_queries.csv", all_query_rows)
    write_csv(out_dir / "ledger_e2_paired_diffs.csv", paired_query_diffs)
    write_csv(out_dir / "summary_e2_paired_gains.csv", paired_summaries)
    write_csv(out_dir / "ledger_e2_writes.csv", all_write_rows)
    write_csv(out_dir / "summary_e3_strata.csv", all_strata_rows)

    latex_snippet = render_latex_table1(cell_summaries)
    (out_dir / "table1_e2_populated.tex").write_text(latex_snippet + "\n", encoding="utf-8")

    meta = {
        "n_hist_selected_crops": len(hist_crops),
        "n_hist_body_clusters": len(body_clusters),
        "n_eligible_faceless_queries": len(eligible_queries),
        "n_write_opportunity_tracks": len(write_candidates),
        "config": dataclasses.asdict(cfg),
    }
    (out_dir / "e2_run_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"manifest": meta, "cells": cell_summaries, "paired": paired_summaries, "latex": latex_snippet}


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 Chronological Replay & Paired Ledger Evaluator")
    parser.add_argument("--observations-csv", type=pathlib.Path, help="Input crop & track metadata CSV")
    parser.add_argument("--embeddings-npz", type=pathlib.Path, help="Input NPZ with body_embs and face_embs")
    parser.add_argument("--config-json", type=pathlib.Path, help="Optional JSON overriding ReplayConfig fields")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True, help="Output directory for CSV ledgers & LaTeX")
    parser.add_argument("--selftest", action="store_true", help="Run built-in synthetic multi-camera E2 verification")
    args = parser.parse_args()

    cfg_dict = json.loads(args.config_json.read_text(encoding="utf-8")) if args.config_json else {}
    cfg = ReplayConfig(**cfg_dict)

    if args.selftest:
        observations = generate_synthetic_selftest_dataset(cfg)
    else:
        if not args.observations_csv or not args.embeddings_npz:
            parser.error("Provide --observations-csv and --embeddings-npz, or pass --selftest.")
        observations = load_observations_from_csv_and_npz(args.observations_csv, args.embeddings_npz)

    res = run_full_e2_pipeline(observations, cfg, args.out_dir)
    print("=== E2 RUN MANIFEST ===")
    print(json.dumps(res["manifest"], indent=2))
    print("\n=== GENERATED LATEX TABLE I ===")
    print(res["latex"])
    print("\n=== KEY PAIRED COMPARISONS ===")
    for p in res["paired"]:
        print(
            f"  {p['condition_a']} -> {p['condition_b']}: "
            f"net={p['net_change']:+d} (+{p['newly_correct']} / -{p['ceased_correct']}), "
            f"95% person-CI=[{p['person_bootstrap_ci95_lo']}, {p['person_bootstrap_ci95_hi']}]"
        )


if __name__ == "__main__":
    main()