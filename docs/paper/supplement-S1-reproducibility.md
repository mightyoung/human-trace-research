# Supplementary Material

**Companion to** *How Much Identity Can a Camera Network Deliver? A Three-Probe Diagnostic and a Tiered Delivery Protocol for Cross-Camera Re-Identification*

This document lists every script, result file and command behind the numbers reported in the main article, together with the environment and the data splits. It is referenced from the article's Reproducibility Statement and from its Data Availability statement.

## S1. Entry points by experiment

All experiments are deterministic given the track tables and embeddings; no training is involved. Entry points are listed below (repository paths); `EVIDENCE.md` maps every number in this article to its source file and command.

**Diagnostic probes**

- Cross-camera AUC sweep (2 representations × 3 alignments): `tmp/full/emb_auc.py` → `emb_auc_out.txt`
- Per-pair stratification: same file, `--by-pair`
- Counterfactual gate (co-visibility map, attribute veto): `tmp/full/attr_filter_ingraph.py`

**Tiered protocol**

- Identity graph construction and merge-strategy comparison: `tmp/full/spacetime_ingraph.py` (`MODES=base,osd,covis60,covis300 THR=0.75`)
- Temporal audit on the OSD clock: `tmp/full/spacetime_audit.py`
- Review-queue generation: `tools/build_spacetime_review.py`
- Adjudication ingestion: `tools/ingest_spacetime_review.py`
- Constrained minimum colouring: `tools/apply_spacetime_truth.py`
- Pose-conditioned templates and pool: `tmp/full/exp_pose_pool.py`, `tmp/full/pose_pr.py`
- Pool with a measured (rather than oracle) face channel: `tools/exp_pose_pool_real.py` → `tmp/full/pose_pool_real.json`; four write rules (`gt`, `gt_usable`, `gt_gallery`, `real`) run inside one causal loop, so the comparison shares a single query set and denominator
- Face-anchor pool: `tmp/full/exp_anchor_pool.py`
- Runtime selector: `tmp/full/exp_selector_runtime.py`

**Non-composability experiments**

- Pairwise verifier AUC and threshold sweep: `tmp/full/spacetime_verifier.py`
- Partition-level splitter: `tmp/full/verifier_split.py`

**Capture-change acceptance**

- Nine-cell driver, geometric audit, fixed-θ table: `tools/e21_xday.py`, `tools/e21_xday_audit.py`, `tools/e21_xday_table.py` → `tmp/e21_xday/matrix.json`, `tmp/e21_xday/audit.json`
- The driver reuses the deployed pipeline unchanged (`tools/d7_sweep.py`) and changes only the gallery and query days

**Machine-readable results**: `tmp/e21_xday/matrix.json`, `tmp/e21_xday/audit.json`, `spacetime_audit.json`, `spacetime_ingraph.json`, `spacetime_signal.json`, `verifier_split.json`, `long35_crosscam_attr8.json`, `long35_crosscam_attr8_v1.json`, `emb_auc_out.txt`, `verifier_out.txt`.

The raw video, track tables, and crops are not releasable (privacy and site authorisation). Reviewers requiring access for verification may request it under an agreement.

## S2. Environment

Python 3.11; `numpy`, `scipy`, `scikit-learn`, `opencv-python`, `onnxruntime-gpu`. Detector: RF-DETR (the deployed pipeline, unchanged; no fine-tuning for any result reported here). Tracker: ByteTrack with the two configs named in the article (`bytetrack_1012.yaml` for gallery segments, `bytetrack_b100.yaml` for query segments). Appearance encoder: OSNet-x1.0 trained on MSMT17, deployed as-is, no fine-tuning. Face channel: SCRFD-2.5G detection, five-point alignment, ArcFace `w600k_mbf` (512-d).

## S3. Data splits

| Domain | Role | Days | Cameras | Window |
|---|---|---|---|---|
| D1 (long35) | main experiments, all tiers | 1 | 6 (305, 316, 318, 320, 323, 343) | 35 min, native 2560×1440 / 25 fps |
| D2 (8-segment set) | calibration of the merge rule | 1 | 8 segments | per-segment |
| D3 (e21) | capture-change acceptance only | 3 (10-22, 10-23, 10-26) | 6 per day | `tr` 0–720 s, `te` 810 s → window end |

No identity truth exists for D3; all D3 statistics are surrogate statistics (coverage, co-occurrence conflict, same-person merge rate) and are labelled as such in the main text.
