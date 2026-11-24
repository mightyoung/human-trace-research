# Derived results — r3（2026-11-24）

Paper: *How Much Identity Can a Camera Network Deliver? A Three-Probe Diagnostic and a Tiered Delivery Protocol for Cross-Camera Re-Identification* (IEEE T-BIOM submission, `docs/paper/tbiom-r3/`).

This package exists so that a reader can **check the paper's numbers without the site video**. Every file here is a derived artefact: scores, ranks, verdicts, audit records, configuration. No raw video, no crops, no face images, no embedding matrices.

Size: 51 files, about 0.41 MB. Everything is covered by `MANIFEST.csv` (path, bytes, sha256) and `SHA256SUMS.txt`.

## Layout

| Path | Contents | Paper location |
|---|---|---|
| `10-selector/` | the 179-query table for the L2+ selector (per-query top-1, score, hit), the causal face-written pool (19 writes, 18 correct), the run summary, the configuration that defines σ, τ, k and the face gate, and EXP-003a/b/c (kernel decomposition, cross-camera transfer, geometry/calibration) | §VI-C, Table V |
| `20-graph/` | the 76 adjudicated merge operations with their re-check verdicts, T1 (126 pairs) and T2 (27 pairs), the attribute-gate runs (434/76 and 443/67), graphs 443 and 450, the four merge strategies, the verifier sweep, the temporal audit, and the six camera offsets | §III, §V, Table II, VI, VII |
| `30-wildtrack/` | the 21-pair table (co-visibility, shared identities, cross-time AUC, equal-*k* AUC, region in/out, Δ), the pixel sweep, the recovered camera mapping, the sampled frames, the raw result files and all scripts | §V-B, Table IV, supplementary S8 |
| `40-env/` | detector / tracker / encoder / face configuration (from supplementary S2), package versions, determinism statement | S2 |

Each subdirectory has its own `README.md` that names the paper number each file backs.

## What was recomputed while assembling this package

Nothing was recalculated from scratch except one re-run, used as a consistency check:

```
tools/exp_pose_pool_real.py … --dump-queries --dump-pools
```

It reproduces the paper exactly: S0 = 101/179, S1 = 122/179, S1 + measured pool = 125/179, oracle-pool upper bounds 142/179 and 149/179, 19 pool writes of which 18 correct. The per-query rows in `10-selector/selector_queries_179.csv` are produced by that run.

**That check found one inconsistency in the manuscript**, now fixed: the paper described the face similarity gate as 0.80, while the run that produces 19 writes uses 0.35. The text now reads 0.35. The same pass corrected the order of the six camera offsets in Table I (they had been listed in ascending order without naming cameras, which reads as camera order: 305 is +25 s, not 0).

## What this package does not settle

1. **No second site.** Every boundary is calibrated on one office floor. Nothing here can test transfer.
2. **No by-identity resampling.** Identity names were not recorded at adjudication, so the AUC intervals in the paper are optimistic and cannot be re-derived by identity.
3. **Region-level test inherits the recovered camera mapping.** `30-wildtrack/camera_mapping.csv` is one assignment out of 7!; no alternative assignment was evaluated, so ΔAUC is conditional on it.
4. **The 21 pairs share seven cameras and one person pool.** They are 21 view pairs, not 21 independent experiments.
5. **No crop-level manifest.** The 1,855 crops (bbox per detection) are not included; `30-wildtrack/wt_crops.py` regenerates them from the public WILDTRACK archive.

## Reproducing from this package alone

Most numbers are re-readable without any computation: open the CSV next to the table in the paper. Two files need one step:

- `30-wildtrack/pairs_21.csv` → the Δ column is `auc_in_region − auc_out_region`; median over the 17 pairs with both estimates is +0.042, 11 positive.
- `10-selector/selector_queries_179.csv` → the `*_hit` columns are `top1 == gt_identity and score >= 0.80`; column sums are 101, 122, 125, 142, 149 for S0, S1, S1+measured pool, S0+oracle, S1+oracle.

## Provenance

Assembled 2026-11-24 from existing artefacts under `deliverable/`, `tmp/full/` and `tmp/derived/`. The build script is `tmp/derived/build_pkg.py`; it copies files and reshapes JSON into CSV, and does not alter any number. The analysis machine has no VCS checkout, so files are identified by sha256 rather than a commit hash.
