# Code index

Ordered by the paper's sections. All scripts read local track tables and
embeddings; no network access and no training.

| Paper § | Script | Produces |
|---|---|---|
| §5.2 (Table 2, 3) | `emb_auc.py` | cross-camera AUC for 2 representations × 3 alignments, plus per-pair stratification |
| §5.3, §6.3 (Table 4) | `attr_filter_ingraph.py` | attribute-gate counterfactual (434 → 443, precision .730 → .930) |
| §6.5 (Table 9) | `spacetime_ingraph.py` | merge-strategy comparison (`MODES=base,osd,covis60,covis300 THR=0.75`) |
| §6.5, §7.4 | `spacetime_audit.py` | 27 physical contradictions on the OSD clock |
| §6.4 (Table 5) | `exp_pose_pool.py`, `pose_pr.py` | pose-conditioned template weighting and pool (56.4 → 68.2 → 79.3%) |
| §6.4 | `exp_anchor_pool.py` | face-anchor pool and selector coverage |
| §6.4 | `exp_selector_runtime.py` | runtime selector (coverage 53% → 73%) |
| §6.5 | `build_spacetime_review.py` | offline single-file adjudication surface (27 items) |
| §6.5 | `ingest_spacetime_review.py` | verdict JSON → must-link / must-differ constraints |
| §7.2 | `apply_spacetime_truth.py` | constrained minimum colouring (443 → 450) |
| §7.3 (Table 7, 8) | `spacetime_verifier.py`, `verifier_split.py` | pairwise AUC and partition-level splitter |
| Appendix | `md2paper_html.py` | renders `main.md` → printable HTML (needs `markdown`) |

Notes carried over from the project's own discipline:

- Cross-run joins by track id **must** be time-checked; the same id recurs.
- Every temporal statement uses the OSD absolute clock, never `frame / fps`.
- `spacetime_ingraph.py` and `apply_spacetime_truth.py` take `--segs` separated
  by **spaces**; `cooc_conflict.py` and `eval_full.py` use **commas**.
- Do not compare numbers across detector versions (see `EVIDENCE.md` §八).
