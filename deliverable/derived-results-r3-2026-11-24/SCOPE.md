# Scope and data-use statement

## What is in the package

Derived results only: per-query scores and ranks, merge verdicts, adjudication records, audit outputs, configuration values, and the scripts that produced them.

## What is not in the package, and why

| Excluded | Reason |
|---|---|
| Raw video (six cameras, 35 min; three further days) | Site authorisation forbids redistribution; footage identifies employees |
| Track and crop images, face crops | Same; faces are biometric data under the site agreement |
| Embedding matrices (`.npz`) | Reconstructible only with the video; excluded with it |
| Person-level identity list | The site supplied names for 26 identities under an internal agreement; not redistributable |
| The 1,855 WILDTRACK crops | Public data, but 26 MB and regenerable with `30-wildtrack/wt_crops.py` |

## Requesting the excluded material

The paper's Data Availability statement says scripts and derived result files are available from the corresponding author on request under the data-use agreement, and that nothing accompanies the submission. This package is consistent with that wording: it is a derived subset, not the site data.

## Numbers that cannot be checked from this package

- Anything requiring the video: the pixel-budget study (face-box widths), the pose-change measurements, the D1 graph itself.
- Anything requiring identity names: by-identity bootstrap intervals.
- The concentration table in supplementary S5 (denominators 766/957/300) — the generating run predates the invalid-segment filter and its configuration was not preserved; the reproducible variant is the nine-cell table in the main text.

## Versioning

Package: `derived-results-r3-2026-11-24`, 51 files, `MANIFEST.csv` + `SHA256SUMS.txt`.
Manuscript: `docs/paper/tbiom-r3/` (main 10 pages, supplement 5 pages).
