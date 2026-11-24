# Paper package v3 — 2026-11-12

**Title.** How Much Identity Can a Camera Network Deliver? A Three-Probe Diagnostic
and a Tiered Delivery Protocol for Cross-Camera Re-Identification

This package contains everything needed to check the paper's numbers: the
manuscript, the adjudicated truth sets, the graph and result files, and the
scripts that produced them. Raw video, track images and crops are **not**
included (privacy and site authorisation); reviewers may request access under
an agreement.

## Layout

| Path | Contents |
|---|---|
| `00-paper/` | manuscript (`main.md`, `main.html`, `main.pdf`), `EVIDENCE.md` (number → file → command), `POSITIONING.md` (venue analysis), Chinese reading guide, figures |
| `01-submission/` | Neurocomputing submission set: double-spaced review copy (`manuscript-review-copy.pdf`, 34 pp.), `highlights.txt`, `title-page.md`, `cover-letter.md`, `CHECKLIST.md`, per-figure PDFs |
| `02-zh/` | Chinese translation of the full manuscript (`main-zh.md`, `main-zh.html`, `main-zh.pdf`, 23 pp.) with three Chinese-labelled figures; same numbers, same table numbering |
| `10-truth/` | T1 (126 pairs) and T2 (27 pairs) with raw adjudication, write-back report and constrained-colouring report |
| `20-data/` | identity graphs (443 / 450), temporal audit, merge-strategy comparison, verifier signal and partition sweep, AUC output, co-visibility table |
| `30-code/` | the 14 scripts behind every table; see `30-code/README.md` |

## Reproducing a number

Start with `00-paper/EVIDENCE.md`. Every figure and table row is mapped to a
source file and to the command in `30-code/`. All experiments are deterministic
given the track tables and embeddings; nothing is trained.

## Environment

Python 3.11 with `numpy`, `scipy`, `opencv-python`, `markdown`. The paper was
rendered with `30-code/md2paper_html.py`, which requires the `markdown` package.

## Checksums

`SHA256SUMS.txt` lists every file; `MANIFEST.csv` gives size and hash per file.

## Scope statement

Two evidence domains are used and **never pooled**: D1 (six cameras, 35 min,
1,574 tracks) supplies deployment-scale structure and merge precision; D2
(eight segments, 26 named identities) supplies identity-level recall and
selector results. The detector/tracker configuration differs between them.
