# -*- coding: utf-8 -*-
"""构建论文 v3 发布包：论文 + 真值 + 数据 + 代码 + 清单 + 校验和。

用法:
    python tools/build_paper_package_v3.py

产物:
    deliverable/paper-package-v3/
    deliverable/dist/paper-package-v3_<date>.zip

设计约束:
  * 不含原始视频与裁图（隐私 + 站点授权）
  * 文本文件统一脱敏（IP / 远程路径 / 端口 / 凭据）
  * 每个文件进 MANIFEST.csv 与 SHA256SUMS.txt
"""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverable" / "paper-package-v3"
DIST = ROOT / "deliverable" / "dist"

TODAY = date.today().isoformat()

# ---------------------------------------------------------------- 脱敏规则
REDACT = [
    (re.compile(r"rtsp://[^\s\"'<>]+"), "rtsp://<user>:<pass>@<camera-ip>:554/..."),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/yhh/human-count"), "<REMOTE_WORKDIR>"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/yhh"), "<REMOTE_HOME>"),
    (re.compile(r"/share/data\d/[A-Za-z0-9_.-]+/data/human-count"), "<DATA_ROOT>"),
    (re.compile(r"/share/data\d/[A-Za-z0-9_.-]+"), "<SHARE_ROOT>"),
    (re.compile(r"~/miniconda3/envs/[A-Za-z0-9_.-]+/bin/python"), "<REMOTE_PYTHON>"),
    (re.compile(r"miniconda3/envs/[A-Za-z0-9_.-]+"), "conda-env"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<IP>"),
    (re.compile(r"(?<![\d.])10022(?![\d.])"), "<SSH_PORT>"),
    (re.compile(r"(?<![\d.])8791(?![\d.])"), "<PORT>"),
    (re.compile(r"Swai@\S+"), "<REDACTED-SECRET>"),
]

TEXT_EXT = {".py", ".md", ".txt", ".json", ".csv", ".toml", ".yaml", ".yml", ".sh", ".html", ".cjs"}

# ---------------------------------------------------------------- 文件清单
# (源路径, 包内相对路径)
FILES: list[tuple[str, str]] = [
    # ---- 论文 ----
    ("docs/paper/main.md", "00-paper/main.md"),
    ("docs/paper/main.html", "00-paper/main.html"),
    ("docs/paper/main.pdf", "00-paper/main.pdf"),
    ("docs/paper/EVIDENCE.md", "00-paper/EVIDENCE.md"),
    ("docs/paper/POSITIONING.md", "00-paper/POSITIONING.md"),
    ("docs/paper/README-zh.md", "00-paper/README-zh.md"),
    ("docs/paper/figs/fig1_pipeline.svg", "00-paper/figs/fig1_pipeline.svg"),
    ("docs/paper/figs/fig2_pixel_budget.svg", "00-paper/figs/fig2_pixel_budget.svg"),
    ("docs/paper/figs/fig3_protocol.svg", "00-paper/figs/fig3_protocol.svg"),
    # ---- Neurocomputing 投稿件 ----
    ("docs/paper/main-review.html", "01-submission/manuscript-review-copy.html"),
    ("docs/paper/main-review.pdf", "01-submission/manuscript-review-copy.pdf"),
    ("docs/paper/submission/highlights.txt", "01-submission/highlights.txt"),
    ("docs/paper/submission/title-page.md", "01-submission/title-page.md"),
    ("docs/paper/submission/cover-letter.md", "01-submission/cover-letter.md"),
    ("docs/paper/submission/CHECKLIST.md", "01-submission/CHECKLIST.md"),
    ("docs/paper/submission/figures/Figure_1.pdf", "01-submission/figures/Figure_1.pdf"),
    ("docs/paper/submission/figures/Figure_2.pdf", "01-submission/figures/Figure_2.pdf"),
    ("docs/paper/submission/figures/Figure_3.pdf", "01-submission/figures/Figure_3.pdf"),
    # ---- 中文对照版 ----
    ("docs/paper/main-zh.md", "02-zh/main-zh.md"),
    ("docs/paper/main-zh.html", "02-zh/main-zh.html"),
    ("docs/paper/main-zh.pdf", "02-zh/main-zh.pdf"),
    ("docs/paper/figs/fig1_pipeline_zh.svg", "02-zh/figs/fig1_pipeline_zh.svg"),
    ("docs/paper/figs/fig2_pixel_budget_zh.svg", "02-zh/figs/fig2_pixel_budget_zh.svg"),
    ("docs/paper/figs/fig3_protocol_zh.svg", "02-zh/figs/fig3_protocol_zh.svg"),
    # ---- 真值 ----
    ("deliverable/gt_long35/gt_long35.json", "10-truth/T1_gt_long35.json"),
    ("deliverable/gt_long35/gt_pairs.csv", "10-truth/T1_gt_pairs.csv"),
    ("deliverable/gt_long35/gt_summary.json", "10-truth/T1_gt_summary.json"),
    ("deliverable/spacetime_review/gt_spacetime_items.json", "10-truth/T2_gt_spacetime_items.json"),
    ("deliverable/spacetime_review/labels_20261112.json", "10-truth/T2_labels_20261112.json"),
    ("deliverable/spacetime_review/items.json", "10-truth/T2_items.json"),
    ("deliverable/spacetime_review/ingest_report.md", "10-truth/T2_ingest_report.md"),
    ("deliverable/spacetime_review/apply_v1_report.md", "10-truth/T2_apply_v1_report.md"),
    ("deliverable/spacetime_review/README.md", "10-truth/T2_README.md"),
    # ---- 数据与结果 ----
    ("tmp/full/long35_crosscam.json", "20-data/graph_510_prelim.json"),
    ("tmp/full/long35_crosscam_attr8.json", "20-data/graph_443_v0.json"),
    ("tmp/full/long35_crosscam_attr8_v1.json", "20-data/graph_450_v1.json"),
    ("tmp/full/spacetime_audit.json", "20-data/spacetime_audit.json"),
    ("tmp/full/spacetime_ingraph.json", "20-data/spacetime_ingraph.json"),
    ("tmp/full/spacetime_signal.json", "20-data/spacetime_signal.json"),
    ("tmp/full/verifier_split.json", "20-data/verifier_split.json"),
    ("tmp/full/emb_auc_out.txt", "20-data/emb_auc_out.txt"),
    ("tmp/full/verifier_out.txt", "20-data/verifier_out.txt"),
    ("deliverable/gt_long35/attr_filter_ingraph.json", "20-data/attr_filter_ingraph.json"),
    ("deliverable/gt_long35/covis_table.json", "20-data/covis_table.json"),
    # ---- 代码 ----
    ("tmp/full/emb_auc.py", "30-code/emb_auc.py"),
    ("tmp/full/spacetime_audit.py", "30-code/spacetime_audit.py"),
    ("tmp/full/spacetime_ingraph.py", "30-code/spacetime_ingraph.py"),
    ("tmp/full/spacetime_verifier.py", "30-code/spacetime_verifier.py"),
    ("tmp/full/verifier_split.py", "30-code/verifier_split.py"),
    ("tmp/full/attr_filter_ingraph.py", "30-code/attr_filter_ingraph.py"),
    ("tools/build_spacetime_review.py", "30-code/build_spacetime_review.py"),
    ("tools/ingest_spacetime_review.py", "30-code/ingest_spacetime_review.py"),
    ("tools/apply_spacetime_truth.py", "30-code/apply_spacetime_truth.py"),
    ("tmp/full/exp_pose_pool.py", "30-code/exp_pose_pool.py"),
    ("tmp/full/pose_pr.py", "30-code/pose_pr.py"),
    ("tmp/full/exp_anchor_pool.py", "30-code/exp_anchor_pool.py"),
    ("tmp/full/exp_selector_runtime.py", "30-code/exp_selector_runtime.py"),
    ("tools/md2paper_html.py", "30-code/md2paper_html.py"),
]

README_PKG = """# Paper package v3 — {today}

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
"""

CODE_README = """# Code index

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
"""


def redact(text: str) -> str:
    for pat, repl in REDACT:
        text = pat.sub(repl, text)
    return text


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    (OUT / "README.md").write_text(README_PKG.format(today=TODAY), encoding="utf-8")
    (OUT / "30-code").mkdir(parents=True, exist_ok=True)
    (OUT / "30-code" / "README.md").write_text(CODE_README, encoding="utf-8")

    rows: list[tuple[str, int, str]] = []
    missing: list[str] = []

    for src_rel, dst_rel in FILES:
        src = ROOT / src_rel
        dst = OUT / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            missing.append(src_rel)
            continue
        if src.suffix.lower() in TEXT_EXT:
            try:
                raw = src.read_text(encoding="utf-8")
                dst.write_text(redact(raw), encoding="utf-8", newline="\n")
            except UnicodeDecodeError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        rows.append((dst_rel.replace("\\", "/"), dst.stat().st_size, sha256(dst)))

    # 自建文件也进清单
    for extra in ["README.md", "30-code/README.md"]:
        p = OUT / extra
        rows.append((extra, p.stat().st_size, sha256(p)))

    rows.sort(key=lambda r: r[0])

    with (OUT / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "sha256"])
        w.writerows(rows)

    with (OUT / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as f:
        for path, _, digest in rows:
            f.write(f"{digest}  {path}\n")

    total = sum(r[1] for r in rows)
    print(f"files: {len(rows)}   total: {total/1024:.0f} KB")
    if missing:
        print("MISSING:")
        for m in missing:
            print("  -", m)

    DIST.mkdir(parents=True, exist_ok=True)
    zip_path = DIST / f"paper-package-v3_{TODAY}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, Path("paper-package-v3") / p.relative_to(OUT))
    print(f"zip: {zip_path}  ({zip_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
