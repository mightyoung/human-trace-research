#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建「完整研究包」（research_package/）。

产物结构：
    research_package/
    ├─ README.md                    交付说明（结构/形式/优先级/脱敏范围与原则）
    ├─ MANIFEST.json                文件清单（路径/大小/sha1/是否脱敏/分组）
    ├─ 01_研究路径/                  （手写正文 + 详细报告全文）
    ├─ 02_阶段总结/                  （手写正文）
    ├─ 03_代码/                      CODE_INDEX.md + 代码本体
    └─ 04_试验数据/                  DATA_INDEX.md + 脱敏说明.md + 脱敏后数据

脱敏：文本文件统一过 REDACT 规则；**图像类默认不入包**（避免人物影像/生物特征）；
      仅收录经人工确认的"指标图"（白名单）。

用法：
    python tools/build_research_package.py            # 全量重建（保留手写正文）
    python tools/build_research_package.py --scan     # 只做泄漏扫描
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "research_package"

# ============================ 脱敏规则 ============================
# 顺序敏感：长模式/更具体的在前。规则沿用 build_handoff_pkg.py 并经本轮扩充。
REDACT = [
    # --- 凭据 / 流地址 ---
    (re.compile(r"rtsp://[^\s\"'<>]+"),
     "rtsp://<user>:<pass>@<camera-ip>:554/Streaming/Channels/102"),
    (re.compile(r"Swai@\S+"), "<REDACTED-SECRET>"),
    (re.compile(r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key)\b\s*[:=]\s*[\"']?[^\s\"',}{]{3,}"),
     r"\1=<REDACTED-SECRET>"),
    # --- 主机 / 路径 / 账号 ---
    (re.compile(r"/home/r4900g3/yhh/human-count"), "<REMOTE_WORKDIR>"),
    (re.compile(r"/home/r4900g3/yhh"), "<REMOTE_WORKDIR>"),
    (re.compile(r"/share/data4/yhh/data/human-count"), "<DATA_ROOT>"),
    (re.compile(r"/share/data4/yhh"), "<SHARE_ROOT>"),
    (re.compile(r"~/yhh/human-count"), "<REMOTE_WORKDIR>"),
    (re.compile(r"~/yhh"), "<REMOTE_WORKDIR>"),
    (re.compile(r"r4900g3-UniServer-R4900-G3"), "<REMOTE_HOST>"),
    (re.compile(r"r4900g3"), "<REMOTE_USER>"),
    (re.compile(r"(?<![A-Za-z0-9_])yhh(?![A-Za-z0-9_])"), "<REMOTE_USER>"),
    (re.compile(r"~/miniconda3/envs/rfdetr/bin/python"), "<REMOTE_PYTHON>"),
    (re.compile(r"miniconda3/envs/rfdetr"), "conda-env(rfdetr)"),
    (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+might"), "<LOCAL_HOME>"),
    (re.compile(r"[A-Za-z]:[\\/]+dev[\\/]+human-count"), "<PROJECT_ROOT>"),
    (re.compile(r"[A-Za-z]:[\\/]+hc2"), "<RUST_TARGET_DIR>"),
    # --- IP / 端口 ---
    (re.compile(r"192\.168\.\d{1,3}\.\d{1,3}"), "<REMOTE_IP>"),
    (re.compile(r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"), "<INTERNAL_IP>"),
    (re.compile(r"(?<![\d.])10022(?![\d.])"), "<REMOTE_PORT>"),
    (re.compile(r"(?<![\d.])8791(?![\d.])"), "<PORT>"),
    (re.compile(r"(?<![\d.])8099(?![\d.])"), "<CONSOLE_PORT>"),
    # --- 基础设施指纹 ---
    (re.compile(r"\bL20\b"), "GPU"),
    (re.compile(r"\basr_server\b"), "other-service-1"),
    (re.compile(r"\blocate_server\b"), "other-service-2"),
]

# --- ★ 场景 / 机构语义（可定位楼层布局与单位身份）→ 匿名代号 ---
# 规则表外置（tools/_redact_sites.json，**不随包交付**），避免打包器自身泄漏原词。
_SITE_FILE = Path(__file__).resolve().parent / "_redact_sites.json"
_SITE_MAP, _ORG_MAP, _DESC_MAP = {}, {}, {}
if _SITE_FILE.exists():
    _cfg = json.loads(_SITE_FILE.read_text(encoding="utf-8"))
    _SITE_MAP = _cfg.get("sites", {})
    _ORG_MAP = _cfg.get("orgs", {})
    _DESC_MAP = _cfg.get("descriptors", {})
_ALL_LABEL = {**_SITE_MAP, **_ORG_MAP, **_DESC_MAP}
# 长词优先，避免 "行政走廊" 先命中 "行政走廊通道"
for _k in sorted(_ALL_LABEL, key=len, reverse=True):
    REDACT.append((re.compile(re.escape(_k)), _ALL_LABEL[_k]))

# 兜底泄漏扫描（正则 + 边界，避免把 sim=0.8791 这类数值误判为端口）
SCAN_PATTERNS = [
    (r"(?<![A-Za-z0-9_])r4900g3(?![A-Za-z0-9_])", "远程主机名"),
    (r"(?<![A-Za-z0-9_])yhh(?![A-Za-z0-9_])", "远程账号"),
    (r"Swai@", "凭据"),
    (r"rtsp://\d{1,3}\.\d{1,3}", "流地址(真实 IP)"),
    (r"192\.168\.\d{1,3}\.\d{1,3}", "内网 IP"),
    (r"(?<![\d.])10022(?![\d.])", "SSH 端口"),
    (r"(?<![\d.])8791(?![\d.])", "服务端口"),
    (r"(?<![\d.])8099(?![\d.])", "控制台端口"),
] + [(re.escape(k), "场景/机构/示例标签语义") for k in _ALL_LABEL]

TEXT_EXT = {".py", ".sh", ".md", ".json", ".jsonl", ".csv", ".toml", ".txt",
            ".yaml", ".yml", ".cfg", ".ini", ".rs", ".ts", ".vue", ".cjs",
            ".js", ".css", ".html"}

# ============================ 代码分拣 ============================
CODE_SPEC = [
    # (包内子目录, 源目录, 允许扩展名, 排除模式)
    ("python-pipeline", "tools", {".py", ".sh"}, None),
    ("console-backend", "webapp/backend", {".py"}, None),
    ("console-frontend", "webapp/frontend/src", {".ts", ".vue", ".js", ".css"}, None),
    ("console-frontend-root", "webapp/frontend", {".json", ".ts", ".js"}, "src/"),
    ("verification", "webapp/deploy", {".cjs", ".sh", ".js"}, None),
    ("rust-service", "src", {".rs"}, None),
]

CODE_SINGLE = [  # (包内子目录, 文件)
    ("rust-service", "Cargo.toml"),
    ("rust-service", "config.toml"),
    ("python-pipeline", "tools/_remote_encode6.sh"),
]

CODE_EXCLUDE_NAMES = {".env", ".env.local", "labels.json", "faces_labels.json",
                      "faces_labels_remote_1014.json", "package-lock.json",
                      "_redact_sites.json"}
CODE_EXCLUDE_PAT = re.compile(r"(\.bak|__pycache__|node_modules|/dist/|" 
                              r"\.pyc$|_remote_.*\.json$)")

# ============================ 数据分拣 ============================
DATA_SPEC = {
    "01_数据链与口径": [
        "output_crosscam/cleanset_v2/clean_manifest_v2.json",
        "output_crosscam/analysis/splits_v5.json",
        "output_crosscam/analysis/splits_v4.json",
        "output_crosscam/analysis/audit/data_chain_audit.json",
        "output_crosscam/analysis/multicam_overview.json",
    ],
    "02_跨机位接力与门户": [
        "output_crosscam/analysis/relay_v2/edges_v2_gated_human.json",
        "output_crosscam/analysis/relay_v2/edges_v2_gated.json",
        "output_crosscam/analysis/relay_v2/edge_states.json",
        "output_crosscam/analysis/relay_v2/metrics_v1.json",
        "output_crosscam/analysis/relay_v2/next_portal_head_v1.json",
        "output_crosscam/analysis/relay_v2/portal_prior_recheck_v1.json",
        "output_crosscam/analysis/relay_v2/visit_portal_link_v1.json",
        "output_crosscam/analysis/relay_clean2/relay_chains_cradio_v6.json",
        "output_crosscam/analysis/relay_clean2/relay_scored_cradio_v5.json",
        "output_crosscam/analysis/c0/c0_baseline_v9.json",
        "output_crosscam/analysis/c0/c0_baseline_v8.json",
        "output_crosscam/analysis/topology/topology_signature.json",
        "output_crosscam/analysis/topology/portal_graph_v2.json",
    ],
    "03_身份选择器与意图": [
        "output_crosscam/analysis/selector_1012/eval_test.json",
        "output_crosscam/analysis/selector_1012/crossseg.json",
        "output_crosscam/analysis/selector_1012/cluster_report.json",
        "output_crosscam/analysis/selector_1012/clf_test.json",
        "output_crosscam/analysis/camgroup_study/study.json",
        "output_crosscam/analysis/intent/1012_all_stats.json",
        "output_crosscam/analysis/intent/long35_all_stats.json",
    ],
    "04_反光倒影治理": [
        "output_crosscam/analysis/mirror_kinematics/mirror_kinematics.json",
        "output_crosscam/analysis/mirror_flip/mirror_flip.json",
        "output_crosscam/analysis/reflection_p1/reflection_depth_scores.json",
        "output_crosscam/analysis/gil/reflection_cluster_v1.json",
        "output_crosscam/analysis/eval3/reflection_gate_ab.json",
    ],
    "05_身份层与人工裁决": [
        "output_crosscam/analysis/gil/gperson_human_v1.json",
        "output_crosscam/analysis/gil/human_adjudication_v1.json",
        "output_crosscam/analysis/gil/human_adjudication_v2.json",
        "output_crosscam/analysis/gil/frag_link_v1.json",
        "output_crosscam/analysis/gil/visit_owner_human_v1.json",
        "output_crosscam/analysis/gil/face_constraint_absorption_v1.json",
        "output_crosscam/analysis/face/face_embed_review_v2.json",
        "output_crosscam/analysis/face/face_mustlink_v1.json",
    ],
    "06_人工真值": [
        "output_crosscam/analysis/truth/TRUTH-FINDINGS-2026-10-14.md",
        "output_crosscam/analysis/truth/ADJUDICATION-2026-10-14.md",
        "output_crosscam/analysis/truth/events_all.csv",
        "output_crosscam/analysis/truth/pair_candidates_labeled.csv",
    ],
    "07_评测与审计": [
        "output_crosscam/analysis/eval_registry/registry_v1.json",
        "output_crosscam/analysis/eval2/learned_reviewer.json",
        "output_crosscam/analysis/eval3/self_train_arms.json",
        "output_crosscam/analysis/eval4/next_portal.json",
        "output_crosscam/analysis/mrl/mrl_validation.json",
        "output_crosscam/analysis/mrl/mrl_retrieval_check.json",
        "output_crosscam/analysis/unsup_reid/exp_cradio_v3.json",
    ],
    "08_人脸与活体_仅汇总指标": [
        "output_crosscam/analysis/face/face_embed_p0c.json",
        "output_crosscam/analysis/face/face_live_flags_v0.json",
    ],
}

# 图像白名单（经人工确认的"指标图"；人物影像一律不入包）
FIG_SPEC = {
    "09_图表": [
        "output_crosscam/analysis/c0/c0_baseline_v9.png",
        "output_crosscam/analysis/c0/fig_clean2_version_compare.png",
        "output_crosscam/analysis/topology/topology_signature.png",
        "output_crosscam/analysis/_pkg_fig/fig_multicam_timeline__desensitized.png",
    ],
}


def scrub(text: str) -> str:
    for pat, rep in REDACT:
        text = pat.sub(rep, text)
    return text


def sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def emit_text(src: Path, dst: Path, manifest, group):
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = src.read_text(encoding="utf-8", errors="replace")
    out = scrub(raw)
    dst.write_text(out, encoding="utf-8")
    manifest.append(dict(group=group, path=str(dst.relative_to(PKG)),
                         src=str(src.relative_to(ROOT)), size=dst.stat().st_size,
                         sha1=sha1(dst), redacted=bool(out != raw)))


def copy_code(manifest):
    for sub, srcdir, exts, skip in CODE_SPEC:
        base = ROOT / srcdir
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            if CODE_EXCLUDE_PAT.search(rel) or p.name in CODE_EXCLUDE_NAMES:
                continue
            if skip and rel.startswith(f"{srcdir}/{skip}"):
                continue
            emit_text(p, PKG / "03_代码" / sub / p.relative_to(base), manifest, "03_代码")
    for sub, rel in CODE_SINGLE:
        p = ROOT / rel
        if p.exists():
            emit_text(p, PKG / "03_代码" / sub / p.name, manifest, "03_代码")


def copy_docs(manifest):
    """74 篇研究报告全文 —— 归入 01_研究路径/详细报告/。"""
    d = ROOT / "docs"
    for p in sorted(d.rglob("*.md")):
        if CODE_EXCLUDE_PAT.search(str(p)):
            continue
        emit_text(p, PKG / "01_研究路径" / "详细报告" / p.relative_to(d),
                  manifest, "01_研究路径")


def copy_data(manifest):
    for group, files in DATA_SPEC.items():
        for rel in files:
            p = ROOT / rel
            if not p.exists():
                manifest.append(dict(group=group, path="", src=rel, size=0,
                                     sha1="", redacted=False, missing=True))
                print(f"  [miss] {rel}")
                continue
            emit_text(p, PKG / "04_试验数据" / group / p.name, manifest, group)
    for group, files in FIG_SPEC.items():
        for rel in files:
            p = ROOT / rel
            if not p.exists():
                print(f"  [miss-fig] {rel}")
                continue
            dst = PKG / "04_试验数据" / group / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            manifest.append(dict(group=group, path=str(dst.relative_to(PKG)),
                                 src=rel, size=dst.stat().st_size,
                                 sha1=sha1(dst), redacted=False))


def scan():
    """泄漏扫描：对包内全部文本按 SCAN_PATTERNS 检查（正则 + 词边界）。"""
    hits = []
    pats = [(re.compile(p), name) for p, name in SCAN_PATTERNS]
    for p in sorted(PKG.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT:
            continue
        if p.name == "MANIFEST.json":   # 清单自身只含规则名与路径，不参与扫描
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        for rx, name in pats:
            n = len(rx.findall(txt))
            if n:
                hits.append((str(p.relative_to(PKG)), name, n))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="只做泄漏扫描")
    args = ap.parse_args()

    if args.scan:
        hits = scan()
        print(f"[scan] 命中 {len(hits)} 条")
        for h in hits[:60]:
            print("   ", h)
        sys.exit(1 if hits else 0)

    manifest = []
    # 只清「生成目录」，保留手写正文与索引（*.md 位于父层）
    clean_dirs = ([f"03_代码/{s[0]}" for s in CODE_SPEC] +
                  [f"04_试验数据/{g}" for g in list(DATA_SPEC) + list(FIG_SPEC)] +
                  ["01_研究路径/详细报告"])
    for sub in clean_dirs:
        d = PKG / sub
        if d.exists():
            shutil.rmtree(d)
    print("[1/4] 代码本体 ...")
    copy_code(manifest)
    print("[2/4] 研究报告全文 ...")
    copy_docs(manifest)
    print("[3/4] 脱敏试验数据 ...")
    copy_data(manifest)
    print("[4/4] 清单 + 泄漏扫描 ...")
    meta = dict(
        version="research_package_v1",
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        redaction_rules=len(REDACT),
        scan_patterns=[n for _, n in SCAN_PATTERNS],
        site_map_entries=len(_ALL_LABEL),
        n_files=len([m for m in manifest if not m.get("missing")]),
        n_redacted=len([m for m in manifest if m.get("redacted")]),
        total_bytes=sum(m["size"] for m in manifest if not m.get("missing")),
    )
    hits = scan()
    meta["leak_hits"] = len(hits)
    (PKG / "MANIFEST.json").write_text(
        json.dumps(dict(meta=meta, files=manifest), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    if hits:
        print("\n!! 泄漏命中（需修复）：")
        for h in hits[:40]:
            print("   ", h)
    else:
        print("\n✓ 泄漏扫描通过（禁字 0 命中）")


if __name__ == "__main__":
    main()
