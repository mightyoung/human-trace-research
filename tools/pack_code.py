#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把项目源码打成一个「不含任何人像像素」的代码包。

只收源码与配置：.py / .rs / .ts / .vue / .js / .css / .html / .json / .toml /
.sh / .sql / .md。图片、视频、模型权重、二进制、压缩包、日志、数据库一律不收；
`node_modules` / `dist` / `target` / `data` / `public` / `__pycache__` 整目录跳过。

打包后立刻用 `scan_sensitive.py` 复扫 zip，断言 A（图片）与 V（视频）与 B（内嵌）都为 0。

用法
  PY tools/pack_code.py [--tag 2026-10-23]
"""
import argparse
import csv
import datetime as dt
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "deliverable", "dist")
NAME = "human-count-code"

# ---- 收哪些目录（相对 ROOT）
ROOTS = [
    "src",                 # Rust 核心
    "configs",             # 生产/实验配置
    "tools",               # 全部分析与管线脚本（主体）
    "tests-pure",          # Rust 纯逻辑测试（跳过 target）
    "webapp/backend",      # 联调后端（跳过 data/）
    "webapp/dataprep",     # 数据准备脚本
    "webapp/deploy",       # 部署脚本（跳过截图与 tgz）
    "webapp/frontend",     # 前端源码（跳过 node_modules / dist / public）
]

# ---- 直接按名收的根文件
TOP_FILES = ["README.md", "OVERVIEW.md", "HANDOFF.md",
             "Cargo.toml", "Cargo.lock", "config.toml"]

# ---- 扩展名黑名单：像素、音视频、模型、二进制、压缩包、字体、日志库
BAD_EXT = {
    # 像素
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".gif", ".webp", ".bmp",
    ".tif", ".tiff", ".ico", ".svg", ".heic", ".avif", ".psd", ".ai",
    # 视频 / 音频
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg",
    ".mp3", ".wav", ".flac", ".m4a", ".aac",
    # 模型权重与张量
    ".onnx", ".pt", ".pth", ".engine", ".npz", ".npy", ".pkl", ".pickle",
    ".h5", ".safetensors", ".ckpt", ".bin", ".weights",
    # 编译产物 / 二进制
    ".pyc", ".pyo", ".o", ".obj", ".rlib", ".rmeta", ".pdb", ".exe", ".dll",
    ".so", ".dylib", ".a", ".lib", ".class", ".jar",
    # 压缩包
    ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".xz", ".bz2",
    # 字体
    ".ttf", ".otf", ".woff", ".woff2",
    # 数据 / 日志
    ".log", ".db", ".sqlite", ".sqlite3", ".parquet", ".feather", ".pdf",
}

# ---- 整目录跳过
BAD_DIRS = {"__pycache__", "node_modules", "dist", "build", "target", ".git",
            ".cache", ".venv", "venv", "site-packages", "data", "media",
            "crops", "clips", "public", ".pytest_cache", ".mypy_cache",
            "tmp", ".idea", ".vscode", "assets.bak"}


def skip_dir(name):
    if name in BAD_DIRS:
        return True
    return name.startswith("data.bak") or name.startswith(".")


def is_bak(fn):
    low = fn.lower()
    return ".bak" in low or low.endswith(".orig") or low.endswith(".rej")


def walk_root(rel_root):
    base = os.path.join(ROOT, rel_root)
    out = []
    for dp, dns, fs in os.walk(base):
        dns[:] = sorted(d for d in dns if not skip_dir(d))
        for f in sorted(fs):
            if os.path.splitext(f)[1].lower() in BAD_EXT:
                continue
            if is_bak(f):
                continue
            p = os.path.join(dp, f)
            out.append(p)
    return out


def lines_of(p):
    try:
        with open(p, "rb") as fh:
            return fh.read().count(b"\n") + 1
    except OSError:
        return 0


DOC1 = re.compile(r'^\s*(?:"""|\'\'\')(.{0,110})')


def doc_first(p):
    if not p.lower().endswith((".py", ".rs", ".sh")):
        return ""
    try:
        txt = open(p, encoding="utf-8", errors="replace").read(4000)
    except OSError:
        return ""
    for ln in txt.splitlines()[:12]:
        m = DOC1.match(ln)
        if m:
            return m.group(1).strip().rstrip('"').strip()
    return ""


def build_stage(stage):
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    rows = []
    for rel in ROOTS:
        for p in walk_root(rel):
            dst = os.path.join(stage, os.path.relpath(p, ROOT))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(p, dst)
            rows.append((os.path.relpath(p, ROOT).replace("\\", "/"),
                         os.path.getsize(p), lines_of(p), doc_first(p)))
    for f in TOP_FILES:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(stage, f))
            rows.append((f, os.path.getsize(p), lines_of(p), ""))
    # 设计文档：只收 docs/ 顶层的 md，commit-015 等外部提案目录不收
    docdir = os.path.join(ROOT, "docs")
    if os.path.isdir(docdir):
        os.makedirs(os.path.join(stage, "docs"), exist_ok=True)
        for f in sorted(os.listdir(docdir)):
            p = os.path.join(docdir, f)
            if os.path.isfile(p) and f.lower().endswith(".md"):
                shutil.copy2(p, os.path.join(stage, "docs", f))
                rows.append((f"docs/{f}", os.path.getsize(p), lines_of(p), ""))
    rows.sort()
    return rows


CODE_MAP = """# 代码地图

> 生成于 {date}｜共 {n} 个源码文件、{lines:,} 行。

本包是**纯源码包**，不含任何数据集、模型权重或影像。数字口径见配套数据包
`data-request-v2_*_noimg.zip` 的 `NUMBERS.md`。完整模块清单见 `CODE_INVENTORY.csv`
（每个文件的字节数、行数、文档首行）。

## 主线管线（按执行顺序）

| 步骤 | 入口 | 说明 |
|---|---|---|
| 0 素材切片 | `tools/cut_1012_segments.py`、`tools/make_splits_v3.py` | 按训练/测试段切视频，stride 抽帧 |
| 1 检测 | `tools/local_detect.py`、`tools/run_yolo_*.sh`、`tools/run_6cam_det.py` | YOLO26s-seg；存档必须 `--iou 0.7` |
| 2 去重 | `tools/dedup_detections.py` | 同帧同人重复框，`--iou 0.5` |
| 3 轨迹 / a·b 分片 | `tools/make_ab_jsonl.py`、`tools/yolo_track_det.py` | ByteTrack 关联；轨迹前后半切分成 a/b 两片 |
| 4 编码 | `tools/encode_1012_osnet_seq.py`、`tools/embed_batch.py`、`tools/embed_fast.py` | OSNet 冻结权重，逐轨迹平均池化 |
| 5 聚类 | `tools/cluster_camera_aware.py` | 机位内 average 0.15 → 跨机位 complete 0.25，同机位禁并 |
| 6 参数扫描 | `tools/d7_sweep.py`、`tools/session_selector_v1.py` | θ 扫描 + self-cal 选点 + `--robust-margin` |
| 7 簇真值 | `tools/derive_clusters_gt.py` | 由逐成员标注派生簇主体、混入者、d_max |
| 8 精度评估 | `tools/eval_precision.py`、`tools/compare_cfgs.py` | 逐条 C/W/U、McNemar + bootstrap |
| 9 基线冻结 | `tools/freeze_baseline.py` | 产出 `BASELINE_FREEZE.json` 供逐位复跑校验 |

## 诊断与专项

| 主题 | 脚本 |
|---|---|
| 簇有害性 / d_max | `tools/diag_cluster_harm.py`、`tools/recompute_purity_stats.py` |
| 分数与门限 | `tools/diag_score.py`、`tools/diag_ceiling.py`、`tools/eval_theta_curve_precision.py` |
| 误差归因 | `tools/diag_errors.py`、`tools/diag_box.py`、`tools/diag_320_unanswered.py`、`tools/analyze_loss_decomp.py` |
| 机位 / 素材 | `tools/diag_320_gallery.py`、`tools/sample_density_material.py`、`tools/audit_video_material.py` |
| 6 机位数据集 | `tools/run_6cam_det.py`、`tools/run_6cam_pipeline.py`、`tools/long35_analysis.py` |
| 串人复核片段 | `tools/switch_windows.py`、`tools/render_switch_clips.py`、`tools/build_switch_review.py` |

## 标注与信度

| 用途 | 脚本 |
|---|---|
| 标注页生成 | `tools/build_annotator_html.py`（`--embed` 出单文件版）、`tools/build_review_d3.py`、`tools/build_d3_tail_html.py`、`tools/build_kappa_review.py` |
| 标注合并 | `tools/merge_review_d3.py`、`tools/merge_d3_tail.py`、`tools/merge_kappa_verdict.py` |
| 一致性 | `tools/eval_kappa.py`、`tools/eval_kappa_a3.py`、`tools/align_annotators.py`、`tools/verify_with_d5.py` |
| 抽样 | `tools/make_a3_sample.py`、`tools/make_gt_templates.py`、`tools/make_query_sheets.py` |

## 交付与自检

| 用途 | 脚本 |
|---|---|
| 台账导出 | `tools/export_paper_ledger.py`（D1/D2/D6 全部 CSV） |
| 去图打包 | `tools/pack_noimage.py`（数据包）、`tools/pack_code.py`（本包） |
| 隐私扫描 | `tools/scan_sensitive.py`（按魔数 + `data:image` payload 双路判定） |
| 包自检 | `tools/verify_package.py`（行数、占位符、路径引用、HTML） |
| 清单哈希 | `tools/make_manifest.py` |

## Web 联调（与论文主线无关，保留备查）

`webapp/backend`（FastAPI 服务：`main.py` / `algo.py` / `render.py` / `predict.py` / `evaluate.py`）、
`webapp/dataprep`（把真值转成前端数据）、`webapp/frontend/src`（Vue 3 + Vite）、
`webapp/deploy`（部署与远程渲染脚本）。
"""

README_CODE = """# human-count 代码包

> 生成于 {date}｜{n} 个文件｜{size}｜{lines:,} 行（代码 + 文档）

## 这是什么

跨机位行人重识别（ReID）实验的**全部源码与配置**：4 机位（305 / 318 / 320 / 323）
＋ 6 机位（加 343 / 316）两条数据线，含检测、跟踪、编码、聚类、参数扫描、
真值派生、逐条精度评估、诊断、标注页生成与交付自检。

**本包不含任何影像**：无视频、无帧截图、无人像裁图、无内嵌 base64 图片。
详见文末「隐私核查」。

## 目录

```
src/            Rust 检测/预处理核心（Cargo.toml、config.toml 在根）
configs/        selector_deploy.json 等生产配置
tools/          {ntools} 个 Python 脚本 + {nsh} 个 shell（主体，见 CODE_MAP.md）
tests-pure/     Rust 纯逻辑测试
webapp/         Vue 3 前端 + FastAPI 后端 + 数据准备 + 部署脚本
docs/           {ndocs} 份设计/实验记录（仅 md，无图）
CODE_MAP.md     按流程组织的入口索引
CODE_INVENTORY.csv  逐文件清单（路径、字节、行数、文档首行）
```

## 环境

```bash
# Python（Windows 本机）
PY=C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe
export PYTHONPATH=D:/human-count/pylibs
export TMPDIR=D:/human-count/tmp TEMP=D:/human-count/tmp
```

- `torch 2.4.1+cpu`；本机**无 NVIDIA**（AMD RX 5700 XT）。`torch_directml` 可跑编码，
  但 ultralytics 在 DirectML 上崩溃，**检测只能 CPU**（3.3 f/s）。
- 依赖：`ultralytics`、`torch`、`numpy`、`scipy`、`scikit-learn`、`opencv-python`、
  `supervision`、`ffmpeg`（渲染片段用）。
- 长任务建议后台跑；单机全量重跑约 2–3 小时（检测占 80%）。

## 数据不在本包内

代码需要的数据（轨迹台账、真值 CSV、42 配置判决）在配套数据包里：
`data-request-v2_*_noimg.zip`（`ledger/`、`gt/`、`results/`）。
原始视频与裁图**不外发**，仅在源机器上保留。

## 复跑主结果

```bash
# 1) 重建台账（D1/D2/D6）
$PY tools/export_paper_ledger.py

# 2) 全配置扫描（42 配置 × 63 个 θ）
$PY tools/d7_sweep.py --run deliverable/data-request-v2/results

# 3) 接入真值算逐条精度
$PY tools/eval_precision.py --run deliverable/data-request-v2/results

# 4) 与 baseline 做配对检验
$PY tools/compare_cfgs.py --run deliverable/data-request-v2/results

# 5) 基线冻结校验（逐位比对）
$PY tools/freeze_baseline.py

# 6) 交付前自检 + 隐私扫描
$PY tools/verify_package.py
$PY tools/scan_sensitive.py deliverable/dist/data-request-v2_*_noimg.zip
```

生产基线：YOLO26s-seg + 同帧去重 → 机位感知聚类（机位内 average 0.15 /
跨机位 complete 0.25 / 同机位禁并）。**不要**再加 `veto_dmax=0.30`——逐条真值显示
正确数 −28（p<1e-4）。

## 未包含

| 类别 | 规模 | 原因 |
|---|---|---|
| `media-resource/`（原始监控视频） | 39 GB | 人像 |
| `models/`（YOLO / OSNet / DINOv3 权重） | 20 GB | 二进制、可从公开源重新下载 |
| `output*/`（检测 jsonl、npz、轨迹） | 2.3 GB | 数据，且体积大 |
| `deliverable/`（数据包与裁图） | 282 MB | 另行单独交付 |
| `docs/commit-015/`（外部方案与补丁） | — | 第三方代码，非本项目产出 |
| `research_package/`、`handoff_out/` | 23 MB | 旧快照，与 `tools/` 重复 |
| `pylibs/`、`vendor/`、`wheels_*/`、`target/` | 392 MB | 依赖与编译产物 |
| `tools/*.bak_*` | {bak} 个 | 历史备份，已被现行版本取代 |

## 隐私核查

打包后自动复扫（`tools/scan_sensitive.py`，按文件魔数 + `data:image` payload 双路判定）：

- 图片文件（A 类）：**{na}**（期望 0）
- 视频文件（V 类）：**{nv}**（期望 0）
- 内嵌图片（B 类）：**{nb}**（期望 0）

源码里出现的 `data:image/webp;base64,` 只是**拼接字符串的代码字面量**，不含像素；
扫描器按「后随 ≥1.5 KB 连续 base64」判为真实内嵌，本包命中 0。

需要看图核对时，走 `PRIVACY.md` 的授权渠道取最小子集，不要整包转发。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=dt.datetime.now().strftime("%Y-%m-%d"))
    a = ap.parse_args()

    os.makedirs(DIST, exist_ok=True)
    stage_root = os.path.join(DIST, "_code_stage")
    stage = os.path.join(stage_root, NAME)
    print(f"[1/4] 收集源码 → {stage}")
    rows = build_stage(stage)

    total_bytes = sum(r[1] for r in rows)
    total_lines = sum(r[2] for r in rows)
    n_bak = len([f for f in os.listdir(os.path.join(ROOT, "tools"))
                 if ".bak" in f.lower()])

    # 先扫 staging，拿到真实命中数再写 README（避免文档里写「0」却没验证）
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import scan_sensitive as ss
    pre = ss.scan_dir(stage)
    n_pre_a = len([r for r in pre if r["kind"] == "A"])
    n_pre_v = len([r for r in pre if r["kind"] == "V"])
    n_pre_b = len([r for r in pre if r["kind"] == "B"])

    with open(os.path.join(stage, "CODE_INVENTORY.csv"), "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "bytes", "lines", "doc"])
        w.writerows(rows)

    with open(os.path.join(stage, "CODE_MAP.md"), "w",
              encoding="utf-8", newline="") as fh:
        fh.write(CODE_MAP.format(date=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                 n=len(rows), lines=total_lines))

    n_tools_py = sum(1 for r in rows if r[0].startswith("tools/") and r[0].endswith(".py"))
    n_tools_sh = sum(1 for r in rows if r[0].startswith("tools/") and r[0].endswith(".sh"))
    n_docs = sum(1 for r in rows if r[0].startswith("docs/"))
    with open(os.path.join(stage, "README_CODE.md"), "w",
              encoding="utf-8", newline="") as fh:
        fh.write(README_CODE.format(
            date=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            n=len(rows) + 3,
            size=f"{total_bytes/1e6:.1f} MB",
            lines=total_lines, bak=n_bak,
            ntools=n_tools_py, nsh=n_tools_sh, ndocs=n_docs,
            na=n_pre_a, nv=n_pre_v, nb=n_pre_b))
    print(f"  {len(rows)} 个文件｜{total_bytes/1e6:.1f} MB｜{total_lines:,} 行")
    print(f"  预扫 staging：图片 {n_pre_a}｜视频 {n_pre_v}｜内嵌 {n_pre_b}")
    if n_pre_a or n_pre_v or n_pre_b:
        raise SystemExit("staging 里仍有影像，先修排除规则")

    print("[2/4] 打包")
    zp = os.path.join(DIST, f"{NAME}_{a.tag}.zip")
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for dp, _, fs in os.walk(stage):
            for f in fs:
                p = os.path.join(dp, f)
                z.write(p, os.path.join(NAME, os.path.relpath(p, stage)))
    print(f"  {zp}  {os.path.getsize(zp)/1e6:.2f} MB")

    print("[3/4] 隐私复扫")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import scan_sensitive as ss
    rows2 = ss.scan_zip(zp)
    A = [r for r in rows2 if r["kind"] == "A"]
    V = [r for r in rows2 if r["kind"] == "V"]
    B = [r for r in rows2 if r["kind"] == "B"]
    print(f"  文件 {len(rows2)}｜图片 {len(A)}｜视频 {len(V)}｜内嵌 {len(B)}")
    if A or V or B:
        print("  !! 命中：", [(r["kind"], r["path"]) for r in (A + V + B)][:10])
        raise SystemExit(1)

    print("[4/4] 校验关键文件")
    names = {r["path"] for r in rows2}
    need = ["tools/d7_sweep.py", "tools/cluster_camera_aware.py",
            "tools/eval_precision.py", "tools/scan_sensitive.py",
            "src/main.rs", "configs/selector_deploy.json", "CODE_MAP.md"]
    miss = [f for f in need if f"{NAME}/{f}" not in names]
    print(f"  关键文件缺失：{miss if miss else '无'}")
    if miss:
        raise SystemExit(1)
    print("  ✓ 无影像、关键文件齐备")

    shutil.rmtree(stage_root, ignore_errors=True)
    print(f"\n[done] {zp}")


if __name__ == "__main__":
    main()
