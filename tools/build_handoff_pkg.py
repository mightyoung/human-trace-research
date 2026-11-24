# -*- coding: utf-8 -*-
"""构建「研究交接包」：脱敏 + 分拣 + 打包 zip。

用法: python tools/build_handoff_pkg.py
产物: handoff_out/human-count-research-handoff.zip
"""
import os
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "handoff_out" / "human-count-research-handoff"
ANA = ROOT / "output_crosscam" / "analysis"

# ---------------- 脱敏规则（顺序敏感：长模式在前） ----------------
REDACT = [
    (re.compile(r"rtsp://[^\s\"'<>]+"), "rtsp://<user>:<pass>@<camera-ip>:554/Streaming/Channels/102"),
    (re.compile(r"/home/r4900g3/yhh/human-count"), "<REMOTE_WORKDIR>"),
    (re.compile(r"/home/r4900g3/yhh"), "<REMOTE_WORKDIR>"),
    (re.compile(r"~/miniconda3/envs/rfdetr/bin/python"), "<REMOTE_PYTHON>"),
    (re.compile(r"miniconda3/envs/rfdetr"), "conda-env(rfdetr)"),
    (re.compile(r"/share/data4/yhh/data/human-count"), "<DATA_ROOT>"),
    (re.compile(r"/share/data4/yhh"), "<SHARE_ROOT>"),
    (re.compile(r"r4900g3-UniServer-R4900-G3"), "<REMOTE_HOST>"),
    (re.compile(r"r4900g3"), "<REMOTE_USER>"),
    # ~ 简写形式的远程家目录（此前只覆盖了绝对路径 /home/<user>/...）
    (re.compile(r"~/yhh/human-count"), "<REMOTE_WORKDIR>"),
    (re.compile(r"~/yhh"), "<REMOTE_WORKDIR>"),
    (re.compile(r"(?<![A-Za-z0-9_])yhh(?![A-Za-z0-9_])"), "<REMOTE_USER>"),
    (re.compile(r"Swai@\S+"), "<REDACTED-SECRET>"),
    (re.compile(r"(?<![\d.])10022(?![\d.])"), "<REMOTE_PORT>"),
    (re.compile(r"(?<![\d.])8791(?![\d.])"), "<PORT>"),  # embed_server 端口
    # 基础设施指纹（非个人隐私，但可定位具体部署环境）
    (re.compile(r"\bL20\b"), "GPU"),
    (re.compile(r"\basr_server\b"), "other-service-1"),
    (re.compile(r"\blocate_server\b"), "other-service-2"),
    (re.compile(r"192\.168\.\d{1,3}\.\d{1,3}"), "<REMOTE_IP>"),
    (re.compile(r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"), "<INTERNAL_IP>"),
    (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+might"), "<LOCAL_HOME>"),
    (re.compile(r"[A-Za-z]:[\\/]+dev[\\/]+human-count"), "<PROJECT_ROOT>"),
    (re.compile(r"[A-Za-z]:[\\/]+hc2"), "<RUST_TARGET_DIR>"),
]

TEXT_EXT = {".py", ".md", ".json", ".jsonl", ".csv", ".sh", ".toml",
            ".txt", ".yaml", ".yml", ".cfg", ".ini"}

# ---------------- 内容清单 ----------------
CODE_FILES = [
    # 检测/跟踪/基础设施
    "tools/infer_supervision.py", "tools/containment_nms.py", "tools/det_view.py",
    "tools/remote_run.py", "tools/make_splits_v3.py", "tools/run_wave4.sh",
    "tools/long35_analysis.py", "tools/multicam_overview.py",
    # 反光/倒影治理
    "tools/mirror_kinematics.py", "tools/mirror_flip_test.py", "tools/track_cleaning.py",
    "tools/glass_mask.py", "tools/reflection_audit.py", "tools/reflection_depth_test.py",
    "tools/depth_gate_test.py", "tools/crosscam_mirror_rule.py", "tools/build_cleanset_v2.py",
    # ReID / 嵌入 / 缓存 / 服务
    "tools/crosscam_reid.py", "tools/embed_fast.py", "tools/encode_tracks.py",
    "tools/global_reid_store.py", "tools/artifact_io.py", "tools/store_admin.py",
    "tools/combo_retrieval.py", "tools/embed_server.py",
    "tools/reid_embgemma_eval.py", "tools/mrl_validate.py", "tools/mrl_retrieval_check.py",
    "tools/cradio_scale_remote.py", "tools/fusion_dinogemma.py",
    # 跨机位接力 / 判定
    "tools/crosscam_spatiotemporal.py", "tools/crosscam_relay.py", "tools/crosscam_transfer.py",
    "tools/crosscam_pipeline.py", "tools/crosscam_depth.py", "tools/crosscam_appearance.py",
    "tools/crosscam_reid_compare.py", "tools/relay_triage.py",
    # 拼接 / 拓扑 / 评估
    "tools/track_stitching.py", "tools/topology_signature.py", "tools/c0_baseline.py",
    "tools/replay_eval.py",
    # 数据链（clean2 闭环与审计）
    "tools/det_view.py", "tools/rebuild_clean2_entities.py", "tools/audit_data_chain.py",
    # 身份选择器 / 意图预测
    "tools/selector.py", "tools/intent_model.py", "tools/intent_dataset.py",
]

DATA_FILES = (
    [f"output_crosscam/cleanset_v2/{c}_detections.jsonl" for c in
     ["305", "316", "318", "320", "323", "343"]] +
    ["output_crosscam/cleanset_v2/clean_manifest_v2.json",
     "output_crosscam/cleanset/clean_manifest.json",
     "output_crosscam/analysis/splits_v5.json",
     "output_crosscam/analysis/splits_v4.json",
     "output_crosscam/analysis/splits_v3.json",
     "output_crosscam/analysis/topology/topology_signature.json",
     "output_crosscam/analysis/c0/c0_baseline_v5.json",
     "output_crosscam/analysis/c0/c0_baseline_v4.json",
     "output_crosscam/analysis/c0/c0_baseline_v3.json",
     "output_crosscam/analysis/audit/data_chain_audit.json",
     "output_crosscam/analysis/multicam_overview.json",
     "output_crosscam/analysis/relay_clean2/relay_chains_cradio_v5.json",
     "output_crosscam/analysis/relay/relay_triage_cradio_v3.json",
     "output_crosscam/analysis/relay/relay_chains_cradio_v4.json",
     "output_crosscam/analysis/candidates_review.csv",
     "output_crosscam/analysis/confirmed_pairs.csv",
     "output_crosscam/analysis/DATA_DISCIPLINE.md",
     "output_crosscam/analysis/transfer_report.md",
     "docs/clean2-rebuild-report.md",
     ] +
    # 规范实体图（clean2）；旧无掩膜口径 stitch/ 仅保留 labels，不入包以免误导
    [f"output_crosscam/analysis/stitch_clean2/net_tracks_{c}.json" for c in
     ["305", "316", "318", "320", "323", "343"]] +
    ["output_crosscam/analysis/stitch_clean2/manifest.json",
     "output_crosscam/analysis/stitch/labels_v2.json"]
)

RESULT_FILES = [
    "output_crosscam/analysis/topology/topology_signature.png",
    "output_crosscam/analysis/c0/c0_baseline_v5.png",
    "output_crosscam/analysis/c0/fig_clean2_version_compare.png",
    "output_crosscam/analysis/c0/c0_baseline_v3.png",
    "output_crosscam/analysis/fig_multicam_timeline.png",
    "output_crosscam/analysis/fig_spatiotemporal.png",
    "output_crosscam/analysis/fig_reid_eval.png",
    "output_crosscam/analysis/fig_depth_summary.png",
    "output_crosscam/analysis/long35/long35_overview.png",
]


def scrub(text):
    for pat, rep in REDACT:
        text = pat.sub(rep, text)
    return text


def copy_file(rel, subdir, manifest, note=""):
    src = ROOT / rel
    if not src.exists():
        manifest.append((subdir, os.path.basename(rel), "—", "MISSING", rel))
        return False
    dst = STAGE / subdir / os.path.basename(rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in TEXT_EXT:
        try:
            raw = src.read_text(encoding="utf-8", errors="replace")
            dst.write_text(scrub(raw), encoding="utf-8")
        except Exception as e:
            shutil.copy2(src, dst)
            print(f"  [warn] 原样复制（读取失败 {e}）: {rel}")
    else:
        shutil.copy2(src, dst)
    size = dst.stat().st_size
    manifest.append((subdir, os.path.basename(rel), size, "ok", rel))
    return True


def main():
    import sys
    zip_only = "--zip-only" in sys.argv
    rescrub = "--rescrub" in sys.argv
    if rescrub:
        # 就地重脱敏：保留既有结构（含手写 HANDOFF/SCENE/REFERENCES），仅对文本再跑一遍规则
        n = 0
        for p in sorted(STAGE.rglob("*")):
            if p.is_file() and p.name != "MANIFEST.md" and p.suffix.lower() in TEXT_EXT:
                raw = p.read_text(encoding="utf-8", errors="replace")
                new = scrub(raw)
                if new != raw:
                    p.write_text(new, encoding="utf-8")
                    n += 1
        print(f"[rescrub] 就地重脱敏 {n} 个文件")
    if STAGE.exists() and not (zip_only or rescrub):
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True, exist_ok=True)
    manifest = []

    if zip_only or rescrub:
        # 只重建清单 + 重打包（保留手动编写的 HANDOFF/SCENE/REFERENCES）
        for p in sorted(STAGE.rglob("*")):
            if p.is_file() and p.name != "MANIFEST.md":
                manifest.append((str(p.relative_to(STAGE).parent), p.name,
                                 p.stat().st_size, "ok", ""))
    else:
        # docs（全部研究报告）
        for p in sorted((ROOT / "docs").glob("*.md")):
            copy_file(str(p.relative_to(ROOT)), "docs", manifest)

        # code
        for rel in CODE_FILES:
            copy_file(rel, "code", manifest)

        # data
        for rel in DATA_FILES:
            copy_file(rel, "data", manifest)

        # results
        for rel in RESULT_FILES:
            copy_file(rel, "results", manifest)

    # ---- 写 MANIFEST ----
    lines = ["# 交接包内容清单（自动生成）", "",
             f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
             "| 目录 | 文件 | 大小 | 状态 |", "|---|---|---|---|"]
    total = 0
    for sub, name, size, st, _ in sorted(manifest, key=lambda x: (x[0], x[1])):
        s = f"{size/1024:.1f} KB" if isinstance(size, int) else "—"
        if isinstance(size, int):
            total += size
        lines.append(f"| {sub} | {name} | {s} | {st} |")
    lines += ["", f"**合计 {len(manifest)} 个文件，约 {total/1024/1024:.1f} MB**", ""]
    (STAGE / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- 打包 ----
    zip_path = ROOT / "handoff_out" / "human-count-research-handoff.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(STAGE.parent))
    print(f"[zip] {zip_path}  ({zip_path.stat().st_size/1024/1024:.2f} MB)")
    missing = [m for m in manifest if m[3] != "ok"]
    if missing:
        print(f"[warn] 缺失 {len(missing)} 个:")
        for m in missing:
            print("   ", m[4])
    print(f"[ok] {len(manifest)-len(missing)}/{len(manifest)} 文件已打包")


if __name__ == "__main__":
    main()
