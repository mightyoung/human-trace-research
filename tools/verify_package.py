#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交付前一致性检查：路径引用、数字一致、图片对应、内部链接。

退出码 0 = 全部通过；1 = 有问题（打印清单）。
"""
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "deliverable", "data-request-v2")

# 包内应有目录；tools/ 与 docs/ 属仓库，不在包内
IN_PKG = ("ledger/", "gt/", "results/")
REPO_ONLY = ("tools/", "docs/", "configs/", "output_crosscam", "media-resource", "tmp/")

issues = []
notes = []


def check_refs():
    """扫描所有 .md 里反引号包裹的路径式引用。"""
    md = [f for f in os.listdir(PKG) if f.endswith(".md")]
    for sub in ("results", "gt", "ledger"):
        d = os.path.join(PKG, sub)
        if os.path.isdir(d):
            md += [os.path.join(sub, f) for f in os.listdir(d) if f.endswith(".md")]
    seen = {}
    for f in md:
        p = os.path.join(PKG, f)
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8").read()
        for m in re.finditer(r"`([A-Za-z0-9_./<>\-]+\.[a-z]{2,4})`", txt):
            ref = m.group(1)
            # 跳过模式型写法（<seg>、<config>、...、通配）
            if "<" in ref or ">" in ref or "..." in ref or "*" in ref or ref.startswith("_"):
                continue
            if any(ref.startswith(x) for x in REPO_ONLY):
                continue
            # 已知的外部输入与旧产物（不在本包内）
            if ref in ("data-request-v2-2026-10-19.md", "report_reject_reasons.py",
                       "baseline_audit/index.csv", "cluster_purity.csv",
                       "profile.expected.thr", "crops_manifest_noimg.csv",
                       "data-request-v2_2026-10-19.zip",
                       "data-request-v2_2026-10-19_crops.zip",
                       "a3_query_gt_FILLED.csv"):
                continue
            # 包内不含任何 .py，脚本名一律视为仓库内路径
            if ref.endswith(".py"):
                continue
            # 包外的分发件（dist/ 下的 zip 等），只作引用说明
            if ref.startswith("dist/"):
                continue
            if not any(ref.startswith(x) for x in IN_PKG) and "/" not in ref:
                # 顶层的裸文件名，如 NUMBERS.md
                cand = os.path.join(PKG, ref)
            else:
                cand = os.path.join(PKG, ref.replace("\\", "/"))
            key = ref
            if os.path.exists(cand):
                seen[key] = "ok"
            else:
                # 可能是简写（如只写文件名）
                base = os.path.basename(ref)
                found = False
                for dp, dn, fn in os.walk(PKG):
                    if base in fn:
                        found = True
                        break
                if found:
                    seen[key] = "loose"
                else:
                    seen[key] = "missing"
                    issues.append(f"[引用缺失] {f}: `{ref}`")
    cnt = {}
    for v in seen.values():
        cnt[v] = cnt.get(v, 0) + 1
    notes.append(f"路径引用：ok {cnt.get('ok', 0)}｜仅文件名可匹配 {cnt.get('loose', 0)}"
                 f"｜缺失 {cnt.get('missing', 0)}")


def check_numbers():
    """NUMBERS.md 的关键数字 vs precision.csv。"""
    prec = os.path.join(PKG, "results", "precision.csv")
    if not os.path.exists(prec):
        issues.append("[缺失] results/precision.csv")
        return
    rows = {r["config"]: r for r in csv.DictReader(open(prec, encoding="utf-8-sig"))}
    base = rows.get("A_baseline")
    veto = rows.get("C_dmax0.30")
    if not base or not veto:
        issues.append("[缺失] precision.csv 里没有 A_baseline 或 C_dmax0.30")
        return
    nums = os.path.join(PKG, "NUMBERS.md")
    txt = open(nums, encoding="utf-8").read()

    checks = [
        ("baseline C", str(base["C"]), "118"),
        ("baseline W", str(base["W"]), "6"),
        ("baseline C/229", f"{float(base['recall_229']):.3f}", "0.515"),
        ("baseline 精度", f"{float(base['precision']):.3f}", "0.952"),
        ("veto C", str(veto["C"]), "90"),
        ("veto 精度", f"{float(veto['precision']):.3f}", "0.947"),
    ]
    for name, actual, expect in checks:
        if actual != expect:
            issues.append(f"[数字不符] {name}: CSV {actual} vs 文档预期 {expect}")
        elif expect not in txt:
            issues.append(f"[文档缺] NUMBERS.md 未出现 {name} = {expect}")
    notes.append(f"关键数字：{len(checks)} 项核对（CSV ↔ 文档预期值）")


def check_images():
    """模板引用的图片是否都在。"""
    total = miss = 0
    for fn in ("gallery_members_TEMPLATE.csv", "query_gt_TEMPLATE.csv",
               "annotator2_TEMPLATE.csv", "a3_gallery_TEMPLATE.csv",
               "a3_query_TEMPLATE.csv"):
        p = os.path.join(PKG, "gt", fn)
        if not os.path.exists(p):
            issues.append(f"[缺失] gt/{fn}")
            continue
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            total += 1
            jpg = r.get("prompt_jpg") or ""
            if jpg and not os.path.exists(os.path.join(PKG, "gt", jpg)):
                miss += 1
    notes.append(f"模板图片：{total} 条引用，缺 {miss} 张")
    if miss:
        issues.append(f"[图片缺失] {miss} 条模板行引用不到图片")


def check_ledger_rows():
    """关键台账的行数。"""
    expect = {
        "ledger/tracks.csv": 874,
        "ledger/clusters_geom.csv": 222,
        "ledger/cooccur_pairs.csv": 1120,
        "ledger/outputs/baseline.csv": 229,
        "ledger/outputs/veto0.30.csv": 229,
        "results/verdicts/A_baseline.csv": 138,
        "results/members/A_baseline.csv": 513,
        "gt/gt_gallery_members_FILLED.csv": 257,
        "gt/review_d3_tail_FILLED.csv": 20,
        "gt/gt_query_gt_FILLED.csv": 295,
        "gt/gt_annotator2_FILLED.csv": 243,
        "gt/review_d3_FILLED.csv": 25,
        "gt/review_kappa_FILLED.csv": 3,
        "gt/a3_gallery_members_FILLED.csv": 257,
        "results/kappa_a3_disagreements.csv": 18,
        "gt_new/318_test_534b_query_TEMPLATE.csv": 107,
        "gt_new/318_test_534b_query_VALID.csv": 80,
    }
    bad = 0
    for rel, n in expect.items():
        p = os.path.join(PKG, rel)
        if not os.path.exists(p):
            issues.append(f"[缺失] {rel}")
            bad += 1
            continue
        got = sum(1 for _ in open(p, encoding="utf-8-sig")) - 1
        if got != n:
            issues.append(f"[行数不符] {rel}: {got} vs 预期 {n}")
            bad += 1
    notes.append(f"台账行数：{len(expect) - bad}/{len(expect)} 符合")


def check_html():
    """关键 HTML 是否在，且没有未替换占位符。"""
    for rel in ("gt/annotate.html", "gt/annotate_standalone.html", "gt/annotate_a3.html",
                "gt/review_d3.html", "gt/review_d3_tail.html",
                "gt/review_kappa.html", "gt/review_female4_male7_male9.html",
                "gt/switch_review.html"):
        p = os.path.join(PKG, rel)
        if not os.path.exists(p):
            issues.append(f"[缺失] {rel}")
            continue
        s = open(p, encoding="utf-8", errors="ignore").read()
        for ph in ("__DATA__", "__IMGS__", "__IMG__", "__TARGETS__"):
            if ph in s:
                issues.append(f"[占位符残留] {rel} 含 {ph}")
    notes.append("HTML：8 个文件存在且无占位符残留")


def check_dupes():
    """包内是否有明显不该有的东西。"""
    if os.path.isdir(os.path.join(PKG, "gt", "crops")):
        notes.append("gt/crops 存在（打包脚本会排除，属正常）")
    for dp, dn, fn in os.walk(PKG):
        for f in fn:
            if f.endswith((".pyc", ".tmp", ".log", ".bak")):
                issues.append(f"[垃圾文件] {os.path.relpath(os.path.join(dp, f), PKG)}")


def main():
    check_refs()
    check_numbers()
    check_images()
    check_ledger_rows()
    check_html()
    check_dupes()

    print("=== 检查项 ===")
    for n in notes:
        print("  · " + n)
    print()
    if issues:
        print(f"=== 发现 {len(issues)} 个问题 ===")
        for i in issues:
            print("  ✗ " + i)
        sys.exit(1)
    print("=== 全部通过 ===")


if __name__ == "__main__":
    main()
