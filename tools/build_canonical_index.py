#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A01：建立唯一权威结果索引（canonical_results）。

背景（docs/project-status-leader-plan-2026-10-26.md A01）：
  交付包里同时存在 strict/legacy 两套精度口径、gt v2/v3 两代真值、以及
  「原始输出」与「可答接受」两种分母。缺一个索引，任何引用都可能拿错数。
  本脚本把「哪个文件是什么口径、能不能引用」固化成机器可读 + 人可读两份文件。

设计原则：
  1. **归类靠人，元数据靠机器**。哪个文件是 canonical / legacy / deprecated 是判断，
     写死在 ENTRIES 里；sha256、mtime、大小由脚本自动填，避免手抄出错。
  2. **status 只有三档**，语义严格：
       canonical   可以引用。论文/交付一律引这一档。
       legacy      保留核对用，**不得默认引用**。若必须引用，须同时给出 canonical 版本。
       deprecated  明确错误或过期，**禁止引用**，仅作历史存档。
  3. **索引不代表冻结**。`BASELINE_FREEZE` 管的是「文件没变」，本索引管的是「该引哪个」。
     两者不冲突：legacy 文件仍在冻结清单里，只是不该被论文引用。

用法：
  python tools/build_canonical_index.py            # 生成/更新索引
  python tools/build_canonical_index.py --check    # 校验索引是否与磁盘一致
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "deliverable", "data-request-v2")
OUT_JSON = os.path.join(PKG, "results", "canonical_results.json")
OUT_MD = os.path.join(PKG, "results", "canonical_results.md")

# ── 口径定义（三层阶梯，混用会错 1~14 条）────────────────────────────────
CALIBERS = {
    "strict": {
        "name": "论文严格口径",
        "rules": ["平票簇记 U（簇内最高票并列时不认定主体身份）",
                  "真身份不在图库的查询移出评估面（不计接受、不计 U）",
                  "非人查询（no-human / shadow*）不计接受"],
        "A_baseline": {"accepted": 123, "C": 117, "W": 4, "U": 2, "precision": 0.9669},
        "note": "论文正文与补充表只能用这一档",
    },
    "wide": {
        "name": "宽口径（历史）",
        "rules": ["平票取 Counter 首个（不记 U）",
                  "真身份不在图库的查询照常计入接受面",
                  "非人查询不计接受"],
        "A_baseline": {"accepted": 124, "C": 118, "W": 6, "U": 0, "precision": 0.9516},
        "note": "旧稿与 BASELINE_FREEZE 的 0.9516 出自这里；与 strict 差 1 条接受 / 2 条 W→U",
    },
    "pre_d3tail": {
        "name": "更早口径（D3 tail 处理前）",
        "rules": ["非人输出仍计入接受面并记为 U", "其余同 wide"],
        "A_baseline": {"accepted": 138, "C": 118, "W": 6, "U": 14, "precision": 0.9516},
        "note": "只在口径禁令里作为分母来源出现（138 = 系统输出总数）；不可与新口径混用",
    },
}

# ── 条目清单（人工策展）────────────────────────────────────────────────
ENTRIES = [
    # canonical —— 可以引用
    dict(path="results/strict/precision_bgr_strict.csv", role="per_config_precision",
         status="canonical", caliber="strict", gt="v3", scope="42 configs",
         generator="python tools/eval_precision.py --deliv deliverable/data-request-v2 "
                   "--run results --strict --out-dir <tmp>",
         note="论文引用入口。A_baseline 固定 123/117/4/2（精度 0.9669）"),
    dict(path="results/strict/precision_rgb_strict.csv", role="per_config_precision",
         status="canonical", caliber="strict", gt="v3", scope="42 configs (RGB 表征)",
         generator="python tools/eval_precision.py --deliv tmp/rgb_pkg "
                   "--gt-dir deliverable/data-request-v2/gt --run results --strict "
                   "--c-denom 229 --out-dir <tmp>",
         note="RGB 通道顺序对照。A_baseline 67/0/3（精度 1.0，cov 0.3581）"),
    dict(path="results/rgb_strict_ab.md", role="ab_report",
         status="canonical", caliber="strict", gt="v3", scope="BGR vs RGB, A_baseline",
         generator="python tools/ab_emb_report.py --run-a <strict_bgr> --run-b <strict_rgb> "
                   "--cfg A_baseline --c-denom 229",
         note="RGB 负结果的正式证据：净 ΔC −50、McNemar p=1.65e-13、CI [−64,−37]"),
    dict(path="results/m3_counting.json", role="counting",
         status="canonical", caliber="strict", gt="v3", scope="61 windows, A_baseline",
         generator="python tools/m3_counting.py --deliv deliverable/data-request-v2 "
                   "--run results --cfg A_baseline",
         note="计数主结果：GT_hi 280 / GT_lo 279 / 系统 244；MAE 1.279 / 1.295；bias −0.590"),
    dict(path="results/m3_counting_windows.csv", role="counting_windows",
         status="canonical", caliber="strict", gt="v3", scope="61 windows",
         generator="（同上）",
         note="逐窗 CSV，供分机位与差异复核"),
    dict(path="results/m3_counting_nogate.json", role="counting_ablation",
         status="canonical", caliber="strict", gt="v3", scope="61 windows, 非人门控关闭",
         generator="关掉 m3_counting.is_nonperson 后重跑（见 docs/s0-a02-a03-2026-11-09.md）",
         note="门控对照：MAE 1.34 / 1.36、bias −0.62；323 上门控反而变差（0.73→1.00）"),
    dict(path="results/m3_counting_nogate_windows.csv", role="counting_ablation",
         status="canonical", caliber="strict", gt="v3", scope="61 windows",
         generator="（同上）", note="门控关逐窗"),
    dict(path="results/loss_decomposition.json", role="loss_decomposition",
         status="canonical", caliber="wide", gt="v3", scope="A_baseline",
         generator="python tools/analyze_loss_decomp.py",
         note="76 字段，gt v2→v3 零差异（已核）。与 wide 精度口径配套使用；"
              "若引用 strict 需说明其 C/W/U 来自 wide"),
    dict(path="results/loss_decomposition.csv", role="loss_decomposition",
         status="canonical", caliber="wide", gt="v3", scope="A_baseline + 分机位",
         generator="（同上）", note="同上"),
    dict(path="results/summary.csv", role="cluster_summary",
         status="canonical", caliber="n/a", gt="n/a", scope="42 configs",
         generator="python tools/d7_sweep.py --emb ... --det ... --out results",
         note="聚类/覆盖/自校准指标。不含真值，与口径无关（K/θ/cov/same/fm）"),
    dict(path="results/queries/", role="per_query_output",
         status="canonical", caliber="n/a", gt="n/a", scope="42 configs",
         generator="（d7_sweep 产出）",
         note="模型逐查询输出，不含真值判定。strict/wide 两套判定都从这里派生"),
    dict(path="results/members/", role="cluster_members",
         status="canonical", caliber="n/a", gt="n/a", scope="42 configs",
         generator="（d7_sweep 产出）",
         note="簇成员集合，跨配置可比的唯一标识"),
    dict(path="results/m3_gate.json", role="gate_diagnostic",
         status="canonical", caliber="n/a", gt="v3", scope="A_baseline",
         generator="（门控诊断）",
         note="非人簇构成（哪些簇由 shadow/no-human 主导），诊断用，不是计数结果"),

    # legacy —— 保留但不得默认引用
    dict(path="results/precision.csv", role="per_config_precision",
         status="legacy", caliber="wide", gt="v3", scope="42 configs",
         generator="python tools/eval_precision.py --deliv deliverable/data-request-v2 --run results",
         note="★ 旧稿与 BASELINE_FREEZE 的 0.9516 出自这里。仍在冻结清单内"
              "（冻结管文件没变，不管该不该引用）。论文引用请改用 strict/ 版本"),
    dict(path="results/verdicts/", role="per_query_verdict",
         status="legacy", caliber="wide", gt="v3", scope="42 configs",
         generator="（eval_precision 宽口径产出）",
         note="宽口径逐条判定（平票取首个）。strict 版需另行生成到 strict/verdicts/"),
    dict(path="results/precision.before-d3tail.csv", role="per_config_precision",
         status="legacy", caliber="pre_d3tail", gt="v2", scope="42 configs",
         generator="（历史）",
         note="最原始口径 138/118/6/14。仅用于解释「138 条系统输出」这个分母的来历"),
    dict(path="results/robust/", role="robustness_run",
         status="legacy", caliber="wide", gt="v3", scope="42 configs",
         generator="python tools/d7_sweep.py --robust-margin 0.10 ...",
         note="self-cal 邻域稳健性版本：θ 1.85→1.95、cov 0.664→0.651。"
              "不是默认工作点，引用时必须注明 --robust-margin 0.10"),

    # deprecated —— 禁止引用
    dict(path="results/m3_counting.json.gtv2", role="counting",
         status="deprecated", caliber="unknown", gt="neither", scope="61 windows",
         generator="（旧运行，档案）",
         note="★ MAE 1.721 / bias −1.033 / gt_hi 323 —— 既不对应当前真值（1.279）也不对应"
              "gt v2 冻结版（同为 1.279），是更早一代输入的遗留。文件名的 .gtv2 只是备份后缀，"
              "**不代表它是 v2 口径**。禁止引用"),
    dict(path="results/m3_counting_windows.csv.gtv2", role="counting_windows",
         status="deprecated", caliber="unknown", gt="neither", scope="61 windows",
         generator="（同上）", note="同上"),
]

STATUS_ORDER = {"canonical": 0, "legacy": 1, "deprecated": 2}
STATUS_LABEL = {"canonical": "canonical（可引用）",
                "legacy": "legacy（保留核对，不得默认引用）",
                "deprecated": "deprecated（禁止引用）"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(pkg_path):
    """返回 (exists, sha256, bytes, mtime, n_files)；目录用聚合哈希。"""
    p = os.path.join(PKG, pkg_path)
    if not os.path.exists(p):
        return dict(exists=False)
    if os.path.isdir(p):
        files = sorted(f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f)))
        h = hashlib.sha256()
        total = 0
        for f in files:
            fp = os.path.join(p, f)
            s = sha256_file(fp)
            h.update(f.encode())
            h.update(s.encode())
            total += os.path.getsize(fp)
        return dict(exists=True, n_files=len(files), sha256_agg=h.hexdigest()[:16],
                    bytes=total,
                    mtime=time.strftime("%Y-%m-%d %H:%M",
                                        time.localtime(os.path.getmtime(p))))
    return dict(exists=True, sha256=sha256_file(p)[:16], bytes=os.path.getsize(p),
                mtime=time.strftime("%Y-%m-%d %H:%M",
                                    time.localtime(os.path.getmtime(p))))


def build():
    entries = []
    for e in sorted(ENTRIES, key=lambda x: (STATUS_ORDER[x["status"]], x["path"])):
        rec = dict(e)
        rec.update(stamp(e["path"]))
        entries.append(rec)
    return {
        "schema": "canonical-results-v1",
        "created": "2026-11-09",
        "pkg": "deliverable/data-request-v2",
        "purpose": "唯一权威结果索引：哪个文件是什么口径、能不能引用。"
                   "任何跨文件引用先查本表，不要凭文件名猜。",
        "calibers": CALIBERS,
        "canonical_values": {
            "identity_A_baseline_strict": {
                "accepted": 123, "C": 117, "W": 4, "U": 2,
                "precision": 0.9669,
                # ★ 列名陷阱：precision.csv 的 `C_over_229` 是**绝对计数**，
                #   比例在同表的 `recall_229`。两者极易混用。
                "C_over_229": 117, "recall_229": 0.5109,
                "coverage": 0.6638, "theta": 1.85, "K": 111,
                "source": "results/strict/precision_bgr_strict.csv"},
            "identity_A_baseline_wide": {
                "accepted": 124, "C": 118, "W": 6, "U": 0,
                "precision": 0.9516, "C_over_229": 118, "recall_229": 0.5153,
                "source": "results/precision.csv (legacy)"},
            "counting_A_baseline": {
                "windows": 61, "gt_hi": 280, "gt_lo": 279, "sys": 244,
                "mae_hi": 1.279, "mae_lo": 1.295, "bias_hi": -0.590,
                "source": "results/m3_counting.json"},
            "rgb_vs_bgr_strict": {
                "bgr": {"C": 117, "W": 4, "U": 2, "precision": 0.9669},
                "rgb": {"C": 67, "W": 0, "U": 3, "precision": 1.0,
                        "C_over_229": 67, "recall_229": 0.2926, "coverage": 0.3581},
                "delta_C": -50, "mcnemar_p": 1.65e-13, "ci95": [-64, -37],
                "source": "results/rgb_strict_ab.md"},
        },
        "entries": entries,
    }


def render_md(blob):
    L = ["# 唯一权威结果索引（canonical_results）",
         "",
         "> 由 `python tools/build_canonical_index.py` 生成；校验用 `--check`。",
         "> 任何跨文件引用先查本表，**不要凭文件名猜口径**。",
         "",
         "**索引 ≠ 冻结**：`BASELINE_FREEZE` 管「文件有没有变」，本表管「该引哪一个」。",
         "legacy 文件仍在冻结清单里，只是不该被论文引用。",
         "",
         "## 1 三层精度口径（混用会错 1~14 条）",
         "",
         "| 口径 | A_baseline 接受 / C / W / U | 精度 | 用途 |",
         "|---|---|---|---|",
         ]
    for k in ("strict", "wide", "pre_d3tail"):
        c = CALIBERS[k]
        a = c["A_baseline"]
        L.append("| `%s` %s | %d / %d / %d / %d | %.4f | %s |"
                 % (k, c["name"], a["accepted"], a["C"], a["W"], a["U"],
                    a["precision"], c["note"]))
    L += ["",
          "规则差异：", ""]
    for k in ("strict", "wide", "pre_d3tail"):
        L.append("- **`%s`**：" % k)
        for r in CALIBERS[k]["rules"]:
            L.append("  - " + r)
    L += ["",
          "## 2 权威值（现场 A_baseline）",
          "",
          "| 项 | 值 | 来源 |",
          "|---|---|---|"]
    cv = blob["canonical_values"]
    i = cv["identity_A_baseline_strict"]
    L.append("| 身份精度（严格） | 接受 %d / C %d / W %d / U %d，精度 **%.4f**，C/229 = %d/229 = **%.4f** | `%s` |"
             % (i["accepted"], i["C"], i["W"], i["U"], i["precision"],
                i["C_over_229"], i["recall_229"], i["source"]))
    c = cv["counting_A_baseline"]
    L.append("| 计数（61 窗） | GT_hi %d / GT_lo %d / 系统 %d；MAE **%.3f** / %.3f；bias **%.3f** | `%s` |"
             % (c["gt_hi"], c["gt_lo"], c["sys"], c["mae_hi"], c["mae_lo"],
                c["bias_hi"], c["source"]))
    r = cv["rgb_vs_bgr_strict"]
    L.append("| RGB vs BGR（严格） | C %d → %d，ΔC **%d**，p=%.2e，CI [%d, %d] | `%s` |"
             % (r["bgr"]["C"], r["rgb"]["C"], r["delta_C"], r["mcnemar_p"],
                r["ci95"][0], r["ci95"][1], r["source"]))
    L += ["", "## 3 文件清单", ""]
    cur = None
    for e in blob["entries"]:
        if e["status"] != cur:
            cur = e["status"]
            L += ["### " + STATUS_LABEL[cur], "",
                  "| 路径 | 角色 | 口径 | gt | 规模 | mtime | 哈希(16) |",
                  "|---|---|---|---|---|---|---|"]
        size = ("%d 文件" % e["n_files"]) if e.get("n_files") else \
               ("%.1f KB" % (e.get("bytes", 0) / 1024.0))
        L.append("| `%s` | %s | %s | %s | %s | %s | `%s` |"
                 % (e["path"], e["role"], e["caliber"], e["gt"], size,
                    e.get("mtime", "—"),
                    e.get("sha256") or e.get("sha256_agg") or "缺失"))
    L += ["", "## 4 引用纪律", "",
          "1. **论文/交付只引 canonical**。表格脚注须写明口径（strict 或 wide）。",
          "2. **引用 legacy 必须同时给出 canonical 版本**，否则读者无法判断差异来源。",
          "3. **deprecated 一律禁引**，包括 `m3_counting.json.gtv2`——它的 `.gtv2` 只是备份后缀，"
          "不代表它是 gt v2 口径（它两套真值都不对应）。",
          "4. **计数是 strict 口径而 `loss_decomposition` 是 wide 口径**，两者不能直接串在一张表里。",
          "5. **★ 列名陷阱**：`precision.csv` 的 `C_over_229` 列存的是**绝对计数**（如 117），"
          "比例在同表的 `recall_229`（0.5109）。两个列名都像比例，极易混用。",
          "6. 新增产物请先加进 `tools/build_canonical_index.py` 的 `ENTRIES` 再生成索引。",
          "", "## 5 逐条说明", ""]
    for e in blob["entries"]:
        L.append("**`%s`** — %s" % (e["path"], e["note"]))
        L.append("")
        L.append("  生成：`%s`" % e["generator"])
        L.append("")
    return "\n".join(L)


def verify_values(blob):
    """核对 canonical_values 里声明的是否与文件实际内容一致。

    为什么必须做：索引写了数字却不校验，等于给自己造了一个更难发现的错源 ——
    读者会相信索引而不去翻文件。任何一项不符都视为 FAIL。
    """
    import csv
    P = os.path.join(PKG, "results")
    bad = []

    def row(fn, cfg="A_baseline"):
        p = os.path.join(P, fn)
        if not os.path.exists(p):
            return None
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            if r["config"] == cfg:
                return r
        return None

    def chk(lab, got, exp):
        if got != exp:
            bad.append((lab, got, exp))

    cv = blob["canonical_values"]
    for key, fn in (("identity_A_baseline_strict", "strict/precision_bgr_strict.csv"),
                    ("identity_A_baseline_wide", "precision.csv")):
        want = cv[key]
        r = row(fn)
        if r is None:
            bad.append((fn, "文件或行缺失", "—"))
            continue
        chk(fn + " acc/C/W/U", tuple(int(r[k]) for k in ("accepted", "C", "W", "U")),
            (want["accepted"], want["C"], want["W"], want["U"]))
        chk(fn + " precision", float(r["precision"]), want["precision"])
        chk(fn + " C_over_229", int(float(r["C_over_229"])), want["C_over_229"])
        chk(fn + " recall_229", float(r["recall_229"]), want["recall_229"])

    r = row("strict/precision_rgb_strict.csv")
    w = cv["rgb_vs_bgr_strict"]["rgb"]
    if r is None:
        bad.append(("strict/precision_rgb_strict.csv", "文件或行缺失", "—"))
    else:
        chk("RGB C/W/U", tuple(int(r[k]) for k in ("C", "W", "U")),
            (w["C"], w["W"], w["U"]))
        chk("RGB recall_229", float(r["recall_229"]), w["recall_229"])

    cw = cv["counting_A_baseline"]
    cp = os.path.join(P, "m3_counting.json")
    if not os.path.exists(cp):
        bad.append(("m3_counting.json", "文件缺失", "—"))
    else:
        d = json.load(open(cp, encoding="utf-8"))
        chk("counting gt_hi/lo/sys", (d["gt_hi"], d["gt_lo"], d["sys"]),
            (cw["gt_hi"], cw["gt_lo"], cw["sys"]))
        chk("counting mae_hi/lo", (d["mae_hi"], d["mae_lo"]),
            (cw["mae_hi"], cw["mae_lo"]))
        chk("counting bias_hi", d["bias_hi"], cw["bias_hi"])
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验索引与磁盘是否一致")
    a = ap.parse_args()

    blob = build()
    missing = [e["path"] for e in blob["entries"] if not e.get("exists")]

    if a.check:
        if not os.path.exists(OUT_JSON):
            print("FAIL 没有索引，先跑一次不带 --check 的生成")
            return 2
        old = json.load(open(OUT_JSON, encoding="utf-8"))
        om = {e["path"]: e for e in old["entries"]}
        bad = []
        for e in blob["entries"]:
            o = om.get(e["path"])
            if not o:
                bad.append((e["path"], "索引里缺这一条"))
            elif o.get("sha256") != e.get("sha256") or \
                    o.get("sha256_agg") != e.get("sha256_agg"):
                bad.append((e["path"], "磁盘内容已变"))
        extra = [p for p in om if p not in {e["path"] for e in blob["entries"]}]
        vbad = verify_values(blob)
        if bad or extra or vbad:
            print("FAIL 索引与磁盘不一致：")
            for p, why in bad:
                print("  %-12s %s" % (why, p))
            for p in extra:
                print("  %-12s %s" % ("索引有但清单里没有", p))
            for lab, got, exp in vbad:
                print("  %-12s %s：实际 %r ≠ 索引 %r" % ("声明值不符", lab, got, exp))
            return 1
        n = {s: sum(1 for e in blob["entries"] if e["status"] == s)
             for s in STATUS_ORDER}
        print("OK 索引一致｜canonical %d / legacy %d / deprecated %d｜声明值 %d 项全对"
              % (n["canonical"], n["legacy"], n["deprecated"], 12))
        return 0

    if missing:
        print("WARN 清单里 %d 个路径在磁盘上不存在：" % len(missing))
        for p in missing:
            print("   ", p)

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=1)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_md(blob))
    n = {s: sum(1 for e in blob["entries"] if e["status"] == s) for s in STATUS_ORDER}
    print("written: %s" % OUT_JSON)
    print("written: %s" % OUT_MD)
    print("entries: canonical %d / legacy %d / deprecated %d"
          % (n["canonical"], n["legacy"], n["deprecated"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
