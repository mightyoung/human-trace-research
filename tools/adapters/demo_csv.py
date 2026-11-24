#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""demo_csv.py — 第二个适配器，用来证明「内核不依赖论文」。

场景：一份虚构的推荐系统离线评估报告（Markdown 表格）+ 原始指标 CSV。
流程与 paper_tbiom 完全相同：discover → claim → claim_anchor → check → eval_run。
**不碰 tex、不碰 EVIDENCE.md、不碰任何 ReID 名词。**

跑法：
  python tools/adapters/demo_csv.py          # 写 tmp/demo-ledger.db，不碰生产库

判定：4 条 claim 全部 match，即为通用性通过。若这里需要改 claim/claim_anchor/eval_run
的字段才能跑，说明内核仍带着论文假设，要回去修内核，不给 demo 打补丁。
"""
import csv
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA = os.path.join(ROOT, "ops", "schema.sql")
DB = os.path.join(ROOT, "tmp", "demo-ledger.db")
REPORT = "examples/demo-csv/report.md"
CSV_PATH = "examples/demo-csv/metrics.csv"
STUDY = "st_demo-recsys"

# 报告里的表格列名 → CSV 列名。适配器自己知道这个映射，内核不知道。
COLS = {"Recall@10": "recall_at_10", "NDCG@10": "ndcg_at_10"}


def now():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def sid(prefix, *parts):
    raw = "|".join(str(p) for p in parts)
    return prefix + "_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def sha256_file(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.isfile(p):
        return None
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---------------------------------------------------------------- adapter 三接口

def discover():
    """从 Markdown 表格里抽数值。返回 claim 草案。"""
    out = []
    lines = open(os.path.join(ROOT, REPORT), encoding="utf-8").read().split("\n")
    header = None
    for i, ln in enumerate(lines):
        if not ln.strip().startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        for j, v in enumerate(cells[1:], start=1):
            if not v.replace(".", "", 1).replace("-", "", 1).isdigit():
                continue
            out.append({"locator": "%s-%s" % (header[j], cells[0]),
                        "statement": "%s 的 %s" % (cells[0], header[j]),
                        "value": v, "method": cells[0], "column": header[j],
                        "row": i})
    return out


def anchor(claim):
    return [{"type": "md_heading",
             "ref_json": {"file": REPORT, "heading": "## 1. 离线指标", "row": claim["row"]}}]


def resolve(anc):
    """返回锚点所在行号（用于定位；本 demo 只用它记哈希，不解析数字）。"""
    lines = open(os.path.join(ROOT, anc["ref_json"]["file"]), encoding="utf-8").read().split("\n")
    want = anc["ref_json"]["heading"]
    for i, ln in enumerate(lines):
        if ln.strip() == want:
            return i + 1, lines
    return None, lines


# ---------------------------------------------------------------- 通用算子（K1 移入内核）

def table_cell(rel, row_key, column, key_field="method"):
    with open(os.path.join(ROOT, rel), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r[key_field] == row_key:
                return r[column]
    raise KeyError("%s 里没有 %s=%s" % (rel, key_field, row_key))


# ---------------------------------------------------------------- 跑

def main():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    if os.path.exists(DB):
        os.remove(DB)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    with open(SCHEMA, encoding="utf-8") as f:
        conn.executescript(f.read())

    conn.execute("INSERT INTO study(id,name,question,owner,status,started_at,note,created_at) "
                 "VALUES(?,?,?,?,?,?,?,?)",
                 (STUDY, "示例：两路召回融合", "候选方案是否在 Recall@10 上优于基线，代价是多少覆盖率？",
                  "demo", "active", "2026-11-24", "通用性冒烟测试，与生产库隔离", now()))

    drafts = discover()
    print("[demo] discover: %d 条" % len(drafts))
    ok = 0
    for d in drafts:
        cid = sid("cl", STUDY, d["locator"], d["value"])
        conn.execute("INSERT INTO claim(id,study_id,card_id,locator,kind,statement,value_text,unit,"
                     "stratum,evidence_grade,verdict,provenance_json,ext_json,created_at,updated_at) "
                     "VALUES(?,?,NULL,?,?,?,?,?,?,?, 'pending', ?, ?, ?, ?)",
                     (cid, STUDY, d["locator"], "metric", d["statement"], d["value"], None,
                      "offline", "data", json.dumps({"adapter": "demo_csv"}), 
                      json.dumps({"method": d["method"], "column": d["column"]}, ensure_ascii=False),
                      now(), now()))
        for a in anchor(d):
            ref = json.dumps(a["ref_json"], ensure_ascii=False, sort_keys=True)
            conn.execute("INSERT INTO claim_anchor(id,claim_id,type,ref_json,resolver,resolved_sha1,created_at) "
                         "VALUES(?,?,?,?,?,?,?)",
                         (sid("ca", cid, a["type"], ref), cid, a["type"], ref, "demo_csv",
                          hashlib.sha1(open(os.path.join(ROOT, a["ref_json"]["file"]), "rb")
                                       .read()).hexdigest()[:12], now()))
        # check：从 CSV 复算
        got = table_cell(CSV_PATH, d["method"], COLS[d["column"]])
        verdict = "match" if got == d["value"] else "mismatch"
        ok += verdict == "match"
        er = sid("er", cid, CSV_PATH)
        conn.execute("INSERT INTO eval_run(id,claim_ids_json,script,script_sha1,as_is,args_json,cwd,"
                     "started_at,ended_at,exit_code,code_version_id,inputs_json,outputs_json,env_json,created_at) "
                     "VALUES(?,?,?,?,1,?,?,?,?,?,NULL,?,?,?,?)",
                     (er, json.dumps([cid]), "demo_csv.table_cell",
                      hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12],
                      json.dumps({"row": d["method"], "col": COLS[d["column"]]}),
                      ROOT, now(), now(), 0 if verdict == "match" else 1,
                      json.dumps([{"path": CSV_PATH, "sha256": sha256_file(CSV_PATH)}]),
                      json.dumps({"recomputed": got, "run": True}), json.dumps({"python": sys.version.split()[0]}),
                      now()))
        conn.execute("UPDATE claim SET verdict=?, recomputed_text=?, eval_run_id=?, reason=?, updated_at=? "
                     "WHERE id=?",
                     (verdict, got, er, "demo: CSV %s[%s]" % (d["method"], COLS[d["column"]]), now(), cid))
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM claim").fetchone()[0]
    by = dict(conn.execute("SELECT verdict,COUNT(*) FROM claim GROUP BY verdict").fetchall())
    anchors = conn.execute("SELECT COUNT(*) FROM claim_anchor").fetchone()[0]
    runs = conn.execute("SELECT COUNT(*) FROM eval_run WHERE exit_code=0").fetchone()[0]
    print("[demo] claim=%d anchor=%d run(exit0)=%d verdict=%s" % (total, anchors, runs,
                                                                  json.dumps(by, ensure_ascii=False)))
    print("[demo] %s" % ("PASS：内核无需为第二场景改动" if ok == total else "FAIL：%d/%d 条不一致" % (ok, total)))
    conn.close()
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
