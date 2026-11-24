#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ledger_migrate.py — 把论文专用台账升级为通用流程层（K0）。

做四件事（幂等，可重复跑）：
  1. 建 study（本项目的唯一一行）与 deliverable（r2 包 / r3 稿 / 台账报告）；
  2. paper_claim（表）→ claim + claim_anchor，id 沿用 pc_* 不破坏已有引用；
  3. 老表改名 paper_claim_v1 备份，建视图 ops/views/paper_tbiom.sql 供老工具读；
  4. 输出守恒诊断：多少 claim 挂到了实验卡（C1）、多少有 run（C2）。

用法：
  python tools/ledger_migrate.py            # 只报告，不写库
  python tools/ledger_migrate.py --apply    # 写库（自动备份 ops/db_backup/）

设计：docs/workbench-flow-model-2026-11-24.md §6
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "ops", "ops.db")
SCHEMA = os.path.join(ROOT, "ops", "schema.sql")
VIEW = os.path.join(ROOT, "ops", "views", "paper_tbiom.sql")
BACKUP_DIR = os.path.join(ROOT, "ops", "db_backup")

STUDY_ID = "st_multicam-reid"

STUDY = {
    "id": STUDY_ID,
    "name": "多机位行人重识别 · 全程身份连续性",
    "question": "在真实办公场景的多机位录像里，能否在不重训表征的前提下，把同一个人的轨迹从头到尾接起来，"
                "并把「接得对不对」量化到可复核的程度？",
    "owner": "might",
    "status": "active",
    "note": "工作台的第一个实例；论文 T-BIOM 是该 study 的一个交付物（kind=paper），不是工作台的主语。",
}

# 交付物：路径 + 用于算 sha256 的主文件
DELIVERABLES = [
    {"name": "T-BIOM r2（投稿稿）", "kind": "paper", "version": "r2",
     "path": "docs/paper/tbiom-r2", "hash_file": "docs/paper/tbiom-r2/main.tex",
     "status": "submitted"},
    {"name": "T-BIOM r3（在改稿）", "kind": "paper", "version": "r3",
     "path": "docs/paper/tbiom-r3", "hash_file": "docs/paper/tbiom-r3/main.tex",
     "status": "draft"},
    {"name": "论文数字台账报告", "kind": "report", "version": "2026-11-24",
     "path": "ops/reports/paper_claims_v1", "hash_file": "ops/reports/paper_claims_v1/report.md",
     "status": "draft"},
]


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha1_file(path, n=12):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def sha256_file(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sid(prefix, *parts):
    raw = "|".join(str(p) for p in parts)
    return prefix + "_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def has_number(s):
    return bool(s) and bool(re.search(r"\d", s))


def table_kind(conn, name):
    row = conn.execute("SELECT type FROM sqlite_master WHERE name=?", (name,)).fetchone()
    return row[0] if row else None


def ensure_schema(conn):
    with open(SCHEMA, encoding="utf-8") as f:
        conn.executescript(f.read())


def ensure_view(conn):
    with open(VIEW, encoding="utf-8") as f:
        conn.executescript(f.read())


def plan(conn):
    """盘点要做什么，不写库。"""
    out = {"study": 0, "claims": 0, "anchors": 0, "deliverables": 0, "refs": 0,
           "legacy_table": False, "orphan": 0, "with_run": 0}
    t = table_kind(conn, "paper_claim")
    out["legacy_table"] = (t == "table")
    if t == "table":
        rows = conn.execute("SELECT * FROM paper_claim").fetchall()
        out["claims"] = len(rows)
        out["anchors"] = sum(len(json.loads(r["tex_anchor_json"])) for r in rows
                             if r["tex_anchor_json"] and r["tex_anchor_json"] not in ("[]", "null"))
        out["with_run"] = sum(1 for r in rows if r["eval_run_id"])
        out["orphan"] = len(rows)  # 老表没有 card_id 概念，全部按 orphan 起算
    else:
        out["claims"] = conn.execute("SELECT COUNT(*) FROM claim").fetchone()[0] if table_kind(conn, "claim") else 0
    out["deliverables"] = len(DELIVERABLES)
    return out


def migrate(conn):
    """执行迁移。调用方负责 commit。返回统计。"""
    t = table_kind(conn, "paper_claim")
    stats = {"claims": 0, "anchors": 0, "refs": 0, "renamed": False}

    # 1) study
    conn.execute(
        "INSERT INTO study(id,name,question,owner,status,started_at,note,created_at) "
        "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
        "question=excluded.question, note=excluded.note",
        (STUDY["id"], STUDY["name"], STUDY["question"], STUDY["owner"], STUDY["status"],
         "2026-10-10", STUDY["note"], now()))

    # 2) claim + claim_anchor（仅当老表还在）
    if t == "table":
        rows = conn.execute("SELECT * FROM paper_claim").fetchall()
        for r in rows:
            ext = {
                "paper_kind": r["kind"],
                "section": r["section"],
                "evidence_line": r["evidence_line"],
                "label": r["label"],
                "adapter": "paper_tbiom",
            }
            conn.execute(
                "INSERT INTO claim(id,study_id,card_id,locator,kind,statement,value_text,unit,"
                "stratum,evidence_grade,verdict,reason,recomputed_text,eval_run_id,"
                "provenance_json,ext_json,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "statement=excluded.statement, value_text=excluded.value_text, "
                "stratum=excluded.stratum, evidence_grade=excluded.evidence_grade, "
                "verdict=excluded.verdict, reason=excluded.reason, "
                "recomputed_text=excluded.recomputed_text, eval_run_id=excluded.eval_run_id, "
                "provenance_json=excluded.provenance_json, ext_json=excluded.ext_json, "
                "updated_at=excluded.updated_at",
                (r["id"], STUDY_ID, None, r["display_id"],
                 "metric" if has_number(r["value_text"]) else "fact",
                 r["label"], r["value_text"], None,
                 r["domain"] or None, r["source_level"],
                 r["verdict"], r["reason"], r["recomputed_text"], r["eval_run_id"],
                 r["provenance_json"], json.dumps(ext, ensure_ascii=False),
                 r["created_at"] or now(), r["updated_at"] or now()))
            stats["claims"] += 1

            anchors = []
            if r["tex_anchor_json"] and r["tex_anchor_json"] not in ("[]", "null"):
                try:
                    anchors = json.loads(r["tex_anchor_json"])
                except Exception:
                    anchors = []
            for a in anchors:
                ref = json.dumps(a, ensure_ascii=False, sort_keys=True)
                aid = sid("ca", r["id"], "tex_table", ref)
                f = a.get("file") or ""
                conn.execute(
                    "INSERT INTO claim_anchor(id,claim_id,type,ref_json,resolver,resolved_sha1,created_at) "
                    "VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
                    (aid, r["id"], "tex_table", ref, "paper_tbiom",
                     sha1_file(os.path.join("docs", "paper", "tbiom-r2", f)) or
                     sha1_file(os.path.join("docs", "paper", "tbiom-r3", f)) or
                     sha1_file(f), now()))
                stats["anchors"] += 1
        conn.execute("ALTER TABLE paper_claim RENAME TO paper_claim_v1")
        stats["renamed"] = True

    # 3) 视图（老工具读 paper_claim 仍可用）
    ensure_view(conn)

    # 4) deliverable + deliverable_ref
    for d in DELIVERABLES:
        did = sid("dl", STUDY_ID, d["name"], d["version"])
        hf = os.path.join(ROOT, d["hash_file"])
        conn.execute(
            "INSERT INTO deliverable(id,study_id,name,kind,version,path,sha256,status,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "sha256=excluded.sha256, status=excluded.status",
            (did, STUDY_ID, d["name"], d["kind"], d["version"], d["path"],
             sha256_file(hf), d["status"], now()))
        # 只有 r2 的锚点能定位（pca.TEX 指向 r2）
        if d["version"] == "r2":
            for (cid, ref, aid) in conn.execute(
                    "SELECT a.claim_id, a.ref_json, a.id FROM claim_anchor a "
                    "JOIN claim c ON c.id=a.claim_id WHERE c.study_id=?", (STUDY_ID,)):
                try:
                    obj = json.loads(ref)
                except Exception:
                    continue
                loc = "%s · %s · 第 %s 行 · %s" % (
                    obj.get("file", "?"), obj.get("table", "?"),
                    obj.get("row", "?"), obj.get("key", ""))
                rid = sid("dr", did, cid, loc)
                conn.execute(
                    "INSERT INTO deliverable_ref(id,deliverable_id,claim_id,location,anchor_id,created_at) "
                    "VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET anchor_id=excluded.anchor_id",
                    (rid, did, cid, loc, aid, now()))
                stats["refs"] += 1

    return stats


def diagnose(conn):
    """守恒诊断：C1（挂到实验卡）/ C2（有 run 或已裁定）/ C4（进了交付物）。"""
    q = lambda sql: conn.execute(sql).fetchone()[0]
    total = q("SELECT COUNT(*) FROM claim")
    if not total:
        return {"total": 0}
    return {
        "total": total,
        "orphan_C1": q("SELECT COUNT(*) FROM claim WHERE card_id IS NULL"),
        "with_run_C2": q("SELECT COUNT(*) FROM claim WHERE eval_run_id IS NOT NULL"),
        "in_deliverable_C4": q("SELECT COUNT(DISTINCT claim_id) FROM deliverable_ref"),
        "metric": q("SELECT COUNT(*) FROM claim WHERE kind='metric'"),
        "fact": q("SELECT COUNT(*) FROM claim WHERE kind='fact'"),
        "verdict": dict(conn.execute("SELECT verdict, COUNT(*) FROM claim GROUP BY verdict").fetchall()),
        "anchors": q("SELECT COUNT(*) FROM claim_anchor"),
        "studies": q("SELECT COUNT(*) FROM study"),
        "deliverables": q("SELECT COUNT(*) FROM deliverable"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写库（默认只报告）")
    ap.add_argument("--db", default=DB)
    a = ap.parse_args()

    conn = sqlite3.connect(a.db)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    p = plan(conn)
    print("[migrate] 现状：paper_claim 是%s；待迁 claim %d 条、锚点 %d 个；deliverable %d 个"
          % ("表（需迁移）" if p["legacy_table"] else "视图（已迁移）",
             p["claims"], p["anchors"], p["deliverables"]))

    if not a.apply:
        d = diagnose(conn) if not p["legacy_table"] else {}
        if d:
            print("[migrate] 诊断：", json.dumps(d, ensure_ascii=False))
        print("[migrate] 未写库。加 --apply 执行。")
        conn.close()
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    bak = os.path.join(BACKUP_DIR, "ops-before-claim-%s.db" % dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    shutil.copy(a.db, bak)
    print("[migrate] 备份 →", os.path.relpath(bak, ROOT))

    try:
        st = migrate(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        print("[migrate] 失败，已回滚；备份在", os.path.relpath(bak, ROOT))
        raise
    print("[migrate] 写入：claim %d、anchor %d、ref %d、老表改名 %s"
          % (st["claims"], st["anchors"], st["refs"], st["renamed"]))
    print("[migrate] 诊断：", json.dumps(diagnose(conn), ensure_ascii=False))
    conn.close()


if __name__ == "__main__":
    sys.exit(main())
