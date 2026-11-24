#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ledger_assign.py — C1 归属：容器级建议表 + 人工确认 + 批量写库。

三层：
  ops/claims/card_map.csv         容器（12 section + 14 表）→ 卡。人工改 decision 列。
  ops/claims/claim_candidates.csv 混合章节里的逐条候选（section 无法继承的那部分）。
  写库                             只认 decision 列；空 = 不动。

用法：
  python tools/ledger_assign.py --suggest              # 生成/刷新两份 CSV（保留已填的 decision）
  python tools/ledger_assign.py --check                # 校验 decision 合法性
  python tools/ledger_assign.py --apply --yes          # 写 claim_card / card_state='n/a'

决策依据：Q7（两级容器、表优先）、Q8（脚本出候选、人改）、Q9（primary 唯一）。
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "ops", "ops.db")
CARD_MAP = os.path.join(ROOT, "ops", "claims", "card_map.csv")
CAND = os.path.join(ROOT, "ops", "claims", "claim_candidates.csv")
STUDY = "st_multicam-reid"

# 容器初值（作者判断，供人改）。格式：容器 → [(卡, role, 置信, 说明)]
# 空卡 = 该容器不服务单张卡（study 级 / 备料 / 验收 / 负面清单）。
SEED = {
    ("section", "一、两域与图统计（§3.1, Table 1）"): [("", "", "", "数据集描述，study 级")],
    ("section", "二、真值（§4, Appendix A）"): [("", "", "", "真值备料，属 ③ 而非某张卡")],
    ("section", "三、外观天花板与 P2（§5.2, Table 2 / Table 3）"): [("EXP-002", "primary", "high", "外观表征上限")],
    ("section", "四、人脸像素预算与 P1（§5.1, Figure 2）"): [("EXP-003", "primary", "high", "人脸锚定的像素前提")],
    ("section", "五、服装多样性与 P3（§5.3, §6.3, Table 5）"): [("EXP-002", "primary", "medium", "外观鲁棒性，归跨镜关联")],
    ("section", "六、分级协议各级证据（§6）"): [("EXP-001", "primary", "high", "分级协议即「能否接回」的判据")],
    ("section", "七、不可折叠（§7）"): [("EXP-001", "primary", "high", "分级不可折叠是等级定义的一部分")],
    ("section", "八、跨采集验收（§8, Table 11 / Table 12）"): [("", "", "", "跨采集验收，对应 D 关")],
    ("section", "九、已知不可用 / 不要引用"): [("", "", "", "负面清单")],
    ("section", "十一、r2 新增/核对的事实（2026-11-14）"): [("", "", "", "混合章节，逐条判")],
    ("section", "十二、V2 补记（2026-11-24）"): [("", "", "", "混合章节，逐条判")],
    ("section", ""): [("", "", "", "无章节归属")],

    ("table", "tab:d1"): [("", "", "", "数据集统计")],
    ("table", "tab:agg"): [("EXP-002", "primary", "high", "外观聚合结果")],
    ("table", "tab:pair"): [("EXP-002", "primary", "high", "机位对 AUC")],
    ("table", "tab:protocol"): [("EXP-001", "primary", "high", "分级协议主表"),
                                ("EXP-002", "supporting", "medium", "跨镜去重也按此协议计")],
    ("table", "tab:merge"): [("EXP-001", "primary", "high", "合并结果"),
                             ("EXP-002", "supporting", "medium", "合并是跨镜去重的落地")],
    ("table", "tab:gate"): [("EXP-002", "primary", "medium", "属性闸门影响合并纯度")],
    ("table", "tab:verifier"): [("EXP-001", "primary", "high", "验证器分级")],
    ("table", "tab:xday"): [("", "", "", "跨采集验收")],
    ("table", "tab:selector"): [("", "", "", "运行时选择器，未立卡")],
    ("table", "The form in which a system"): [("EXP-001", "primary", "high", "五种协议形态")],
    ("table", "Domains"): [("", "", "", "域定义")],
    ("table", "Adjudicated pairs"): [("", "", "", "真值备料")],
    ("table", "Annotation reliability"): [("", "", "", "标注质量，属 ③")],
    ("table", "Same cells as the nine-cell"): [("", "", "", "补充材料核对表")],
}

# 逐条候选用的卡关键词（只生成候选，不自动定）
KW = {
    "EXP-001": ["交汇", "接回", "断口", "碎片", "分级", "协议", "L0", "L1", "L2", "L3", "合并", "簇", "纯度"],
    "EXP-002": ["跨机位", "跨镜", "去重", "外观", "表征", "AUC", "重复框", "共视", "机位对"],
    "EXP-003": ["人脸", "锚定", "无脸", "像素", "图库", "face", "SCRFD"],
    "EXP-004": ["人工", "纠错", "复核", "判读", "标注", "κ", "kappa", "分钟"],
    "EXP-005": ["训练", "微调", "更新", "权重", "记忆", "跨天", "OSNet 更新"],
}


def now():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def cc_id(cid, card, role):
    return "cc_" + hashlib.sha1(("%s|%s|%s" % (cid, card, role)).encode()).hexdigest()[:10]


def containers(conn):
    """返回 {(type, name): [claim_id]}，type ∈ section|table。"""
    out = collections.defaultdict(list)
    for r in conn.execute("SELECT id, ext_json FROM claim WHERE study_id=?", (STUDY,)):
        sec = (json.loads(r["ext_json"] or "{}").get("section") or "")
        out[("section", sec)].append(r["id"])
    for r in conn.execute("""SELECT a.claim_id, a.ref_json FROM claim_anchor a
                             JOIN claim c ON c.id=a.claim_id WHERE c.study_id=?""", (STUDY,)):
        try:
            t = json.loads(r["ref_json"]).get("table") or ""
        except Exception:
            continue
        out[("table", t)].append(r["claim_id"])
    return out


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with io.open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def do_suggest(conn):
    keep_map = {(r["container_type"], r["container"]): r for r in read_csv(CARD_MAP)}
    keep_cand = {r["claim_id"]: r for r in read_csv(CAND)}
    cont = containers(conn)

    rows = []
    for (typ, name), cids in sorted(cont.items(), key=lambda x: (x[0][0], -len(x[1]))):
        seed = SEED.get((typ, name), [("", "", "", "")])
        for card, role, conf, note in seed:
            old = keep_map.get((typ, name))
            # 预填：有候选卡填卡；混合章节留空待逐条；其余不服务单张卡的填 n/a
            default = "" if "逐条" in note else ("n/a" if not card else card)
            rows.append({
                "container_type": typ,
                "container": name,
                "n_claims": len(cids),
                "candidate_card": card,
                "role": role,
                "confidence": conf,
                "decision": (old or {}).get("decision") or default,
                "note": note,
            })
    write_csv(CARD_MAP, ["container_type", "container", "n_claims", "candidate_card",
                         "role", "confidence", "decision", "note"], rows)

    by_id = {r["id"]: r for r in conn.execute(
        "SELECT id, locator, value_text, ext_json FROM claim WHERE study_id=?", (STUDY,))}
    cand_rows = []
    for (typ, name), cids in cont.items():
        seed = SEED.get((typ, name), [])
        if typ != "section" or not any("逐条" in s[3] for s in seed):
            continue
        for cid in sorted(cids, key=lambda x: by_id[x]["locator"] or ""):
            r = by_id[cid]
            txt = "%s %s" % (json.loads(r["ext_json"] or "{}").get("label") or "", r["value_text"] or "")
            hits = []
            for card, words in KW.items():
                m = [w for w in words if w.lower() in txt.lower()]
                if m:
                    hits.append((card, len(m), "/".join(m[:2])))
            hits.sort(key=lambda x: (-x[1], x[0]))
            old = keep_cand.get(cid, {})
            cand_rows.append({
                "claim_id": cid,
                "locator": r["locator"] or "",
                "section": name[:40],
                "value": (r["value_text"] or "")[:40],
                "cand1": hits[0][0] if hits else "",
                "why1": hits[0][2] if hits else "",
                "cand2": hits[1][0] if len(hits) > 1 else "",
                "why2": hits[1][2] if len(hits) > 1 else "",
                "decision": old.get("decision", ""),
                "role": old.get("role", "primary"),
                "note": old.get("note", ""),
            })
    write_csv(CAND, ["claim_id", "locator", "section", "value", "cand1", "why1",
                     "cand2", "why2", "decision", "role", "note"], cand_rows)
    print("[assign] %s：%d 行容器" % (os.path.relpath(CARD_MAP, ROOT), len(rows)))
    print("[assign] %s：%d 条待逐条" % (os.path.relpath(CAND, ROOT), len(cand_rows)))
    inherit = sum(r["n_claims"] for r in rows if r["candidate_card"] and r["container_type"] == "section")
    print("[assign] section 可继承 %d 条、逐条 %d 条；两处都空着等人工填 decision"
          % (inherit, len(cand_rows)))


def cards(conn):
    return {r[0] for r in conn.execute("SELECT id FROM exp_card")}


def do_check(conn):
    ok = True
    maps = read_csv(CARD_MAP)
    if not maps:
        print("[check] 先跑 --suggest")
        return 1
    cont = containers(conn)
    seen = set()
    for r in maps:
        key = (r["container_type"], r["container"])
        seen.add(r["container_type"] + "|" + r["container"][:70])
        d = (r["decision"] or "").strip()
        if d and d != "n/a" and d not in cards(conn):
            print("  ! 未知卡 %s（容器 %s）" % (d, r["container"]))
            ok = False
        if d and d != "n/a" and d != r["candidate_card"]:
            print("  · 改判：%s → %s" % (r["container"][:30], d))
    have = {t + "|" + n[:70] for t, n in cont}
    miss = have - seen
    if miss:
        print("  ! 未覆盖容器 %d 个：%s" % (len(miss), list(miss)[:3]))
        ok = False
    for r in read_csv(CAND):
        d = (r["decision"] or "").strip()
        if d and d != "n/a" and d not in cards(conn):
            print("  ! 未知卡 %s（%s）" % (d, r["locator"]))
            ok = False
    print("[check] %s（容器 %d、逐条 %d）" % ("通过" if ok else "有问题", len(maps), len(read_csv(CAND))))
    return 0 if ok else 1


def do_apply(conn, yes):
    if not yes:
        print("[apply] 需要 --yes 确认（会写 claim_card）")
        return 1
    cont = containers(conn)
    pri = {}                                    # cid -> (card, rank)
    na = {}                                     # cid -> rank（只在没有 primary 时生效）
    support = collections.defaultdict(set)

    for r in read_csv(CARD_MAP):
        d = (r["decision"] or "").strip()
        if not d:
            continue
        rank = 2 if r["container_type"] == "table" else 1   # 表优先（Q7）
        for cid in cont.get((r["container_type"], r["container"])) or []:
            if d == "n/a":
                if rank > na.get(cid, 0):
                    na[cid] = rank
            elif (r["role"] or "primary") == "supporting":
                support[cid].add(d)
            elif rank > pri.get(cid, ("", 0))[1]:
                pri[cid] = (d, rank)

    for r in read_csv(CAND):
        d = (r["decision"] or "").strip()
        if not d:
            continue
        cid = r["claim_id"]
        if (r["role"] or "primary") == "supporting":
            support[cid].add(d)
        elif d == "n/a":
            na[cid] = 3
        else:
            pri[cid] = (d, 3)                   # 人工最高

    conn.execute("DELETE FROM claim_card")
    conn.execute("UPDATE claim SET card_state=NULL WHERE study_id=?", (STUDY,))
    n_p = n_na = 0
    for cid, (card, rank) in pri.items():
        conn.execute("INSERT INTO claim_card(id,claim_id,card_id,role,source,created_at) VALUES(?,?,?,?,?,?)",
                     (cc_id(cid, card, "primary"), cid, card, "primary",
                      "manual" if rank == 3 else "container", now()))
        n_p += 1
    for cid in na:
        if cid in pri:                          # 有明确卡就不豁免
            continue
        conn.execute("UPDATE claim SET card_state='n/a' WHERE id=?", (cid,))
        n_na += 1
    n_s = 0
    for cid, cards_ in support.items():
        for card in cards_:
            if pri.get(cid, ("", 0))[0] == card:
                continue
            conn.execute("INSERT OR IGNORE INTO claim_card(id,claim_id,card_id,role,source,created_at) "
                         "VALUES(?,?,?,?,?,?)",
                         (cc_id(cid, card, "supporting"), cid, card, "supporting", "manual", now()))
            n_s += 1
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM claim WHERE study_id=?", (STUDY,)).fetchone()[0]
    print("[apply] primary %d、supporting %d、n/a %d、orphan %d / 总计 %d → 已归属 %.0f%%"
          % (n_p, n_s, n_na, total - n_p - n_na, total, 100.0 * (n_p + n_na) / total))
    return 0


def do_status(conn):
    q = lambda s: conn.execute(s).fetchone()[0]
    total = q("SELECT COUNT(*) FROM claim WHERE study_id='%s'" % STUDY)
    p = q("SELECT COUNT(DISTINCT claim_id) FROM claim_card WHERE role='primary'")
    s = q("SELECT COUNT(*) FROM claim_card WHERE role='supporting'")
    na = q("SELECT COUNT(*) FROM claim WHERE card_state='n/a'")
    orph = total - p - na
    print("[status] 总计 %d | primary %d (%.0f%%) | n/a %d | orphan %d | supporting 边 %d"
          % (total, p, 100.0 * p / total, na, orph, s))
    for r in conn.execute("""SELECT card_id, COUNT(*) n FROM claim_card WHERE role='primary'
                             GROUP BY card_id ORDER BY n DESC"""):
        print("   %-8s %d" % (r[0], r[1]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--suggest", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    conn.row_factory = sqlite3.Row
    if a.suggest:
        return do_suggest(conn) or 0
    if a.check:
        return do_check(conn)
    if a.apply:
        return do_apply(conn, a.yes)
    return do_status(conn)


if __name__ == "__main__":
    sys.exit(main())
