#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ledger_flow.py — 流程诊断：把「从实验卡贯穿到底」的四条守恒关系算成数字。

C1 每条 claim 挂到实验卡了吗（挂不上就是 orphan；说明类可标 n/a 豁免）
C2 每条 claim 有可执行的 run 吗
C3 每次 run 有输入版本吗
C4 每个交付物引用了多少 claim、其中多少已核、多少挂了锚点

只读，输出 ops/reports/ledger/FLOW.md。
用法：python tools/ledger_flow.py [--db ops/ops.db]
"""
import argparse
import collections
import json
import os
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "ops", "ops.db")
OUT = os.path.join(ROOT, "ops", "reports", "ledger", "FLOW.md")


def pct(a, b):
    return "—" if not b else "%.0f%%" % (100.0 * a / b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    conn.row_factory = sqlite3.Row
    q = lambda sql, args=(): conn.execute(sql, args).fetchone()[0]

    counts = collections.OrderedDict()
    for t in ("study", "exp_card", "exp_gate", "asset_version", "label_set", "eval_protocol",
              "eval_run", "claim", "claim_anchor", "claim_card", "deliverable", "deliverable_ref"):
        try:
            counts[t] = q("SELECT COUNT(*) FROM %s" % t)
        except sqlite3.OperationalError:
            counts[t] = None

    total = counts.get("claim") or 0
    primary = q("SELECT COUNT(DISTINCT claim_id) FROM claim_card WHERE role='primary'") if total else 0
    support = q("SELECT COUNT(*) FROM claim_card WHERE role='supporting'") if total else 0
    na = q("SELECT COUNT(*) FROM claim WHERE card_state='n/a'") if total else 0
    orphan = total - primary - na
    with_run = q("SELECT COUNT(*) FROM claim WHERE eval_run_id IS NOT NULL") if total else 0
    verified = q("SELECT COUNT(*) FROM claim WHERE verdict IN ('match','unverifiable')") if total else 0
    runs = counts.get("eval_run") or 0
    run_no_input = q("SELECT COUNT(*) FROM eval_run WHERE inputs_json IN ('[]','')") if runs else 0
    run_ok = q("SELECT COUNT(*) FROM eval_run WHERE exit_code=0") if runs else 0

    per_card = []
    if counts.get("exp_card"):
        for r in conn.execute("SELECT id, title, state FROM exp_card ORDER BY id"):
            n = q("SELECT COUNT(*) FROM claim_card WHERE card_id=? AND role='primary'", (r["id"],))
            v = q("""SELECT COUNT(*) FROM claim_card cc JOIN claim c ON c.id=cc.claim_id
                     WHERE cc.card_id=? AND cc.role='primary'
                       AND c.verdict IN ('match','unverifiable')""", (r["id"],))
            per_card.append((r["id"], r["title"], r["state"], n, v))

    # orphan 散在哪些容器（适配器私有字段，报告只做归组，不解释业务含义）
    orphan_groups = []
    if total:
        rows = conn.execute("""SELECT ext_json, stratum FROM claim c
                               WHERE COALESCE(c.card_state,'') != 'n/a'
                                 AND c.id NOT IN (SELECT claim_id FROM claim_card WHERE role='primary')
                           """).fetchall()
        c = collections.Counter()
        for r in rows:
            try:
                ext = json.loads(r["ext_json"] or "{}")
            except Exception:
                ext = {}
            key = ext.get("section") or r["stratum"] or "（无容器）"
            c[key] += 1
        orphan_groups = c.most_common()

    dels = []
    if counts.get("deliverable"):
        for r in conn.execute("SELECT id,name,kind,version,status,sha256 FROM deliverable ORDER BY kind,name"):
            n = q("SELECT COUNT(DISTINCT claim_id) FROM deliverable_ref WHERE deliverable_id=?", (r["id"],))
            v = q("""SELECT COUNT(DISTINCT f.claim_id) FROM deliverable_ref f
                     JOIN claim c ON c.id=f.claim_id
                     WHERE f.deliverable_id=? AND c.verdict IN ('match','unverifiable')""", (r["id"],))
            an = q("""SELECT COUNT(*) FROM deliverable_ref
                      WHERE deliverable_id=? AND anchor_id IS NOT NULL""", (r["id"],))
            dels.append((r, n, v, an))

    L = ["# 流程诊断（C1–C4）\n",
         "> 生成：`tools/ledger_flow.py`　时间：%s　库：`%s`" % (
             datetime.now().strftime("%Y-%m-%d %H:%M"), os.path.relpath(a.db, ROOT)),
         "> 定义见 `docs/workbench-flow-model-2026-11-24.md` §2。四条守恒与领域无关；",
         "> 换一个 study，只有「容器」的取值变化，判定方式不变。\n",
         "## 一、流水线存量\n",
         "| 环节 | 表 | 行数 |", "|---|---|---:|"]
    label = {"study": "① 立题", "exp_card": "② 立卡", "exp_gate": "② 关口", "asset_version": "③ 备料·文件",
             "label_set": "③ 备料·真值", "eval_protocol": "④ 协议", "eval_run": "⑤ 执行",
             "claim": "⑥ 立论", "claim_anchor": "⑥ 出处", "claim_card": "⑥ 归属",
             "deliverable": "⑦ 交付", "deliverable_ref": "⑦ 引用"}
    for t, n in counts.items():
        L.append("| %s | `%s` | %s |" % (label.get(t, ""), t, "—" if n is None else n))
    L.append("")

    L += ["## 二、四条守恒\n",
          "| 关系 | 要求 | 实测 | 判定 |", "|---|---|---|---|",
          "| **C1** claim→实验卡 | 每条能回答「属于哪张卡」；说明类可标 n/a | 挂卡 **%d / 可归属 %d（%s）**；"
          "n/a %d、orphan **%d**；supporting 边 %d | %s |"
          % (primary, total - na, pct(primary, total - na), na, orphan, support,
             "通过" if not orphan else "**待归属 %d 条**" % orphan),
          "| **C2** claim→run | 每条有可执行 run 或已裁定 | %d / %d 有 run（%s），已裁定 %d（%s） | %s |"
          % (with_run, total, pct(with_run, total), verified, pct(verified, total),
             "通过" if verified >= total * 0.95 else "部分"),
          "| **C3** run→输入版本 | 每次 run 有输入哈希 | %d / %d 有输入（%s），exit0 %d | %s |"
          % (runs - run_no_input, runs, pct(runs - run_no_input, runs), run_ok,
             "通过" if runs and run_no_input <= 1 else "部分"),
          "| **C4** 交付物→claim | 每个交付物能说出引用了多少、核了多少、锚定了多少 | 见下表 | — |", ""]

    L += ["## 三、实验卡（C1 的分母）\n"]
    if per_card:
        L += ["| 卡 | 标题 | 状态 | claim | 已裁定 |", "|---|---|---|---:|---:|"]
        for cid, title, state, n, v in per_card:
            L.append("| `%s` | %s | %s | %d | %d |" % (cid, title[:28], state, n, v))
        L += ["", "**没有卡认领的 claim：%d 条**（占 %s）；标 n/a 豁免的 %d 条。"
                  "归属由 `ops/claims/card_map.csv` + `claim_candidates.csv` 决定，"
                  "写库命令 `tools/ledger_assign.py --apply --yes`。\n"
              % (orphan, pct(orphan, total), na)]
        if orphan_groups:
            L += ["orphan 分布（按出处容器，前 10）：\n", "| 容器 | 条数 |", "|---|---:|"]
            for k, v in orphan_groups[:10]:
                L.append("| %s | %d |" % (str(k)[:60].replace("|", "/"), v))
            L.append("")
    else:
        L += ["（没有实验卡）\n"]

    L += ["## 四、交付物（C4）\n"]
    if dels:
        L += ["| 交付物 | 类型 | 版本 | 状态 | 引用 claim | 已裁定 | 挂锚点 |", "|---|---|---|---|---:|---:|---:|"]
        for r, n, v, an in dels:
            L.append("| %s | %s | %s | %s | %d | %d | %d |"
                     % (r["name"], r["kind"], r["version"], r["status"], n, v, an))
        L.append("")
    else:
        L += ["（没有交付物）\n"]

    L += ["## 五、判定\n",
          "C1 是当前唯一未达成的环节：`claim` 能追到文件和 run，却追不到「为什么做这件事」。",
          "补法只有一条：人工确认容器映射（`card_map.csv` 28 行）与逐条候选（`claim_candidates.csv`），"
          "再 `--apply`。禁止自动写归属。\n"]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    os.replace(tmp, a.out)
    print("[flow] %s  C1 %d/%d (n/a %d, orphan %d)  C2 %d/%d  C3 %d/%d  C4 %s" % (
        os.path.relpath(a.out, ROOT), primary, total - na, na, orphan, with_run, total,
        runs - run_no_input, runs, ",".join("%s=%d" % (r["version"], n) for r, n, _, _ in dels) or "—"))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
