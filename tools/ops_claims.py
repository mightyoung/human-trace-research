#!/usr/bin/env python
"""ops_claims.py — 论文数字台账（v2.1 / V1）。

把 docs/paper/EVIDENCE.md 解析成 `paper_claim` 索引行，把 tex 表格的每一行锚定到 claim，
并把来源文件登记为 `asset_version`。权威始终是文件；本工具只写索引表，不改 EVIDENCE、不改 tex。

子命令
  import   解析 EVIDENCE + 登记来源文件 sha256 → paper_claim / asset_version
  anchor   按 ops/claims/tex_anchor.json 把 14 张 tex 表的行挂到 claim，并算出“无 claim 覆盖”的数字
  truth    登记 5 批真值（label_set）与论文包协议（eval_protocol）；条数一律实测
  report   生成 ops/reports/paper_claims_v1/report.md
  all      import + anchor + truth + report
  status   只看库内计数

写真实库需 --apply（默认只演练）；写临时库不需要。
用法：python tools/ops_claims.py all --db <path> [--apply]
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops_db                      # noqa: E402
import paper_claims_audit as pca   # noqa: E402

ROOT = pca.ROOT
ANCHOR_PATH = os.path.join(ROOT, "ops", "claims", "tex_anchor.json")
SETS_PATH = os.path.join(ROOT, "ops", "claims", "label_sets.json")
PROTO_PATH = os.path.join(ROOT, "ops", "claims", "eval_protocols.json")
VERIFY_PATH = os.path.join(ROOT, "ops", "claims", "verify_specs.json")
REPORT_DIR = os.path.join(ROOT, "ops", "reports", "paper_claims_v1")
BACKUP_DIR = os.path.join(ROOT, "ops", "db_backup")
# 通用流程层（K0）里本适配器所属的 study。写入一律走 claim / claim_anchor，
# 读仍走视图 paper_claim（= ops/views/paper_tbiom.sql，见 docs/workbench-flow-model-2026-11-24.md）。
STUDY_ID = "st_multicam-reid"
ADAPTER = "paper_tbiom"
KIND_BY_EXT = {".mp4": "media", ".jpg": "media", ".jpeg": "media", ".png": "media",
               ".json": "derived", ".jsonl": "derived", ".npz": "derived", ".npy": "derived",
               ".csv": "derived", ".txt": "derived", ".md": "doc", ".py": "script",
               ".sh": "script", ".cmd": "script", ".pt": "weights", ".pth": "weights",
               ".onnx": "weights", ".html": "derived"}


def now():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def clean(s):
    return s.replace("**", "").replace("`", "").strip()


def claim_id(section, label, value, k=0, kind="table"):
    # k：同一 (section, label, value) 的第几条。9 格表与集中度表前两格都是日期，需要靠 k 区分。
    key = "%s\x1f%s\x1f%s\x1f%d\x1f%s" % (section, label, value, k, kind)
    return "pc_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def parsed_claims():
    ev_path = os.path.join(ROOT, pca.EVIDENCE)
    with open(ev_path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")
    res = pca.Resolver()
    claims, _ = pca.parse_evidence(text, res)
    out = []
    seen = {}
    for c in claims:
        cells = [clean(x) for x in c["cells"]]
        label = cells[0][:60]
        value = cells[1] if len(cells) > 1 else ""
        dom = next((x for x in cells[2:] if x in ("D1", "D2", "D3")), "")
        k = seen.get((c["section"], label, value), 0)
        seen[(c["section"], label, value)] = k + 1
        out.append({"id": claim_id(c["section"], label, value, k, c["kind"]),
                    "display_id": c["id"], "kind": c["kind"],
                    "section": c["section"], "line": c["line"], "label": label,
                    "value": value, "cells": cells, "domain": dom,
                    "level": c["level"], "src_ok": c["src_ok"], "repro": c["repro"],
                    "scripts": c["scripts"], "paths": c["paths"], "missing": c["missing"]})
    return out, res, hashlib.sha256(raw).hexdigest()


def prov_json(c):
    return json.dumps({
        "src_file": bool(c["src_ok"]),
        "script": bool(c["repro"]),
        "step": bool(c["repro"]),
        "recompute": None,          # V3 由 ops_verify 填
        "n": None,
        "domain": bool(c["domain"]),
        "paths": c["paths"],
        "scripts": c["scripts"],
        "missing": c["missing"],
    }, ensure_ascii=False)


def claim_ext(c):
    """paper_tbiom 的私有字段打进 claim.ext_json；内核不解释这些键。"""
    return json.dumps({"adapter": ADAPTER, "paper_kind": c["kind"], "section": c["section"],
                       "evidence_line": c["line"], "label": c["label"]}, ensure_ascii=False)


def tex_sha1(fname):
    for sub in ("tbiom-r2", "tbiom-r3"):
        p = os.path.join(ROOT, "docs", "paper", sub, fname or "")
        if os.path.isfile(p):
            return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]
    return None


def put_anchors(conn, cid, lst):
    """替换一条 claim 的全部 tex 锚点（claim_anchor 是权威，视图再从它聚合回 tex_anchor_json）。"""
    conn.execute("DELETE FROM claim_anchor WHERE claim_id=? AND resolver=?", (cid, ADAPTER))
    for a in lst:
        ref = json.dumps(a, ensure_ascii=False, sort_keys=True)
        aid = "ca_" + hashlib.sha1(("%s|tex_table|%s" % (cid, ref)).encode()).hexdigest()[:10]
        conn.execute("INSERT INTO claim_anchor(id,claim_id,type,ref_json,resolver,resolved_sha1,created_at) "
                     "VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
                     (aid, cid, "tex_table", ref, ADAPTER, tex_sha1(a.get("file")), now()))


def do_import(conn, commit=True, rebuild=False):
    claims, res, ev_sha = parsed_claims()
    n_new = n_upd = 0
    if rebuild:
        busy = conn.execute("SELECT COUNT(*) FROM claim WHERE study_id=? AND verdict != 'pending'",
                            (STUDY_ID,)).fetchone()[0]
        if busy:
            raise SystemExit("拒绝重建：已有 %d 条非 pending 裁定，先导出再重建" % busy)
        conn.execute("DELETE FROM claim_anchor WHERE claim_id IN (SELECT id FROM claim WHERE study_id=?)",
                     (STUDY_ID,))
        conn.execute("DELETE FROM claim WHERE study_id=?", (STUDY_ID,))
    for c in claims:
        row = conn.execute("SELECT id FROM claim WHERE id=?", (c["id"],)).fetchone()
        if row:
            conn.execute("""UPDATE claim SET locator=?, kind=?, value_text=?, stratum=?, evidence_grade=?,
                            provenance_json=?, ext_json=?, updated_at=? WHERE id=?""",
                         (c["display_id"], "metric" if re.search(r"\d", c["value"] or "") else "fact",
                          c["value"], c["domain"] or None, c["level"], prov_json(c), claim_ext(c), now(), c["id"]))
            n_upd += 1
        else:
            conn.execute("""INSERT INTO claim(id, study_id, card_id, locator, kind, statement, value_text,
                            unit, stratum, evidence_grade, provenance_json, ext_json, verdict, created_at, updated_at)
                            VALUES(?,?,NULL,?,?,?,?,NULL,?,?,?,?, 'pending', ?, ?)""",
                         (c["id"], STUDY_ID, c["display_id"],
                          "metric" if re.search(r"\d", c["value"] or "") else "fact",
                          c["label"], c["value"], c["domain"] or None, c["level"], prov_json(c), claim_ext(c),
                          now(), now()))
            n_new += 1
    # 剪枝：EVIDENCE 改写某行文本后，旧 hash id 不再出现，留着就是指向已消失行的孤儿。
    # 只删「当前解析结果里没有」且「没有 tex 锚点」的行；有锚点的留给 do_anchor 报错。
    cur_ids = {c["id"] for c in claims}
    n_orph = 0
    for (oid,) in conn.execute("SELECT id FROM claim WHERE study_id=? AND id NOT IN "
                               "(SELECT claim_id FROM claim_anchor)", (STUDY_ID,)).fetchall():
        if oid not in cur_ids:
            conn.execute("DELETE FROM claim WHERE id=?", (oid,))
            n_orph += 1
    # 来源文件 → asset_version
    n_av = 0
    for path, rec in res.cache.items():
        if not rec:
            continue
        ext = os.path.splitext(path)[1].lower()
        vid = "av_" + hashlib.sha1(("%s|%s" % (rec["path"], rec["sha256"])).encode()).hexdigest()[:10]
        exists = conn.execute("SELECT 1 FROM asset_version WHERE id=?", (vid,)).fetchone()
        if not exists:
            conn.execute("""INSERT INTO asset_version(id, path, sha256, bytes, mtime, kind, hash_alg, weak, first_seen_at)
                            VALUES(?,?,?,?,?,?,?,?,?)""",
                         (vid, rec["path"], rec["sha256"], rec["bytes"], rec["mtime"],
                          KIND_BY_EXT.get(ext, "other"),
                          "sha256" if rec["sha256"] else "size-mtime", int(rec["weak"]), now()))
            n_av += 1
    for k, v in (("paper_claim_source_sha256", ev_sha), ("paper_claim_imported_at", now())):
        conn.execute("INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))
    if commit:
        conn.commit()
    return {"claims": len(claims), "new": n_new, "updated": n_upd, "pruned": n_orph, "assets_new": n_av,
            "evidence_sha256": ev_sha}


def load_tex_tables():
    out = []
    for rel in pca.TEX:
        lines = open(os.path.join(ROOT, rel), encoding="utf-8").read().split("\n")
        _, tables, _ = pca.parse_tex(lines)
        for t in tables:
            out.append({"file": os.path.basename(rel), **t})
    return out


def do_anchor(conn, commit=True):
    anchor = json.load(open(ANCHOR_PATH, encoding="utf-8"))
    built = (anchor.get("_built_from") or {}).get("sha256")
    cur = conn.execute("SELECT v FROM meta WHERE k='paper_claim_source_sha256'").fetchone()
    warn = "" if (cur and built and cur[0] == built) else \
        "锚定文件绑定的是另一版 EVIDENCE（built_from 与库内 sha256 不一致），请复核 ops/claims/tex_anchor.json"
    claims, _, _ = parsed_claims()
    by_display = {c["display_id"]: c for c in claims}
    by_id = {c["id"]: c for c in claims}
    all_nums = set()
    for c in claims:
        all_nums |= pca.nums(" ".join(c["cells"]))
    tables = load_tex_tables()

    conn.execute("DELETE FROM claim_anchor WHERE claim_id IN (SELECT id FROM claim WHERE study_id=?)",
                 (STUDY_ID,))
    anchors, gaps, errors, declared = {}, [], [], []
    for spec in anchor["tables"]:
        key = spec.get("label") or spec.get("caption")
        cand = [t for t in tables if t["file"] == spec["file"] and
                ((spec.get("label") and t["label"] == spec["label"]) or
                 (spec.get("caption") and t["caption"].startswith(spec["caption"])))]
        if not cand:
            errors.append("找不到表：%s %s" % (spec["file"], key))
            continue
        t = cand[0]
        if os.environ.get("HC_CLAIMS_DEBUG"):
            print("  [dbg]", key, "->", t["caption"][:20], len(t["rows"]),
                  [(x["caption"][:12], repr(spec.get("caption")), x["caption"].startswith(spec.get("caption") or "\0")) for x in cand][:2])
        for r in spec["rows"]:
            i = r["i"]
            if i >= len(t["rows"]):
                errors.append("%s 缺第 %d 行" % (key, i))
                continue
            raw = " ".join(t["rows"][i])
            if r.get("expect") and r["expect"].replace("--", "-") not in raw.replace("--", "-"):
                errors.append("%s r%d 期望「%s」未命中，实际：%s" % (key, i, r["expect"], raw[:60]))
                continue
            rn = sorted(pca.nums(re.sub(r"\d{2}-\d{2}", " ", raw)))
            if not r["claims"]:
                declared.append({"file": spec["file"], "table": key, "row": i,
                                 "key": r.get("expect", ""), "numbers": rn})
                continue
            covered = set()
            for did in r["claims"]:
                c = by_id.get(did) or by_display.get(did)
                if not c:
                    errors.append("%s r%d 引用了不存在的 claim %s" % (key, i, did))
                    continue
                anchors.setdefault(c["id"], []).append(
                    {"file": spec["file"], "table": key, "row": i, "key": r.get("expect", "")})
                text = " ".join(c["cells"])
                covered |= pca.nums(text)
            rest = [n for n in rn if n not in covered]
            if rest:
                # 区分两件事：EVIDENCE 全文都没有这个数（真要补出处）／只是本行没挂对 claim。
                # 后者只报带小数或 ≥3 位的数：1、2、10 这类短整数在全文里到处撞，报出来是噪声。
                g = {"file": spec["file"], "table": key, "row": i, "key": r.get("expect", ""),
                     "true_gap": [n for n in rest if n not in all_nums],
                     "anchor_gap": [n for n in rest if n in all_nums and ("." in n or len(n) >= 3)]}
                if g["true_gap"] or g["anchor_gap"]:
                    gaps.append(g)
    for cid, lst in anchors.items():
        put_anchors(conn, cid, lst)
    conn.execute("INSERT INTO meta(k, v) VALUES('tex_anchor_built_from', ?) "
                 "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (built or "",))
    if commit:
        conn.commit()
    return {"anchored_claims": len(anchors), "cells": sum(len(v) for v in anchors.values()),
            "gaps": gaps, "declared": declared, "errors": errors, "warn": warn}


# ---------------------------------------------------------------- V2：真值集与评测协议

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register_asset(conn, rel, kind="labels"):
    """把一个文件登记进 asset_version（全内容 sha256）。返回 sha256 或 None。"""
    p = os.path.join(ROOT, rel)
    if not os.path.isfile(p):
        return None
    sha = sha256_file(p)
    vid = "av_" + hashlib.sha1(("%s|%s" % (rel, sha)).encode("utf-8")).hexdigest()[:10]
    if not conn.execute("SELECT 1 FROM asset_version WHERE id=?", (vid,)).fetchone():
        st = os.stat(p)
        conn.execute("""INSERT INTO asset_version(id, path, sha256, bytes, mtime, kind, hash_alg,
                        weak, first_seen_at) VALUES(?,?,?,?,?,?,?,0,?)""",
                     (vid, rel.replace("\\", "/"), sha, st.st_size, int(st.st_mtime), kind, "sha256", now()))
    return sha


def item_labels(d, spec):
    """按 spec 取出 (逐条标签列表, 分组字典)。三种取值形态统一成 list。"""
    items = d
    for part in (spec.get("path") or "").split("."):
        if part:
            items = items[part]
    fld = spec["label_field"]
    labels = [it.get(fld) for it in items]
    groups = None
    if spec.get("group_field"):
        groups = {}
        for it in items:
            groups.setdefault(it.get(spec["group_field"]), []).append(it.get(fld))
    return labels, groups


def measure_set(spec):
    """打开 uri 实测条数；与 declared 不一致由调用方报错。"""
    p = os.path.join(ROOT, spec["uri"])
    d = json.load(open(p, encoding="utf-8"))
    kind = spec["kind"]
    same_v = [str(x) for x in spec.get("same") or []]
    diff_v = [str(x) for x in spec.get("diff") or []]
    uns_v = [str(x) for x in spec.get("unsure") or []]
    out = {"n_items": None, "n_same": None, "n_diff": None, "n_unlabeled": None, "rule": {}}
    if kind in ("binary_items", "three_way"):
        labels, _ = item_labels(d, spec)
        ls = [str(x) for x in labels]
        out["n_items"] = len(ls)
        out["n_same"] = sum(1 for x in ls if x in same_v)
        out["n_diff"] = sum(1 for x in ls if x in diff_v)
        out["n_unlabeled"] = sum(1 for x in ls if uns_v and x in uns_v) or \
            sum(1 for x in ls if x not in same_v and x not in diff_v and x not in uns_v)
        out["rule"] = {"field": spec["label_field"],
                       "dist": {k: ls.count(k) for k in sorted(set(ls))}}
    elif kind == "groups_of":
        _, groups = item_labels(d, spec)
        pure = [g for g, v in groups.items() if not any(str(x) in diff_v for x in v)]
        out["n_items"] = len(groups)
        out["n_same"] = len(pure)
        out["n_diff"] = len(groups) - len(pure)
        out["n_unlabeled"] = 0
        out["rule"] = {"group_field": spec["group_field"],
                       "rule": "组内至少一条判异 → 混人组",
                       "sizes": sorted((len(v) for v in groups.values()), reverse=True)}
    elif kind == "kappa_layer":
        lay = d["layers"]
        for part in str(spec["layer"]).split("."):     # 支持 gallery.person_only
            lay = lay[part]
        out["n_items"] = lay["n"]
        out["rule"] = {"layer": spec["layer"], "po": round(lay["po"], 4),
                       "k": round(lay["k"], 4), "ci95": [round(x, 4) for x in lay.get("ci95") or []],
                       "variants": {k: (v.get("k") if isinstance(v, dict) else v)
                                    for k, v in lay.items() if isinstance(v, dict) and "k" in v}}
    else:
        raise SystemExit("未知 kind: %s" % kind)
    out["sha256"] = sha256_file(p)
    out["abs"] = p
    return out


def do_truth(conn, commit=True):
    """登记 label_set 与 eval_protocol；条数一律实测，与 declared 不符即报错。"""
    sets = json.load(open(SETS_PATH, encoding="utf-8"))["sets"]
    protos = json.load(open(PROTO_PATH, encoding="utf-8"))["protocols"]
    errors, rows, prows = [], [], []

    for spec in sets:
        try:
            m = measure_set(spec)
        except Exception as e:                                   # noqa: BLE001
            errors.append("%s 读取失败：%s" % (spec["name"], e))
            continue
        dec = spec.get("declared") or {}
        for key in ("n_items", "n_same", "n_diff", "n_unlabeled"):
            if key in dec and dec[key] != m[key]:
                errors.append("%s.%s 登记 %s ≠ 实测 %s" % (spec["name"], key, dec[key], m[key]))
        lid = "ls_" + hashlib.sha1(("%s|%s" % (spec["name"], m["sha256"])).encode()).hexdigest()[:10]
        rule = dict(m["rule"])
        rule["declared"] = dec
        rule["uri_bytes"] = os.path.getsize(m["abs"])
        conn.execute("""INSERT INTO label_set(id, name, sampling, domain, uri, sha256, n_items, n_same,
                        n_diff, n_unlabeled, annotator, reviewer, rule_json, note, created_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(name, sha256) DO UPDATE SET
                        sampling=excluded.sampling, domain=excluded.domain, uri=excluded.uri,
                        n_items=excluded.n_items, n_same=excluded.n_same, n_diff=excluded.n_diff,
                        n_unlabeled=excluded.n_unlabeled, annotator=excluded.annotator,
                        reviewer=excluded.reviewer, rule_json=excluded.rule_json, note=excluded.note""",
                     (lid, spec["name"], spec.get("sampling", ""), spec.get("domain", ""), spec["uri"],
                      m["sha256"], m["n_items"], m["n_same"], m["n_diff"], m["n_unlabeled"],
                      spec.get("annotator", ""), spec.get("reviewer", ""),
                      json.dumps(rule, ensure_ascii=False), spec.get("note", ""), now()))
        register_asset(conn, spec["uri"], "labels")
        rows.append((spec["name"], m["n_items"], m["n_same"] or 0, m["n_diff"] or 0,
                     m["n_unlabeled"] or 0, m["sha256"][:10]))

    for spec in protos:
        man = os.path.join(ROOT, spec["manifest_uri"])
        root = os.path.join(ROOT, spec.get("root") or os.path.dirname(spec["manifest_uri"]))
        checked = ok = 0
        missing, mismatch = [], []
        if os.path.isfile(man):
            for line in open(man, encoding="utf-8"):
                line = line.strip()
                if not line or " " not in line:
                    continue
                sha, rel = re.split(r"\s+", line, maxsplit=1)
                checked += 1
                fp = os.path.join(root, rel)
                if not os.path.isfile(fp):
                    missing.append(rel)
                elif sha256_file(fp) != sha:
                    mismatch.append(rel)
                else:
                    ok += 1
        else:
            errors.append("%s 清单不存在：%s" % (spec["name"], spec["manifest_uri"]))
        msha = sha256_file(man) if os.path.isfile(man) else None
        pid = "ep_" + hashlib.sha1(("%s|%s" % (spec["name"], msha)).encode()).hexdigest()[:10]
        verify = {"checked": checked, "ok": ok, "missing": missing[:20], "mismatch": mismatch[:20],
                  "n_missing": len(missing), "n_mismatch": len(mismatch)}
        conn.execute("""INSERT INTO eval_protocol(id, name, status, sampling, identity_isolation,
                        params_json, manifest_uri, manifest_sha256, file_count, verify_json, note, created_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(id) DO UPDATE SET status=excluded.status, sampling=excluded.sampling,
                        identity_isolation=excluded.identity_isolation, params_json=excluded.params_json,
                        manifest_uri=excluded.manifest_uri, manifest_sha256=excluded.manifest_sha256,
                        file_count=excluded.file_count, verify_json=excluded.verify_json, note=excluded.note""",
                     (pid, spec["name"], spec.get("status", "recorded"), spec.get("sampling", ""),
                      spec.get("identity_isolation", ""),
                      json.dumps(spec.get("params_json") or {}, ensure_ascii=False),
                      spec["manifest_uri"], msha, checked, json.dumps(verify, ensure_ascii=False),
                      spec.get("note", ""), now()))
        register_asset(conn, spec["manifest_uri"], "doc")
        prows.append((spec["name"], spec.get("status"), checked, ok, len(missing), len(mismatch), pid))

    if commit:
        conn.commit()
    return {"sets": rows, "protocols": prows, "errors": errors}


def _cp(k, n, a=0.05):
    """Clopper-Pearson 精确区间（二项）。与 V2 手算口径一致。"""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def _fmt(x, like):
    """把复算值截到论文写法的小数位，便于逐位比对。"""
    d = len(like.split(".")[1]) if "." in like else 0
    return ("%." + str(d) + "f") % x


def _set_vals(conn, name):
    row = conn.execute("SELECT n_items, n_same, n_diff, n_unlabeled FROM label_set "
                       "WHERE name=?", (name,)).fetchone()
    if not row:
        raise SystemExit("label_set 缺 %s，先跑 truth" % name)
    return dict(zip(("n_items", "n_same", "n_diff", "n_unlabeled"), [x or 0 for x in row]))


def _set_row(conn, name):
    row = conn.execute("SELECT uri, rule_json FROM label_set WHERE name=?", (name,)).fetchone()
    if not row:
        raise SystemExit("label_set 缺 %s，先跑 truth" % name)
    return row


def _sum_sets(conn, names, field):
    """跨 label_set 求和：用于「合计」行（T1 + T2），每个集合各自实测后再相加。"""
    total = 0
    for n in names:
        total += _set_vals(conn, n)[field]
    return [total]


def _set_rule(conn, name, path):
    """取 label_set 的某个字段：先认 rule_json（支持 a.b 与列表下标），再认表列（n_items 等）。"""
    row = _set_row(conn, name)
    if path in ("n_items", "n_same", "n_diff", "n_unlabeled"):
        v = conn.execute("SELECT %s FROM label_set WHERE name=?" % path, (name,)).fetchone()[0]
        return [float(v or 0)]
    cur = json.loads(row["rule_json"] or "{}")
    for part in str(path).split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return [float(cur)]


def _map_stats(file, path=""):
    """扁平 {键: 组名} 映射的分组统计：[条目数, 组数, 组大小中位, 最大组]。"""
    d = json.load(open(os.path.join(ROOT, file), encoding="utf-8"))
    for k in str(path or "").split("."):
        if k:
            d = d[k]
    vals = list(d.values()) if isinstance(d, dict) else list(d)
    cnt = {}
    for v in vals:
        cnt[v] = cnt.get(v, 0) + 1
    sizes = sorted(cnt.values())
    n = len(sizes)
    med = 0 if not n else (sizes[n // 2] if n % 2 else (sizes[n // 2 - 1] + sizes[n // 2]) / 2.0)
    return [len(vals), len(sizes), med, sizes[-1] if sizes else 0]


def _camera_pairs(conn, names, field="pair"):
    """真值条目的机位对并集大小（论文 Total 行的 12）。"""
    seen = set()
    for n in names:
        uri = _set_row(conn, n)["uri"]
        d = json.load(open(os.path.join(ROOT, uri), encoding="utf-8"))
        for it in d["items"]:
            if it.get(field):
                seen.add(str(it[field]))
    return [len(seen)]


def _json_filter_count(file, path, where):
    """按 where 的键值（AND）数条目：用于「41 条异判」「17 判异」这类分解。"""
    d = json.load(open(os.path.join(ROOT, file), encoding="utf-8"))
    for k in str(path or "").split("."):
        if k:
            d = d[k]
    return [sum(1 for it in d
                if all(str(it.get(k)) == str(v) for k, v in (where or {}).items()))]


def _code_const(file, name=None, expr=None):
    """读生成脚本里的常量（TRAIN_DUR=720 等），支持 A, B = 1, 2 与 A-B 形式的派生量。"""
    txt = open(os.path.join(ROOT, file), encoding="utf-8").read()
    consts = {}
    pat = (r"^\s*([A-Z][A-Z0-9_]*(?:\s*,\s*[A-Z][A-Z0-9_]*)*)\s*=\s*"
           r"(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*)\s*$")
    for m in re.finditer(pat, txt, re.M):
        names = [x.strip() for x in m.group(1).split(",")]
        vals = [float(x) for x in m.group(2).split(",")]
        if len(names) == len(vals):
            consts.update(dict(zip(names, vals)))
    if expr:
        parts = re.split(r"([+-])", expr.replace(" ", ""))
        if parts[0] not in consts:
            raise ValueError("%s 里没有常量 %s" % (file, parts[0]))
        total = consts[parts[0]]
        for i in range(1, len(parts), 2):
            if parts[i + 1] not in consts:
                raise ValueError("%s 里没有常量 %s" % (file, parts[i + 1]))
            v = consts[parts[i + 1]]
            total = total + v if parts[i] == "+" else total - v
        return [total]
    if name not in consts:
        raise ValueError("%s 里没有常量 %s" % (file, name))
    return [consts[name]]


def _within(got, exp, tol):
    try:
        return abs(float(got) - float(exp)) <= tol + 1e-12
    except (TypeError, ValueError):
        return False


def _eval_expr(expr, vals):
    """只支持 字段 与 字段±字段；刻意不用 eval。"""
    parts = re.split(r"([+-])", expr.replace(" ", ""))
    total = vals[parts[0]]
    for i in range(1, len(parts), 2):
        sign, fld = parts[i], parts[i + 1]
        total = total + vals[fld] if sign == "+" else total - vals[fld]
    return total


def _ordered_nums(s):
    """按出现顺序取数字（用于 tex 行逐列比对，不能像 nums() 那样丢顺序）。"""
    return [m.replace(",", "") for m in re.findall(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", s)]


def _tex_row_numbers(file, label, row, skip_first=0):
    for t in load_tex_tables():
        if t["file"] == file and (t["label"] == label or
                                 (not t["label"] and t["caption"].startswith(label))):
            if row >= len(t["rows"]):
                raise ValueError("tex 表 %s 只有 %d 行" % (label, len(t["rows"])))
            raw = " ".join(t["rows"][row][skip_first:])
            raw = re.sub(r"\d{2,3}--\d{2,3}", " ", raw)      # 机位对标签先去掉，免把 320 读成 -320
            raw = re.sub(r"\d{2}-\d{2}", " ", raw)          # 日期标签（10-22）
            raw = raw.replace("$-$", "-").replace("\u2212", "-")   # 论文负号写法
            return [n.replace(",", "") for n in re.findall(
                r"(?<![\w.\-])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", raw)]
    raise ValueError("tex 表未找到 %s %s" % (file, label))


def _gate_run(file, mode):
    """按 mode 读闸门实验的一次运行，返回与 tab:gate 列序一致的数字。"""
    d = json.load(open(os.path.join(ROOT, file), encoding="utf-8"))
    for r in d["runs"]:
        if r["mode"] == mode:
            return [r["n_after"], r["merged"], r["pos_hit"], r["pos_n"],
                    r["neg_hit"], r["neg_n"], r["good"], r["judged"], r["prec"]]
    raise ValueError("%s 无 mode=%s" % (file, mode))


def _run_by_mode(file, mode, fields, extra=None):
    """按 mode 读一次运行，按 fields 顺序取值。兼容两种布局：{runs:[{mode:...}]} 与 {mode:{...}}。"""
    d = json.load(open(os.path.join(ROOT, file), encoding="utf-8"))
    rec = None
    if isinstance(d, dict) and isinstance(d.get("runs"), list):
        rec = next((r for r in d["runs"] if r.get("mode") == mode), None)
    elif isinstance(d, dict):
        rec = d.get(mode)
    if rec is None:
        raise ValueError("%s 无 mode=%s" % (file, mode))
    miss = [f for f in fields if f not in rec]
    if miss:
        raise ValueError("%s mode=%s 缺字段 %s" % (file, mode, ",".join(miss)))
    return [rec[f] for f in fields] + list(extra or [])


KNOWN_KINDS = {"prop", "set_expr", "matrix_row", "gate_run", "gate_field",
               "run_fields", "pool_cell", "pack", "graph", "text_value",
               "json_count", "constraints", "auc_block", "auc_pair", "unverifiable",
               "sum_sets", "set_rule", "camera_pairs", "json_filter_count", "code_const",
               "map_stats"}
NEED_EXPECT = {"prop", "set_expr", "gate_field", "text_value", "json_count", "constraints",
               "sum_sets", "set_rule", "camera_pairs", "json_filter_count", "code_const",
               "map_stats"}


def _text_value(file, label, col=-1):
    """从存证文本里取一行，按 label 前缀匹配，返回第 col 个数字。用于“重跑脚本、输出存档”的复算。"""
    p = os.path.join(ROOT, file)
    for ln in open(p, encoding="utf-8"):
        if ln.strip().startswith(label):
            nums = re.findall(r"\d+(?:\.\d+)?", ln)
            if nums:
                return [float(nums[col])]
    raise ValueError("%s 里没有以「%s」开头的行" % (file, label))


def _json_count(file, path):
    d = json.load(open(os.path.join(ROOT, file), encoding="utf-8"))
    for k in path.split("."):
        d = d[k]
    return [len(d)]


def _constraints(gt_file, graph_file):
    """L3 约束遵守：判异对必须不同组、判同对必须同组。返回 [负对通过, 负对总数, 正对通过, 正对总数]。"""
    items = json.load(open(os.path.join(ROOT, gt_file), encoding="utf-8"))["items"]
    groups = json.load(open(os.path.join(ROOT, graph_file), encoding="utf-8"))["groups"]
    where = {}
    for gi, gr in enumerate(groups):
        for cam, tid in gr:
            where["%s:%s" % (cam, tid)] = gi
    neg = [0, 0]
    pos = [0, 0]
    for it in items:
        a, b, lab = it["a"], it["b"], it["label"]
        same = where.get(a) is not None and where.get(a) == where.get(b)
        bucket = neg if int(lab) == 0 else pos
        bucket[1] += 1
        bucket[0] += int((not same) if int(lab) == 0 else same)
    return [neg[0], neg[1], pos[0], pos[1]]


def _auc_artifact(file):
    return open(os.path.join(ROOT, file), encoding="utf-8").read().split("\n")


def _auc_block(file, block):
    """tab:agg 一行：<block> 跨机位 同人 n=.. 中位 a | 异人 n=.. 中位 b | AUC c [lo,hi]。"""
    for ln in _auc_artifact(file):
        s = ln.strip()
        if re.match(r"^%s\s+跨机位" % re.escape(block), s):
            m1 = re.search(r"同人 n=\s*\d+ 中位\s*(-?\d+\.\d+)", s)
            m2 = re.search(r"异人 n=\s*\d+ 中位\s*(-?\d+\.\d+)", s)
            m3 = re.search(r"AUC\s*(-?\d+\.\d+)\s*\[(-?\d+\.\d+),\s*(-?\d+\.\d+)\]", s)
            if not (m1 and m2 and m3):
                raise ValueError("解析不了：%s" % s[:80])
            return [float(m1.group(1)), float(m2.group(1)),
                    float(m3.group(1)), float(m3.group(2)), float(m3.group(3))]
    raise ValueError("%s 无 %s 行" % (file, block))


def _auc_pair(file, block, pair):
    """tab:pair 一行：  <pair>  同  异  同人中位 异人中位  AUC。"""
    inside, out = False, None
    for ln in _auc_artifact(file):
        s = ln.strip()
        if re.match(r"^%s\s+跨机位" % re.escape(block), s):
            inside = True
            continue
        if inside and re.match(r"^\S+\s+跨机位", s):
            break
        if inside and s.startswith(pair):
            parts = s.split()
            out = [int(parts[1]), int(parts[2]), float(parts[5])]
            break
    if out is None:
        raise ValueError("%s 的 %s 块里没有机位对 %s" % (file, block, pair))
    return out


def _pack_metric(metric, subdir=os.path.join("webapp", "backend", "data", "tracks")):
    """D1 采集统计：直接读每机位 tracks 元数据，不依赖任何汇总产物。"""
    files = sorted(glob.glob(os.path.join(ROOT, subdir, "*.json")))
    if not files:
        raise ValueError("找不到 %s/*.json" % subdir)
    recs = [json.load(open(p, encoding="utf-8")) for p in files]
    if metric == "cams":
        return [len(recs)] + sorted(int(r["cam"]) for r in recs)
    if metric == "resolution":
        cnt = {}
        for r in recs:
            cnt[(int(r["w"]), int(r["h"]))] = cnt.get((int(r["w"]), int(r["h"])), 0) + 1
        out = []
        for (w, h), c in sorted(cnt.items(), key=lambda kv: -kv[1]):
            out += [c, w, h]
        return out
    if metric == "duration":
        dur = [r["nframes"] / float(r["fps"]) for r in recs]
        fps = sorted(round(float(r["fps"])) for r in recs)
        return [round(sum(dur) / len(dur)), fps[len(fps) // 2]]
    if metric == "tracks":
        return [sum(len(r["tracks"]) for r in recs)]
    raise ValueError("未知 pack metric %s" % metric)


def _graph_metric(file, metric):
    """身份图统计：读 groups（成员为 [cam, tid]）。"""
    d = json.load(open(os.path.join(ROOT, file), encoding="utf-8"))
    g = d["groups"]
    if metric == "groups":
        return [len(g)]
    if metric == "members":
        return [sum(len(x) for x in g)]
    if metric == "crosscam":
        return [sum(1 for x in g if len({str(c) for c, _ in x}) > 1)]
    if metric == "largest":
        return [max(len(x) for x in g)]
    raise ValueError("未知 graph metric %s" % metric)


def _pool_cell(file, tau, key, pool_field=None, extra=None):
    """选择器表：table[tau][key] = [k, n] → 召回百分比；pool_field 取 stat 里的池大小。"""
    d = json.load(open(os.path.join(ROOT, file), encoding="utf-8"))
    tb = (d.get("table") or {}).get(tau) or {}
    if key not in tb:
        raise ValueError("%s tau=%s 无 %s" % (file, tau, key))
    k, n = tb[key]
    out = [100.0 * k / n]
    if pool_field:
        out.append(d["stat"][pool_field])
    out.extend(extra or [])
    return out


def validate_specs(specs):
    """启动时校验配方：未知 kind、缺 expect、缺字段、重复登记 —— 全部一次报清。"""
    errs, seen = [], {}
    for i, spec in enumerate(specs, 1):
        tag = spec.get("claim") or ("#%d" % i)
        if not spec.get("claim"):
            errs.append("#%d 缺 claim" % i)
        if not spec.get("checks"):
            errs.append("%s 没有 checks" % tag)
        seen.setdefault(tag, []).append(i)
        for j, chk in enumerate(spec.get("checks") or [], 1):
            k = chk.get("kind")
            if k not in KNOWN_KINDS:
                errs.append("%s check#%d 未知 kind %r" % (tag, j, k))
                continue
            if k in NEED_EXPECT and chk.get("expect") in (None, "", []):
                errs.append("%s check#%d 缺 expect" % (tag, j))
            if k == "prop" and not ("k" in chk and "n" in chk):
                errs.append("%s check#%d prop 缺 k/n" % (tag, j))
            if k == "run_fields" and not chk.get("fields"):
                errs.append("%s check#%d run_fields 缺 fields" % (tag, j))
            if k in ("gate_run", "run_fields", "matrix_row", "pool_cell", "pack", "graph",
                     "auc_block", "auc_pair") and not chk.get("row"):
                errs.append("%s check#%d 缺 tex row" % (tag, j))
            if k == "auc_block" and not chk.get("block"):
                errs.append("%s check#%d auc_block 缺 block" % (tag, j))
            if k == "auc_pair" and not chk.get("pair"):
                errs.append("%s check#%d auc_pair 缺 pair" % (tag, j))
            if k == "pool_cell" and not chk.get("key"):
                errs.append("%s check#%d pool_cell 缺 key" % (tag, j))
            if k in ("pack", "graph") and not chk.get("metric"):
                errs.append("%s check#%d %s 缺 metric" % (tag, j, k))
            if k in ("sum_sets", "camera_pairs") and not (isinstance(chk.get("sets"), list)
                                                          and len(chk["sets"]) >= 2):
                errs.append("%s check#%d %s 需要 ≥2 个 sets" % (tag, j, k))
            if k == "sum_sets" and not chk.get("field"):
                errs.append("%s check#%d sum_sets 缺 field" % (tag, j))
            if k == "set_rule" and not (chk.get("set") and chk.get("path")):
                errs.append("%s check#%d set_rule 缺 set/path" % (tag, j))
            if k == "json_filter_count" and not chk.get("file"):
                errs.append("%s check#%d json_filter_count 缺 file" % (tag, j))
            if k == "code_const" and not (chk.get("name") or chk.get("expr")):
                errs.append("%s check#%d code_const 缺 name/expr" % (tag, j))
            if k == "map_stats" and not chk.get("file"):
                errs.append("%s check#%d map_stats 缺 file" % (tag, j))
    for tag, where in seen.items():
        if len(where) > 1:
            errs.append("%s 重复登记 %d 次（#%s）→ 后一条会覆盖前一条裁定"
                        % (tag, len(where), ",".join(map(str, where))))
    return errs


def _matrix_cell(cell, variant, theta):
    p = os.path.join(ROOT, "tmp", "e21_xday", "matrix.json")
    d = json.load(open(p, encoding="utf-8"))["metrics"]
    key = "%s|%s|%s" % (variant, cell, theta)
    if key not in d:
        raise ValueError("matrix.json 无键 %s" % key)
    v = d[key]
    return [v["coverage"], v["same"], v["fm"] * 100, v["ev"]]


def _check_inputs(chk, conn):
    """一条检查实际读到的文件（写进 eval_run.inputs_json）。tex 文件不算输入。"""
    outs = []
    for key in ("file", "gt", "graph"):
        v = chk.get(key)
        if v and not str(v).endswith(".tex"):
            outs.append(str(v))
    if chk.get("kind") in ("set_expr", "sum_sets", "set_rule"):
        names = ([chk["set"]] if chk.get("set") else []) + list(chk.get("sets") or [])
        for n in names:
            row = conn.execute("SELECT uri FROM label_set WHERE name=?", (n,)).fetchone()
            if row:
                outs.append(row[0])
    seen, uniq = set(), []
    for p in outs:
        p = p.replace("\\", "/")
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _code_version(conn):
    """把执行器与两份配方文件的 sha256 冻结成一个 code_version，返回其 id。"""
    files = []
    for rel in ("tools/ops_claims.py", "ops/claims/verify_specs.json", "ops/claims/label_sets.json"):
        p = os.path.join(ROOT, rel)
        files.append({"path": rel, "sha256": sha256_file(p) if os.path.isfile(p) else None})
    sha = hashlib.sha1(json.dumps(files, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cid = "cv_" + sha[:10]
    conn.execute("INSERT OR IGNORE INTO code_version(id, sha1, files_json, as_is, created_at) "
                 "VALUES(?,?,?,1,?)",
                 (cid, sha, json.dumps(files, ensure_ascii=False), now()))
    return cid, sha


def do_verify(conn, commit=True):
    """按 ops/claims/verify_specs.json 逐条复算，写 verdict / recomputed_text。不改任何论文文件。

    每次复算同时落一行 eval_run（输入文件 sha256、退出码、code_version）：
    verdict=match 只允许来自 exit_code=0 的 run（设计稿 §5 约束）。
    """
    specs = json.load(open(VERIFY_PATH, encoding="utf-8"))["specs"]
    bad_specs = validate_specs(specs)
    if bad_specs:
        raise SystemExit("verify_specs.json 有 %d 处结构问题，先修再跑：\n  - %s"
                         % (len(bad_specs), "\n  - ".join(bad_specs)))
    cv_id, cv_sha = _code_version(conn)
    tool_sha = sha256_file(os.path.join(ROOT, "tools", "ops_claims.py"))
    rows, errors = [], []
    for spec in specs:
        did = spec["claim"]
        hit = conn.execute("SELECT id, value_text FROM paper_claim WHERE id=? OR display_id=?",
                           (did, did)).fetchone()
        if not hit:
            errors.append("%s 不在 paper_claim" % did)
            continue
        cid, vtext = hit
        parts, bad, unver = [], [], []
        inputs, _seen = [], set()
        for chk in spec["checks"]:
            for p in _check_inputs(chk, conn):
                if p not in _seen:
                    _seen.add(p)
                    full = os.path.join(ROOT, p)
                    inputs.append({"path": p, "role": "read",
                                   "sha256": sha256_file(full) if os.path.isfile(full) else None})
        _prov = conn.execute("SELECT provenance_json FROM paper_claim WHERE id=?", (cid,)).fetchone()
        if _prov and _prov[0]:
            for p in (json.loads(_prov[0]).get("paths") or []):   # 声明来源，未直接读取
                p = str(p).replace("\\", "/")
                if p in _seen:
                    continue
                _seen.add(p)
                full = os.path.join(ROOT, p)
                inputs.append({"path": p, "role": "declared",
                               "sha256": sha256_file(full) if os.path.isfile(full) else None})
        for p in (spec.get("inputs") or []):                  # 配方显式声明的输入（provenance 为空时用）
            p = str(p).replace("\\", "/")
            if p in _seen:
                continue
            _seen.add(p)
            full = os.path.join(ROOT, p)
            inputs.append({"path": p, "role": "declared",
                           "sha256": sha256_file(full) if os.path.isfile(full) else None})
        for chk in spec["checks"]:
            raw_exp = chk.get("expect")
            exp = raw_exp if isinstance(raw_exp, list) else ([raw_exp] if raw_exp else [])
            try:
                if chk["kind"] == "unverifiable":
                    unver.append(chk.get("reason", ""))
                    continue
                if chk["kind"] == "prop":
                    k, n = int(chk["k"]), int(chk["n"])
                    p = k / n
                    lo, hi = _cp(k, n)
                    scale = 100.0 if chk.get("percent") else 1.0
                    cand = [p * scale, lo * scale, hi * scale][:len(exp)]
                    got = [_fmt(c, e) for c, e in zip(cand, exp)]
                    parts.append("%d/%d = %.4f, CP[%.4f, %.4f]%s" % (
                        k, n, p, lo, hi, "（×100）" if chk.get("percent") else ""))
                elif chk["kind"] == "set_expr":
                    v = _eval_expr(chk["expr"], _set_vals(conn, chk["set"]))
                    got = [_fmt(v, exp[0])]
                    parts.append("%s.%s = %s" % (chk["set"], chk["expr"], got[0]))
                elif chk["kind"] == "matrix_row":
                    raw = _matrix_cell(chk["cell"], chk.get("variant", "剔除无效段"),
                                       chk.get("theta", "0.9"))
                    exp = _tex_row_numbers(chk["file"], chk["label"], int(chk["row"]))
                    if len(exp) != len(raw):
                        raise ValueError("tex 行 %d 个数、matrix 给 %d 个" % (len(exp), len(raw)))
                    got = [_fmt(g, e) for g, e in zip(raw, exp)]
                    parts.append("%s: %s" % (chk["cell"][5:], ", ".join(got)))
                elif chk["kind"] == "gate_field":
                    raw = _gate_run(chk["file"], chk["mode"])
                    got = [_fmt(raw[int(chk["idx"])], exp[0])]
                    parts.append("%s[%d] = %s" % (chk["mode"], chk["idx"], got[0]))
                elif chk["kind"] in ("auc_block", "auc_pair"):
                    if chk["kind"] == "auc_block":
                        raw = _auc_block(chk["file"], chk["block"])
                    else:
                        raw = _auc_pair(chk["file"], chk.get("block", "osnet"), chk["pair"])
                    tex = _tex_row_numbers(chk["file_tex"], chk["label"], int(chk["row"]),
                                           skip_first=int(chk.get("skip_first", 1)))
                    pick = chk.get("pick") or list(range(len(raw)))
                    raw = [raw[i] for i in pick]
                    if chk.get("tex_idx"):
                        exp = [tex[i] for i in chk["tex_idx"]]
                    else:
                        exp = tex
                    if len(exp) != len(raw):
                        raise ValueError("tex 定位 %d 个数、auc 给 %d 个" % (len(exp), len(raw)))
                    got = [_fmt(g, e) for g, e in zip(raw, exp)]
                    parts.append(", ".join(got))
                elif chk["kind"] == "text_value":
                    raw = _text_value(chk["file"], chk["label"], int(chk.get("col", -1)))
                    if len(exp) != len(raw):
                        raise ValueError("expect %d 个、text 给 %d 个" % (len(exp), len(raw)))
                    got = [_fmt(g, e) for g, e in zip(raw, exp)]
                    parts.append("%s = %s" % (chk["label"], got[0]))
                elif chk["kind"] == "json_count":
                    raw = _json_count(chk["file"], chk["path"])
                    got = [_fmt(g, e) for g, e in zip(raw, exp)]
                    parts.append("%s.%s = %s" % (os.path.basename(chk["file"]), chk["path"], got[0]))
                elif chk["kind"] == "constraints":
                    raw = _constraints(chk["gt"], chk["graph"])
                    if len(exp) != len(raw):
                        raise ValueError("expect %d 个、constraints 给 %d 个" % (len(exp), len(raw)))
                    got = [_fmt(g, e) for g, e in zip(raw, exp)]
                    parts.append("负 %s/%s、正 %s/%s" % (got[0], got[1], got[2], got[3]))
                elif chk["kind"] == "sum_sets":
                    raw = _sum_sets(conn, chk["sets"], chk["field"])
                    got = [_fmt(g, exp[0]) for g in raw]
                    parts.append("%s.%s = %s" % ("+".join(chk["sets"]), chk["field"], got[0]))
                elif chk["kind"] == "set_rule":
                    raw = _set_rule(conn, chk["set"], chk["path"])
                    got = [_fmt(g, exp[0]) for g in raw]
                    parts.append("%s.%s = %s" % (chk["set"], chk["path"], got[0]))
                elif chk["kind"] == "camera_pairs":
                    raw = _camera_pairs(conn, chk["sets"], chk.get("field", "pair"))
                    got = [_fmt(g, exp[0]) for g in raw]
                    parts.append("机位对并集 = %s" % got[0])
                elif chk["kind"] == "json_filter_count":
                    raw = _json_filter_count(chk["file"], chk.get("path", "items"), chk.get("where"))
                    got = [_fmt(g, exp[0]) for g in raw]
                    parts.append("%s = %s" % (chk.get("where") or chk.get("path"), got[0]))
                elif chk["kind"] == "code_const":
                    raw = _code_const(chk["file"], chk.get("name"), chk.get("expr"))
                    got = [_fmt(g, exp[0]) for g in raw]
                    parts.append("%s = %s" % (chk.get("expr") or chk.get("name"), got[0]))
                elif chk["kind"] == "map_stats":
                    raw = _map_stats(chk["file"], chk.get("path"))
                    if len(exp) != len(raw):
                        raise ValueError("expect %d 个、map_stats 给 %d 个" % (len(exp), len(raw)))
                    got = [_fmt(g, e) for g, e in zip(raw, exp)]
                    parts.append("%s: %s" % (os.path.basename(chk["file"]), ", ".join(got)))
                elif chk["kind"] in ("pack", "graph"):
                    if chk["kind"] == "pack":
                        raw = _pack_metric(chk["metric"], chk.get("subdir") or
                                           os.path.join("webapp", "backend", "data", "tracks"))
                    else:
                        raw = _graph_metric(chk["file"], chk["metric"])
                    exp = _tex_row_numbers(chk["file_tex"], chk["label"], int(chk["row"]),
                                           skip_first=int(chk.get("skip_first", 1)))
                    if len(exp) != len(raw):
                        raise ValueError("tex 行 %d 个数、%s 给 %d 个" % (len(exp), chk["metric"], len(raw)))
                    got = [_fmt(g, e) for g, e in zip(raw, exp)]
                    parts.append("%s: %s" % (chk["metric"], ", ".join(got)))
                elif chk["kind"] == "pool_cell":
                    raw = _pool_cell(chk["file"], chk.get("tau", "0.8"), chk["key"],
                                     chk.get("pool"), chk.get("extra"))
                    exp = _tex_row_numbers(chk["file_tex"], chk["label"], int(chk["row"]),
                                           skip_first=int(chk.get("skip_first", 1)))
                    if len(exp) != len(raw):
                        raise ValueError("tex 行 %d 个数、pool 给 %d 个" % (len(exp), len(raw)))
                    got = [_fmt(g, e) for g, e in zip(raw, exp)]
                    parts.append("%s: %s" % (chk["key"], ", ".join(got)))
                elif chk["kind"] == "run_fields":
                    raw = _run_by_mode(chk["file"], chk["mode"], chk["fields"],
                                       chk.get("extra"))
                    exp = _tex_row_numbers(chk["file_tex"], chk["label"], int(chk["row"]),
                                           skip_first=int(chk.get("skip_first", 1)))
                    if len(exp) != len(raw):
                        raise ValueError("tex 行 %d 个数、run 给 %d 个" % (len(exp), len(raw)))
                    got = [_fmt(g, e) for g, e in zip(raw, exp)]
                    parts.append("%s: %s" % (chk["mode"], ", ".join(got)))
                elif chk["kind"] == "gate_run":
                    raw = _gate_run(chk["file"], chk["mode"])
                    exp = _tex_row_numbers(chk["file_tex"], chk["label"], int(chk["row"]),
                                           skip_first=1)
                    if len(exp) != len(raw):
                        raise ValueError("tex 行 %d 个数、gate 给 %d 个" % (len(exp), len(raw)))
                    got = [_fmt(g, e) for g, e in zip(raw, exp)]
                    parts.append("%s: %s" % (chk["mode"], ", ".join(got)))
                else:
                    raise ValueError("未知 check kind %s" % chk["kind"])
            except Exception as e:                                   # noqa: BLE001
                errors.append("%s: %s" % (did, e))
                bad.append(str(e))
                continue
            for g, e in zip(got, exp):
                if g != e:
                    if chk.get("tol") and _within(g, e, float(chk["tol"])):
                        continue
                    bad.append("期望 %s，复算 %s" % (e, g))
        if bad:
            verdict = "mismatch"
        elif unver:
            verdict = "unverifiable"
        else:
            verdict = "match"
        reason = spec.get("why", "")
        if bad:
            reason += "｜不一致：" + "；".join(bad)
        if unver:
            reason += "｜不可复算：" + "；".join(unver)
        exit_code = 0 if not bad else 1
        run_key = hashlib.sha1(json.dumps([did, spec["checks"]], sort_keys=True,
                                          ensure_ascii=False).encode()).hexdigest()[:10]
        er_id = "er_" + run_key
        conn.execute("""INSERT INTO eval_run(id, claim_ids_json, script, script_sha1, as_is, args_json,
                        cwd, started_at, ended_at, exit_code, code_version_id, inputs_json,
                        outputs_json, env_json, created_at)
                        VALUES(?,?,?,?,1,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(id) DO UPDATE SET ended_at=excluded.ended_at,
                        exit_code=excluded.exit_code, outputs_json=excluded.outputs_json,
                        inputs_json=excluded.inputs_json, code_version_id=excluded.code_version_id""",
                     (er_id, json.dumps([cid]), "tools/ops_claims.py:verify", tool_sha,
                      json.dumps(spec["checks"], ensure_ascii=False), ROOT, now(), now(),
                      exit_code, cv_id, json.dumps(inputs, ensure_ascii=False),
                      json.dumps({"recomputed": "；".join(parts), "verdict": verdict},
                                 ensure_ascii=False),
                      json.dumps({"python": sys.version.split()[0], "tool_sha256": tool_sha,
                                  "code_version": cv_sha}), now()))
        if verdict == "match" and exit_code != 0:            # 设计稿 §5 约束兜底
            verdict = "mismatch"
            reason += "｜无成功的 eval_run"
        conn.execute("UPDATE claim SET verdict=?, recomputed_text=?, reason=?, eval_run_id=?, "
                     "updated_at=? WHERE id=?",
                     (verdict, "；".join(parts), reason, er_id, now(), cid))
        rows.append((did, (vtext or "")[:30], "；".join(parts), verdict, "；".join(bad)[:60]))
    if commit:
        conn.commit()
    return {"rows": rows, "errors": errors}


def do_migrate_ids(conn):
    """把锚点与复算配方里的 display_id 换成 hash id：display_id 随 EVIDENCE 增行位移，hash id 不会。"""
    claims, _, _ = parsed_claims()
    by_disp = {c["display_id"]: c["id"] for c in claims}

    a = json.load(open(ANCHOR_PATH, encoding="utf-8"))
    n_a, bad = 0, []
    for t in a["tables"]:
        key = t.get("label") or t.get("caption", "")
        for r in t["rows"]:
            out = []
            for tok in r["claims"]:
                if tok.startswith("pc_"):
                    out.append(tok)
                    continue
                cid = by_disp.get(tok)
                if not cid:
                    bad.append("%s r%s: %s" % (key[:16], r["i"], tok))
                    out.append(tok)
                else:
                    out.append(cid)
                    n_a += 1
            r["claims"] = out
    a["_id_note"] = ("claims 存 hash id（pc_*），display_id 只是人读标签；"
                     "EVIDENCE 改了某行文本会让该行 hash id 变化，从而在锚定时报错而不是静默指错")
    a["_id_migrated_at"] = now()
    json.dump(a, open(ANCHOR_PATH, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=2)

    v = json.load(open(VERIFY_PATH, encoding="utf-8"))
    n_v = 0
    for spec in v["specs"]:
        tok = spec["claim"]
        if tok.startswith("pc_"):
            continue
        cid = by_disp.get(tok)
        if not cid:
            bad.append("spec %s" % tok)
            continue
        spec["_was_display"] = tok
        spec["claim"] = cid
        n_v += 1
    v["_id_note"] = "claim 存 hash id；_was_display 是迁移时的 display_id，仅供人读"
    json.dump(v, open(VERIFY_PATH, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=2)
    return {"anchor_ids": n_a, "spec_ids": n_v, "unresolved": bad}


def do_doctor(conn, max_bytes=200 * 1024 * 1024):
    """体检（只读）：登记过的文件是否还在、内容是否变、run 的输入是否漂移、代码版本是否过期。

    不写库；结果同时落到 ops/reports/paper_claims_v1/DOCTOR.md。
    """
    assets = conn.execute("SELECT path, sha256, bytes, kind, first_seen_at FROM asset_version "
                          "ORDER BY path, first_seen_at").fetchall()
    by_path = {}
    for path, sha, nbytes, kind, seen in assets:
        by_path.setdefault(path, []).append((sha, nbytes, kind, seen))
    a_missing, a_drift, a_large, a_ok, a_hist = [], [], [], 0, 0
    for path, vers in by_path.items():
        sha, nbytes, kind, _seen = vers[-1]          # 最新一行才是当前期望值，旧行是历史
        a_hist += max(0, len(vers) - 1)
        full = os.path.join(ROOT, path)
        if not os.path.isfile(full):
            a_missing.append((path, kind, (sha or "")[:10]))
            continue
        if nbytes and nbytes > max_bytes:
            a_large.append((path, kind, nbytes))
            continue
        if sha and sha256_file(full) != sha:
            a_drift.append((path, kind, sha[:10], sha256_file(full)[:10]))
        else:
            a_ok += 1

    run_paths = {}
    for rid, inputs in conn.execute("SELECT id, inputs_json FROM eval_run"):
        for it in json.loads(inputs or "[]"):
            p, rec = it.get("path"), it.get("sha256")
            d = run_paths.setdefault(p, {"n": 0, "sha": rec, "null": rec is None})
            d["n"] += 1
    r_missing, r_drift, r_null = [], [], []
    for p, d in sorted(run_paths.items()):
        full = os.path.join(ROOT, p)
        if d["null"]:
            r_null.append((p, d["n"]))
        elif not os.path.isfile(full):
            r_missing.append((p, d["n"]))
        elif os.path.getsize(full) <= max_bytes and sha256_file(full) != d["sha"]:
            r_drift.append((p, d["n"]))

    cv_stale = []
    for cid, sha1, files in conn.execute("SELECT id, sha1, files_json FROM code_version"):
        cur = []
        for f in json.loads(files or "[]"):
            full = os.path.join(ROOT, f["path"])
            cur.append({"path": f["path"],
                        "sha256": sha256_file(full) if os.path.isfile(full) else None})
        now_sha = hashlib.sha1(json.dumps(cur, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if now_sha != sha1:
            cv_stale.append((cid, sha1[:10], now_sha[:10]))

    L = ["# 台账体检（doctor）\n",
         "> 只读检查：登记过的文件是否还在、内容是否变、run 的输入是否漂移、代码版本是否过期。",
         "> 大文件（>%d MB）跳过哈希，只验存在性。\n" % (max_bytes // (1024 * 1024)),
         "| 检查项 | 结果 |", "|---|---|",
         "| `asset_version` 行数 | %d 行 / %d 个路径（当前一致 %d、缺失 %d、内容变化 %d、跳过 %d；历史版本 %d） |" % (
             len(assets), len(by_path), a_ok, len(a_missing), len(a_drift), len(a_large), a_hist),
         "| run 引用的唯一路径 | %d（缺失 %d、内容变化 %d、登记时已缺失 %d） |" % (
             len(run_paths), len(r_missing), len(r_drift), len(r_null)),
         "| `code_version` 过期 | %d / %d |" % (len(cv_stale), conn.execute(
             "SELECT COUNT(*) FROM code_version").fetchone()[0])]
    for title, rows, fmt in (
            ("资产缺失（登记过、现在找不到）", a_missing, lambda r: "| `%s` | %s | %s |" % r),
            ("资产内容变化（sha256 不一致）", a_drift, lambda r: "| `%s` | %s | %s → %s |" % r),
            ("资产跳过大文件", a_large, lambda r: "| `%s` | %s | %d B |" % r),
            ("run 输入缺失", r_missing, lambda r: "| `%s` | %d 条 run |" % r),
            ("run 输入内容变化", r_drift, lambda r: "| `%s` | %d 条 run |" % r),
            ("run 登记时已缺失", r_null, lambda r: "| `%s` | %d 条 run |" % r)):
        if not rows:
            continue
        L += ["", "## %s（%d）\n" % (title, len(rows)), "| 路径 | 类型/条数 | 详情 |", "|---|---|---|"]
        L += [fmt(r) for r in rows[:40]]
        if len(rows) > 40:
            L.append("| … | 另 %d 条 | |" % (len(rows) - 40))
    if cv_stale:
        L += ["", "## code_version 与当前文件不一致（%d）\n" % len(cv_stale),
              "| id | 登记 sha1 | 当前 sha1 |", "|---|---|---|"]
        L += ["| `%s` | %s | %s |" % r for r in cv_stale]

    path = os.path.join(REPORT_DIR, "DOCTOR.md")
    os.makedirs(REPORT_DIR, exist_ok=True)
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    return {"assets": len(assets), "asset_ok": a_ok, "asset_missing": a_missing,
            "asset_drift": a_drift, "asset_large": a_large,
            "run_paths": len(run_paths), "run_missing": r_missing, "run_drift": r_drift,
            "run_null": r_null, "cv_stale": cv_stale, "path": path}


def do_report(conn, rep):
    os.makedirs(REPORT_DIR, exist_ok=True)
    rows = conn.execute("""SELECT verdict, source_level, COUNT(*) FROM paper_claim
                           GROUP BY 1,2""").fetchall()
    total = conn.execute("SELECT COUNT(*) FROM paper_claim").fetchone()[0]
    kc = dict(conn.execute("SELECT kind, COUNT(*) FROM paper_claim GROUP BY 1").fetchall())
    anchored = conn.execute("SELECT COUNT(*) FROM paper_claim WHERE tex_anchor_json IS NOT NULL").fetchone()[0]
    av = conn.execute("SELECT kind, COUNT(*), SUM(weak) FROM asset_version GROUP BY 1").fetchall()
    gaps = rep["gaps"]
    true_gap = sum(len(g["true_gap"]) for g in gaps)
    anchor_gap = sum(len(g["anchor_gap"]) for g in gaps)
    L = ["# 论文数字台账（V1 登记锚定 · V2 真值与协议）\n",
         "> 由 `tools/ops_claims.py` 生成；权威是 `docs/paper/EVIDENCE.md` 与 tex，本报告只反映索引状态。\n",
         "## 1. claim 与锚定\n",
         "| 项目 | 数值 |\n|---|---|",
         "| `paper_claim` 行数 | %d（表格行 %d / 说明行 %d） |" % (total, kc.get("table", 0), kc.get("prose", 0)),
         "| 已挂 tex 锚点的 claim | %d（%d 个表格行） |" % (anchored, rep["cells"]),
         "| 登记的文件版本 `asset_version` | %d |" % sum(a[1] for a in av),
         "| 表格数字中 EVIDENCE 全文没有的 | **%d** |" % true_gap,
         "| 表格数字中有证据但本行未挂到的 | %d |" % anchor_gap,
         "| 整行声明“无证据行”：%d 行 / %d 个数字 | **合计需补 %d 个** |" % (
             len(rep.get("declared") or []), sum(len(d["numbers"]) for d in (rep.get("declared") or [])),
             true_gap + sum(len(d["numbers"]) for d in (rep.get("declared") or []))),
         "", "## 2. 按来源等级与裁定\n",
         "| 裁定 | 来源等级 | 条数 |\n|---|---|---:|"]
    L += ["| %s | %s | %d |" % (v, lv, n) for v, lv, n in rows]
    L += ["", "## 3. 文件版本分布\n", "| kind | 条数 | 其中弱身份 |\n|---|---:|---:|"]
    L += ["| %s | %d | %d |" % (k, n, w or 0) for k, n, w in av]
    L += ["", "## 4. 表格数字里 EVIDENCE 没有的（V2 要补行）\n",
          "| 文件 | 表 | 行 | 数字 |\n|---|---|---|---|"]
    for g in gaps:
        if g["true_gap"]:
            L.append("| %s | %s | %s | %s |" % (g["file"], g["table"], g["key"] or ("r%d" % g["row"]),
                                               " ".join(g["true_gap"])))
    L += ["", "## 5. 有证据但本行未挂到的（改锚点即可）\n",
          "| 文件 | 表 | 行 | 数字 |\n|---|---|---|---|"]
    for g in gaps:
        if g["anchor_gap"]:
            L.append("| %s | %s | %s | %s |" % (g["file"], g["table"], g["key"] or ("r%d" % g["row"]),
                                               " ".join(g["anchor_gap"])))
    L += ["", "## 6. 整行声明“无证据行”的（补 EVIDENCE 或改口径说明）\n",
          "| 文件 | 表 | 行 | 数字 |\n|---|---|---|---|"]
    for d in (rep.get("declared") or []):
        L.append("| %s | %s | %s | %s |" % (d["file"], d["table"], d["key"] or ("r%d" % d["row"]),
                                           " ".join(d["numbers"][:14]) + (" …" if len(d["numbers"]) > 14 else "")))
    if rep["errors"]:
        L += ["", "## 5. 锚定错误\n"] + ["- %s" % e for e in rep["errors"]]
    if rep.get("warn"):
        L += ["", "> ⚠ %s" % rep["warn"]]

    ls_rows = conn.execute("""SELECT name, n_items, n_same, n_diff, n_unlabeled, sha256, uri, sampling
                              FROM label_set ORDER BY name""").fetchall()
    if ls_rows:
        L += ["", "## 7. 真值集（V2）\n",
              "> 条数由 `ops_claims truth` 打开文件实测；与登记值不符会在导入时直接报错。"
              "T1 与 T2 是两个总体，**不可合并计算精度**。\n",
              "| 集合 | 条数 | 同 | 异 | 不决 | 抽样方式 | 文件 sha256 |\n|---|---:|---:|---:|---:|---|---|"]
        for name, n, s, d_, u, sha, uri, samp in ls_rows:
            L.append("| %s | %d | %d | %d | %d | %s | `%s` |" % (
                name, n, s or 0, d_ or 0, u or 0, (samp or "")[:28], (sha or "")[:12]))
    pr = conn.execute("""SELECT name, status, identity_isolation, file_count, manifest_uri, verify_json
                         FROM eval_protocol ORDER BY name""").fetchall()
    if pr:
        L += ["", "## 8. 评测协议（V2）\n",
              "| 协议 | 状态 | 身份隔离 | 清单文件数 | 核验 |\n|---|---|---|---:|---|"]
        for name, status, iso, fc, muri, vj in pr:
            v = json.loads(vj or "{}")
            L.append("| %s | %s | %s | %d | ok %d / 缺 %d / 不符 %d |" % (
                name, status, iso, fc or 0, v.get("ok", 0), v.get("n_missing", 0), v.get("n_mismatch", 0)))

    vr = conn.execute("""SELECT display_id, value_text, recomputed_text, verdict, reason
                         FROM paper_claim WHERE recomputed_text IS NOT NULL ORDER BY display_id""").fetchall()
    if vr:
        n_match = sum(1 for x in vr if x[3] == "match")
        n_unv = sum(1 for x in vr if x[3] == "unverifiable")
        L += ["", "## 9. 复算（V3）\n",
              "> 由 `ops_claims verify` 按 `ops/claims/verify_specs.json` 重算；"
              "「论文值」列是 EVIDENCE 的写法，**不因复算结果改动**。\n",
              "| claim | 论文值 | 复算 | 判定 | 差异 |\n|---|---|---|---|---|"]
        for did, vt, rt, vd, rs in vr:
            rs = rs or ""
            if "不一致" in rs:
                diff = rs.split("｜不一致：")[-1]
            elif "不可复算" in rs:
                diff = "不可复算：" + rs.split("｜不可复算：")[-1]
            else:
                diff = ""
            L.append("| %s | %s | %s | **%s** | %s |" % (
                did, (vt or "")[:26], (rt or "")[:64], vd, diff[:60]))
        L += ["", "已登记 **%d** 条：一致 **%d**、不一致 **%d**、不可复算 **%d**。" % (
            len(vr), n_match, len(vr) - n_match - n_unv, n_unv)]

        # 覆盖的表格数字个数：按配方指向的 tex 行统计（没有 tex 位置的条目不计数）
        try:
            specs = json.load(open(VERIFY_PATH, encoding="utf-8"))["specs"]
        except Exception:                                            # noqa: BLE001
            specs = []
        verd = {}
        for _id, _disp, _vd in conn.execute(
                "SELECT id, display_id, verdict FROM paper_claim").fetchall():
            verd[_id] = _vd
            verd[_disp] = _vd
        # 覆盖的表格数字：按 (文件,表,行,列) 去重，避免同一行被多条 claim 重复计数
        all_cells, ok_cells = set(), set()
        for spec in specs:
            v = verd.get(spec["claim"])
            if not v:
                continue
            for chk in spec["checks"]:
                ft, lb, rw = chk.get("file_tex"), chk.get("label"), chk.get("row")
                if not ft and chk.get("kind") == "matrix_row":
                    ft = chk.get("file")          # matrix_row 的 file 就是 tex 文件
                if not (ft and lb and rw):
                    continue
                sk = int(chk.get("skip_first", 1))
                try:
                    if chk.get("tex_idx"):
                        idx = chk["tex_idx"]
                    else:
                        idx = range(len(_tex_row_numbers(ft, lb, int(rw), skip_first=sk)))
                except Exception:                                    # noqa: BLE001
                    continue
                for i in idx:
                    all_cells.add((ft, lb, int(rw), sk, int(i)))
                    if v in ("match", "mismatch"):
                        ok_cells.add((ft, lb, int(rw), sk, int(i)))
        tot_num = len(all_cells)
        cover = {"match": len(ok_cells), "mismatch": 0,
                 "unverifiable": len(all_cells - ok_cells)}
        tot_cells = 0
        for t in load_tex_tables():
            for i, r in enumerate(t["rows"]):
                sk = 0 if i == 0 else 1
                try:
                    tot_cells += len(_tex_row_numbers(t["file"], t["label"] or t["caption"][:12],
                                                      i, skip_first=sk))
                except Exception:                                    # noqa: BLE001
                    continue
        pct = (100.0 * len(ok_cells) / tot_cells) if tot_cells else 0.0
        L += ["", "覆盖表格单元格 **%d / %d**（已核 **%d = %.0f%%**、不可复算 **%d**）；"
              "只统计配方里带 tex 行列位置的条目，按单元格去重。" % (
                  tot_num, tot_cells, cover["match"], pct, cover["unverifiable"])]

    er = conn.execute("SELECT COUNT(*), SUM(exit_code=0) FROM eval_run").fetchone()
    cv_n = conn.execute("SELECT COUNT(*) FROM code_version").fetchone()[0]
    read_n = conn.execute("SELECT COUNT(*) FROM eval_run WHERE inputs_json LIKE '%\"role\": \"read\"%'").fetchone()[0]
    decl_n = conn.execute("SELECT COUNT(*) FROM eval_run WHERE inputs_json LIKE '%\"role\": \"declared\"%'").fetchone()[0]
    no_in = conn.execute("SELECT COUNT(*) FROM eval_run WHERE inputs_json IN ('[]','')").fetchone()[0]
    bad_run = conn.execute("SELECT COUNT(*) FROM paper_claim p LEFT JOIN eval_run e "
                           "ON p.eval_run_id=e.id WHERE p.verdict='match' "
                           "AND (e.id IS NULL OR e.exit_code!=0)").fetchone()[0]
    L += ["", "## 10. 复算运行（eval_run / code_version）\n",
          "> 设计稿 §5 约束：`verdict=match` 只能由 `exit_code=0` 的 run 写入。",
          "> `inputs_json` 的 `role`：`read` = 检查真正打开的文件；`declared` = 该 claim 在 EVIDENCE 声明的来源（未直接读取，仅记哈希）。\n",
          "| 项目 | 值 |", "|---|---|",
          "| `eval_run` 行数 | %d（exit_code=0：%d） |" % (er[0] or 0, er[1] or 0),
          "| `code_version` 行数 | %d |" % cv_n,
          "| 有 `read` 输入文件的 run | %d |" % (read_n or 0),
          "| 仅有 `declared` 来源的 run | %d |" % ((decl_n or 0) - (read_n or 0)),
          "| 无任何输入的 run（纯字面量，如 `prop` 的 k/n） | %d |" % (no_in or 0),
          "| `match` 但无成功 run 的 claim | **%d** |" % bad_run]

    path = os.path.join(REPORT_DIR, "report.md")
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L))
    return path, true_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["import", "anchor", "truth", "verify", "migrate-ids",
                                    "report", "all", "status", "doctor"])
    ap.add_argument("--db", default=None)
    ap.add_argument("--apply", action="store_true", help="写真实库时必须显式给出")
    ap.add_argument("--rebuild", action="store_true", help="清空 paper_claim 后重建（仅当没有非 pending 裁定）")
    ap.add_argument("--rebind", action="store_true", help="EVIDENCE 改动后，把锚定文件的 _built_from 更新为当前 sha256")
    a = ap.parse_args()

    real = os.path.abspath(a.db or ops_db.DB_PATH)
    is_real = real == os.path.abspath(ops_db.DB_PATH)
    if is_real and a.cmd not in ("status", "doctor") and not a.apply:
        print("[dry-run] 目标是真实库 %s；加 --apply 才会写入" % real)
        return 0
    if is_real and a.cmd in ("import", "all", "anchor", "truth"):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        dst = os.path.join(BACKUP_DIR, "ops_%s.db" % datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.copy2(real, dst)
        print("[backup]", os.path.relpath(dst, ROOT))

    conn = ops_db.connect(real)
    ops_db.init_db(conn)
    if a.rebind:
        _, _, ev_sha = parsed_claims()
        blob = json.load(open(ANCHOR_PATH, encoding="utf-8"))
        blob["_built_from"] = {"sha256": ev_sha, "date": now()[:10],
                               "note": "由 ops_claims --rebind 更新"}
        json.dump(blob, open(ANCHOR_PATH, "w", encoding="utf-8", newline="\n"),
                  ensure_ascii=False, indent=2)
        print("[rebind]", ev_sha[:12])
    rep = {"gaps": [], "cells": 0, "errors": [], "warn": ""}
    if a.cmd in ("import", "all"):
        print("[import]", do_import(conn, rebuild=a.rebuild))
    if a.cmd in ("anchor", "all"):
        rep = do_anchor(conn)
        print("[anchor] claims=%d cells=%d gaps=%d errors=%d %s" % (
            rep["anchored_claims"], rep["cells"], len(rep["gaps"]), len(rep["errors"]), rep["warn"]))
        for e in rep["errors"]:
            print("  ! ", e)
    if a.cmd in ("truth", "all"):
        tr = do_truth(conn)
        print("[truth] sets=%d protocols=%d errors=%d" % (
            len(tr["sets"]), len(tr["protocols"]), len(tr["errors"])))
        for r in tr["sets"]:
            print("   set %-16s n=%-4d same=%-3d diff=%-3d unlabeled=%-3d sha=%s" % r)
        for r in tr["protocols"]:
            print("   proto %-14s %-9s files=%-3d ok=%-3d missing=%d mismatch=%d %s" % r)
        for e in tr["errors"]:
            print("  ! ", e)
    if a.cmd in ("verify", "all"):
        vr = do_verify(conn)
        vc = {}
        for r in vr["rows"]:
            vc[r[3]] = vc.get(r[3], 0) + 1
        print("[verify] claims=%d %s errors=%d" % (len(vr["rows"]), vc, len(vr["errors"])))
        for r in vr["rows"]:
            print("   %-9s %-8s %s" % (r[0], r[3], r[2][:76]))
        for e in vr["errors"]:
            print("  ! ", e)
    if a.cmd == "migrate-ids":
        mr = do_migrate_ids(conn)
        print("[migrate-ids] anchor=%d specs=%d unresolved=%d" % (
            mr["anchor_ids"], mr["spec_ids"], len(mr["unresolved"])))
        for b in mr["unresolved"]:
            print("  ! ", b)
    if a.cmd in ("report", "all"):
        path, n = do_report(conn, rep)
        print("[report]", os.path.relpath(path, ROOT), "无覆盖数字", n)
    if a.cmd == "doctor":
        dr = do_doctor(conn)
        print("[doctor] assets=%d ok=%d missing=%d drift=%d large=%d | run_paths=%d missing=%d drift=%d null=%d | cv_stale=%d" % (
            dr["assets"], dr["asset_ok"], len(dr["asset_missing"]), len(dr["asset_drift"]),
            len(dr["asset_large"]), dr["run_paths"], len(dr["run_missing"]), len(dr["run_drift"]),
            len(dr["run_null"]), len(dr["cv_stale"])))
        print("[doctor]", os.path.relpath(dr["path"], ROOT))
    if a.cmd == "status":
        for t in ("claim", "claim_anchor", "study", "deliverable", "deliverable_ref",
                  "asset_version", "label_set", "eval_protocol", "eval_run", "code_version"):
            print(t, conn.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0])
        print("paper_claim（视图）", conn.execute("SELECT COUNT(*) FROM paper_claim").fetchone()[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
