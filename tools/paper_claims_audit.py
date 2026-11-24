#!/usr/bin/env python
"""paper_claims_audit.py — V0 只读体检：论文数字的证据覆盖。

把 docs/paper/EVIDENCE.md 里每一行带数字的内容当作一条 claim，核对三件事：
  ① 来源文件是否存在（≤ 512 MB 记 sha256；更大的只记 size/mtime 并标 weak）
  ② 行内/表级是否引用了复算脚本，且脚本存在
  ③ EVIDENCE.md 与 main.tex / supplement.tex 的数字集合差异（粗匹配）

引用的三个来源，按优先级：
  行内反引号 → 同一张表上方紧邻的说明行与小节标题 →（仅当都没有时）同小节表后的“来源/文件/复现”行
  “同上”继承上一行的文件引用。

只读：除 --out 目录外不写任何文件，不碰 ops/ops.db，不执行任何复算脚本。
用法：python tools/paper_claims_audit.py [--out ops/reports/paper_claims_v0]
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, OrderedDict
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE = "docs/paper/EVIDENCE.md"
TEX = ["docs/paper/tbiom-r2/main.tex", "docs/paper/tbiom-r2/supplement.tex"]

CAMS = ["305", "316", "318", "320", "323", "343"]
BRACE = {"cam": CAMS, "cell": ["*"]}
HASH_LIMIT = 512 * 1024 * 1024
KNOWN_EXT = {".json", ".py", ".md", ".txt", ".npz", ".csv", ".npy", ".sh", ".mp4", ".jpg",
             ".jpeg", ".png", ".pt", ".pth", ".onnx", ".toml", ".yaml", ".yml", ".cmd", ".html"}
SCRIPT_EXT = {".py", ".sh", ".cmd"}
LATE_MARK = ("来源", "文件", "复现")
EXTERNAL_SECTIONS = {"Introduction", "Related Work"}
ROOT_DIRS = {"docs", "tmp", "tools", "deliverable", "webapp", "output_crosscam", "media-resource",
             "models", "datasets", "results", "ops", "remote-results", "src", "analysis"}


def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


# ---------------------------------------------------------------- 路径引用
def is_pathlike(tok):
    if any(ch in tok for ch in " \n\t") or tok.startswith("§") or "[]" in tok:
        return False
    if tok.startswith(("~", "./", "../")):
        return True
    if "/" in tok:
        # 带斜杠但不是项目目录下的文件 → 字段指针（如 surv/n_rev/judged）
        return tok.split("/")[0] in ROOT_DIRS or os.path.splitext(tok)[1].lower() in KNOWN_EXT
    return os.path.splitext(tok)[1].lower() in KNOWN_EXT


def expand_braces(tok):
    """`2026-10-2{2,6}_x_{tr,te}.json` → 4 个具体路径；`{cam}` → 6 个机位；`{cell}` → 通配。"""
    out = [tok]
    for _ in range(4):
        nxt = []
        for t in out:
            m = re.search(r"\{([^{}]+)\}", t)
            if not m:
                nxt.append(t)
                continue
            alts = BRACE.get(m.group(1)) or m.group(1).split(",")
            nxt += [t[:m.start()] + a + t[m.end():] for a in alts]
        if nxt == out:
            break
        out = nxt
    return out


class Resolver:
    def __init__(self):
        self.cache = {}
        self.by_name = {}

    def stat(self, path):
        if path in self.cache:
            return self.cache[path]
        try:
            st = os.stat(path)
        except OSError:
            self.cache[path] = None
            return None
        rec = {"path": rel(path), "bytes": st.st_size, "mtime": int(st.st_mtime),
               "sha256": None, "weak": True}
        if st.st_size <= HASH_LIMIT:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            try:
                st2 = os.stat(path)
            except OSError:
                st2 = st
            if (st2.st_size, st2.st_mtime) == (st.st_size, st.st_mtime):
                rec["sha256"] = h.hexdigest()
                rec["weak"] = False
        self.cache[path] = rec
        return rec

    def resolve(self, tok, hints):
        """返回 (存在的具体文件列表, 展开后的候选数)。"""
        t = os.path.expanduser(tok)
        if os.path.isabs(t):
            cands = [t]
        elif "/" in t:
            cands = [os.path.join(ROOT, t)]
        else:
            cands = ([os.path.join(h, t) for h in hints]
                     + [os.path.join(d, t) for d in self.by_name.get(os.path.basename(t), [])]
                     + [os.path.join(ROOT, "tools", t), os.path.join(ROOT, "tmp", "full", t),
                        os.path.join(ROOT, "docs", t)])
        hits, n = [], 0
        for c in cands:
            for p in expand_braces(c):
                n += 1
                if any(ch in p for ch in "*?["):
                    hits += sorted(glob(p))
                elif os.path.isfile(p):
                    hits.append(p)
        hits = sorted(set(hits))
        for h in hits:
            self.by_name.setdefault(os.path.basename(h), []).append(os.path.dirname(h))
        return hits, n


def refs_in(text, hints, res):
    """返回 (文件引用, 符号引用)。`→ field` 形式视为字段指针而非文件。"""
    files, syms = [], []
    for m in re.finditer(r"(→\s?)?`([^`]+)`", text):
        tok = m.group(2)
        if m.group(1) or not is_pathlike(tok):
            syms.append(tok)
            continue
        hits, n = res.resolve(tok, hints)
        files.append({"tok": tok, "hits": hits, "n_cand": n})
    return files, syms


def hints_of(files):
    hs = []
    for f in files:
        for h in f["hits"]:
            d = os.path.dirname(h)
            if d not in hs:
                hs.append(d)
    return hs


def split_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def is_sep(line):
    return bool(re.match(r"^\|\s*:?-{2,}", line.strip()))


# ---------------------------------------------------------------- 解析 EVIDENCE.md
def parse_evidence(text, res):
    lines = text.split("\n")
    claims, commands = [], []
    sec_no = sub_no = 0
    sec = sub = ""
    in_code = False
    code_buf, pending, prev_files = [], [], []
    tctx = {"refs": []}
    i = 0
    while i < len(lines):
        l = lines[i]
        ln = i + 1
        if l.startswith("```"):
            if in_code and code_buf:
                commands.append("\n".join(code_buf))
                code_buf = []
            in_code = not in_code
            i += 1
            continue
        if in_code:
            code_buf.append(l)
            i += 1
            continue
        if l.startswith("## ") or l.startswith("### ") or l.startswith("---"):
            if l.startswith("## "):
                sec_no += 1
                sub_no = 0
                sec, sub = l[3:].strip(), ""
            elif l.startswith("### "):
                sub_no += 1
                sub = l[4:].strip()
                f, _ = refs_in(sub, [], res)
                tctx = {"refs": f, "from": "heading"}
            pending = []
            prev_files = []
            i += 1
            continue
        if l.startswith("|"):
            cells = split_row(l)
            if i + 1 < len(lines) and is_sep(lines[i + 1]):
                files = []
                for pl in pending:
                    f, _ = refs_in(pl, [], res)
                    files += f
                if tctx.get("from") == "heading":
                    files = tctx["refs"] + files
                tctx = {"refs": files, "from": "above" if pending else "heading"}
                pending, prev_files = [], []
                i += 2
                continue
            if is_sep(l) or not any(re.search(r"\d", c) for c in cells):
                i += 1
                continue
            files, syms = refs_in(l, hints_of(tctx["refs"]), res)
            same = any(c.startswith("同上") for c in cells)
            # 「同上」或只写了字段指针（summary.x、runs[].y）的行，继承上一行解析到的文件
            if not files and prev_files and (same or (syms and not tctx["refs"])):
                files = [dict(f, inherited=True) for f in prev_files]
            claims.append({"sec": sec_no, "sub": sub_no, "section": sec, "subsection": sub,
                           "line": ln, "kind": "table", "cells": cells, "label": cells[0][:60],
                           "files": files, "symbols": syms, "table_refs": tctx["refs"],
                           "same_as_above": same})
            if files and not same:
                prev_files = files
            i += 1
            continue
        if l.strip():
            pending.append(l)
            if re.search(r"\d", l):
                kind = "code_cmd" if l.strip().startswith(("$PY", "MODES=")) else "prose"
                files, syms = refs_in(l, hints_of(tctx["refs"]) + [os.path.join(ROOT, "tmp", "full")], res)
                claims.append({"sec": sec_no, "sub": sub_no, "section": sec, "subsection": sub,
                               "line": ln, "kind": kind, "cells": [l.strip()[:160]],
                               "label": l.strip()[:60], "files": files, "symbols": syms,
                               "table_refs": [], "same_as_above": False})
        i += 1

    # 表后「来源」行：只补给完全无引用的表格行
    by_sub = {}
    for c in claims:
        by_sub.setdefault((c["sec"], c["sub"]), []).append(c)
    for c in claims:
        if c["kind"] != "table" or c["files"] or c["table_refs"]:
            continue
        end = max(x["line"] for x in by_sub[(c["sec"], c["sub"])]) + 1
        for j in range(c["line"], min(end, len(lines))):
            lj = lines[j]
            if lj.startswith(("#", "---")):
                break
            if any(k in lj for k in LATE_MARK) and "`" in lj:
                f, _ = refs_in(lj, [], res)
                c["files"] = c["files"] + [dict(x, late=True) for x in f]

    for n, c in enumerate(claims, 1):
        c["id"] = "E%02d-%03d" % (c["sec"], n)
        hits = [h for f in c["files"] + c["table_refs"] for h in f["hits"]]
        data = [h for h in hits if os.path.splitext(h)[1].lower() != ".md"]
        docs = [h for h in hits if os.path.splitext(h)[1].lower() == ".md"]
        scripts = [h for h in hits if os.path.splitext(h)[1].lower() in SCRIPT_EXT]
        miss = [f["tok"] for f in c["files"] if not f["hits"]]
        weak = [rel(h) for h in data if res.stat(h)["weak"]]
        c["level"] = "data" if data else ("doc" if docs else "none")
        c["missing"], c["weak"] = miss, weak
        c["scripts"] = sorted({rel(h) for h in scripts})
        c["src_ok"] = c["level"] == "data" and not miss and not weak
        c["repro"] = bool(scripts)
        c["late"] = any(f.get("late") for f in c["files"])
        c["paths"] = sorted({rel(h) for h in hits})
    return claims, commands


# ---------------------------------------------------------------- 解析 tex
NUM = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|(?<![\w.])\.\d+")


def norm_num(s):
    s = s.replace(",", "")
    return "0" + s if s.startswith(".") else s


def nums(s):
    return {norm_num(m) for m in NUM.findall(s)}


def tex_clean(t):
    t = re.sub(r"(?<!\\)%.*", "", t)
    for c in ("label", "ref", "eqref", "cite", "citep", "citet", "autoref", "cref", "Cref",
              "pageref", "url", "usepackage", "include", "input", "bibliography",
              "bibliographystyle", "vspace", "hspace", "cline", "captionsetup", "arraystretch"):
        t = re.sub(r"\\" + c + r"\*?(\[[^\]]*\])?\{[^{}]*\}", " ", t)
    t = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^{}]*\}", " ", t)
    t = re.sub(r"\\href\{[^{}]*\}", " ", t)
    t = re.sub(r"\\cmidrule\([a-z]*\)\{[^{}]*\}", " ", t)
    t = re.sub(r"\\multicolumn\{[^{}]*\}\{[^{}]*\}", " ", t)
    t = re.sub(r"\\multirow\{[^{}]*\}", " ", t)
    t = re.sub(r"\\(setlength|resizebox)\{[^{}]*\}\{[^{}]*\}", " ", t)
    t = re.sub(r"@(?:\{[^{}]*\}|\|)", " ", t)
    t = re.sub(r"[pmb]\{[^{}]*\}", " ", t)           # 列宽 p{0.6cm}
    t = re.sub(r"\\begin\{[^{}]*\}(\[[^\]]*\])?", " ", t)
    t = re.sub(r"\\end\{[^{}]*\}", " ", t)
    t = re.sub(r"\\documentclass(\[[^\]]*\])?\{[^{}]*\}", " ", t)
    t = re.sub(r"\\addlinespace(\[[^\]]*\])?", " ", t)
    t = re.sub(r"\\(toprule|midrule|bottomrule|hline|footnotesize|scriptsize|small|centering|noindent)", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    return t


def parse_tex(lines):
    ctx, tables, clean = OrderedDict(), [], []
    cur = None
    sec, ext = "", False
    for ln, raw in enumerate(lines, 1):
        if "\\begin{thebibliography}" in raw:
            break
        m = re.search(r"\\section\*?\{([^}]*)\}", raw)
        if m:
            sec = m.group(1)
            ext = sec in EXTERNAL_SECTIONS
        t = tex_clean(raw)
        clean.append(t)
        if re.search(r"\\begin\{table", raw):
            cur = {"line": ln, "label": "", "caption": "", "text": "", "rows_raw": []}
        if cur is not None:
            cur["text"] += " " + t
            if "&" in raw and not raw.lstrip().startswith("%"):
                cur["rows_raw"].append(raw)
            m = re.search(r"\\label\{([^}]*)\}", raw)
            if m and not cur["label"]:
                cur["label"] = m.group(1)
            m = re.search(r"\\caption\{(.*)", raw)
            if m and not cur["caption"]:
                cur["caption"] = re.sub(r"\s+", " ", tex_clean(m.group(1)))[:70]
            if re.search(r"\\end\{table", raw):
                cur["nums"] = sorted(nums(cur["text"]))
                cur["rows"] = [[re.sub(r"\s+", " ", tex_clean(c)).strip() for c in
                                r.strip().rstrip("\\").split("&")] for r in cur.pop("rows_raw")]
                tables.append(cur)
                cur = None
        if ext:
            continue
        for n in nums(t):
            if n not in ctx:
                ctx[n] = "%s §%s" % (re.sub(r"\s+", " ", t).strip()[:46], sec)
    allnums = set()
    for t in clean:
        allnums |= nums(t)
    return allnums, tables, ctx


# ---------------------------------------------------------------- 主流程
def pct(a, b):
    return "%d/%d" % (a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default=EVIDENCE)
    ap.add_argument("--tex", nargs="*", default=TEX)
    ap.add_argument("--out", default="ops/reports/paper_claims_v0")
    a = ap.parse_args()
    os.chdir(ROOT)

    res = Resolver()
    ev_text = open(a.evidence, encoding="utf-8").read()
    claims, commands = parse_evidence(ev_text, res)
    ev_nums = nums(ev_text)

    cmd_checks = []
    for block in commands:
        for ln in block.split("\n"):
            for m in re.findall(r"(?:\$PY\s+|PY=)(\S+)", ln):
                hits, _ = res.resolve(m, [os.path.join(ROOT, "tools")])
                cmd_checks.append({"script": m, "ok": bool(hits)})

    tex_all, tex_ctx, tex_tables = set(), {}, []
    for p in a.tex:
        allnums, tables, ctx = parse_tex(open(p, encoding="utf-8").read().split("\n"))
        tex_all |= allnums
        for n, c in ctx.items():
            tex_ctx.setdefault(n, "%s:%s" % (os.path.basename(p), c))
        for tb in tables:
            tex_tables.append({"file": os.path.basename(p),
                               **{k: tb[k] for k in ("line", "label", "caption", "rows")},
                               "nums": sorted(tb["nums"])})
    tab_nums = set().union(*[set(tb["nums"]) for tb in tex_tables]) if tex_tables else set()

    for c in claims:
        if c["kind"] == "table" and len(c["cells"]) >= 2:
            v = nums(c["cells"][1])
            c["value_absent"] = sorted(v - tex_all)
    only_ev = sorted(ev_nums - tex_all)
    only_tex = sorted(tex_all - ev_nums)
    only_tex_tab = sorted(tab_nums - ev_nums)

    n = len(claims)
    by = Counter(c["level"] for c in claims)
    src_ok = sum(c["src_ok"] for c in claims)
    rep = sum(c["repro"] for c in claims)
    missing = Counter(m for c in claims for m in c["missing"])
    assets = {r["path"]: r for r in res.cache.values() if r}

    os.makedirs(a.out, exist_ok=True)
    json.dump({"generated_from": a.evidence, "claims": claims, "commands": cmd_checks,
               "assets": assets, "only_evidence": only_ev, "only_tex": only_tex,
               "only_tex_tables": only_tex_tab, "tex_ctx": tex_ctx, "tex_tables": tex_tables},
              open(os.path.join(a.out, "claims.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    L = ["# 论文数字证据体检 V0（只读）\n",
         "> 解析 `%s` 得 **%d 条 claim**；与 %s 做数字集合比对。本报告**不执行任何复算**；"
         "“有脚本”只表示引用了存在的脚本，不表示脚本复算过这个数字。\n" % (
             a.evidence, n, "、".join(os.path.basename(t) for t in a.tex)),
         "## 1. 总览\n", "| 指标 | 数值 |\n|---|---|",
         "| claim 总数 | %d |" % n,
         "| ① 来源含数据/结果文件（非 md） | %s |" % pct(by["data"], n),
         "| ① 来源只有文档（md） | %s |" % pct(by["doc"], n),
         "| ① 没有任何可解析来源 | %s |" % pct(by["none"], n),
         "| ① 严格达标（引用全部存在、已记内容 sha） | %s |" % pct(src_ok, n),
         "| ② 引用了存在的复算脚本 | %s |" % pct(rep, n),
         "| ③ tex 表格内数字 / 其中 EVIDENCE 全文找不到 | %d / **%d** |" % (len(tab_nums), len(only_tex_tab)),
         "| ③ EVIDENCE 有、tex 没有的数字 | %d |" % len(only_ev),
         "| 已登记来源文件（去重） | %d（未记内容 sha 的 %d） |" % (len(assets), sum(r["weak"] for r in assets.values())),
         "", "## 2. 分节\n",
         "| 节 | claim | 数据文件 | 仅文档 | 无来源 | 有脚本 | 严格达标 |\n|---|---:|---:|---:|---:|---:|---:|"]
    secs = OrderedDict()
    for c in claims:
        secs.setdefault(c["section"], []).append(c)
    for s, cs in secs.items():
        b = Counter(x["level"] for x in cs)
        L.append("| %s | %d | %d | %d | %d | %d | %d |" % (s[:38], len(cs), b["data"], b["doc"],
                                                         b["none"], sum(x["repro"] for x in cs), sum(x["src_ok"] for x in cs)))

    L += ["", "## 3. 每张 tex 表里、EVIDENCE 全文找不到的数字\n",
          "| 文件 | 表 | 标题 | 表内数字 | 缺 |\n|---|---|---|---:|---|"]
    for tb in tex_tables:
        miss = sorted(set(tb["nums"]) - ev_nums)
        L.append("| %s | `%s` | %s | %d | %s |" % (tb["file"], tb["label"] or "-", tb["caption"][:52],
                                                 len(tb["nums"]), " ".join(miss[:14]) + (" …" if len(miss) > 14 else "") if miss else ""))
    L += ["", "## 4. 解析不到的引用\n", "| 引用 | 次数 |\n|---|---:|"]
    L += ["| `%s` | %d |" % (m.replace("|", "\\|"), k) for m, k in missing.most_common(30)] or ["| 无 | 0 |"]
    L += ["", "## 5. §十 复现清单的脚本\n", "| 脚本 | 存在 |\n|---|---|"]
    L += ["| `%s` | %s |" % (c["script"], "是" if c["ok"] else "**否**") for c in cmd_checks]
    L += ["", "## 6. tex 正文有、EVIDENCE 没有的数字（不含 Introduction / Related Work / 参考文献；按出现位置）\n",
          "| 数字 | 出现位置 |\n|---|---|"]
    L += ["| %s | %s |" % (k, v.replace("|", "\\|")[:80]) for k, v in tex_ctx.items() if k in set(only_tex)]
    L += ["", "## 7. EVIDENCE 有、tex 没有的数字（%d 个）\n" % len(only_ev), " ".join(only_ev) + "\n",
          "## 8. 值单元数字在 tex 全文找不到的表格行\n", "| id | 行 | 值 | 找不到 |\n|---|---|---|---|"]
    L += ["| %s | %s | %s | %s |" % (c["id"], c["label"].replace("|", "\\|")[:28], c["cells"][1].replace("|", "\\|")[:36],
                                    " ".join(c["value_absent"][:8])) for c in claims if c.get("value_absent")]
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L))

    print("claims", n, dict(by), "src_ok", src_ok, "repro", rep, "tab_nums", len(tab_nums),
          "only_tex_tab", len(only_tex_tab), "only_tex", len(only_tex), "only_ev", len(only_ev),
          "assets", len(assets), "weak", sum(r["weak"] for r in assets.values()),
          "cmds", sum(c["ok"] for c in cmd_checks), "/", len(cmd_checks), "->", a.out)


if __name__ == "__main__":
    sys.exit(main())
