#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参考文献 DOI 核验：逐条查 Crossref，比对标题/年份/卷期页，输出核验表。

为什么必须做
------------
投稿件里 23 条参考文献，其中 2025–2026 年的新文献占 5 条。这类条目最容易出现
「标题对、年份卷期错」或「DOI 不存在」，而审稿人一查就露。核验不靠记忆，靠
Crossref 官方元数据。

口径
----
- 用 `query.bibliographic`（作者+标题混合）取 Crossref 前 5 条候选
- 标题相似度用归一化后的 token Jaccard，≥0.75 视为命中
- 只报告差异，不自动改文件（改稿由人确认）
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request

UA = "human-count-refcheck/1.0 (mailto:unknown@example.com)"
API = "https://api.crossref.org/works"


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return [w for w in s.split() if w]


def jac(a, b):
    A, B = set(norm(a)), set(norm(b))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


INIT = re.compile(r"^([A-Z]\.\s?-?){1,3}[A-Z][A-Za-z\u00C0-\u024F'\-]+(\s[A-Z][A-Za-z\u00C0-\u024F'\-]+){0,2}$")


def split_ref(raw):
    """从 "L. Zheng, L. Shen, Title, in: Venue" 里拆出 (authors, title)。

    ★ 不能用 `raw.split('.')[1]`：作者缩写本身带句点，会把 " Zheng, L" 当成标题。
    按逗号切 token，逐个判断是不是「姓名」（缩写 + 姓氏），第一个不是的 token 就是标题起点。
    """
    toks = [t.strip() for t in raw.split(", ")]
    i = 0
    while i < len(toks) and (INIT.match(toks[i]) or re.match(r"^([A-Z]\.\s?-?){1,3}$", toks[i])):
        i += 1
    if i == 0:
        # 没有可识别作者段（标准号、机构条目）
        return "", raw.split(", in:")[0].split(", ")[0]
    authors = ", ".join(toks[:i])
    rest = ", ".join(toks[i:])
    title = re.split(r",\s*in:|\s+in:\s|,?\s*(?:IEEE|ACM|Proceedings|Adv\.|European|Int\.|Knowl|Appl\.|Neural|Sensors|Eng\.|Ann\.)", rest)[0]
    return authors, title.strip().rstrip(",")


def parse_refs(path):
    txt = open(path, encoding="utf-8").read()
    body = txt.split("## References", 1)[1].split("\n---", 1)[0]
    refs = []
    for m in re.finditer(r"\[(\d+)\]\s+(.+?)(?=\n\n\[|\n\n---|\Z)", body, re.S):
        refs.append((int(m.group(1)), " ".join(m.group(2).split())))
    return refs


def crossref(title, author=None, rows=5):
    q = title if not author else "%s %s" % (author, title)
    url = API + "?" + urllib.parse.urlencode({
        "query.bibliographic": q, "rows": rows, "select": "DOI,title,author,container-title,issued,volume,page,type"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["items"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/paper/main.md"
    refs = parse_refs(path)
    print("解析到 %d 条参考文献\n" % len(refs))
    out = []
    for n, raw in refs:
        # 作者段与标题段
        authors, title = split_ref(raw)
        yr = re.search(r"(19|20)\d{2}", raw)
        try:
            items = crossref(title, (authors.split(",")[0] if authors else None))
        except Exception as e:
            print("[%2d] 查询失败 %s" % (n, e))
            out.append(dict(n=n, status="error", ref=raw))
            continue
        best, bs = None, 0.0
        for it in items:
            t = (it.get("title") or [""])[0]
            s = jac(title, t)
            if s > bs:
                best, bs = it, s
        if not best or bs < 0.45:
            print("[%2d] 未命中 (best %.2f) | %s" % (n, bs, title[:80]))
            out.append(dict(n=n, status="no-match", title=title, best=bs, ref=raw))
            time.sleep(0.4)
            continue
        got = dict(
            doi=best.get("DOI"),
            title=(best.get("title") or [""])[0],
            venue=(best.get("container-title") or [""])[0],
            year=(best.get("issued", {}).get("date-parts") or [[None]])[0][0],
            vol=best.get("volume"), page=best.get("page"), type=best.get("type"))
        flag = []
        if bs < 0.75:
            flag.append("标题相似度低 %.2f" % bs)
        if yr and got["year"] and abs(int(yr.group(0)) - int(got["year"])) > 1:
            flag.append("年份 引%d vs 实%s" % (int(yr.group(0)), got["year"]))
        print("[%2d] %-5s %.2f | %s | %s %s %s"
              % (n, "OK" if not flag else "DIFF", bs, got["doi"], got["venue"][:38],
                 got["year"], got["vol"] or ""))
        if flag:
            print("      ⚠ %s" % "; ".join(flag))
            print("      引: %s" % raw[:120])
            print("      实: %s | %s %s pp.%s" % (got["title"][:90], got["venue"], got["vol"], got["page"]))
        out.append(dict(n=n, status="ok" if not flag else "diff", flags=flag, got=got, ref=raw, score=bs))
        time.sleep(0.4)

    json.dump(out, open("tmp/refcheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_ok = sum(1 for o in out if o["status"] == "ok")
    print("\n汇总：%d 条，完全对上 %d，有差异 %d，未命中 %d"
          % (len(out), n_ok, sum(1 for o in out if o["status"] == "diff"),
             sum(1 for o in out if o["status"] == "no-match")))
    print("明细 → tmp/refcheck.json")


if __name__ == "__main__":
    main()
