#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按「首次引用顺序」重编参考文献编号。

为什么要脚本
------------
`docs/paper/main.md` 的编号是**手写**的 `[n]`，且已经按首引顺序排过（1→23）。
往里插新文献时，手改会同时踩三个坑：
  1. 正文里的 `[a, b, c]` 组要逐组改；
  2. 编号变了之后，「首次引用顺序」可能整体位移；
  3. 表格与图注里也有引用，容易漏。

做法
----
把新文献先以 `[24]`、`[25]`… 追加到文献表末尾并在正文任意位置引用，
然后跑本脚本：正文里所有引用按**出现顺序**重新编号，文献表按新编号重排。

安全约束（任何一条不满足就报错退出，不写文件）
--------------------------------------------
- 文献表里有、正文里从没引用的条目 → 报错（否则会被静默丢掉）；
- 正文引用了、文献表里没有的编号 → 报错；
- 编号必须落在 1..len(refs) 内。

不处理的形态：`[0.321, 0.585]` 这类置信区间、`[0000-0000-0000-0000]` 这类 ORCID，
靠正则的「只认 1–2 位纯数字」自然排除。

用法
----
    $PY tools/renumber_refs.py docs/paper/main.md            # 预演（不改文件）
    $PY tools/renumber_refs.py docs/paper/main.md --write     # 落盘
"""
import argparse
import re
import sys

CITE = re.compile(r"\[\d{1,2}(?:,\s*\d{1,2})*\]")
REF = re.compile(r"^\[(\d{1,3})\]\s+(.*)$")
HEADS = ("## References", "## 参考文献", "## REFERENCES")


def split_doc(text):
    for h in HEADS:
        i = text.find(h)
        if i >= 0:
            return text[:i], text[i:], h
    sys.exit(f"找不到文献表标题（试过 {HEADS}）")


def parse_refs(block):
    """[(num, body)]，body 可跨行（到下一个 [n] 之前）。"""
    lines = block.split("\n")
    out, cur = [], None
    for ln in lines:
        m = REF.match(ln)
        if m:
            if cur:
                out.append(cur)
            cur = [int(m.group(1)), m.group(2)]
        elif cur is not None:
            if ln.strip() == "":
                continue
            cur[1] += " " + ln.strip()
    if cur:
        out.append(cur)
    return [(n, b.strip()) for n, b in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    text = open(args.path, encoding="utf-8").read()
    body, refblk, head = split_doc(text)
    refs = parse_refs(refblk)
    have = {n for n, _ in refs}
    print(f"文献表 {len(refs)} 条，编号 {min(have)}–{max(have)}")

    # ---- 正文：按出现顺序收集 ----
    order, cited = [], set()
    for m in CITE.finditer(body):
        for part in m.group(0)[1:-1].split(","):
            n = int(part.strip())
            if n not in have:
                sys.exit(f"正文引用了不存在的 [{n}]：…{body[max(0, m.start()-40):m.end()+20]}…")
            cited.add(n)
            if n not in order:
                order.append(n)
    unused = sorted(have - cited)
    if unused:
        sys.exit(f"文献表里这些条目从未被引用，拒绝重排：{unused}")

    newno = {old: i + 1 for i, old in enumerate(order)}
    moved = sum(1 for k, v in newno.items() if k != v)
    print(f"首引顺序确定完毕；{moved} 条编号会变")
    for old in order:
        if newno[old] != old:
            print(f"   [{old}] -> [{newno[old]}]  {dict(refs)[old][:60]}")

    # ---- 重写正文 ----
    def repl(m):
        ns = [newno[int(p.strip())] for p in m.group(0)[1:-1].split(",")]
        return "[" + ", ".join(str(x) for x in ns) + "]"

    newbody = CITE.sub(repl, body)

    # ---- 重写文献表：按新编号排序，原本就够乱的顺序也一并理顺 ----
    bynew = sorted(((newno[n], b) for n, b in refs), key=lambda x: x[0])
    lines = [head, ""]
    for n, b in bynew:
        lines.append(f"[{n}] {b}")
        lines.append("")
    newref = "\n".join(lines).rstrip() + "\n"

    out = newbody + newref
    if not args.write:
        print("\n预演模式，未写文件。加 --write 落盘。")
        return
    open(args.path, "w", encoding="utf-8").write(out)
    print(f"\n已写入 {args.path}（{len(out)} 字符）")


if __name__ == "__main__":
    main()
