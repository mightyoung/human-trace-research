"""把 docs/paper/main.md 迁到 IEEE 口径（一次性迁移；重复运行前先恢复备份）。

做的事：
1. 期刊名 Neurocomputing -> IEEE T-BIOM（含 Article type 行）
2. Keywords -> Index Terms
3. 章节号阿拉伯 -> 罗马；子节 1.1 -> A
4. 正文交叉引用 Sections 5 to 7 -> Sections V–VII；Section 5.3 -> Section V-C
5. 正文 Table 1 -> Table I（Table B1 -> Table B-I）；Figure 1 -> Fig. 1
   注意：以 **Table / **Figure 开头的整行是图题表题，交给渲染器处理，这里跳过
6. 尾部（CRediT 起）整段替换为：附录 A / 附录 B / IEEE 声明五段 / IEEE 格式 26 条文献

用法：python tools/paper_ieee_migrate.py [--dry]
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs/paper/main.md"
TAIL = ROOT / "tmp/paper_tail_ieee.md"
BACKUP = ROOT / "docs/paper/main.md.bak_preieee"

ROMAN = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def roman(n: int) -> str:
    s = ""
    for v, r in ROMAN:
        while n >= v:
            s += r
            n -= v
    return s


def letters(i: int) -> str:
    return chr(ord("A") + i - 1)


def main() -> int:
    dry = "--dry" in sys.argv
    text = MD.read_text(encoding="utf-8")
    stats: dict[str, int] = {}

    # ---------- 1. 期刊名 ----------
    old_journal = "*Article type: Full Length Article. Submitted to Neurocomputing.*"
    stats["journal"] = text.count(old_journal)
    text = text.replace(
        old_journal,
        "*Article type: Regular Paper. Submitted to IEEE Transactions on Biometrics, "
        "Behavior, and Identity Science.*",
    )

    # ---------- 2. Keywords -> Index Terms ----------
    stats["keywords"] = text.count("**Keywords:**")
    text = text.replace("**Keywords:**", "**Index Terms:**")

    # ---------- 3. 章节号 -> 罗马 / 字母 ----------
    sec_map: dict[int, str] = {}
    sub_map: dict[tuple[int, int], str] = {}
    cur_sec, sub_i = 0, 0
    for line in text.splitlines():
        m = re.match(r"^## (\d+) (.+)$", line)
        if m:
            cur_sec, sub_i = int(m.group(1)), 0
            sec_map[cur_sec] = roman(cur_sec)
            continue
        m = re.match(r"^### (\d+)\.(\d+) (.+)$", line)
        if m and int(m.group(1)) == cur_sec:
            sub_i += 1
            sub_map[(cur_sec, int(m.group(2)))] = letters(sub_i)

    out = []
    for line in text.splitlines():
        m = re.match(r"^## (\d+) (.+)$", line)
        if m:
            out.append(f"## {roman(int(m.group(1)))}. {m.group(2)}")
            continue
        m = re.match(r"^### (\d+)\.(\d+) (.+)$", line)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            out.append(f"### {sub_map.get(key, '?')}. {m.group(3)}")
            continue
        out.append(line)
    text = "\n".join(out) + "\n"
    stats["sections"] = len(sec_map)
    stats["subsections"] = len(sub_map)

    # ---------- 4/5. 正文交叉引用（逐行，跳过图题表题与代码块） ----------
    def sec_label(num: int, sub: int | None) -> str:
        s = sec_map.get(num, roman(num))
        if sub is None:
            return f"Section {s}"
        return f"Section {s}-{sub_map.get((num, sub), '?')}"

    def tab_label(tag: str) -> str:
        if tag.upper().startswith("B"):
            return f"Table B-{roman(int(tag[1:]))}"
        return f"Table {roman(int(tag))}"

    n = {"range": 0, "sec": 0, "tab": 0, "fig": 0}

    out = []
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        # 图题表题：只保护开头的 "Table N." / "Figure N."，行内其余交叉引用照转
        prefix = ""
        cap = re.match(r"^(\*\*(?:Table|Figure) (?:B?\d+)\.)", line)
        if cap:
            prefix = cap.group(1)
            line = line[len(prefix):]

        # 区间：Sections 5 to 7 / Sections 5 and 7
        def _range(m: re.Match) -> str:
            n["range"] += 1
            return f"Sections {roman(int(m.group(1)))}–{roman(int(m.group(2)))}"

        line = re.sub(r"\bSections (\d+) (?:to|and) (\d+)\b", _range, line)

        # 多子节并列：Sections 7.2 and 7.4 -> Sections VII-B and VII-D
        def _mulsub(m: re.Match) -> str:
            n["range"] += 1
            a = sec_label(int(m.group(1)), int(m.group(2)))
            b = sec_label(int(m.group(3)), int(m.group(4)))
            return f"Sections {a.split(' ', 1)[1]} and {b.split(' ', 1)[1]}"

        line = re.sub(
            r"\bSections (\d+)\.(\d+) (?:and|to) (\d+)\.(\d+)\b", _mulsub, line
        )

        # 单引用：Section 5.3 / Section 5
        def _sec(m: re.Match) -> str:
            n["sec"] += 1
            sub = int(m.group(2)) if m.group(2) else None
            return sec_label(int(m.group(1)), sub)

        line = re.sub(r"\bSection (\d+)(?:\.(\d+))?\b", _sec, line)

        def _tab(m: re.Match) -> str:
            n["tab"] += 1
            return tab_label(m.group(1))

        line = re.sub(r"\bTable (B?\d+)\b", _tab, line)

        def _fig(m: re.Match) -> str:
            n["fig"] += 1
            return f"Fig. {m.group(1)}"

        line = re.sub(r"\bFigures? (\d+)\b", _fig, line)
        out.append(prefix + line)

    text = "\n".join(out) + "\n"
    stats.update(n)

    # ---------- 6. 替换尾部 ----------
    marker = "## CRediT authorship contribution statement"
    idx = text.find(marker)
    if idx < 0:
        print("！找不到 CRediT 段，尾部未替换")
        return 2
    tail = TAIL.read_text(encoding="utf-8")
    head = text[:idx].rstrip()
    if head.endswith("---"):
        head = head[:-3].rstrip()
    new_text = head + "\n\n---\n\n" + tail

    for k, v in stats.items():
        print(f"  {k:<14}{v}")
    print(f"  体积          {len(text.encode())/1024:.1f} KB -> {len(new_text.encode())/1024:.1f} KB")

    if dry:
        print("\n--dry：未写盘")
        return 0

    if not BACKUP.exists():
        shutil.copy2(MD, BACKUP)
        print(f"备份 -> {BACKUP.name}")
    MD.write_text(new_text, encoding="utf-8")
    print(f"写入 {MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
