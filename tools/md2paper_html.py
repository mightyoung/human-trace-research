"""把 docs/paper/main.md 渲染成可打印的 A4 论文页（无外部依赖，可离线打开）。

用法：
    python tools/md2paper_html.py
    python tools/md2paper_html.py docs/paper/main.md docs/paper/main.html
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]

CSS = """
:root {
  --ink: #1a1a1a; --ink-2: #444; --rule: #d8d8d8; --accent: #8a1c1c;
  --bg: #fff; --code-bg: #f5f3ef;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: #ececec; color: var(--ink);
  font-family: "Iowan Old Style", "Palatino Linotype", "Songti SC", "Source Han Serif SC", "Noto Serif CJK SC", "SimSun", Georgia, serif;
  font-size: 15.5px; line-height: 1.72;
}
.page {
  max-width: 820px; margin: 0 auto; background: var(--bg);
  padding: 62px 78px 80px; box-shadow: 0 2px 18px rgba(0,0,0,.10);
}
h1 {
  font-size: 27px; line-height: 1.32; margin: 0 0 18px; letter-spacing: -.2px;
  font-weight: 700;
}
h2 {
  font-size: 19px; margin: 40px 0 12px; padding-top: 10px;
  border-top: 1px solid var(--rule); font-weight: 700;
}
h3 { font-size: 16px; margin: 26px 0 8px; font-weight: 700; }
h4 { font-size: 15px; margin: 20px 0 6px; font-weight: 700; font-style: italic; }
p { margin: 0 0 13px; text-align: justify; hyphens: auto; orphans: 2; widows: 2; }
strong { font-weight: 700; }
em { font-style: italic; }
a { color: var(--accent); text-decoration: none; border-bottom: 1px dotted #c9a; }
blockquote {
  margin: 16px 0; padding: 10px 18px; border-left: 3px solid var(--accent);
  background: #faf7f4; color: var(--ink-2); font-size: 14.5px;
}
blockquote p:last-child { margin-bottom: 0; }
code {
  font-family: "SF Mono", "Cascadia Mono", Consolas, monospace; font-size: 13px;
  background: var(--code-bg); padding: 1.5px 5px; border-radius: 3px;
  overflow-wrap: anywhere;
}
pre {
  background: var(--code-bg); padding: 14px 16px; border-radius: 4px;
  overflow-x: auto; font-size: 12.6px; line-height: 1.55;
}
pre code { background: none; padding: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 18px 0 20px; font-size: 13.6px;
  font-variant-numeric: tabular-nums;
  break-before: avoid; page-break-before: avoid;
}
th {
  text-align: left; font-weight: 700; border-top: 1.5px solid var(--ink);
  border-bottom: 1px solid var(--ink); padding: 7px 9px; vertical-align: bottom;
}
td { padding: 6px 9px; border-bottom: 1px solid #e9e9e9; vertical-align: top; }
th { white-space: nowrap; }
tbody tr:last-child td { border-bottom: 1.5px solid var(--ink); }
hr { border: 0; border-top: 1px solid var(--rule); margin: 34px 0; }
img { display: block; max-width: 100%; height: auto; margin: 22px auto 10px; }
p > img + strong, p:has(> img) { text-align: center; }
figure { margin: 22px 0; }
ul, ol { margin: 0 0 13px; padding-left: 24px; }
li { margin-bottom: 5px; text-align: left; }
/* 首行元信息（Authors / 单位 / 通讯 / 文章类型） */
.page > p:nth-of-type(-n+5) { color: var(--ink-2); font-size: 14px; margin-bottom: 4px; text-align: left; }
.abstract-label { }
@media print {
  body { background: #fff; font-size: 10.3pt; line-height: 1.5; }
  .page { box-shadow: none; margin: 0; padding: 0; max-width: none; }
  h2 { page-break-after: avoid; }
  table, pre, blockquote { page-break-inside: avoid; }
  @page { size: A4; margin: 20mm 18mm; }
}
@media (max-width: 700px) {
  .page { padding: 26px 20px 50px; }
  body { font-size: 15px; }
  table { font-size: 12.4px; }
  th, td { padding: 5px 6px; }
}
"""

# Elsevier review format: Times New Roman 12 pt, double spacing, wide margins,
# single column, no screen decorations. Mirrors the Neurocomputing guide's
# "double spacing with wide margins" requirement for the submitted manuscript.
CSS_REVIEW = """
:root { --ink: #000; --rule: #bbb; }
* { box-sizing: border-box; }
body {
  margin: 0; background: #ececec; color: var(--ink);
  font-family: "Times New Roman", Times, "Songti SC", "SimSun", "Microsoft YaHei", serif;
  font-size: 12pt; line-height: 2.0;
}
.page {
  max-width: 210mm; margin: 0 auto; background: #fff;
  padding: 25mm 25mm 25mm;
  box-shadow: 0 2px 18px rgba(0,0,0,.10);
}
h1 { font-size: 16pt; line-height: 1.4; margin: 0 0 16px; font-weight: 700; text-align: left; }
h2 { font-size: 13pt; margin: 20px 0 8px; font-weight: 700; border: 0; padding: 0; }
h3 { font-size: 12pt; margin: 16px 0 6px; font-weight: 700; }
h4 { font-size: 12pt; margin: 14px 0 6px; font-weight: 700; font-style: italic; }
p { margin: 0 0 6px; text-align: justify; hyphens: auto; }
strong { font-weight: 700; }
em { font-style: italic; }
a { color: var(--ink); text-decoration: none; }
blockquote {
  margin: 12px 0 12px 18px; padding: 0; border: 0; font-size: 12pt; font-style: italic;
}
blockquote p { margin-bottom: 4px; }
code {
  font-family: "Courier New", Courier, monospace; font-size: 10.5pt;
  background: none; padding: 0;
}
pre {
  background: none; padding: 0; margin: 10px 0; overflow-x: visible;
  font-family: "Courier New", Courier, monospace; font-size: 10pt; line-height: 1.35;
}
pre code { background: none; padding: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 10px 0 14px;
  font-size: 9.5pt; line-height: 1.25; font-variant-numeric: tabular-nums;
}
th {
  text-align: left; font-weight: 700; border-top: 1px solid #000;
  border-bottom: 1px solid #000; padding: 3px 5px; vertical-align: bottom;
}
td { padding: 2.5px 5px; border-bottom: 0; vertical-align: top; }
tbody tr:last-child td { border-bottom: 1px solid #000; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 16px 0; }
img { display: block; max-width: 100%; height: auto; margin: 14px auto 6px; }
p:has(> img) { text-align: center; }
ul, ol { margin: 0 0 6px; padding-left: 22px; }
li { margin-bottom: 2px; }
.page > p:nth-of-type(-n+8) { text-align: left; }
@media print {
  body { background: #fff; }
  .page { box-shadow: none; margin: 0; padding: 0; max-width: none; }
  h2, h3 { page-break-after: avoid; }
  table, pre, blockquote { page-break-inside: avoid; }
  img { page-break-inside: avoid; }
  @page { size: A4; margin: 25mm 22mm; }
}
"""


# IEEE Transactions two-column layout (IEEEtran-like), reproduced in CSS because
# no TeX distribution is available locally. Title and author block span the page;
# abstract and Index Terms open the first column; body runs in two columns;
# wide tables break out to full width the way `table*` does in IEEEtran.
CSS_IEEE = """
:root { --ink: #000; --rule: #000; }
* { box-sizing: border-box; }
body {
  margin: 0; background: #fff; color: var(--ink);
  font-family: "Times New Roman", Times, "Nimbus Roman", "Liberation Serif", serif;
  font-size: 10pt; line-height: 1.14;
  -webkit-font-smoothing: antialiased;
}
.masthead { text-align: center; margin: 0 0 10pt; }
.masthead h1 {
  font-size: 23pt; font-weight: 400; line-height: 1.14; letter-spacing: -.2px;
  margin: 0 0 12pt; text-align: center;
}
.masthead p { text-align: center; margin: 0 0 2pt; text-indent: 0; }
.masthead p:nth-of-type(1) { font-size: 11pt; margin-bottom: 8pt; }
.masthead p:nth-of-type(2),
.masthead p:nth-of-type(3) { font-size: 10pt; font-style: italic; margin-bottom: 2pt; }
.masthead p:nth-of-type(4),
.masthead p:nth-of-type(5) { font-size: 9pt; }
.body2 { column-count: 2; column-gap: 0.20in; text-align: justify; hyphens: auto; }
.body2 > *:first-child { margin-top: 0; }
h2 {
  font-size: 10pt; font-weight: 400; text-align: center; font-variant: small-caps;
  letter-spacing: .2px; margin: 11pt 0 4pt; break-after: avoid; page-break-after: avoid;
}
h3 {
  font-size: 10pt; font-weight: 400; font-style: italic; text-align: left;
  margin: 8pt 0 2pt; break-after: avoid; page-break-after: avoid;
}
h4 {
  font-size: 10pt; font-weight: 400; font-style: italic; text-align: left;
  margin: 6pt 0 2pt; break-after: avoid;
}
p { margin: 0 0 0; text-indent: 1.1em; text-align: justify; orphans: 2; widows: 2; }
p.abs, p.idx { text-indent: 0; }
p.abs { font-size: 9pt; font-weight: 700; margin: 0 0 6pt; }
p.idx { font-size: 9pt; font-style: italic; margin: 0 0 8pt; }
p.abs .lead, p.idx .lead { font-style: italic; font-weight: 700; }
strong { font-weight: 700; }
em { font-style: italic; }
a { color: var(--ink); text-decoration: none; }
blockquote {
  margin: 6pt 0 6pt 0; padding: 0 0 0 10pt; border-left: 2px solid #999;
  font-size: 9pt; font-style: italic;
}
blockquote p { text-indent: 0; }
code {
  font-family: "Courier New", Courier, monospace; font-size: 8.4pt;
  background: none; padding: 0; overflow-wrap: anywhere; word-break: break-all;
}
pre {
  background: none; padding: 0; margin: 6pt 0; overflow-x: visible;
  font-family: "Courier New", Courier, monospace; font-size: 6.9pt; line-height: 1.28;
}
pre code { background: none; padding: 0; font-size: 6.9pt; white-space: pre; }
img { display: block; width: 100%; height: auto; margin: 8pt auto 4pt; }
p:has(> img) { text-indent: 0; text-align: center; }
figure { margin: 8pt 0; }
ul, ol { margin: 2pt 0 6pt; padding-left: 16pt; }
li { margin-bottom: 2pt; text-align: justify; }
li p { text-indent: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 0 0 8pt;
  font-size: 7.4pt; line-height: 1.22; font-variant-numeric: tabular-nums;
}
.tblwrap { break-inside: avoid; page-break-inside: avoid; margin: 4pt 0 8pt; }
.tblwrap.wide { column-span: all; margin: 6pt 0 9pt; }
.tblwrap.wide table { font-size: 7.8pt; }
.figwrap { column-span: all; break-inside: avoid; page-break-inside: avoid; margin: 6pt 0 8pt; }
.figwrap img { width: 79%; margin: 0 auto 4pt; }
th {
  text-align: left; font-weight: 400; border-top: 0.8pt solid #000;
  border-bottom: 0.5pt solid #000; padding: 2pt 3pt; vertical-align: bottom;
}
td { padding: 1.6pt 3pt; border-bottom: 0; vertical-align: top; text-align: left; }
tbody tr:last-child td { border-bottom: 0.8pt solid #000; }
hr { display: none; }
.tcap {
  text-align: center; font-size: 8pt; margin: 8pt 0 2pt; text-indent: 0;
  break-after: avoid; page-break-after: avoid;
}
.tcap .tno { display: block; font-variant: small-caps; letter-spacing: .3px; }
.fcap { font-size: 8pt; text-align: justify; margin: 0 0 8pt; text-indent: 0; }
.ref {
  font-size: 7.8pt; text-align: justify; text-indent: -1.5em; padding-left: 1.5em;
  margin: 0 0 1.5pt; break-inside: avoid; page-break-inside: avoid;
}
@media screen {
  body { background: #ececec; }
  .sheet {
    max-width: 200mm; margin: 0 auto; background: #fff; padding: 18mm 15mm;
    box-shadow: 0 2px 18px rgba(0,0,0,.12);
  }
}
@media print {
  @page { size: letter; margin: 0.72in 0.62in 0.85in 0.62in; }
  .sheet { max-width: none; margin: 0; padding: 0; box-shadow: none; }
  table, pre, blockquote, img { page-break-inside: avoid; }
}
"""


ROMAN = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def _roman(n: int) -> str:
    s = ""
    for v, r in ROMAN:
        while n >= v:
            s += r
            n -= v
    return s


def _ieee_caption(tag: str, is_table: bool) -> str:
    """Table 1 -> TABLE I ; Table B1 -> TABLE B-I ; Figure 3 -> Fig. 3."""
    if is_table:
        if tag.upper().startswith("B"):
            return f"TABLE B-{_roman(int(tag[1:]))}"
        return f"TABLE {_roman(int(tag))}"
    return f"Fig. {int(tag)}."


def transform_ieee(body: str) -> tuple[str, str]:
    """返回 (masthead_html, body_html)。"""
    # 摘要：h2 Abstract + 紧跟的段落 -> 粗体 run-in
    body = re.sub(
        r"<h2[^>]*>\s*Abstract\s*</h2>\s*<p>(.*?)</p>",
        r'<p class="abs"><span class="lead">Abstract&mdash;</span>\1</p>',
        body,
        count=1,
        flags=re.S,
    )
    # Index Terms
    body = re.sub(
        r"<p><strong>Index Terms:?</strong>\s*(.*?)</p>",
        r'<p class="idx"><span class="lead">Index Terms&mdash;</span>\1</p>',
        body,
        count=1,
        flags=re.S,
    )

    # 表题：<p><strong>Table 1. 标题</strong>尾文</p>
    def _tcap(m: re.Match) -> str:
        tag, title, rest = m.group(1), m.group(2).strip(), m.group(3)
        html_t = f'<div class="tcap"><span class="tno">{_ieee_caption(tag, True)}</span>{title}'
        if rest.strip():
            html_t += " " + rest.strip()
        return html_t + "</div>"

    body = re.sub(
        r"<p><strong>Table (B?\d+)\.\s*(.*?)</strong>(.*?)</p>", _tcap, body, flags=re.S
    )

    # 图题：<p><strong>Figure 1. 标题</strong>尾文</p>
    def _fcap(m: re.Match) -> str:
        tag, title, rest = m.group(1), m.group(2).strip(), m.group(3)
        out = f'<div class="fcap"><span class="fno">{_ieee_caption(tag, False)}</span> {title}'
        if rest.strip():
            out += " " + rest.strip()
        return out + "</div>"

    body = re.sub(
        r"<p><strong>Figure (\d+)\.\s*(.*?)</strong>(.*?)</p>", _fcap, body, flags=re.S
    )

    # 参考文献段落：h2 References 之后全部 <p> 加 class="ref"
    m = re.search(r"<h2[^>]*>\s*References\s*</h2>", body)
    if m:
        head, tail = body[: m.end()], body[m.end():]
        tail = tail.replace("<p>", '<p class="ref">')
        body = head + tail

    # 宽表：列数 >= 5 时跨栏；同时把表题与表格打包成不可拆的块
    def _wrap_table(m: re.Match) -> str:
        cap, tbl = m.group(1), m.group(2)
        first_row = tbl.split("</tr>")[0]
        # 注意不能用 count("<th")：会把 <thead> 数进去
        ncol = len(re.findall(r"<th[ >]", first_row))
        cls = "tblwrap wide" if ncol >= 5 else "tblwrap"
        tbl = tbl.replace(' class="wide"', "")
        return f'<div class="{cls}">{cap}{tbl}</div>'

    body = re.sub(
        r'(<div class="tcap">.*?</div>)\s*(<table[^>]*>.*?</table>)',
        _wrap_table,
        body,
        flags=re.S,
    )

    # 图：图片段落与图题打包，并跨栏（图都是宽示意图）
    body = re.sub(
        r'(<p[^>]*>(?:\s*<img[^>]*>)+\s*</p>)\s*(<div class="fcap">.*?</div>)',
        r'<div class="figwrap">\1\2</div>',
        body,
        flags=re.S,
    )

    # 拆分报头（第一个 <hr /> 之前的部分）
    parts = re.split(r"<hr\s*/?>", body, maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", body


def build(md_path: Path, out_path: Path, fmt: str = "pretty") -> None:
    src = md_path.read_text(encoding="utf-8")
    # 去掉首行元信息里的反引号与分隔，交给 CSS 淡化
    body = markdown.markdown(
        src,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
        output_format="html5",
    )
    # 参考文献段落：让 [1] 之类不换行折行
    title = "Appearance Is Not Enough"
    m = re.search(r"<h1>(.*?)</h1>", body, re.S)
    if m:
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip()

    # 中英文版共用一套模板：按 CJK 字符占比自动判定 lang
    cjk = len(re.findall(r"[\u4e00-\u9fff]", src))
    lang = "zh" if cjk > len(src) * 0.15 else "en"

    css = CSS_REVIEW if fmt == "review" else (CSS_IEEE if fmt == "ieee" else CSS)
    if fmt == "ieee":
        masthead, rest = transform_ieee(body)
        inner = (
            f'<div class="masthead">\n{masthead}\n</div>\n'
            f'<div class="body2">\n{rest}\n</div>'
        )
        shell = f'<div class="sheet">\n{inner}\n</div>'
    else:
        shell = f'<div class="page">\n{body}\n</div>'

    doc = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{css}</style>
</head>
<body>
{shell}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    kb = len(doc.encode("utf-8")) / 1024
    print(f"wrote {out_path}  ({kb:.0f} KB, {len(src.split())} words in source, fmt={fmt})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fmt = "pretty"
    if "--review" in sys.argv:
        fmt = "review"
    elif "--ieee" in sys.argv:
        fmt = "ieee"
    elif "--format" in sys.argv and "review" in sys.argv:
        fmt = "review"
    md = Path(args[0]) if len(args) > 0 else ROOT / "docs/paper/main.md"
    out = Path(args[1]) if len(args) > 1 else ROOT / "docs/paper/main.html"
    build(md, out, fmt)
