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

    css = CSS_REVIEW if fmt == "review" else CSS
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{css}</style>
</head>
<body>
<div class="page">
{body}
</div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    kb = len(doc.encode("utf-8")) / 1024
    print(f"wrote {out_path}  ({kb:.0f} KB, {len(src.split())} words in source, fmt={fmt})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fmt = "review" if "--format" in sys.argv and "review" in sys.argv else "pretty"
    if "--review" in sys.argv:
        fmt = "review"
    md = Path(args[0]) if len(args) > 0 else ROOT / "docs/paper/main.md"
    out = Path(args[1]) if len(args) > 1 else ROOT / "docs/paper/main.html"
    build(md, out, fmt)
