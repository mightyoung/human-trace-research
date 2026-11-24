# Retarget: Neurocomputing → IEEE T-BIOM

日期：2026-11-12
决策依据：`docs/journal-target-decision-2026-11-12.md`（用户选择方案 A）

---

## 1. 为什么换

Neurocomputing 的 Article types 第一条对 Regular article 有明文排除：

> "Regular submissions must be directly relevant to Neural Networks and/or Learning Systems.
> Please note that Neurocomputing does NOT accept submissions on ... pure image processing or
> pure combination of existing algorithms ..."

本文五个贡献里没有一个是新的网络或学习算法，两个关键词（off-the-shelf、frozen）在文里出现多次，
正落在排除条款上。desk reject 风险是条款级的，不是审稿意见级的。

T-BIOM 的 Call for Papers 主题列表里有一条直接命中：

> "Novel (challenging) biometric databases, protocol and benchmarking, large scale evaluation,
> confidence interval estimation, and performance modeling and prediction"

另有 "Person re-identification"、"Detection, tracking, identification, and de-identification in
surveillance" 两条。本文的主干（三探针 + 五级协议 + 两条不可互替的数字 + κ 与 CI）就是这条 bullet。

---

## 2. 本轮已做的改动（英文 `main.md` 与中文 `main-zh.md` 同步）

| # | 位置 | 改动 | 目的 |
|---|---|---|---|
| 1 | 摘要前两句 | "We answer with a deployment-time measurement procedure." → "We treat this as a measurement problem in identity science and answer with a deployment-time protocol." | 把定位从「部署方法」挪到「身份科学里的测量」 |
| 2 | 摘要 | 词数 249 → **250**，全部数字未动 | 守住 250 词上限 |
| 3 | 关键词 | `Cross-camera ReID; Deployment protocol; Selective prediction; Human-in-the-loop; Co-visibility; Field study` → `Person re-identification; Biometric evaluation protocol; Benchmarking; Confidence interval estimation; Surveillance identity analysis; Human-in-the-loop` | 逐词对齐 T-BIOM 主题列表，便于编辑分派审稿人 |
| 4 | §1 第一段 | 新增一句：「The contribution is a measurement instrument, and we treat it as one — what is measured is identity completeness, what is reported is a population together with an interval, and what is delivered is a set of obligations that can be written into a contract and checked after acceptance.」 | 让「测量」这个身份在第一屏就出现 |
| 5 | **新增 §2.6「Evaluation protocols and benchmarking in identity science」** | 三段：FERET 的协议范式 → Best-Rowden 的时间维度 → **本刊 Kang 的基准有效性** → 本文是它在现场采集上的对应物 | 直接接上 T-BIOM 的评估传统，并引用一篇 T-BIOM 论文 |
| 6 | 参考文献 | 新增 3 条（[20] FERET / [21] Best-Rowden & Jain / [22] Kang），全部经 Crossref 核 DOI，23 条 → **26 条** | 补 I&M/身份科学脉络 |
| 7 | 编号 | 按首引顺序重编：新增三条落在 **[20]–[22]**，原 [20]–[23] 顺延为 [23]–[26] | 新章节在 §3 之前，所以首引更早 |
| 8 | `submission/highlights.txt` | 第 1 条改为「Benchmark mAP does not transfer; three label-free probes predict what a site supports」 | 首条即点「基准不可迁移」 |

### 新增三条文献（DOI 已核）

```
[20] P.J. Phillips, H. Moon, S.A. Rizvi, P.J. Rauss, The FERET evaluation methodology for
     face-recognition algorithms, IEEE TPAMI 22 (2000) 1090–1104. 10.1109/34.879790
[21] L. Best-Rowden, A.K. Jain, Longitudinal study of automatic face recognition,
     IEEE TPAMI 40 (2018) 148–162. 10.1109/TPAMI.2017.2652466
[22] C. Kang, Are synthetic datasets reliable for benchmarking generalizable person
     re-identification?, IEEE T-BIOM 7 (2025) 146–155. 10.1109/TBIOM.2024.3459828
```

★ Kang 是**单作者**（Cuicui Kang），不能写 et al.。★ 正文只陈述「该文问的是数据来源合法性」，
**没有替它下结论**——因为只核了标题与元数据，没读全文。

---

## 3. 编号是怎么改的（可复现）

手改编号必错。新增文献先以临时号 `[24]`–`[26]` 追加到文献表并在正文引用，然后：

```bash
$PY tools/renumber_refs.py docs/paper/main.md --write
$PY tools/renumber_refs.py docs/paper/main-zh.md --write
```

脚本按**首次引用顺序**重编正文与文献表，并在以下情况**报错退出、不写文件**：

- 文献表有条目从未被引用（防止静默丢文献）；
- 正文引用了不存在的编号；
- 编号越界。

跑完再跑一次应报「0 条编号会变」，即自洽。英文与中文两次跑出的映射一致（都是 [24]→[20]）。
置信区间 `[0.321, 0.585]` 与 ORCID `[0000-0000-...]` 不会被当作引用（正则只认 1–2 位纯数字）。

---

## 4. 重建产物（本轮）

```bash
$PY tools/md2paper_html.py docs/paper/main.md       docs/paper/main.html
$PY tools/md2paper_html.py docs/paper/main.md       docs/paper/main-review.html --review
$PY tools/md2paper_html.py docs/paper/main-zh.md    docs/paper/main-zh.html
# 再对三个 html 跑 Chrome --headless=new --print-to-pdf --no-pdf-header-footer
```

页数（按 PDF `/Count` 读）：

| 文件 | 页数 | 说明 |
|---|---:|---|
| `main.pdf` | **20** | 阅读版 |
| `main-review.pdf` | **36** | 双倍行距审稿版 |
| `main-zh.pdf` | **22** | 中文全译 |

⚠ **与上一轮记的 21 / 35 / 23 不可直接比**：上一轮页数的测量口径没有记录，
本轮改用 PDF 根 `/Count` 读取，且内容是净增的（+1 节 +3 文献 +1 句摘要）。
以后报页数一律注明「按 `/Count`」。

---

## 5. 还没做的（换 IEEE 必须做，本轮不做）

1. **版面**：现在是 md → HTML → Chrome PDF 的 A4 单栏散文版，**不是 IEEEtran 双栏**。
   真投稿要么换 `IEEEtran.cls` 重排，要么用 IEEE Author Portal 的模板服务。这是机械工作但必须做。
2. **页数**：TIM/多数 IEEE 汇刊对 regular paper 有隐含页数约束，T-BIOM 没有硬上限，
   但 36 页双倍行距偏长（约合 18 页双栏）。建议压到 ≤14 页双栏，压缩点在第一优先是 §3 的数据描述与附录表格。
3. **IEEE 必备件**：作者 ORCID、IEEE copyright 表单、`\begin{IEEEkeywords}` 形态的关键词、
   图注格式（Fig. 1. 而非 Figure 1）、参考文献 IEEE 风格（现在是 Elsevier 风格，需整体换格式）。
4. **`submission/` 里按 Elsevier 做的六段声明**（CRediT、competing interest、funding、data availability、
   generative AI、acknowledgements）在 IEEE 下要改成 IEEE 的表述，CRediT 可保留。
5. **Cover letter** 已按 T-BIOM 重写（`submission-tbiom/cover-letter.md`），核心是把
   T-BIOM 主题列表那条 bullet 原文引出来，说明本文命中的是哪一条。
6. **对级 κ 仍缺**（153 条，单标注人）。T-BIOM 的审稿人比 Neurocomputing 更可能盯这一条，
   因为评估协议正是该刊的正题。盲标页已就绪：`deliverable/spacetime_review/kappa2/kappa2_blind.html`。
   **这是投稿前唯一一条「花钱也补不上、只能找人」的缺口。**

---

## 6. 一句话交接

正文与文献已按 T-BIOM 定位改完并重建（20/36/22 页）；剩下的是**版面（IEEEtran 双栏）+ 参考文献 IEEE 风格**
这两件机械活，以及**对级 κ 需要一个真人**。前两件我可以做，第三件只能等你安排第二标注人。

---

# 第二轮：IEEE 版面与文献风格（2026-11-13）

上面 §5 的 1、2、3（部分）、4 已完成。**仍未做**：ORCID / copyright 表单（须作者本人）、对级 κ（须第二标注人）。

## 7. 本轮改了什么

本地**没有 TeX 发行版**（`pdflatex` / `xelatex` / `tectonic` 均不存在），因此没有走 `IEEEtran.cls`，
而是用 CSS 复刻 IEEEtran 双栏版式，仍由 Chrome 打印成 PDF。产物：

| 文件 | 说明 |
|---|---|
| `main.md` | 正文源（已改 IEEE 口径） |
| `main-ieee.html` | 双栏版式，可直接浏览器打开校对 |
| `main-ieee.pdf` | **15 页双栏**，提交用 |

### 7.1 内容（`main.md`）

| # | 改动 | 证据 |
|---|---|---|
| 1 | Article type：`Submitted to Neurocomputing` → `Regular Paper. Submitted to IEEE Transactions on Biometrics, Behavior, and Identity Science` | 1 处 |
| 2 | `Keywords:` → `Index Terms:`（IEEE 用词） | 1 处 |
| 3 | 章节号 阿拉伯 → 罗马：`## 5` → `## V.`，11 个正文章 + 附录 A/B | 11 章 |
| 4 | 子节 `### 5.3` → `### C.`，按父节内出现顺序给字母 | 35 个子节 |
| 5 | 正文交叉引用 `Section 5.3` → `Section V-C`；`Sections 5 to 7` → `Sections V–VII`；`Sections 7.2 and 7.4` → `Sections VII-B and VII-D` | 34 处单引 + 2 处区间 |
| 6 | 正文 `Table 10` → `Table X`，`Table B1` → `Table B-I` | 13 处 |
| 7 | 正文 `Figure 2` → `Fig. 2` | 5 处 |
| 8 | 参考文献 26 条整体换 IEEE 风格：作者缩写前置、题名加引号、期刊名斜体、`in Proc. …`、`vol./pp./doi:`；**未新增任何原来没有的卷期页码**（只做格式转换，不编造元数据） | 26 条 |
| 9 | 六段 Elsevier 声明换成 IEEE 五段：`Acknowledgment`（合并 funding + AI 使用披露）、**`Ethics Statement`（T-BIOM 必需，原来没有）**、`Conflicts of Interest`、`Data Availability`、`Author Contributions` | 5 段 |
| 10 | **修一个真实结构 bug**：原 `[22]` 之后附录 A/B 被压成一行、且 `[23]–[26]` 排到了附录**后面**。现在改为 IEEE 顺序：正文 → 附录 A → 附录 B → 声明 → 参考文献 | 见下 |

### 7.2 版式（`tools/md2paper_html.py` 新增 `--ieee`）

- 报头（标题 23 pt / 作者 11 pt / 单位斜体 10 pt）通栏；正文 `column-count: 2`，栏间距 0.20 in
- 摘要 `Abstract—` 粗体 run-in 9 pt；`Index Terms—` 斜体 9 pt，均不缩进
- 章节标题居中小型大写、子节斜体左对齐
- 表题 `TABLE I` / `TABLE B-I` 独立成行，居中；图题 `Fig. 1.`；题注与图表打包成 `break-inside: avoid` 的块
- 列数 ≥5 的表跨栏（对应 `table*`），3 张图全部跨栏
- 参考文献 7.8 pt 悬挂缩进，`break-inside: avoid`

### 7.3 版式踩过的两个坑（都是「看起来对但其实错」）

1. **表题与表格被分页拆开**：`TABLE IV` 落在上一页页脚、表格在下一页页首。因为只给 `<table>` 设了
   `page-break-inside: avoid`，题注是兄弟节点。改为把 `题注 + 表格` 包进一个 `.tblwrap` 再设 avoid。
2. **列数统计把 `<thead>` 数成了 `<th`**：`html.count("<th")` 会把 `<thead>` 计入，导致 4 列表格被误判成
   5 列而跨栏（`TABLE III / VI / VII / VIII / IX / B-III` 全部错跨）。改成 `re.findall(r"<th[ >]", …)`。

### 7.4 校验方式

不是靠看截图，而是**数字比对**：把 PDF 文本与 HTML 文本各自抽英文词数，比值 **1.035**
（PDF 略高来自连字符断词），并断言 12 个关键串（`I. Introduction` … `XI. Conclusion`、`Appendix A/B`、
`Ethics Statement`、`References`、`[26] GB/T`）全部命中 —— 用来证明两栏 + 跨栏没有吞内容。

```bash
python tools/md2paper_html.py docs/paper/main.md docs/paper/main-ieee.html --ieee
chrome --headless=new --no-pdf-header-footer \
  --print-to-pdf=docs/paper/main-ieee.pdf docs/paper/main-ieee.html
python tmp/check_ieee_pdf.py --png   # 页数 / 词数比 / 关键串 / 逐页首尾词
```

### 7.5 仍未解决

1. **15 页双栏**（原 36 页双倍行距 ≈ 18 页双栏，已压下来，但比 14 页的理想值多 1 页）。若编辑要求 ≤14，
   优先砍附录 B 的三张表（可移到补充材料）。
2. **无页眉页脚页码**（Chrome 打印不生成），正式投稿版用 IEEE 模板时会自动带上。
3. **对级 κ 仍缺**，同 §5 第 6 条。
4. `main.html` 已按新正文重生成（单栏阅读版，内容与 `main-ieee.html` 一致，仅版式不同）；
   `main-review.html` / `main-review.pdf` 仍是旧的 Elsevier 双倍行距版，**已作废，不要拿去投稿**。

