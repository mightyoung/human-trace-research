# IEEE T-BIOM 投稿合规清单

> 目标刊：*IEEE Transactions on Biometrics, Behavior, and Identity Science*（T-BIOM，EIC: Mark Nixon）
> 稿件类型：**Regular Paper**　|　投稿系统：**IEEE Author Portal**（底层 ScholarOne）
> 核对依据：IEEE Biometrics Council *Author Instructions*（ieee-biometrics.org）、IEEE Author Center 模板与编辑风格手册、IEEE ORCID 政策
> 核对日期：2026-11-14　|　稿件版本：B 案（`main-ieee.pdf`，18 页）

---

## 0. 结论先行

| 类别 | 数量 | 说明 |
|---|---:|---|
| 已就绪，可直接上传 | 8 项 | 主稿、补充材料、三图、cover letter、title page、声明、摘要、关键词 |
| **必须由作者本人填** | 5 项 | 姓名/单位/邮箱/电话、ORCID（全部作者）、推荐审稿人、伦理批准编号、作者简介与照片 |
| **有硬风险，需先决策** | 3 项 | ① 页数 18 vs 基数 10　② 未用 IEEEtran 模板　③ 表编号非罗马数字 |

**三项硬风险按严重度排序**：页数 > 模板 > 表编号。前两项编辑在初审就可能提出，第三项通常在排版/校样阶段被迫改。

---

## 1. 官方硬指标逐条对照

| # | T-BIOM 要求（原文要点） | 现状 | 位置 |
|---|---|---|---|
| 1 | **Regular paper 上限 10 页**（"Regular papers are up to 10 pages in length"），完成最终排版后超出部分按 IEEE MOPC 计费 | ⚠️ **18 页，超 8 页** | `05-manuscript/main-ieee.pdf` |
| 2 | 补充材料**必须独立文件**，"must not be included within the same PDF file as the main paper"；补充材料无页数上限 | ✅ 独立 PDF，2 页 | `06-supplement/` |
| 3 | 数据集：主结果若基于**不公开**数据集，"this fact must be clearly explained and justified in the submission **and in the cover letter**" | ✅ 正文 Data Availability 已述；cover letter §4 专段 | 两处 |
| 4 | 可复现性：鼓励提供代码/模型/演示；不能提供时"detailed documentation (e.g., pseudocode, complete algorithmic descriptions, or scripts) must be included" | ✅ 补充材料 S1 列全部脚本、结果文件、命令 | `06-supplement/` |
| 5 | 必须在引言与文献综述中"provide a clear statement of the problem addressed"及相对 SOTA 的改进 | ✅ §I + 6 条贡献 | `main-ieee.pdf` p1–2 |
| 6 | **立即拒稿**风险：limited novelty（"a straightforward combination of theories and algorithms that are well established, and are repeated on a known scenario"） | ✅ 已把「不可组合性」提到贡献第 1 条，作为结构性结果而非方法组合 | 摘要 / §I / §VII |
| 7 | 实验数据不足或过时 → 直接拒（不送审） | ✅ 3 个采集域、4 天、26 条 2020–2026 文献 | 全文 |
| 8 | 自我引用不得过度（"not gratuitously cite the authors' own previous works"） | ⚠️ 26 条中本课题前作 0 条；作者既有发表需本人复核一遍 | `main-ieee.pdf` p17–18 |
| 9 | 不得一稿多投；会议扩展稿需 ≥30% 新内容 + 引用 + 说明 | ✅ 原创，无会议版 | cover letter §5 |
| 10 | CrossCheck 查重（提交即视为授权） | — 系统自动，无需动作 | — |
| 11 | 英文质量："large number of typographical and/or grammatical errors will also be rejected without review" | ⚠️ 建议投稿前做一次专业润色（IEEE 与 AJE 等有协议折扣） | 全文 |
| 12 | 摘要 ≤ 250 词 | ⚠️ **恰 250 词**，不同计数器可能溢出，建议压到 245 | `main-ieee.pdf` p1 |
| 13 | Index Terms | ✅ 6 个（Person re-identification; Biometric evaluation protocol; Benchmarking; Confidence interval estimation; Surveillance identity analysis; Human-in-the-loop） | p1 |
| 14 | **全部作者需 ORCID**（"All IEEE journals require an ORCID for all authors"） | ⚠️ 占位符待填 | `03-title-page.md` |
| 15 | 单盲评审（作者身份对审稿人可见，无需匿名；以投稿系统页面为准） | ✅ 正文含作者块 | — |
| 16 | 图：单独文件提交，矢量优先 | ✅ 3 张矢量 PDF（已裁到图幅，525 pt 宽）+ SVG 源 | `07-figures/` |
| 17 | 表：IEEE 风格，表号用**罗马数字**（TABLE I） | ⚠️ 现为阿拉伯数字（Table 1–12、A1–A3） | 全文 |
| 18 | 作者简介（biographies）+ 证件照，IEEE Transactions 惯例 | ⚠️ 缺，需模板 | `09-author-biographies.md` |
| 19 | 使用 IEEE 官方模板（IEEEtran / Word 模板） | ⚠️ 现为自制 HTML→PDF，版式接近但非模板 | 见 §2 |
| 20 | 预印本：允许 arXiv，投稿时声明 | ⚠️ 若有 arXiv 版需在系统勾选并注明编号 | `04-declarations.md` |

---

## 2. 三项硬风险的处理建议

### 2.1 页数：18 页 vs 基数 10 页

官方原文只给上限与「超出按 MOPC 计费」，没有写死"最多 14 页"。实务上 T-BIOM 见刊论文多在 12–14 页。

三条路，取舍如下：

| 方案 | 做法 | 代价 | 推荐度 |
|---|---|---|---|
| **A. 接受 MOPC 投 18 页** | 不动内容，投稿时接受超页费 | 费用按页计，投稿系统会先算给你；超 8 页成本最高 | 若经费充足且时间紧，选它 |
| **B. 压到 14–15 页再投** | 见 §2.1 清单 | 需砍约 3–4 页内容（约 2 个表 + 1 节） | **推荐**：14 页是同类论文的常态区间 |
| **C. 改投 14 页基数的刊** | IEEE TIM / Pattern Recognition Letters 等 | 重写叙事与 scope 论证 | 若 B 砍不动时的退路 |

**到 15 页还能砍什么（按损失从小到大）**：

1. Table A2（T2 组级结果，8 行）→ 并入 A1 脚注：−0.15 页
2. §IX 部署成本与运维 → 压成 1 段并入 §X：−0.6 页
3. §II 相关工作 → 每类压 1 句，删 3 条低相关文献：−0.5 页
4. Table 9（验证器扫描）→ 只留 3 个阈值点：−0.3 页
5. §VIII-D/E 两小节合并：−0.5 页
6. Fig. 3（协议全景）与 Table 5 信息重叠 → 删图保表：−0.7 页
7. 表字号 7.4→7.2 pt、参考文献 8→7.5 pt：−0.25 页

1–7 合计约 −2.8 页 → 15 页；再叠一层内容压缩可到 14。

### 2.2 模板

现状：`tools/md2paper_html.py --ieee` 生成双栏 10 pt Times、栏距 0.20 in、页边距 18/15 mm 的 PDF。视觉与 IEEEtran 接近，但**不是模板**：栏宽、页眉页脚、标题字号、表标题位置（IEEE 要求表题在表**上方**、全大写罗马数字）都存在差异。

两条路：

- **投稿初稿阶段**：多数编辑接受"接近 IEEE 双栏"的 PDF，格式问题在 revision 时改。风险低。
- **若要用官方模板**：需把 15 张表 + 3 图 + 26 条文献迁到 LaTeX（IEEEtran.cls）。工作量约 1–2 天，且会重新触发分页与页数核算。

**建议**：初投用现 PDF（并接受 §2.1-A 或 B 的页数决定）；若进入 major revision，再迁 IEEEtran。

### 2.3 表编号罗马数字

IEEE 编辑风格手册：表号用罗马数字（TABLE I、TABLE II），图号用阿拉伯数字（Fig. 1）。现全文 15 张表用阿拉伯数字，正文引用约 50 处。

改法（可脚本化，一次完成、英中同步）：`Table 1..12 → TABLE I..XII`、`Table A1..A3 → TABLE A-I..A-III`，同时改正文引用。**建议在页数方案定下来后一起做**，避免两次重排。

---

## 3. 文件清单与系统字段映射

| 包内文件 | 上传到 Author Portal 的哪个字段 | 备注 |
|---|---|---|
| `05-manuscript/main-ieee.pdf` | **Manuscript**（主文件） | 图表已嵌入正文，勿再传匿名版 |
| `06-supplement/supplement-S1-reproducibility.pdf` | **Supplementary Material** | 必须独立文件，不可并入主 PDF |
| `07-figures/Figure_1.pdf` ~ `Figure_3.pdf` | **Figure**（逐张上传） | 矢量，已裁图幅；同名 SVG 为源文件，可一并上传 |
| `02-cover-letter.md` → 贴入或转 PDF | **Cover Letter** | 含数据集不可及论证（官方强制） |
| `03-title-page.md` → 转 PDF | **Title Page** | 作者、单位、ORCID、通讯方式 |
| `04-declarations.md` | 系统表单逐项填写 | 伦理、数据、COI、资助、AI、预印本 |
| `08-suggested-reviewers.md` | **Suggested Reviewers** | 3–5 位，需机构邮箱 |
| `09-author-biographies.md` | 校样阶段（或随稿提交） | 三段式 + 证件照 |

---

## 4. 只能由作者本人填写的 5 项

| 位置 | 待填 |
|---|---|
| `03-title-page.md` | 全部作者姓名、单位英文全称、城市/邮编/国家、通讯作者邮箱与电话 |
| `03-title-page.md` / 系统 | **每一位作者**的 ORCID（IEEE 强制，缺一位都无法提交） |
| `08-suggested-reviewers.md` | 3–5 位推荐审稿人（机构邮箱） |
| `04-declarations.md` | 伦理审查：人类受试数据的批准机构与编号（无正式批件时，需写明豁免依据与数据使用协议） |
| `09-author-biographies.md` | 三段式简介 + 专业证件照（不接受 QR 码替代） |

---

## 5. scope 论证（编辑最可能问的第一个问题）

T-BIOM 的 broad topics 含 "standards for their deployment"、"novel (challenging) biometric databases, protocol and benchmarking, large scale evaluation, confidence interval estimation, and performance modeling and prediction"、"person re-identification"、"detection, tracking, identification, and de-identification in surveillance"。

本文定位：**部署前的可用性测量协议**——不是新网络，而是把「这套机位能支持哪些身份功能」变成三项可测探针 + 五级带验收条件的交付。这条定位同时落在 "protocol and benchmarking" 与 "performance modeling and prediction" 两项里，也在 surveillance re-identification 的应用语境中。

**desk reject 的最可能理由**（提前想好答法）："limited novelty — 组合已有方法"。答法已写进 §I 与 §VII：不可组合性是结构性的（对全局划分施加局部约束必然伤及未处理的正确合并，三个独立演示），不是把几个模块拼起来；三探针中 P1/P3 承自既有标准与类规模统计，本文的新增在 **P2 的答案单位（按区域而非机位对）** 与 **分级协议本身**——这一点摘要与 §I 已如实声明"新颖度不均等"。

---

## 6. 提交前 24 小时检查表

- [ ] 页数方案已定（18 页接受 MOPC / 压到 15 / 压到 14）
- [ ] 表编号已罗马化（若决定做）
- [ ] 摘要压到 ≤245 词
- [ ] 全部作者 ORCID 已收集并验证姓名拼写与稿件一致
- [ ] 三张图在 PDF 里清晰、无跨页断裂
- [ ] 补充材料 PDF 里没有指向"正文 Appendix A"的失效引用
- [ ] Cover letter 已填日期与署名
- [ ] 系统里作者顺序与稿件一致（改序需全体作者确认）
- [ ] CrossCheck 自查（可用 iThenticate 先跑一次）
- [ ] 预印本状态已确认（有无 arXiv 版）
