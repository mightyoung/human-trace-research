# 投稿包导览 — IEEE T-BIOM

> ⚠️ **本目录已被 `docs/paper/tbiom-r2/` 取代（2026-11-14）。** 这里的稿子是 18 页的自制 HTML→PDF 版；r2 已是 IEEEtran 模板、**10 页**、33 处作者查询只剩 3 处。下文提到的页数、图表数与"迁 LaTeX"工作量均已过期，仅保留声明表、推荐审稿人、biography 模板仍可复用。

> 一句话：**主稿 18 页、补充材料 2 页、3 张矢量图、6 份文字材料**，其中 5 项必须由作者本人填。
> 打开顺序：先看 `01-CHECKLIST.md` 的 §0 结论表，再回来看本页的上传顺序。

---

## 一、包内文件地图

```
submission-tbiom/
├── 00-SUBMIT-GUIDE.md          本文件：上传顺序与字段映射
├── 01-CHECKLIST.md             ★ 合规清单（含 3 项硬风险）
├── 02-cover-letter.md          Cover letter（含数据集不可及论证，官方强制）
├── 03-title-page.md            Title page（作者/单位/ORCID/统计/声明汇总）
├── 04-declarations.md          12 项声明，对应系统逐项表单
├── 05-manuscript/
│   └── main-ieee.pdf           ★ 主稿，18 页，图表已嵌入
├── 06-supplement/
│   └── supplement-S1-reproducibility.pdf   ★ 补充材料，2 页
├── 07-figures/
│   ├── Figure_1.pdf            矢量，已裁到图幅（525×300 pt）
│   ├── Figure_2.pdf            矢量（525×225 pt）
│   ├── Figure_3.pdf            矢量（525×323 pt）
│   └── fig1–3_*.svg            源文件（可一并上传）
├── 08-suggested-reviewers.md   推荐审稿人模板 + 候选池 + 回避规则
├── 09-author-biographies.md    三段式简介模板 + 照片规格
├── MANIFEST.csv                文件清单 + 大小 + SHA-256
├── SHA256SUMS.txt              校验和
└── _archive/                   11-13 的旧版（NeurIPS/Elsevier 口径），仅供对照
```

---

## 二、Author Portal 上传顺序

系统把材料拆成几个必填步骤，按下面顺序走，中途不会卡：

| 步骤 | 系统字段 | 用哪个文件 | 注意 |
|---|---|---|---|
| 1 | Article Type | — | 选 **Regular Paper** |
| 2 | Title / Abstract / Index Terms | 从 `03-title-page.md` 复制 | 摘要 250 词，粘贴后系统会自己数一遍 |
| 3 | Add Authors | `03-title-page.md` 的作者表 | **每位作者都要 Link ORCID**；姓名拼写必须与稿件 byline 逐字符一致 |
| 4 | Upload Files → **Cover Letter** | `02-cover-letter.md`（转 PDF 或粘贴） | 必须先填日期与署名 |
| 5 | Upload Files → **Manuscript** | `05-manuscript/main-ieee.pdf` | 只传这一份，不要另传匿名版 |
| 6 | Upload Files → **Supplementary Material** | `06-supplement/supplement-S1-reproducibility.pdf` | 官方要求独立文件、不得并入主稿；若系统另设 **Graphical Abstract** 字段，用 `07-figures/Figure_3.pdf`（三探针 × 五级台阶） |
| 7 | Upload Files → **Figure** | `07-figures/Figure_1..3.pdf` 逐张 | 若系统提示"图中含文字请上传源文件"，把同名 SVG 也传上 |
| 8 | Suggested Reviewers | `08-suggested-reviewers.md` | 3–5 位，机构邮箱 |
| 9 | Declarations（逐项勾选/填写） | `04-declarations.md` | 与稿件措辞保持一致 |
| 10 | Review & Submit | — | 提交前把 PDF 从头翻一遍，特别是图有没有跨页断裂 |

**接收后还要做**：IEEE Electronic Copyright Form（eCF）、作者简介与照片（`09-author-biographies.md`）、校样（proof）核对、若选开放获取则处理 APC。

---

## 三、三个必须先决策的问题

在点 Submit 之前，这三件事没定就别投：

1. **页数**：18 页对 Regular Paper 的 10 页基数是超 8 页，走 MOPC 按页计费。要么接受，要么按 `01-CHECKLIST.md` §2.1 压到 15 或 14 页。
2. **模板**：现稿是自制双栏 PDF，视觉接近 IEEEtran 但不是模板。初投通常可接受；若想一次到位，需迁 LaTeX（1–2 天）。
3. **表编号**：IEEE 要求罗马数字（TABLE I），现为阿拉伯数字。可脚本化批量改，建议与页数方案一起做，避免二次重排。

---

## 四、这份包里哪些是"成品"，哪些是"骨架"

| 状态 | 文件 |
|---|---|
| **成品，可直接用** | `05-manuscript/`、`06-supplement/`、`07-figures/`、`01-CHECKLIST.md`、`02-cover-letter.md`（除日期署名）、`04-declarations.md`、`08-suggested-reviewers.md` |
| **骨架，需作者填** | `03-title-page.md`（姓名/单位/ORCID/伦理编号）、`08` 的具体人名与邮箱、`09` 的三段简介与照片 |

---

## 五、若被 desk reject，最可能的两句话与应对

- **"Limited novelty — combination of established components."** 应对：把 §VII 的不可组合性作为主贡献前置（已经在 §I 与摘要里做了），并在 cover letter 第 3 节第 1 条给出三个独立演示。若编辑仍坚持，改投 IEEE TIM 更稳（TIM 接受"测量程序 + 不确定度预算"这一叙事）。
- **"Single-site, single-annotator study."** 应对：κ = 0.925 已在图库层给出；对级 153 条缺第二标注人，这是已知短板，如实写在 Limitations 第 2 条。补法：找第二人盲标 40–50 对，2 小时内可完成，若编辑要求再补。
