# Neurocomputing 投稿合规清单

> 目标刊：*Neurocomputing*（Elsevier，ISSN 0925-2312），稿件类型 **Full Length Article**。
> 本清单逐条对照该刊 Guide for Authors 与 Elsevier 通用规范；「已满足」只代表文件层面完成，「待补」代表必须由作者本人填写。
> 更新：2026-11-12。

---

## 一、硬指标对照

| # | 刊物要求 | 现状 | 位置 |
|---|---|---|---|
| 1 | 摘要 ≤ 250 词，可独立阅读，不引参考文献 | **249 词**，无引用、无未定义缩写 | `main.md` → Abstract |
| 2 | Highlights：3–5 条，每条 ≤ 85 字符（含空格），单独可编辑文件 | **5 条，72–80 字符**，独立文件 | `highlights.txt` |
| 3 | 关键词 1–7 个，英文，避免 "and/of" 类组合词 | **6 个** | `main.md` → Keywords |
| 4 | 章节连续编号、层级不超过三级 | 1–11 加两个附录，最深三级（6.3.1 无） | `main.md` |
| 5 | 正文引用用方括号数字，参考文献表按**首次引用顺序**编号 | 23 条，编号 1→23 与正文出现顺序完全一致（已用脚本核验） | `main.md` → References |
| 6 | 参考文献格式：首字母前置、题名句首大写、期刊缩写、卷 (年) 页码 | 23 条全部转换 | `main.md` → References |
| 7 | CRediT 作者贡献声明 | 已写，作者名待填 | `main.md` → CRediT |
| 8 | Declaration of competing interest | 已写 | `main.md` |
| 9 | Funding 声明（含无经费时的固定句式） | 已写，需确认是否属实 | `main.md` |
| 10 | Data availability statement | 已写：视频不可发布，其余全发布 | `main.md` |
| 11 | 生成式 AI 使用声明 | 已写 | `main.md` |
| 12 | Acknowledgements | 已写，人名待填 | `main.md` |
| 13 | 图：单独文件、引用齐全、图注为「短标题 + 描述」 | 3 张，矢量 PDF（单栏 182 mm），另有 SVG 源 | `submission/figures/` |
| 14 | 表：可编辑文本、无竖线、表注在表下 | 10 张三线表，图注在上、表注在下 | `main.md` |
| 15 | 正文双倍行距、宽边距、Times New Roman、单栏 | 审稿版 PDF：Times 12 pt / 行距 2.0 / A4 边距 25 mm | `main-review.pdf` |
| 16 | Cover letter | 已写，一页内，含 scope 适配论证 | `cover-letter.md` |
| 17 | 标题页（作者、单位、通讯作者、ORCID） | 已建模板 | `title-page.md` |
| 18 | 单盲评审（作者身份对审稿人可见） | 正文已含作者块，无需匿名 | — |

审稿版 34 页（双倍行距），阅读版 26 页。

---

## 二、只能由作者填写的内容

| 位置 | 待填项 |
|---|---|
| `main.md` 作者块 / `title-page.md` | 全部作者姓名、单位、城市邮编、国家、通讯作者邮箱与电话、ORCID |
| `main.md` → CRediT | 按实际贡献分配 14 类角色，不要沿用占位符 |
| `main.md` → Funding | 若无经费，保留现有固定句式；若有，按 Elsevier 格式补「Funding: 项目名 [编号]」 |
| `main.md` → Acknowledgements | 现场协调人、看过初稿的同事 |
| `cover-letter.md` | 日期、署名、单位、邮箱 |
| `title-page.md` | 推荐审稿人 3–5 位（同机构、近三年有合作者回避） |

---

## 三、提交前必须处理的完整性问题

1. **参考文献未做 DOI 核验。** 23 条只做了格式转换，没有逐条比对 DOI。Elsevier 编辑部会查，[7]（Appl. Soft Comput. 2026）、[8]（Knowl.-Based Syst. 2026, in press）、[15]（Sensors 2026）三条是当年新文献，风险最高；[12]（AI City Challenge）缺 URL。建议逐条补 DOI，查不到的换掉。
2. **单标注人，没有 Cohen's κ。** 153 条判读由一人完成。审稿人几乎一定会问。补法：另找一人独立判 40–50 条，算 κ，20 分钟。这是目前唯一「问了答不上」的点。
3. **T1 非随机抽样。** 正文 §4.1 已声明，但如果审稿人要求无偏估计，现有数据给不出。
4. **7 卷体检缺失。** 所有合并实验在 D1（单次 35 分钟）上完成，只在 D2 做过分段验证。若被质疑过拟合单场景，可用 7 卷长卷补一次阈值体检。

---

## 四、Editorial Manager 上传顺序

1. `cover-letter.md` → 转 PDF 或直接粘贴
2. `title-page.md` → Title Page
3. `main-review.pdf` → Manuscript（**不要**在系统里再传一份匿名版）
4. `highlights.txt` → Highlights（文件名必须含 "highlights"）
5. `submission/figures/Figure_1.pdf` … `Figure_3.pdf` → Figure，逐张上传
6. 补充材料（可选）：把 `00-paper/` 与 `30-code/` 打成一个 zip 作为 Supplementary material

---

## 五、scope 风险（要提前想好怎么答）

*Neurocomputing* 是神经计算刊，本篇不提出新网络。desk reject 的最可能理由是「方法增量不足」。cover letter 里已按「学习表征的工作包络 + 决策规则」来框定，正文 §9.1 也把可迁移结论限定在「表征在何种采集条件下失效」。

如果编辑仍以 scope 退回，两个改投方向：
- **IEEE TIM**：把叙事改成「测量程序 + 不确定度预算」，契合度最高；
- **ESWA**：需要补一节决策形式化（把分级协议写成带约束的优化问题）。
