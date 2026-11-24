# 投稿目标复评：Neurocomputing 的 scope 风险有多大，该换哪家

日期：2026-11-12
触发：上一轮 6 条待决项第 6 条 ——「Neurocomputing scope 风险 → 备选 IEEE TIM / ESWA」
结论：**风险是真实的、且卡在最容易被 desk reject 的那一条上。建议换主投。但备选不该止于 TIM/ESWA。**

---

## 0. 结论先行

| 候选 | scope 匹配度 | 主要风险 | 改造成本 | 建议 |
|---|---|---|---|---|
| Neurocomputing | **低** | 官方明文不收「纯图像处理」「纯组合已有算法」 | 一行都不用改（现状即可投） | **不建议主投** |
| IEEE T-BIOM | **最高** | 要求「benchmarking / 大样本评估 / 置信区间」——正是本文 | 摘要与小节标题需向身份科学口径靠 | **推荐主投** |
| IEEE TIM | 中高 | 一次筛选就拒掉大量不对口稿件；必须写成「测量」 | 需重写引言的定位段 + 补 I&M 文献 5–8 篇 | 备选一 |
| ESWA | 中 | 要「专家/智能系统」的完整系统贡献 | 要把交付阶梯包装成系统 | 备选二，优先级最低 |

一句话取舍：**这篇文章的贡献是「怎么测量一个相机网络到底能交付多少身份」，不是「一个新的网络」。
所以应该投「身份科学的评估/基准」或「测量科学」，而不是投「神经计算」。**

---

## 1. 证据（官方 scope 原文，2026-11 检索）

### Neurocomputing（Elsevier，IF 6.5–6.7）

Guide for Authors 的 Article types 第一条 **Type 1: Regular article** 原话：

> "Regular submissions must be **directly relevant to Neural Networks and/or Learning Systems**.
> Please note that Neurocomputing does **NOT** accept submissions on pure dynamical systems,
> pure control engineering, pure fuzzy logics, **pure image processing** or
> **pure combination of existing algorithms**, which are not directly relevant to
> neural networks or learning systems."

对照本文：五个贡献（三探针诊断 / 分级交付协议 / 不可组合性 / 151 条对级真值 + 报告纪律 / 复核工具）
**没有一个是新的网络或学习算法**。OSNet 与 EG2 都是拿来用的现成模型。
这正是 "pure combination of existing algorithms" 与 "not directly relevant to neural networks" 的交集。

风险等级：**高**。不是审稿意见层面的争议，而是编辑初筛阶段可能直接退回（desk reject），
理由一行就够，且申辩空间很小——因为条款是明文的。

### IEEE T-BIOM（IF 5.0，CiteScore 11.9）

Call for Papers 的 Broad topics 里有一条（原文）：

> "Novel (challenging) biometric databases, **protocol and benchmarking, large scale evaluation,
> **confidence interval estimation**, and performance modeling and prediction"

另有 "Person re-identification"、"Detection, tracking, identification, and de-identification in
surveillance" 两条直接命中主题。

本文五个贡献里，第 1、4 条直接落在这一条 bullet 上（协议 + 基准 + 置信区间 + 真值集），
第 3 条是评估理论，第 2、5 是部署协议与工具。

风险点：T-BIOM 的作者须知写着 "A paper is a candidate for an Immediate Rejection if it is of
limited novelty, e.g. a straightforward combination of theories and algorithms that are well
established, and are repeated on a known scenario." —— 这条对**提算法**的稿件最致命，
对**提测量与协议**的稿件反而宽松，因为它的判断标准是「有没有给出新的、可复用的评估知识」。
本文恰好给的是「同一站点上三条可复现的探针 + 五级阶梯 + 两条不许互替的数字」。
**这是三个候选里唯一一个把「协议与基准」写进主题列表的。**

风险等级：**低**。

### IEEE TIM（IF 7.0 量级，CiteScore 9.8）

Scope 四条中第 (3) 条：

> "(3) **analysis, representation, display, and preservation of the information obtained from a
> set of measurements**"

以及投稿页的定位要求：

> "Highlight the novelty of your work and what it contributes to: methods or instruments for
> measurement, detection, tracking, monitoring, characterization, identification, sensing,
> estimation, recognition, or diagnosis of a physical phenomenon; ... or **measurement theory
> including uncertainty, calibration, etc.**"

本文能对上：被测量 = 身份完整度；不确定度 = Wilson / bootstrap CI 与 κ；
标定 = 三探针的决策边界（60 px / 1-of-15 / 35 种着装）；仪器验证 = 五级阶梯逐级的验收条件与失败模式。

**但 TIM 有一条硬警告**：

> "A large number of submissions are immediately rejected at first screening stage, because not
> well aligned to TIM's scope."

且 TIM 专门为图像处理类稿件准备了 Topical Guide，要求把工作放进 I&M 的知识脉络里、
与 I&M 文献做对比。本文现有参考文献 23 条**全部是 CV/ReID 文献，没有一条 I&M**。
不改这一项，进 TIM 就是同样的初筛风险，只是理由从「不是神经计算」换成「不是测量」。

风险等级：**中**（改造后可降到低）。

### ESWA

要「专家系统/智能系统」意义上的系统贡献与应用价值。本文的复核队列确实是个系统，
但 ESWA 现在对「应用新颖性」和「系统完整性」的要求比 T-BIOM 严，
而本文刻意不做自动判定、把最后一公里交给人——这在 ESWA 的评价框架里是减分项。

风险等级：**中偏高**。

---

## 2. 为什么之前会选 Neurocomputing（以及那个判断哪里错了）

之前的口径是「首选 Neurocomputing，勿投方法刊」。前半句的直觉没错：本文**不该**投
Pattern Recognition / TPAMI 这类方法刊，因为没有新方法。但推论「那就投 Neurocomputing」
漏了一步——Neurocomputing 虽然名字宽，却明确把「纯图像处理」和「纯组合已有算法」写进了
**不收**的清单，而这两条正好把本文框住。

真正对口的不是「宽的 AI 刊」，而是**把评估与协议当贡献的刊**。T-BIOM 的主题列表里就有这一条。

---

## 3. 换目标的实际成本（不是零，但很小）

| 项 | Neurocomputing 现状 | 换 T-BIOM 要动的 | 换 TIM 要动的 |
|---|---|---|---|
| 摘要 | 249 词 | 改前两句定位：从「诊断+协议」改到「身份科学中的评估」 | 重写：首句必须点出「测量」与「不确定度」 |
| Highlights | 5 条独立文件 | 措辞对齐 T-BIOM 关键词（protocol / benchmarking / CI） | 同上 |
| 参考文献 | 23 条，全 CV | 补 3–5 条 T-BIOM 近三年同类评估/协议文 | 补 5–8 条 I&M 文献，并按 I&M 脉络重排相关工作 |
| 相关工作 | 5 小节，全 CV | 加一小节「生物特征评估与协议」 | 加一小节「测量不确定度与仪器验证」 |
| 格式 | Elsevier | IEEEtran（双栏） | IEEEtran |
| 图表 | 三图矢量 PDF | 无需改 | 无需改 |
| 声明 | 六段齐 | 无需改 | 无需改 |

工作量估计：T-BIOM 半天到一天（摘要 + 一段相关工作 + 3–5 篇文献）；
TIM 一到两天（还要重新定位引言、补 I&M 文献、可能补一节不确定度分析）。

**注意**：现成的 `submission/` 包（`CHECKLIST.md` + 六段声明 + 三图矢量）是按 Elsevier 结构做的，
换 IEEE 需要重排版面。这是纯机械工作，但会动到页数（现 21 页单倍 / 35 页双倍），
T-BIOM 没有硬页数限制，TIM 的 regular paper 通常也在 12–14 页以内，需要压缩。

---

## 4. 建议

**主投 T-BIOM，TIM 作为一次退稿后的第二个去处，Neurocomputing 降为不再考虑。**

理由是三条硬事实，不是偏好：

1. 本文五个贡献里没有一个新网络（Neurocomputing 明文不收这类）；
2. T-BIOM 的主题列表里逐字写着 "protocol and benchmarking, large scale evaluation,
   confidence interval estimation" ——这是本文的主干；
3. TIM 的对口改造量是 T-BIOM 的两倍以上，且它自己承认初筛拒稿率高。

如果时间压力大、不想重排版面，**次优方案是留在 Elsevier 但换刊**：
Image and Vision Computing 或 Journal of Visual Communication and Image Representation
（都是 Elsevier，模板不变，主题接受 surveillance 评估类工作），
代价是这两个刊对「评估类贡献」的容忍度不如 T-BIOM，且影响力更低。

---

## 5. 待用户决定的一件事

换刊会动到摘要、相关工作、参考文献与版面格式，属于**方向变更**，需要一次授权：

- **A（推荐）**：改投 T-BIOM。我按上表改造，产出 `submission-tbiom/` 完整包。
- **B**：留在 Elsevier 换刊（IVC / JVCIR），只改摘要与相关工作，版面不动。
- **C**：坚持 Neurocomputing，接受初筛风险；我把 §1.1 与引言改为更强调
  「以神经网络为测量对象」以降低风险，但这只是对冲，不能消除条款冲突。

---

## 6. 复现

本文所有 scope 引文来自各刊官方 Aims & Scope / Guide for Authors / Call for Papers 页面
（2026-11-12 检索）。原文措辞如与本文件引用有出入，以官方页面为准。
