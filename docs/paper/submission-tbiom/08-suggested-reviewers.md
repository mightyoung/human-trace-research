# Suggested Reviewers — 模板与候选池

> 系统字段：**Suggested Reviewers**（3–5 位）。T-BIOM 单盲评审，推荐人是"建议"不是"指定"，编辑可不用。
> **邮箱必须核实**：用现任单位的官方域名邮箱，不要用 gmail/qq 等个人邮箱，也不要用从论文 PDF 里抄的旧地址（会 bounce，反而扣分）。

## 回避规则（IEEE 与多数期刊一致）

- 不得与任一作者**同一机构**（含近 3 年内曾任职的机构）
- 不得与任一作者**近 3 年有共同署名论文**、共同项目、师生关系
- 不得是作者的**现任或前任导师/学生**
- 不得是同一课题的**直接竞争者**（正在做同一数据集上的同类评测）
- 建议覆盖三个方向：评测协议/benchmark、监控 ReID、人脸与生物特征可用性

## 候选池（来自本文参考文献，需自行核实）

下面这些是本文实际引用、且研究方向与稿件高度重合的研究者。**不是推荐名单，是候选池**——请按回避规则筛，并核实现任单位与邮箱。

| 方向 | 候选（文献号） | 与本文的重合点 |
|---|---|---|
| 评测协议 / benchmark | L. Zheng [1]、M. Ye [5] | ReID benchmark 设计与大规模评测方法学 |
| 监控多目标多机位 | E. Ristani [10][11]、C. Tomasi [11] | MTMC 性能度量与跨机位评测口径 |
| 视频监控中的身份与隐私 | A. Cavallaro [3]、T. Xiang [3][5] | 监控场景下的重识别与身份分析 |
| 不确定度 / 风险控制 | A. N. Angelopoulos [14]、Y. Geifman [13] | 覆盖率—风险控制、选择性预测，与本文"带区间的验收"直接相关 |
| 人脸可用性 / 采集几何 | 从 [26]（GB/T 31488-2015）与其他人脸质量评估文献中另选 | 像素预算与朝向/可见性对识别的影响 |

> 第 4 行那一组（不确定度/风险控制）最值得放一位：本文的核心主张之一是"验收数字必须带总体与区间"，这组审稿人最可能认可这个框架，也最可能提出有用的反驳。

## 补充候选的检索方式

1. 在 IEEE Xplore 搜 T-BIOM 近两年 "evaluation protocol" / "benchmark" / "surveillance re-identification" 的论文作者；
2. 在 Google Scholar 找引用本文参考文献 [1][5][10] 且近三年活跃的作者；
3. 排除名单：把确有利益冲突的人写进 **Excluded Reviewers**（系统有独立字段，写上比不写更专业）。

## 填写模板

```
Suggested Reviewer 1
Name:        [Full name, with title]
Affiliation: [Institution, Department]
E-mail:      [name]@[institution].edu   ← 必须是机构域名
ORCID:       [0000-0000-0000-0000]      ← 有则填，便于编辑核对
Expertise:   Evaluation protocols and benchmarking for person re-identification
Reason:      Author of the benchmark and survey work this manuscript builds on [1][5];
             no joint publication with any author.
```

（第 2–5 位同格式。）

## 常见坑

- **邮箱过期**：从 PDF 抄的邮箱常常是十年前的。用机构主页上的现职邮箱。
- **推荐了同一圈子的人**：如果 5 位里有 3 位互相是合作者，编辑会认为这份名单没有信息量。
- **忘了写 Excluded**：如果某位学者与作者有明确竞争关系，不写进排除名单，被分配到就是风险。
