# 研究议程交付物（research-agenda deliverables）

日期：2026-10-14 ｜ 依据：`docs/external-plan-review-2026-10-14-v2.md` §六 **决策结果**

> **决策（2026-10-14 09:10）**：采纳外部《下一阶段研究计划》框架为**主干**；
> **倒影/镜像治理保留为"困难场景证据链 + 输入质量模块"**，不作主线贡献主张。
> 对应外部计划 §7 要求的五份首轮交付物。

本目录五份文件均为**基于项目已完成资产**的基线文档，不是空模板。全文区分：

- **【已完成】** 本机/远程可复现的运行结果，附产物路径与哈希；
- **【推断】** 由已完成结果或文献推导，尚未独立验证；
- **【建议/未做】** 尚待执行的方案，不得当作已实现能力。

## 索引

| 文件 | 作用 | 主要地基 |
|---|---|---|
| `research-charter.md` | 范围、目标层次、研究假设 H1–H3、**不能主张的结论** | 本目录全部 + v2 复核 |
| `literature-rule-catalog.md` | 18 条已核验引用 + 22 张规则卡（CQ/TG） | 三份外部意见引用逐条核验 |
| `evaluation-protocol.md` | 时间切分、真值隔离、指标与失败条件、发布门槛 | `splits_v5.json`、`analysis/truth/`、`c0/` |
| `open-set-selector-baseline.md` | 冻结编码器、125 原型库、未知拒判、per-camera 校准、伪标签路线 | `analysis/selector_1012/`、`camgroup_study/` |
| `portal-topology-baseline.md` | 门户语义×轨迹融合、图边状态机、H2 对照、最低回放 | `analysis/relay_clean2/`、`analysis/truth/`、`stitch_clean2/` |

## 一页速览：计划五份交付 vs 本项目现状

| 计划要求 | 现状 | 本目录文件 |
|---|---|---|
| 1. `research-charter.md` | 从未系统化撰写 | ✅ 本目录（保留倒影为困难场景） |
| 2. `literature-rule-catalog.md` | 零散在 3 份项目报告里 | ✅ 本目录（规则卡已条目化） |
| 3. `evaluation-protocol.md` | 纪律散见 `DATA_DISCIPLINE.md`/SCENE | ✅ 本目录（含真值/门槛） |
| 4. `open-set-selector-baseline.md` | **已超额完成**（125 原型、AUC 0.9942） | ✅ 本目录（实证版） |
| 5. `portal-topology-baseline.md` | 接力图+C0 已有；**门户语义发现未做** | ✅ 本目录（含 WP2/H2 设计，标未做） |

**当前唯一实质缺口**：WP2 门户发现（语义候选）与 H2 对照实验——本目录给出可直接执行的设计，但**尚未运行**。
