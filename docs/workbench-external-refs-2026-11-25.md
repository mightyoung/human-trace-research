# 外部参考：OpenAI / Anthropic 的科研 Agent 工作流

> 目的：给本工作台的流程设计找对照，判断哪些该学、哪些不该学。
> 采集时间 2026-11-25。**事实**标注来源；**判断**是我的推断，单独标出。

---

## 1. 两个可以直接对标的系统

### Anthropic Claude Science（前身 Claude for Life Sciences，2025-10-20 发布）

定位是"统一科研工作台"，不是聊天框：文献、数据、图、稿件、HPC 提交在同一环境里。

其公开描述中最值得注意的一条（**事实**，来源为发布会报道与官方页）：
**每个图、表、稿件都附带生成它的代码、软件环境、工作流历史，以及一段人话解释**；另有 reviewer agent 持续检查计算、引用、图表的一致性。

对照我们的表：`claim`（断言）＋ `claim_anchor`（出处）＋ `eval_run`（输入哈希、输出、退出码、env）＋ `provenance_json`（6 要件）就是同一件事，只是我们把粒度压到了"单个数字"。差别在于**他们把这套东西做成产品默认输出，我们做成事后台账**。

### OpenAI for Science（2025-09 起）

**事实**：定位是"模型作研究协作者，不是 oracle"；工程上采用 **generator-critic loop**（模型自评后再交人）；2026 年公布的一批数学结果**每一条都附 Lean 4 机器可校验证书**，并在公开仓库里可复建。

**事实（反面）**：2025-10 曾宣称解决若干 Erdős 问题，数学家指出模型是从旧论文里检索出已有解，相关帖子被删除。这是"没有可校验证书就宣布结论"的代价。

**判断**：这印证了我们把 `verdict` 限定为算子复算结果、并把 `unverifiable` 显式标注的做法；也说明"证书要能被第三方独立执行"是关键，我们目前只做到"自己跑一遍"。

---

## 2. 机制映射

| 外部机制 | 来源 | 我们的对应物 | 差距 |
|---|---|---|---|
| 输出附代码＋环境＋历史＋解释 | Claude Science | `claim_anchor` + `eval_run.inputs/env` + `provenance_json` | 缺"单条 claim 的独立复现命令" |
| reviewer 检查计算与引用 | Claude Science | `ops_claims verify`（算子复算）＋ 锚点校验 | 数字有复算，**锚点失效没有定期检查** |
| generator-critic | OpenAI for Science | 人工判读 → 复算 | 缺"claim 与实验卡指标是否一致"的检查 |
| 机器可校验证书 | OpenAI for Science（Lean 4） | `eval_run` + `verify_specs.json` | 证书不可独立执行（要连库） |
| append-only action ledger | OpenAI4S（arXiv 2609.15096） | `exp_gate` 追加、`claim` 只增不删 | `eval_run` 现在是 upsert，不是追加 |
| trace / span 树 | OpenAI Agents SDK | `eval_run.args/outputs` | **不照搬**：trace 是调试信息，不是审计证据 |
| workflow vs agent 分界 | Anthropic《Building effective agents》 | 固定七环节，无自主循环 | 一致：不为"智能"引入不可预测的控制流 |

关于最后一行：Anthropic 的判据是"**控制流由代码预定 = workflow；由模型决定 = agent**"。我们整条流水线属于前者，这是刻意选择——审计要求可预测，不接受"模型自己决定下一步"。

---

## 3. 不照搬的四件事

1. **不用 LLM 判定 verdict**。外部系统的 reviewer 是模型；我们的 `verdict` 只能由算子写出。模型可以生成"建议归属"（本轮的 card_map），但不能写结论。
2. **不把 trace 当证据**。span 树能回答"agent 做了什么"，不能回答"这个数字是不是这么算出来的"。后者需要输入哈希 + 算子 + 期望值。
3. **不用自然语言当唯一记录**。Claude Science 的"人话解释"是给人读的，机器不认；我们每条 claim 必须有可执行 check，解释只写在 `reason`。
4. **不做全自动归属**。OpenAI/Anthropic 都能让模型直接给出结论，但在"这条数字属于哪个实验"上，错的代价是审计链断裂——所以 Q8 定为"脚本出候选、人改"。

---

## 4. 三条可落地改进（按性价比排序）

| # | 改进 | 理由（外部依据） | 代价 |
|---|---|---|---|
| 1 | **`hc verify <claim_id>`：把单条 claim 的核验渲染成一条可复制的命令**，不连台账也能跑 | OpenAI 的证书之所以可信，是因为第三方能独立执行；我们的证书目前只能自己跑 | 每类算子给命令模板，约 1 天 |
| 2 | **doctor 增加锚点漂移检查**：`claim_anchor.resolved_sha1` 与当前 tex 文件比对，失效即报 | Claude Science 的 reviewer 会持续查引用；我们只在迁移时算过一次 | 半天 |
| 3 | **`eval_run` 改 append-only**：同一配方重跑插入新行，旧行标 `superseded`，不再 upsert | OpenAI4S 的 action ledger 是追加式；upsert 会抹掉"当时算出来是什么" | 半天，会改变 run 计数口径 |

前两条不改现有语义，建议直接排进 K1；第三条要先确认"run 计数"在报告里的口径（现在 101 行 = 101 条配方）。

---

## 5. 与我们结论一致的地方

- **provenance 是产品特性，不是合规负担**：他们做在输出侧，我们做在断言侧，方向一致。
- **生成与校验必须同屏**：他们的 reviewer agent、我们的 verify，都是"同一次操作里既有产出又有反驳"。
- **可复现仍是开放问题**：OpenAI4S 论文明确写"environment specification and full rerunnability remain weak for every evaluated system, ours included"——我们的 `as_is=1` 与"不可复算 + 原因"就是对这一点的诚实回应。
