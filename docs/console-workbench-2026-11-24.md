# 控制台功能盘点与实验卡模块（2026-11-24）

> 依据：`multicamera_reid_research_report_2026-11-23.html`（17 章整合版）§08 人机分工、§09 科研工作台与四个确认关口、§10 五张实验卡。
> 交付：`#/workbench`（前端）+ `/api/workbench/*`（后端）+ `ops/ops.db` 两表。

## 一、结论

控制台的**执行面已经齐**（运行台账、训练曲线、任务派发、标注队列、算法配置），缺的是**立项面**：
研究报告把「谁能决定、在哪个点决定、决定后自动做什么」写成了流程，但控制台没有任何位置承载它。
本次补的就是这一层——实验卡 + 四关审批 + 人机分工，不新增算法、不改任何既有判定口径。

## 二、现有 15 页与报告要求的对应关系

| 报告要求 | 控制台现状 | 判定 |
|---|---|---|
| 每次运行附数据版本/提交/配置/耗时/原始预测/失败样例 | `#/runs`（run→dataset→model 血缘、stderr 尾部）、`#/train` | 已有，够用 |
| 任务分派含「谁做、预算、授权、验收」 | `#/jobs`（`assignee_kind` human/agent、`budget`、`authorization`、`acceptance`） | 已有，够用 |
| 标注队列按缺口排序、判定留痕 | `#/annotate`、`#/truth`、`#/e21-label` | 已有，够用 |
| **实验卡**（一个问题 + 基线 + 候选 + 主指标 + 失败条件 + 预算） | 台账 `experiment` 表有 4 条（`ops_ingest_train` 自动汇总的意图分类变体，无假设、无预算），**控制台零入口** | **缺** |
| **四关 A 立项 / B 数据与真值 / C 边界变化 / D 成果验收** | 无表、无接口、无页面 | **缺** |
| **卡级人机分工**（你掌控什么、Agent 自主什么） | 只有 `#/jobs` 的字段级 `assignee_kind`，看不到「这张卡谁拍板」 | **缺** |
| **七件套交付物** | 无 checklist，交付物散落在 `docs/` 与 `deliverable/` | **缺** |
| **状态机**（草稿→待立项→已批准→…→已采纳/需修订/归档） | 无 | **缺** |

## 三、本次落地

**表**（`ops/schema.sql`，幂等建表）

- `exp_card`：`question / hypothesis / baseline_json / candidate_json / metric_json / budget_json / roles_json / evidence_json / runs_json / state / risk / source`
- `exp_gate`：`card_id / gate / decision / note / actor / at`，**append-only**，最新一条即当前决定

**接口**（`webapp/backend/workbench_api.py`，前缀 `/api/workbench`）

| 方法 | 路径 | 语义 |
|---|---|---|
| GET | `/status` | 台账可用性（缺库 503，不阻塞其它页） |
| GET | `/overview` | 计数 + 四关定义 + 待确认清单 |
| GET | `/cards` | 全量卡（含 `gate_state`、`gate_log`、`next_states`） |
| POST | `/cards/{id}/gate` | 记一次关口决定；`退回/暂缓` 无理由返回 400 |
| POST | `/cards/{id}/state` | 推进状态；非法流转 400；`已采纳` 前置 D=同意 |
| POST | `/cards/{id}/evidence` | 七件套勾选（只记交没交 + 路径） |
| POST | `/cards/{id}/run` | 挂载 run（baseline/variant/ablation），只校验存在 |

**页面**（`#/workbench`，`WorkbenchView.vue`）：待确认置顶的卡列表 + 详情（问题/假设、基线 vs 候选、
主指标·同时报·失败条件、预算与风险、四关表单、人机分工、七件套、运行与审批记录）。

**种子**：五张卡正文取自报告 §10，`state` 停在 `proposed`（EXP-001）/ `draft`（其余四张）。
**只种计划、不种成绩**——`metric_json` 里没有任何数值；种子用 `meta.exp_card_seed_20261123` 防重跑。

## 四、验收记录（2026-11-24 14:55，临时库副本）

真实浏览器（headless Chrome + CDP）逐项操作，非静态检查：

1. 首屏 5 张卡渲染，EXP-001「待立项审批」，A 关「未决」；
2. 点「记录决定」→ 表单展开 → 选「同意」→ 填理由 → 提交；
3. 结果：状态自动转「已批准」、A 关变「同意」、页面提示「状态联动为「已批准」」、审批记录 1 条；
4. 钩七件套第 1 项 → 后端 `evidence_done=1`；
5. 状态下拉变为「已归档 / 方案·编码 / 暂停」，与 `approved` 的出边一致；
6. 服务端对账：`state=approved`、`gate_state.A.decision=approve`、note 与时间落库；
7. 反向校验：`D 未同意 → 标为已采纳` 返回 400；`draft → approved` 返回 400；`退回不写理由` 返回 400。

真实库 `ops/ops.db` 只写入种子（5 卡 + 5 条 audit），`exp_gate` 为空 —— 测试的审批记录只落在临时副本。

## 五、边界与未做

- **不自动生成结论**：卡上不出现任何「已验证/有效」字样；结论仍需人写进 `conclusion` 或文档。
- **不做权限**：`actor` 固定 `console`，单机单人使用；多人协作要先有身份再来谈审批有效性。
- **不与 `experiment` 表合并**：自动汇总的训练变体没有假设与预算，硬并会污染两侧口径。
- **未接 CI**：七件套勾选靠人，暂不做产物路径校验（需要先约定产物命名规范）。
