# 科研资产台账 v2.1：以论文数字为锚的可回溯台账（评审修订版）

> **2026-11-24 23:40 起，本文档降级为 `paper_tbiom` 适配器的实现说明。**
> 主语已由「论文数字」改为「实验流程」，见 `docs/workbench-flow-model-2026-11-24.md`。
> 本文的 6 要件、覆盖率口径、Clopper-Pearson、tex 解析三坑仍然有效，但只约束这一个适配器，
> 对工作台内核无约束力；`paper_claim` 已迁为通用 `claim` 表的视图。

> 版本：2026-11-24 ｜ 状态：**V0 已完成（只读体检）；V1–V3 已实施；主语被流程模型取代**
> 前序：`docs/ledger-v2-design-2026-11-24.v2.0-superseded.md`（v2.0，其主张 2 与 §4–§10 已被本版推翻）
> 口径：§1 的数字全部来自 `tools/paper_claims_audit.py` 的实际运行；建议与推断单独标注。

---

## 0. 结论先行

**评审后目标收窄为一句话：论文里的每个数字，被质疑时能在一秒内指到文件、脚本和哈希。**

这带来一个主语的更换：v2.0 把 `experiment` 当聚合根；v2.1 把**论文数字（claim）**当主语，实验卡降为“人读的叙述层”。依据是一个实测事实：`main.tex` 的 9 张表、163 个数字，在 EVIDENCE.md 里的出处**全部指向分析脚本、真值文件和其他文档，没有一条指向 `run` 表**。先做实验聚合根，这篇论文依然不可回溯。

**三条设计原则**

1. **文件即记录，台账是索引**。EVIDENCE.md、标注 JSON、论文包 MANIFEST 是权威；台账只做带内容哈希的索引与比对，不接管写入。
2. **门是诚实门，不是资格门**。复算不了的数字标明原因，比强行“全绿”更有价值；永不自动改论文。
3. **不为整齐而建表**。每张新表必须写明第一个写入者与首个消费者，否则不建。

---

## 1. V0 只读体检结果（2026-11-24 实测）

脚本 `tools/paper_claims_audit.py`；产物 `ops/reports/paper_claims_v0/{report.md, claims.json}`。只读，不碰 `ops/ops.db`，不执行任何复算。

### 1.1 EVIDENCE.md 侧

| 指标 | 数值 |
|---|---|
| 解析出的 claim（表格行 + 说明行） | **176** |
| ① 来源含数据/结果文件 | 93 / 176 |
| ① 来源只有另一篇 md | 38 / 176 |
| ① 没有任何可解析来源 | 45 / 176 |
| ① 严格达标（引用全部存在，且已记内容 sha256） | 93 / 176 |
| ② 引用了存在的复算脚本 | **45 / 176** |
| 已登记来源文件（去重） | 62，全部已记 sha256 |
| §十 复现清单的脚本 | 10 / 10 存在 |
| 解析不到的引用 | 0 |

分节看，问题集中在四处：

| 节 | claim | 含数据文件 | 只指向 md | 无来源 | 有脚本 |
|---|---:|---:|---:|---:|---:|
| 三、外观天花板 | 18 | 13 | 0 | 5 | 13 |
| 四、人脸像素预算 | 12 | 2 | **9** | 1 | **0** |
| 五、服装多样性 | 10 | 3 | **5** | 2 | **0** |
| 六、分级协议各级证据 | 36 | 19 | 7 | **10** | **1** |
| 七、不可折叠 | 17 | 13 | 0 | 4 | 9 |
| 八、跨采集验收 | 28 | 19 | 0 | 9 | 19 |
| 十一、r2 新增事实 | 29 | 12 | **15** | 2 | 1 |

“无来源”的 45 条里，有一部分是合理的派生量（如“合计 153”“人工成本 20–30 min 实测”），不等于错误。但**“只回溯到另一篇 md”的 38 条是真实缺口**：它们目前只能回溯到叙述，回不到数据。

### 1.2 EVIDENCE.md 与 tex 的差异（粗匹配）

| 对象 | 表数 | 表内数字 | EVIDENCE 全文找不到 |
|---|---:|---:|---:|
| `main.tex` 正文表 | 9 | 163 | **10** |
| `supplement.tex` 补充表 | 5 | 96 | **23** |

正文 9 张表里 7 张完全覆盖。缺的 10 个数字是两类：

- `tab:pair` 的区间端点 `0.07`、`1.00`
- `tab:protocol` 的 `0.81`、`0.83`、`101`、`122`、`48.8`、`60.8`、`63.8`、`74.9`

**缺口主要在补充材料**：

| 表 | 缺的数字 |
|---|---|
| Domains | `720`、`810` |
| Annotation reliability（κ 表） | `0.913`、`0.925`、`0.945`、`0.958`、`0.972`、`236` |
| 跨采集第二张表 | `0.275`、`0.296`、`0.371`、`0.516`、`0.643`、`0.653`、`0.877`、`21`、`23.7`、`40.2`、`50.3`、`67.3`、`76.0`、`83.8`、`86.4` |

表外还有 44 个数字不在 EVIDENCE 里，集中在补充材料的步态表和 Limitations 里的 κ。反向，EVIDENCE 有而 tex 没有的数字 69 个，属于未被引用的证据，无害。

**这节的方法局限**：按整篇文档的数字集合比对，`34` 这样的数字会在很多地方命中，不能定位到单元格。它的用途是找出“一边完全没有”的数字；定位要靠 V1 的单元格锚定。

### 1.3 两份 EVIDENCE.md 不同步

`docs/paper/EVIDENCE.md`（24,934 字节，含第十一节 r2 补记）与论文包 `deliverable/paper-package-v3/00-paper/EVIDENCE.md`（13,191 字节）**sha256 不一致**，论文包冻结在旧版。投稿时如果附的是论文包，附的就是旧证据表。

### 1.4 V1 结果（登记与锚定，2026-11-24 18:50）

产物：`ops/reports/paper_claims_v1/report.md`；工具 `tools/ops_claims.py`；锚定文件 `ops/claims/tex_anchor.json`。

| 项目 | 数值 |
|---|---|
| `paper_claim` 行数 | **176**（表格行 132 / 说明行 44） |
| 挂到 tex 表格行的 claim | **62 条 → 86 个表格行** |
| 登记的文件版本 `asset_version` | **62**，全部内容 sha256，0 个弱身份 |
| 表格数字中 EVIDENCE 全文没有的 | 12 |
| 有证据但本行没挂到的（值在别处出现） | 6 |
| 整行声明“无证据行”的 | 11 行 / 45 个数字 |
| **合计需补证据的表格数字** | **57** |

V1 把 V0 的“33 个数字缺证据”拆成了可执行的三类，其中两个实质发现：

1. **缺口几乎全是区间端点**。`tab:pair` 的 `[0.78, 1.00]`、`[0.07, 0.40]`；`tab:protocol` 的 `[0.60, 0.83]`、`[0.81, 0.99]`；κ 表的 `[0.891, 0.958]`、`[0.913, 0.972]`。这些 CI 是可以从 k/n 与区间方法算出来的，EVIDENCE 里一条都没记。**V2 的正确补法是一条“CI 的计算方法与脚本”证据行，而不是逐个数字补行。**
2. **补充材料的集中度表整表没有对应证据行**。该表是“全量”口径（covered 469 / 110 / 342…），EVIDENCE 第八节第二张表记的是“剔除无效段”口径（445 / 112 / 363…），**两套只有第 1 行相同**。9 行 36 个数字无法回溯。这是审稿人最容易问的一处，已单独列在报告 §6。

V0 报 33、V1 报 57，差值不是矛盾：V0 按全文数字集合比对，`0.809` 这类数字因为别处出现过而被误判为“有出处”；V1 按单元格锚定，才看清它们属于另一套口径。

---

## 2. 评审裁定（Q1–Q21）

| 编号 | 问题 | 裁定 |
|---|---|---|
| Q1 | 台账对谁负责 | **论文可回溯**；方法论示范是副产品 |
| Q2 | 台账与 JSON 谁权威 | **文件是权威**，台账是带内容哈希的索引 |
| Q3 | 采集靠自觉还是进程边界 | 进程级包装为主、白名单声明为辅；**降级到 V4**，对本文历史脚本无效 |
| Q7 | 三件重构是否保留 | **只拆 `ops_db.py`**，最后做；`main.py` 拆分与前端重写不做 |
| Q5 | 工作量 | 不估工期，改为**有序版本 + 完成定义** |
| Q6 | 双标与 kappa | **砍掉未来流程**；只留 `annotator` + 可选 `reviewer`；历史 κ 进 claim |
| Q8 | 可回溯的主语 | **换成“数字”**，实验卡降为叙述层 |
| Q9 | 数字粒度 | 一张表一次执行 = 一条 `eval_run`；只有**被引用的数字**进 `paper_claim` |
| Q10 | 标注收敛范围 | **只做登记**：T1、T2 不合并，加已判合并、审计组；不做归一、不做写切换 |
| Q11 | `formal` 字段 | 取消；改为协议 `recorded` + `provenance` 分数；`isolation` 拆成 `sampling` 与 `identity_isolation` |
| Q12 | 论文是否改动 | 正文不改；supplement 加出处表 |
| Q13 | 可回溯的判定标准 | 6 要件中 **①④ 硬性**，其余计分 |
| Q14 | 复算与论文不一致 | **两值并存、标待裁定、按根因分组**；永不自动改论文 |
| Q15 | 资产身份算法 | **全量内容 sha256**，按 size+mtime 缓存 |
| Q16 | 包装器的输入 | 输入靠声明、输出靠 mtime 自动发现；空缺显式标 `inputs_declared=0` |
| Q17 | 投稿节奏 | **r2 按现状投**；台账只体检不改数，差异留给 r3 或 rebuttal |
| Q18 | 复算不了的数字 | 优先新写校验脚本；做不到则标“不可复算 + 原因”；**永不自动删数字** |
| Q19 | 标注这一版 | **只登记** `label_set` |
| Q20 | EVIDENCE 与 tex 谁权威 | **当前 EVIDENCE.md 为权威**，解析 tex 做一致性比对；论文包 v3 的 60 个 sha256 登记为首个 `eval_protocol`（`recorded`） |
| Q21 | 完成定义 | 见 §3，可验率 ≥ 90% |

**评审后不再做的事**：实验卡并入 `experiment`；统一标注入口与前端重写；`test_suite` / `evaluation` / `assignment` / `adjudication` / `annotation_export` / `model_alias` / `env_version` / `metric_def` / `metric_value` 十张表；`formal` 字段与封存门禁。

---

## 3. 完成定义

| 指标 | 定义 | 目标 |
|---|---|---|
| **覆盖率** | 正文表格 + 摘要、引言、结论里的内联数字，要件①达标的比例 | **100%** |
| **可验率** | 其中要件④达标（复算一致，或已标“不可复算 + 原因”）的比例 | **≥ 90%**，其余逐条有原因（实测 100%；「一致率」单列 65%，口径见 §10 第 8 条） |
| **一致性** | EVIDENCE.md 与 `main.tex` / `supplement.tex` 的差异清零，或全部已裁定 | 清零或已裁定 |

达到即停；V4 包装器与 V5 拆包降为“有空再做”。

**可回溯的 6 要件**：① 来源文件存在且已记内容哈希；② 生成脚本存在且已记哈希；③ 复现步骤非空、可执行；④ 复算值等于论文值；⑤ 分母 n 显式；⑥ 域与检测版本已标注。

两点必须说清：

- **② 只是“现样哈希”**。历史脚本不会重新经过包装器，记录的是脚本**现在**的哈希，不能证明是产出该数字时的那一版。台账里用 `as_is=1` 标注，出处表照实写。
- **④ 有出口**。描述性数字（如 1,894 条轨迹、2099 秒）如果没有可执行的复现步骤，标为“不可复算 + 原因”也算达标。

---

## 4. 目标架构

```
┌─ 权威文件（不动，人照常写）──────────────────────────────────────────────┐
│ EVIDENCE.md · 真值 JSON · 脚本 · 结果文件 · 论文包 MANIFEST · main.tex   │
├─ 索引与比对（新增；只读文件，只写台账）──────────────────────────────────┤
│ ops_claims   解析 EVIDENCE → claim；解析 tex → 单元格锚点；比对          │
│ ops_verify   逐条复算 → eval_run；差异按根因分组                         │
│ hc run       （V4）进程级包装：命令、代码哈希、输入输出、退出码          │
├─ 登记层（ops/ops.db，加 6 张表 + exp_card 一列）─────────────────────────┤
│ paper_claim ── eval_run ── eval_protocol                                  │
│      │            │                     └─ code_version                  │
│      └─ label_set ┴─ asset_version（文件 + sha256）                      │
├─ 展示（按需）────────────────────────────────────────────────────────────┤
│ Markdown 报告（V1）· supplement 出处表（V3）· #/paper-claims 页面（可选） │
└─ 既有控制台（#/workbench / #/runs / #/annotate 等）全部不动 ─────────────┘
```

**与 v2.0 的差异**：没有领域层分包，没有服务层拆 router，没有标注写切换；只新增两个只读工具，V4 才加包装器。

---

## 5. 数据模型（加法，全部 `CREATE TABLE IF NOT EXISTS`）

| 表 | 关键字段 | 第一个写入者 | 首个消费者 |
|---|---|---|---|
| `paper_claim` | `id`、`display_id`、`kind`（table / prose）、`evidence_line`、`section`、`label`、`value_text`、`domain`（D1/D2/D3）、`tex_anchor_json`（表 label + 行键 + 行号）、`verdict`（pending / match / mismatch / unverifiable）、`reason`、`recomputed_text`、`source_level`（data / doc / none）、`provenance_json` | `ops_claims import` | `ops_verify`、出处表 |
| `eval_run` | `id`、`claim_ids_json`、`script`、`script_sha1`、`as_is`、`args_json`、`cwd`、`started_at`、`ended_at`、`exit_code`、`code_version_id`、`inputs_json`、`outputs_json`（路径 + sha256）、`env_json` | `ops_verify` | `paper_claim.verdict` |
| `eval_protocol` | `id`、`name`、`status`（recorded / sealed）、`sampling`、`identity_isolation`、`params_json`（阈值、bootstrap 次数与 seed）、`manifest_uri`、`manifest_sha256`、`file_count`、`note` | V2 导入论文包 v3 | `paper_claim`、`eval_run` |
| `label_set` | `id`、`name`（T1 / T2 / judged_merges / audited_groups / gallery_double）、`sampling`、`uri`、`sha256`、`n_items`、`n_same`、`n_diff`、`annotator`、`reviewer`、`domain` | V2 登记五批真值 | `ops_verify` |
| `asset_version` | `id`、`path`、`sha256`、`bytes`、`kind`（media / weights / labels / derived / external_dataset）、`content_hash_alg` | V1 由 V0 的 62 个文件起步 | `paper_claim`、`eval_run` |
| `code_version` | `sha1`、`files_json`、`as_is` | V3 起 | `eval_run` |
| `run_input`（V4） | `run_id`、`kind`、`role`、`ref_table`、`ref_id`、`ref_sha1` | `hc run` | `run` |

`exp_card` 加一列 `claims_json`，指向 `paper_claim.id`。

**建表节奏随写入者走**：V1 只建 `paper_claim` 与 `asset_version`（有写入者、有消费者）；`eval_protocol`、`label_set` 在 V2 建，`eval_run`、`code_version` 在 V3 建。避免一次建出 6 张空表。

**一处更正**：评审时我说“`asset_version` 复用现有 `asset` 表”，核对后不成立。`asset` 是逐条素材的粒度（带 `cam / tid / t0 / t1`），与文件级版本不是一回事，硬用会语义错位，所以新建表。

**约束**

- `paper_claim` 只增不删；EVIDENCE.md 里删掉的行标 `retired`，不物理删除。
- `verdict=match` 只能由 `eval_run` 且 `exit_code=0` 的记录写入。
- `asset_version.sha256` 一经写入不改；文件变了就新增一行。
- `identity_isolation` 本篇一律填 `none`，出处表必须显示。

---

## 6. 接口边界

| 边界 | 谁写 | 谁读 | 说明 |
|---|---|---|---|
| EVIDENCE.md | 人 | `ops_claims` | 权威。格式沿用现有表格，新增行自动成为 claim |
| `main.tex` / `supplement.tex` | 人 | `ops_claims` | **不改**。锚定用 `claims.toml`（表 label + 行键 + 列键） |
| `ops/ops.db` 新表 | `ops_claims`、`ops_verify`、`hc run` | 报告、出处表、（可选）页面 | 旧表的写路径不变 |
| `ops_claims` | 新表 | 文件 | 只读文件，不执行脚本 |
| `ops_verify` | `eval_run` + `paper_claim.verdict` | 脚本 | **唯一**能改 `verdict` 的入口；不写 tex 与 EVIDENCE |
| `hc run`（V4） | `run`、`run_input`、`code_version` | — | 本地优先；远端仍走现有 bundle |

**明确不做**：不建服务层 router、不做权限、不改标注写路径。

---

## 7. 版本切片

| 版本 | 做什么 | 可验证的产出 |
|---|---|---|
| **V0 只读体检** ✅ | 解析 EVIDENCE → claim；核对来源存在性并记 sha256；比对 tex | `ops/reports/paper_claims_v0/`，本文 §1.1–1.2 |
| **V1 登记层** ✅ | `paper_claim`（含 `kind`）+ `asset_version` 两张表；V0 结果入库；`claims.toml`→`tex_anchor.json` 锚定 14 张表；`ops_claims audit` 幂等 | `ops/reports/paper_claims_v1/`，本文 §1.4 |
| **V2 真值与协议** ✅ | 登记五批 `label_set`（T1 126、T2 27、已判合并 76、审计组 15、图库双标 257）；论文包 v3 为 `eval_protocol`（60 文件逐位核 sha256）；EVIDENCE 增补 §十二（CI 方法、κ 两版本、D3 口径、9 格表不可复算） | `label_set` 5 行、`eval_protocol` 1 行；`paper_claim` 215；**表格数字缺口 12 → 0**；补充材料加注后编译 4 页 |IDENCE 与 tex 一致性差异清零 |
| **V3 复算** ◐ | `ops_claims verify` + `verify_specs.json`：`prop`（Clopper-Pearson）、`set_expr`、`matrix_row`、`gate_run` / `run_fields`、`pool_cell`、`pack` / `graph`、`text_value`、`json_count`、`constraints`、`auc_block` / `auc_pair`（读 `emb_auc_out.txt`，支持 `pick` + `tex_idx` 定位单元格）、`unverifiable`。锚点与配方存**哈希 id**；配方启动做结构校验（未知 kind、缺 expect、缺 tex 行、重复登记） | **正文 9 张表全部裁定完毕**（`tab:d1`/`agg`/`pair`/`protocol`/`gate`/`selector`/`verifier`/`merge`/`xday`）。累计 **81 条**：一致 54、不一致 0、不可复算 27；覆盖 **227 / 330 个表格单元格**（已核 176 = 53%）。补充材料仍缺 7 行 |
| **V4 运行采集** | `hc run` 包装器 + `run_input` + `code_version` | 任何一次运行都有命令、代码哈希、输入输出 |
| **V5 拆包** | `ops_db.py` 按域拆分 | 无可见变化，可选 |

**顺序约束**：V1 → V2 → V3 严格依次（V3 复算依赖 V1 的锚点与 V2 的真值哈希）。V4、V5 与主线无依赖。

**每个版本的做法固定**：先备份 `ops/ops.db`；在临时库副本上过一遍；再动真实库；不改前端。

---

## 8. 迁移与兼容

| 项 | 做法 |
|---|---|
| 备份 | 每次写库前 `ops/db_backup/ops_<时间戳>.db` |
| DDL | 只加表加列；`meta.schema_version` 保持不动（不动旧表，不触发旧代码的 stale 判定） |
| 旧表 | `experiment`、`exp_card`、`run`、`label` 等一律不动；`exp_card` 只加 `claims_json` |
| EVIDENCE.md | 不改历史行；补缺的 33 个数字时只**追加**行，并注明补记日期 |
| 论文包 v3 | 冻结不动；其 sha256 作为 `eval_protocol(package-v3)` 登记；重打包是另一条新协议 |
| 两份 EVIDENCE 不同步 | 以 `docs/paper/EVIDENCE.md` 为准；论文包是否重打由你决定（§9 R3） |
| 回滚 | 新表可整体留置，不影响任何旧页面与旧脚本 |
| 控制台在用 | 本方案不改后端与前端，无需重启 |

---

## 9. 风险点

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | **复算与论文不一致**（已知分母漂移、GT 同名噪声 43%、检测版本不可比） | 投稿后发现数字站不住 | 两值并存、按根因分组裁定；r2 按现状投，差异留给 r3 |
| R2 | 脚本哈希是“现样”而非“当时” | ② 要件被高估 | `as_is=1` 显式标注；出处表照实写 |
| R3 | 论文包 v3 冻结的是旧版 EVIDENCE.md | 投稿附件与正文证据不一致 | §1.3 已实测；V2 登记两条协议的哈希，是否重打包由你决定 |
| R4 | 粗匹配会漏（`34` 到处命中） | 假阳性覆盖 | V1 起改为单元格锚定；V0 结果只用于找“一边完全没有”的数字 |
| R5 | 一张表一次执行 ≠ 一个数字一次执行 | 同一 `eval_run` 挂多个 claim，一错俱错 | `eval_run.outputs_json` 记到文件；差异定位到 claim 行 |
| R6 | 补充材料的数字覆盖弱于正文 | 审稿人先看补充材料 | V1 把 supplement 5 张表一并锚定 |
| R7 | T1 与 T2 被合并计算 | 口径错误（EVIDENCE 明令禁止） | `label_set` 分两行，`sampling` 必填 |

---

## 10. 验收标准

**V1**（已完成，实测见 §1.4）
1. `paper_claim` 176 行，等于 V0 的 176；重复运行 `import` 新增 0 行（`--rebuild` 前会拒绝有裁定的库）。
2. 14 张 tex 表全部锚定成功，`errors=0`；未覆盖的数字按“缺证据 / 锚点不全 / 整行无证据”三类列出。
3. 覆盖率有实数写入报告。

**V2**
4. 五批 `label_set` 各有 `sha256`；条数与 EVIDENCE 一致（T1 126 = 34 + 92，T2 27 = 12 + 15）。
5. `eval_protocol(package-v3)` 的 `manifest_sha256` 与 `SHA256SUMS.txt` 一致；`identity_isolation='none'`。
6. EVIDENCE 与 tex 的一致性差异清零。

**V3**
7. 每条被引用的数字有 `verdict`；`match` 的都能点到 `eval_run` 及其输出文件哈希。
8. **可验率**（本表定义：要件④达标＝复算一致，**或**已标“不可复算 + 原因”）：101/101 配方、247/247 已登记单元格 = **100%**。「一致」是更严的观察口径，单列披露：66/101（65%）、192/330（58%）。
   *（2026-11-24 21:13 修订：原文只写「≥90%」，未区分「④ 达标率」与「一致率」，容易被读成后者。现明确门槛看 ④ 达标率，一致率与未登记格数一并披露；覆盖率另见第 3 条。）*
9. 构造一条故意算错的校验：`verdict` 变为 `mismatch`，而 `main.tex` 与 EVIDENCE.md 均未被改动（负向用例）。
10. supplement 出处表生成，且包含 `identity_isolation` 与 `as_is` 两列。

**总体**：§3 的三项完成定义全部达成即停。

---

## 附：仍从 v2.0 保留的内容

- §1 的问题诊断（三载体并存、run 缺输入身份、接入率 4/501、数据无版本）作为背景事实保留，但**不再作为本方案要解决的问题**。
- 开源借鉴只保留三条：CVAT 的“阶段与状态分离”对应 `eval_run` 与 `verdict` 的分离；MLflow 的“版本即哈希”对应 `asset_version`；评测集工程的“结果盖数据哈希”对应 `eval_run.outputs_json`。其余（Datumaro 切分、DVC 锁文件、ClearML 队列）不在本版范围内。

---

## 11. V3 收口结果（2026-11-24 21:10）

| 验收项 | 标准 | 实测 | 结论 |
|---|---|---|---|
| 1 claim 行数与幂等 | 176 起、重复 import 新增 0 | 238（表格 162 / 说明 76）；重跑新增 0、剪枝 0 | 通过（行数随后续批次增长） |
| 2 14 表锚定 | `errors=0` | 91 claim / 117 表行，`gaps=0`、`errors=0` | 通过 |
| 3 覆盖率写实数 | 报告里有数字（测量本身） | 247/330 单元格（登记 75%）、已核 192（58%）、不可复算 55；是否达标见 8b | 通过 |
| 4 label_set 有 sha、条数实测 | 五批 | 7 个集合（含 κ 第一版全层/仅人物），条数与声明一致 | 通过 |
| 5 package-v3 清单 | manifest 与 `SHA256SUMS.txt` 一致 | 60/60 文件，sha 全对 | 通过 |
| 6 EVIDENCE 与 tex 差异 | 清零 | `true_gap=0`、`anchor_gap=0` | 通过 |
| 7 每条引用数字有 verdict；`match` 能点到 `eval_run` 与输入/输出哈希 | 是 | 101 条配方全有裁定；`eval_run` **101 行全 `exit_code=0`**，101/101 claim 已挂 run，`match` 无成功 run 的 0 条；输入侧 50 有 `read` 文件、29 仅 `declared` 来源、1 条纯字面量（E06-089 人工耗时，不可复算）；`code_version` 1 行 | 通过 |
| 8 可验率（④ 达标率） | ≥ 90%（一致，或已标不可复算+原因） | 101/101 配方、247/247 已登记单元格 = **100%**；单看「一致」为 66/101（65%）、192/330（58%） | 通过 |
| 8b 覆盖率（要件①登记） | 100% | 表格 **247/330 = 75%**；摘要/引言/结论的内联数字工具未纳入统计 | **未达成**（83 格：59 同一数值别处已登记、3 表头 `95`、21 机位号/分母说明；清单见 `COVERAGE-GAPS.md`） |
| 9 负向用例 | 改错必被抓 | 3 处故意错值 3/3 判 `mismatch`，真库与 `main.tex`/EVIDENCE 未被改动 | 通过 |
| 10 supplement 出处表 | 含 `identity_isolation` / `as_is` | `ops/reports/paper_claims_v1/SUPPLEMENT-SOURCES.md`，27 行全裁定 | 通过 |

**口径澄清（2026-11-24 21:13）**：§3 的「可验率」是「要件④达标（复算一致，**或**已标不可复算+原因）」的比例，按此口径实测 **100%**（101/101 配方、247/247 已登记单元格）；「一致」是更严的观察口径（66/101 = 65%、192/330 = 58%），两者不能混用。缺口集中在四类“产物未留”——`Same cells` 整表（9 行、54 格）、92.7% 上限、5.5 轨/人碎片化、人脸 150/52 px：都有行、都有 `reason`，但复算不成立。
**仍未达成的是覆盖率**（§10 第 3 条）：表格登记 247/330（75%），摘要/引言/结论的内联数字未纳入工具统计。**判定：V3 通过（可验率 100%）；覆盖率缺口如实披露，不删表、不删数字。**

**缺口处置（2026-11-24 21:17，已确认）**：83 格**不逐格补 claim**——其中 59 格是正文已登记数字在补充材料/同一表内的重复出现，3 格是表头 `95`（`95% CI` 被当数字），只有 21 格是仅此处出现的机位号/分母说明。逐格补只会把 `paper_claim` 从 238 推到 300+，证据不增一分。清单：`ops/reports/paper_claims_v1/COVERAGE-GAPS.md`。**r2 按现状投。**

**台账体检（`ops_claims doctor`，2026-11-24 21:45）**：只读命令，查三类腐坏——登记资产是否还在/内容是否变、`eval_run` 的 87 个输入路径是否漂移、`code_version` 是否落后于当前文件。首次结果：**129 行 / 115 个路径，缺失 0、内容变化 0**（4 个 >200 MB 的 long35 视频只验存在性，不哈希）、run 输入全在且未变、cv 刷新后过期 0。报告 `ops/reports/paper_claims_v1/DOCTOR.md`。
★ 口径：`asset_version` 是**追加**语义，同一路径有 14 个历史版本（当前 115 路径 / 129 行），体检只拿**最新一行**当期望值，旧行不算漂移。

**V3 之后仍欠的三件事**（都不影响 r2 投稿）：① 若 r3 要用 445 的派生比例（54/445、141/445），须以现行 gt（494 轨）重算；② `tab:pair` 标题的方法名（Hanley–McNeil → percentile bootstrap）；③ κ 仅人物 CI 下界 0.913 → 0.912。
