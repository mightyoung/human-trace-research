# 科研工作台的通用流程模型（v1，2026-11-24）

> **一句话**：工作台服务的是「任何一次实证研究」，human-count 只是第一个实例。
> 本轮之前我把主语定成了「论文数字」，结果 `paper_claim` 认 EVIDENCE.md 的行号、核验器认 tex 表格——
> 这是把一个通用流程做成了单项目工具。本文档把主语改回**流程**，论文只是流程末端的一种交付物。

---

## 1. 偏差在哪里（自查）

| 位置 | 现在的样子 | 问题 |
|---|---|---|
| `paper_claim` | 字段 `display_id`(E01-004)、`evidence_line`、`section`、`tex_anchor_json`、`domain`(D1/D2/D3) | 表名与字段都是论文专用，换个场景（产品 A/B、模型评测报告）无法复用 |
| `tools/ops_claims.py` | 1436 行；`parsed_claims()` 解析 EVIDENCE.md，`load_tex_tables()` 解析 tex | 解析器写死在工具里，没有「换一个来源」的接口 |
| 核验算子 | `_pool_cell` / `_matrix_cell` / `_gate_run` / `_auc_block` / `_camera_pairs` / `_pack_metric` | 约 2/3 是**本项目特有**的取数逻辑，混在通用核验调度里 |
| 完成定义 | 「正文表格 + 内联数字的覆盖率」 | 只描述论文，不描述流程 |
| 实验卡 | 5 张 draft，与 claim/run 无任何连接 | **流程起点反而是孤岛**——这正是最初「实验卡形同虚设」的问题，没被解决 |

结论：**v2.1 解决的是「论文数字能不能回溯」，不是「实验能不能被流程管住」。** 前者只是后者的最后一环。

---

## 2. 通用流程：七环节

```
① 立题 Study
    问题是什么、谁负责、时间窗、成功标准
        │
② 立卡 exp_card + exp_gate(A/B/C/D)
    可证伪假设 / 基线(不改) / 候选(单变量) / 主指标 / 预算 / 人机分工
        │
③ 备料 asset_version · label_set · code_version
    数据、真值、代码各是什么版本（一律 sha256，只增不改）
        │
④ 定协议 eval_protocol
    采样怎么取、有没有泄漏、参数与随机种子、冻结包 manifest
        │
⑤ 执行 eval_run
    输入＝③的版本集合 + ④的协议；输出＝指标 + 产物 + 退出码
        │
⑥ 立论 claim（+ claim_anchor 出处 + claim_check 配方）
    一条可核验的陈述：数值 / 事实 / 对比 / 约束
        │
⑦ 交付 deliverable（+ deliverable_ref 引用位置）
    论文 / 报告 / 看板 / 数据集发布
```

**这张图里没有「论文」二字**，但把论文放进去完全成立：论文是 ⑦ 的一种 `kind=paper`，它的表格是 ⑥ 的锚点。

### 四条守恒关系（工作台的真正验收标准）

| # | 不变式 | 违反时的表现 |
|---|---|---|
| **C1** | 每条 claim 能回答「属于哪张实验卡」（空则显式标 `orphan`） | 数字散落，没人知道为什么做这个实验 |
| **C2** | 每条 claim 能回答「依据哪次 run / 哪些资产」 | 数字无法复算，只能靠记忆 |
| **C3** | 每次 run 能回答「用了哪些资产版本、哪个协议、哪份代码」 | 换数据/换代码后结论静默失效 |
| **C4** | 每个交付物能回答「引用了哪些 claim、其中多少已核」 | 投稿/发布时不知道哪句话有底 |

C1–C4 与领域无关：换成推荐系统离线评测、换成材料实验，都是这四条。

---

## 3. 三层分离

```
┌─ 内核（domain-agnostic，不出现任何项目名词）──────────────────┐
│ study · exp_card · exp_gate · asset_version · label_set        │
│ code_version · eval_protocol · eval_run · claim · claim_anchor │
│ deliverable · deliverable_ref                                  │
│ 核验调度：取配方 → 调算子 → 写 verdict + eval_run               │
│ 报告渲染：覆盖率 / 可验率 / 缺口清单 / doctor                   │
└───────────────────────────┬────────────────────────────────────┘
                            │  三个接口（见 §5）
┌─ 适配器（一个场景一个）────▼────────────────────────────────────┐
│ adapters/paper_tbiom.py     EVIDENCE.md + main.tex/supplement.tex │
│ adapters/demo_csv.py        最小示例：CSV + Markdown 报告          │
│ adapters/<未来>              JIRA 看板 / 实验记录 Excel / API       │
└───────────────────────────┬────────────────────────────────────┘
┌─ 实例（数据）──────────────▼────────────────────────────────────┐
│ ops/ops.db（台账）· ops/claims/*.json（配方）· docs/paper/*.tex    │
└─────────────────────────────────────────────────────────────────┘
```

**判断标准**：内核代码里 grep 不到 `paper` / `tex` / `EVIDENCE` / `机位` / `D1|D2|D3`。出现即越界。

---

## 4. 术语映射

| 现名 | 通用名 | 理由 |
|---|---|---|
| `paper_claim` | `claim` | 断言不限于论文 |
| `display_id`（E01-004） | `locator` | 人读定位符，格式由 adapter 定 |
| `section`（中文小节） | `anchor.container` | 容器可以是章节/表格/看板面板 |
| `evidence_line` | `anchor.ref_json.line` | 行号只是锚点的一种坐标 |
| `tex_anchor_json` | `claim_anchor`（独立表） | 一 claim 多锚点，多态 |
| `domain`（D1/D2/D3） | `stratum` | 分层标签，取值由 adapter 定 |
| `source_level` | `evidence_grade` | data / doc / none |
| `verify_specs.json` 的 `kind` | `operator` | 算子名，可注册 |
| `eval_protocol.identity_isolation` | `leakage_control` | 泄漏控制，不限身份 |

### 算子分层（关键）

| 层 | 算子 | 归属 |
|---|---|---|
| **通用** | `prop`(k/n→比例+CP CI)、`ratio`、`mean`、`count`(过滤计数)、`sum`、`diff`、`const`(代码常量)、`expr`、`json_path`、`file_hash`、`text_match`、`table_cell`(按行键取列) | 内核 |
| **场景** | `pool_cell`(τ×池 表)、`matrix_row`(θ 矩阵)、`gate_run`、`auc_block`/`auc_pair`、`pack_metric`(论文包清单)、`graph_metric`(身份图) | `adapters/paper_tbiom.py` 注册 |
| **逃生口** | `script`（跑一条命令，比 stdout） | 内核（通用） |

`table_cell` 是通用化的关键：tex 表行、Excel 行、CSV 行都能表达成「按行键取第 k 列」，本项目的 `matrix_row` 才需要留在 adapter。

---

## 5. 适配器接口（三个方法）

```python
class Adapter:
    name: str                       # "paper_tbiom" / "demo_csv"

    def discover(self) -> list[dict]:
        """返回待登记 claim 草案：locator / container / value_text / stratum / statement"""

    def anchor(self, claim) -> list[dict]:
        """返回锚点：[{type, ref_json}]，type ∈ tex_table|md_heading|csv_cell|json_path|url|file_line"""

    def resolve(self, anchor) -> tuple[int, list[str]]:
        """把锚点解析成 (行号, 该行数字序列)，供 table_cell/tex 类算子定位"""
```

内核只依赖这三个方法。新增场景＝新增一个文件，不动内核。

---

## 6. 数据模型变更（加法）

新增 5 张表，全部有真实写入者与消费者，不建空壳：

| 表 | 关键字段 | 首个写入者 | 消费者 |
|---|---|---|---|
| `study` | `id`、`name`、`question`、`owner`、`status` | 迁移脚本（1 行：multicam-reid） | claim、deliverable |
| `claim` | `id`、`study_id`、`card_id`、`locator`、`kind`、`statement`、`value_text`、`unit`、`stratum`、`evidence_grade`、`verdict`、`reason`、`recomputed_text`、`eval_run_id`、`provenance_json` | 迁移（238 行，沿用 `pc_*` id 不破坏引用） | 报告、交付物 |
| `claim_anchor` | `id`、`claim_id`、`type`、`ref_json`、`resolver`、`resolved_sha1` | 迁移（91 条 tex 锚点展开） | 锚点校验、出处表 |
| `deliverable` | `id`、`study_id`、`name`、`kind`、`version`、`path`、`sha256`、`status` | 迁移（r2 论文包、r3 稿、台账报告） | deliverable_ref |
| `deliverable_ref` | `id`、`deliverable_id`、`claim_id`、`location` | 迁移（按 tex 锚点的 file 反推） | 覆盖率、投稿检查 |

**`paper_claim` 保留为视图**（`CREATE VIEW paper_claim AS SELECT ... FROM claim ...`），老工具、老接口、老报告零改动即可继续读。写入侧改到 `claim`，只有两处：`do_import` 的 INSERT、`do_verify` 的 UPDATE。

**`claim_check` 暂不入库**：配方继续以 `ops/claims/verify_specs.json` 为权威文件——符合本项目「文件即记录、台账是索引」的铁律，也避免再造一张与文件重复的表。等出现「同一 claim 多套配方并存」的真实需求再入库。

---

## 7. 怎么证明「通用」不是口号

唯一可信的证据是**跑通第二个场景**。本轮做一个最小第二适配器：

- `examples/demo-csv/`：3 行实验指标 CSV + 1 份 Markdown 报告（虚构的推荐系统离线实验，与 ReID 无关）
- `adapters/demo_csv.py`：从 md 表格抽 4 个数字，锚点 `md_heading`，用 `table_cell` 算子从 CSV 复算
- 跑通 `discover → anchor → verify → report`，产出 4 条 claim 全部 `match`

如果内核里还有论文假设，这一步会直接失败。**失败即修内核，不为 demo 打补丁。**

---

## 8. 分期

| 期 | 内容 | 判定 |
|---|---|---|
| **K0（本轮）** | 本文档 + 5 张表 + `paper_claim` 视图 + 238 条迁移 + demo 适配器冒烟 | 本项目回归 66/35/0 不变；demo 4/4 一致；内核无项目名词 |
| **K1** | 把 `ops_claims.py` 拆成 `ledger/core.py` + `adapters/paper_tbiom.py`；算子按通用/场景分文件 | 拆分前后报告逐字节一致 |
| **K2** | 前端 `#/workbench` 增加「证据链」页签：卡 → 资产 → run → claim → 交付物 五列钻取 | 从任一张卡能点到它的 claim，从任一 claim 能点回卡 |
| **K3（按需）** | 第二个真实场景接入（非 demo） | 新场景零改内核 |

**不动的东西**：`exp_card`/`exp_gate` 表结构与四关语义、`asset_version` 追加规则、`eval_run` 的 match 约束、`main.tex`/`supplement.tex` 的任何数字、`#/workbench` 现有交互。

---

## 9. 与前一份设计的关系

`docs/ledger-v2-design-2026-11-24.md`（v2.1）**不作废**，降级为「paper_tbiom 适配器的实现说明」：它的 6 要件、覆盖率口径、Clopper-Pearson、tex 解析三坑，都是这个适配器内部的细节，对内核无约束力。

本文档取代它的**主语**：从「论文数字」改为「实验流程」。

---

## 10. 执行准入：从「记录」变成「门禁」（2026-11-25）

第 2 节把流程画成链，但链上一直有个断点：`exp_card`/`exp_gate` 只被工作台读写，
**执行入口 `tools/ops_run.py` 从不读它们**——卡还没批也能跑。现在补上。

**规则**（研究类 `kind ∈ {train, eval, infer, encode}`）：

| 情况 | 行为 |
|---|---|
| `--card EXP-00x` 且卡 `state ∈ {approved, building, testing, running}`、`exp_gate` 中 **A 关最新为 approve** | 放行，run 写入 `card_id/card_role` 并追加到 `exp_card.runs_json` |
| `--card` 但卡不满足上面条件 | **exit 4**，报出当前 state、缺哪一关 |
| `--no-card --note "理由"` | 放行，note 前缀 `[no-card]`，可统计、可审计 |
| 两者都没给 | **exit 4**，列出所有卡及状态 |
| `build / export / other` | 不受限（管道与搬运不是实验） |
| `--record-only`（补历史） | 不受限（登记既往事实，不是发起新执行） |

**为什么只硬门禁 A 关**：A 关（立项）的判据是离散的——「同意/退回/搁置」，
机器可判。B（数据与真值）、C（边界变化）、D（成果验收）的判据是判断性的，
硬编会把"表格没填完"当"不合格"，反而诱导假填。它们保持记录性。

**它暴露的第一件事**：现有 5 张卡**一张都没过 A 关**（EXP-001 `proposed`，EXP-002~005 `draft`）。
所以门禁上线当天，研究类 run 只能走 `--no-card`。这不是流程变严，而是此前
「卡是摆设」这一事实第一次显形。

**下一步若要更严**：把 `--no-card` 的比例做成看板指标（`run.card_id IS NULL` 占比），
高于阈值就说明立项环节没人用，问题在流程而不在门禁。
