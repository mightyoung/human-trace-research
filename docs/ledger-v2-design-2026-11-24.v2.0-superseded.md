# 科研资产台账 v2：以实验为核心对象的重构设计

> 版本：2026-11-24 ｜ 状态：**设计稿，未实施** ｜ 依据：本机代码与 `ops/ops.db` 只读实测 + 开源方案查证
> 前序：`docs/ops-console-design-v1.md`（v1，P0 已上线）、`docs/ops/backlog-plan.md`（G1–G11 缺口清单）
> 口径：§1 的数字全部为只读查询/读码结果；推断与建议单独标注。本次**没有改任何代码与数据**。

---

## 0. 结论先行

**实验卡空转的根因不是页面，而是“实验”在库里没有成为汇聚点。** 现在同一件事有三个互不相认的载体：

| 载体 | 行数 | 实际内容 | 与 run / 数据 / 模型的连接 |
|---|---|---|---|
| `experiment` 表 | 4 | `ops_ingest_train` 自动生成的训练批次 | 35 条 `experiment_run`，**角色全是 variant、0 条 baseline**；`hypothesis`/`dataset_id` 全空 |
| `exp_card` 表（11-24 新增） | 5 | 11-23 报告 §10 的五张研究卡 | **无任何外键**，`runs_json` 全空 |
| `docs/*.md` 叙述 | 237 篇 | 结论与证据 | 靠文件名互指，无 ID |

三条链路各自断开：run 没有数据身份（`dataset_id` 11/47、`input_hash` 2/47、`git_rev` 0/47），指标没有定义（`metric_def` 不存在），测试集没有对象（holdout 只在文档与脚本里）。在这种底座上，卡再漂亮也只能是文本。

**设计主张（三条）**

1. **不换库、不换框架、不搬大文件**：仍是 `ops/ops.db` + FastAPI + Vue，仍是“只登记指针 + 校验和”。v1 的 13 张表全部保留，v2 只做**加法**（§4）。
2. **`experiment` 升级为聚合根**：合并 `exp_card`；向下经 `trial → run → run_input / metric_value / evaluation` 连到数据版本、模型 sha、代码 sha、测试集版本；关口（A–D）与七件套从“手填”改为“系统按事实判定”。
3. **标注与数据集各收敛为一个入口 + 一个登记表**：标注侧用 `label_schema` 把 18 类判定统一进已有的 `annotation_task/item/label`；数据侧新增 `dataset_version`，让“run 用的是哪一版数据”可回答且不可被原地覆盖。

**与 v1 的关系**：v1 回答“有没有账”；v2 回答“账能不能串成一个实验”。`backlog-plan.md` 的 G4/G8/C1/C2/G7 全部并入本方案的 P1–P4，不另起炉灶。

**需要你定的一件事**（§9）：P2 标注写切换的推进方式——逐类灰度（推荐）还是一次性切换。

---

## 1. 问题诊断（证据）

### 1.1 台账实况（`ops/ops.db` 只读，2026-11-24）

| 表 | 行数 | 关键事实 | 证据 |
|---|---|---|---|
| `run` | 47 | train 39（38 条来自 `intent_model*.py`）/ build 4 / encode 2 / eval 1 / export 1；`dataset_id` 非空 **11**、`model_id` 非空 **3**、`input_hash` **2**、`git_rev` **0** | SQL 统计 |
| `experiment` / `experiment_run` | 4 / 35 | 4 条全为自动汇总；35 条链接 `role` 全 `variant` | SQL 统计 |
| `exp_card` / `exp_gate` | 5 / 0 | 五张卡无 trial、无 gate 记录 | SQL 统计 |
| `model` | 9 | 5 个人脸 onnx + 3 个 OSNet + 1 个意图模型；`deployable` 全 0；磁盘上 ≥60 个权重文件，**未登记 ≥51** | `tools/ops_model.py`；`models/` 清单 |
| `dataset` | 5 | **3 行无 `snapshot_hash`**（`ds_intent_synth`、`ds_intent_1012_cc`、`ds_6cam_long35`）；哈希算法 `rel-size-mtime-v1` **含 mtime、不是内容哈希** | `tools/ops_dataset.py:76-104` |
| `asset` / `adjudication` | 0 / 0 | 建了表，没人写 | SQL 统计 |
| `annotation_task` / `item` / `label` | 8 / 704 / 781 | 8 个任务 `mode` 全 `single`、`assignee` 全空；`annotator` 只有 `legacy`(302) / `console`(464) / `seed`(15)，**没有真人身份** | SQL 统计 |
| `run_metric` | 2401 | `val_f1`/`val_acc`/`loss` 各 796，其余是结构指标；没有 n、没有区间 | SQL 统计 |
| `job` | 5 | 全部是 `label` 卡 | SQL 统计 |

### 1.2 六个结构性缺口

| # | 缺口 | 不做会怎样 | 证据 |
|---|---|---|---|
| D1 | **“实验”三载体并存且互不连接** | 卡上写什么都无法被验证；论文数字回溯不到 run | §0 表 |
| D2 | **接入率极低**：`tools/` 501 个 py 中只有 4 个训练脚本接了台账，shell 包装器 0 个 | 台账只覆盖意图分类家族；ReID/聚类/人脸/属性各线的实验都在账外 | `grep -l "ops_db\|ops_run\|Reporter"` |
| D3 | **数据无版本对象**：`dataset` 一行 = 一个名字，重扫即原地改 `snapshot_hash` | 历史 run 引用的“当时的 hash”无处可查；stale 判定只能比“现在” | `ops_dataset.py` `ON CONFLICT DO UPDATE` |
| D4 | **指标无定义、无数值行**：`metrics_json` 是自由文本；`normalize_metrics` 只认 5 个键 | 项目反复踩的“分母漂移 / 三种口径”无法在库里表达 | `ops_db.py:395-426` |
| D5 | **测试集无对象**：holdout 是预注册文档 + `run_holdout.sh` 里锁的 sha256；且**人员不隔离** | 正式结果与开发结果在库里无法区分 | `docs/PREREGISTRATION_holdout_2026-10-24.md:205` |
| D6 | **代码身份缺失**：仓库不是 git，`git_rev` 恒空；`code_fingerprint` 只算入口脚本 | “同一份代码”只能靠人记 | `ops_run.py:58-67` |

### 1.3 标注侧：18 类判定，进台账的只有 8 类

| 组 | 任务类型 | 落盘 | 进台账 |
|---|---|---|---|
| 已入账（8） | pair / mirror / mask / face_person / face_pair / truth_pair / truth_event / s6b | `labels.json`、`faces_labels.json`、`truth/labels.json` | 是（镜像） |
| 未入账（6） | trackpair(117) / clusterver(15) / mergever(33) / attr(756 项、739 判) / E21 标簇(515) / E21 单图 | 各自独立 JSON | **否** |
| 未入账（2） | 选择器人工见证(20) / 自动人脸复核采纳 | `reid/selector_pool.json` | **否** |
| 未入账（2） | 人脸库命名 / 聚类工作台人工改簇 | `faces_labels.json.name`、`clusters_state.json` | **否** |

衍生问题（均有代码位置，见调查记录）：

- **值域五套**表达同一个“是否同人”：`CONF/REJ`、`1/0/2`、`same/diff/unsure`、`MAYBE`、`MODERATE`。
- **判定类写入调用点约 17 处**散在 6 个前端文件（`AnnotateView.vue:94/190/288/358/416/438`、`FacesView.vue:95/97/100/103`、`TruthView.vue:190`、`AttrAnnotator.vue:157`、`E21LabelView.vue` 5 处），全部走老接口；后端 **18 个 POST 路由完全不经台账**（attr / trackpair / clusterver / mergever 各 1、e21 标签 3、选择器池 4、聚类工作台 5、属性查询 2），已镜像的 4 个路由失败时也只打 stderr（`main.py:66-75`）。
- **未入账的 5 类标签没有任何消费者**（`tools/` 全目录 0 命中）：trackpair / clusterver / mergever / e21 标了但不进任何评测。
- **评测侧读的是另一份副本**：≥12 个评测脚本读 `output_crosscam/analysis/truth/remote/labels.json`，不是控制台的那份。
- kappa 有 7 个脚本，全部基于 `deliverable/` 的 CSV，与台账标注人是两套世界。
- 8 个离线 HTML 标注页结果只在浏览器 localStorage，靠“导出”带出，回流情况不一。

### 1.4 工程结构

`main.py` 2412 行单文件；`AnnotateView.vue` 1295 行混了 6 种交互；`tools/` 501 个 py（10 月 419 个、11 月 82 个）。这是“每次新增一类判定就加一个路由 + 一个 JSON + 一个页签”的累积结果，也是 D1–D6 的土壤。

---

## 2. 开源方案调研与借鉴点

| 方案 | 核心思想 | **借鉴到本项目** | 不照搬的理由 |
|---|---|---|---|
| **CVAT**（标注） | `Project → Task → Job` 三层；`JobType = annotation/ground_truth/replica`；`Stage = annotation/validation/acceptance` × `State = new/in progress/completed/rejected`；`ValidationMode = gt/gt_pool`（蜜罐帧）；`QualitySettings → QualityRequirement(target_metric, threshold, filter)`；`QualityReport` 按 job→task→project 逐级汇总；重操作走 RQ 异步 worker | ① 任务类型 = 渲染器 + 值域（对应 CVAT 的 label spec）；② 阶段/状态二维分离；③ 黄金题（`stratum=golden`）与复核作为一等流程；④ 质量报告逐级汇总 | CVAT 面向“画框”，本项目是“判定型”标注；其服务栈（PostgreSQL/Redis/Kvrocks/ClickHouse/OPA）对单机项目过重 |
| **Datumaro**（CVAT 的数据集框架，MIT） | 数据集 = items + categories + **subsets**；`transform / filter / merge / split / diff / stats`；ReID 专用切分“**避免同一身份同时出现在训练与测试**” | ① `dataset_version` 的 `split` 与 `lineage`（由哪版经什么算子派生）；② 身份不重叠的切分规则可直接复用其实现思路；③ `diff` 用于版本变更说明 | 不引入其格式生态，只借模型与切分思想 |
| **MLflow**（模型注册/追踪） | `RegisteredModel → ModelVersion`；版本带 lineage（哪个 run 产出）；**stages 已弃用，改用 alias（如 `@champion`）+ tags**；MLflow 3 的 LoggedModel 把“模型 + 数据集 digest + 评估”绑在一起 | ① `model_alias` 取代 `deployable` 布尔；② 版本 = 权重 sha1，注册时即带 `trained_by_run`；③ 评估结果必须带数据 digest | 不部署 MLflow server；保留一个可选导出适配器 |
| **DVC / lakeFS**（数据版本） | DVC：`.dvc` 指针 + 内容哈希缓存 + `dvc.yaml`/`dvc.lock` 的 DAG 复现；lakeFS：对象存储上的零拷贝分支与不可变 commit，“**记录 commit 而不是分支名**” | ① `dataset_version` 不可变，run 引用版本而不是数据集名；② `trial_key` = 规范化输入的哈希，相当于 `dvc.lock` 的一行 | 单机、数据在本地盘，不需要对象存储层 |
| **ClearML**（训练任务/队列） | agent + queue 模型：任务入队，远端 agent 拉取执行；自带数据集版本与 HPO | P5 远程算力：在 `192.168.11.16` 跑一个极简 worker 从台账拉 `job(kind=run)` | 现阶段单 GPU、任务分钟到小时级，SSH + 包装器足够 |
| **Kubeflow / Ray Train** | K8s 原生 CRD / Actor 模型的分布式训练编排 | 不采用 | v1 §2.2 已明确不做 K8s；规模不匹配 |
| **评测集工程实践**（evaluation harness） | 黄金集不可变 + changelog；每个用例带来源；统计门限用置信下界而非均值；评测结果盖数据哈希 | ① `test_suite` 封存后不可改，改即新版本；② `metric_value` 强制 `n + ci`；③ 回归告警按“基线 − 容差”且带区间 | — |

**可直接复用的组件**：`datumaro`（切分/校验工具，pip）、`scipy.stats`（kappa/区间）、现有 `ops_selftest.py` 框架、`ops_run.py` 采集钩子。**只借思想不引依赖的**：CVAT 流程模型、MLflow 注册表语义、DVC 锁文件。

---

## 3. 目标架构

```
┌─ 交互层──────────────────────────────────────────────────────────────────────┐
│ #/ops 总览 · #/experiments 实验 · #/annotate 标注(统一) · #/datasets 数据集  │
│ #/models 模型 · #/tests 测试集与评测 · #/runs 运行 · #/jobs 派发 · 其余不动  │
├─ 服务层（/api/ops/*，按域拆 router）─────────────────────────────────────────┤
│ experiments · annotations · datasets · models · evaluations · runs · jobs    │
│ 老接口（/api/labels 等 22 个写路由）降级为“台账 → JSON 投影”的适配器         │
├─ 领域层（ledger 包，替代 ops_db.py 的单体）──────────────────────────────────┤
│ core(连接/迁移/audit) · datasets · annotations · experiments · metrics       │
│ evaluations · models · jobs        ← 唯一写入口；CLI 与 API 都调它           │
├─ 登记层（ops/ops.db，schema v2，append-only + 版本不可变）───────────────────┤
│ experiment ─< experiment_run(trial) >─ run ─< run_input >─ dataset_version   │
│      │                                   │            └─ model / code / env  │
│      ├─< exp_gate                        ├─< metric_value >─ metric_def      │
│      └─ evidence(由事实判定)             └─< evaluation >─ test_suite        │
│ annotation_task ─< item ─< label / adjudication ；label_schema ; export      │
├─ 执行层──────────────────────────────────────────────────────────────────────┤
│ ops_run（采集+登记）· ops_eval（封存集评测）· ops_export（标注导出）· bundle │
├─ 资产层（原地不动）──────────────────────────────────────────────────────────┤
│ media-resource/ · datasets/ · models/ · output_crosscam/ · remote-results/   │
└──────────────────────────────────────────────────────────────────────────────┘
```

**与 v1 的差异只在三处**：领域层从单文件变成按域分包；实验成为聚合根；标注与数据集各有唯一的登记表与入口。执行层与资产层不动。

**硬约束（在 v1 三条之上追加四条）**

4. **正式结果只能由 `ops_eval` 写入**：`run.formal=1` 当且仅当所有输入都已登记且测试集处于 `sealed`，否则只能是 `split=dev` 的开发结果。
5. **`dataset_version` 不可变**：封存后任何字段不可改；要改就新建版本并写 `changelog` 与 `parent_id`。
6. **`metric_value` 必须有定义**：`metric` 必须先在 `metric_def` 登记（含分母与方向），否则拒绝写入。
7. **标注只有一种写法**：`POST /api/ops/label`；JSON 文件一律是台账的投影，不再接受直接写。

---

## 4. 核心数据模型

### 4.1 实体关系（新增加粗）

```
experiment ─┬─< experiment_run (trial: role + trial_key) >── run ──< run_input ──> dataset_version / model / code_version / env_version
            ├─< exp_gate        (关口决定，append-only)          ├─< metric_value >── metric_def
            ├─ evidence_json    (七件套，系统判定)                └─< evaluation  >── test_suite ──> dataset_version
            └── parent_id       (分支/子实验)

dataset ──< dataset_version ──< lineage(parent, op, params)      ; run_input.role = train/val/test/detector/reid/face/labels
model   ──< model_alias (champion/baseline/…) ; parent_model_id
annotation_task ──< annotation_item ──< label / adjudication ; assignment ; annotation_export ──> dataset_version(kind=label_set)
label_schema (task_kind → 问题 / 单元 / 值域 / 渲染器 / 归一化)
```

### 4.2 变更清单

**改：`experiment`（吸收 `exp_card`）**

| 新增列 | 来源 | 说明 |
|---|---|---|
| `kind` | 新 | `research`（人立项的卡）/ `sweep`（超参扫描组）/ `ingest`（自动汇总批次） |
| `question`、`metric_plan_json`、`roles_json`、`evidence_json`、`risk`、`source` | `exp_card` 同名列 | 原样迁入 |
| `parent_id`、`updated_at`、`sealed_at` | 新 | 分支与封存时间 |
| `status` | 取 `exp_card` 的状态机 | `draft/proposed/approved/building/testing/running/scored/reviewed/accept_pending/adopted/revise/paused/failed/archived`；既有 4 行 `done` 迁为 `archived`（写 audit） |

**改：`experiment_run` → 试验（trial）**：保留 `(experiment_id, run_id, role)`，加 `trial_key`（规范化输入哈希：数据版本 + 模型 sha + 代码 sha + 参数）、`arm`（候选编号，如 `C1 冻结原型`）、`is_primary`。**同一 `trial_key` 的多条 run = 同一试验的多次执行**，用于漂移检测；不新建 trial 表。

**改：`run`**：加 `split`（`dev/val/test`）、`formal`（0/1）、`attempt_no`。`dataset_id`/`model_id` 保留为“主输入”的便捷列，完整输入在 `run_input`。

**新表**

| 表 | 关键字段 | 约束 |
|---|---|---|
| `run_input` | `run_id, kind(dataset_version/model/code/env/label_set/test_suite), role, ref_id, ref_sha1` | 同 run 内 `(kind, role)` 唯一 |
| `dataset_version` | `dataset_id, version_no, manifest_uri, content_hash, hash_alg, file_count, total_bytes, split, lineage_json, changelog, parent_id, status(draft/sealed/deprecated), sealed_at` | `sealed` 不可更新（触发器） |
| `metric_def` | `name PK, version, definition, unit, direction(higher/lower), denominator, range_json, note` | 无定义不收数 |
| `metric_value` | `run_id, metric, split, subgroup, value, n, ci_low, ci_high, ci_method, evaluation_id` | PK 含 subgroup；`n` 非空 |
| `test_suite` | `name, version, dataset_version_id, label_set_version_id, protocol_json(指标/门限/分组/评测脚本 sha1), isolation(time/identity), status, parent_id, changelog, sealed_at` | 封存后协议不可改 |
| `evaluation` | `run_id, test_suite_id, status, report_uri, started_at, ended_at` | 唯一 `(run_id, test_suite_id)` |
| `code_version` | `sha1 PK, files_json` | 由 `ops_run.code_fingerprint()` 扩展为入口脚本 + 其 import 闭包 |
| `env_version` | `sha1 PK, kind(pip/conda/image), spec_json` | 镜像取 digest、环境取 freeze |
| `model_alias` | `model_id, alias` | `champion/baseline/paper/main` 等，取代 `deployable` |
| `label_schema` | `task_kind PK, version, question, unit, values_json, multi, renderer, normalize_json` | 18 类各一行 |
| `assignment` | `task_id, item_id, annotator, role(annotator/reviewer/adjudicator), assigned_at, done_at` | 支撑双标与 kappa |
| `annotation_export` | `task_id, label_max_id, item_count, content_hash, uri` | 导出即登记为 `dataset_version(kind=label_set)` |

**`label_schema` 的归一语义**（只做映射，不改原值）：`SAME / DIFF / UNSURE / NON_PERSON / UNCLEAR / ID:<subject> / FP / PROPS:<json>`。原值保留在 `label.value` 与 `data_json`，规范化值进 `label.canonical`（新增列）。例：`CONF→SAME`、`REJ→DIFF`、`MODERATE→UNSURE`、`1→SAME`、`0→DIFF`、`2→UNSURE`、`same/diff/unsure` 同名、`reject→NON_PERSON`、`person:姓名→ID:姓名`、`fp→FP`。attr 的多字段与 clusterver 的成员标记走 `PROPS:` + `data_json`。

---

## 5. 接口边界

| 边界 | 谁能写 | 谁只读 | 说明 |
|---|---|---|---|
| 领域层 `ledger/*.` | CLI（`tools/ops_*.py`）、`/api/ops/*` | — | **唯一写入口**；每函数自带 audit |
| 老接口 `/api/labels`、`/api/faces/*`、`/api/truth/labels`、`/api/attr/label`、`/api/trackpair/*`、`/api/clusterver/*`、`/api/mergever/*`、`/api/e21/*`、`/api/selector/witness|revoke`、`/api/selector/auto_face/apply`、`/api/cluster/*` | 无人 | 全部 | P2 起降级为适配器：写 → 转调 `ops_label`；读 → 台账投影。保留路由名一个发布周期 |
| JSON 文件（`backend/data/*.json`、`analysis/truth/remote/labels.json`） | `ops_export` 投影 | 评测脚本 | 投影带 `content_hash`，消费者可核对 |
| `ops_run.py` | 训练/推理脚本 | — | 新参数：`--experiment --role --input kind:role:id --split`；`Reporter.metric(n=, ci=)` |
| `ops_eval.py`（新） | `evaluation` + `metric_value` | — | 封存集哈希不符即拒绝登记正式结果 |
| Agent | 开发数据、草稿、建议 | 封存集、关口决定 | 对应 v1 “人与 agent 同一入口”，封存与审批只对人开放 |

**页面与接口对应**

| 页面 | 读 | 写 |
|---|---|---|
| `#/experiments`（由 `#/workbench` 演进） | `GET /api/ops/experiments/{id}`：卡 + trials + inputs + metrics + evaluations + gates + evidence | `POST .../gate`、`.../state`、`.../trial`（把 run 挂为 trial） |
| `#/annotate`（统一） | `GET /api/ops/queue?task_id=`（已有）+ `label_schema` | `POST /api/ops/label`（已有）+ `.../assign`、`.../adjudicate` |
| `#/datasets` | `GET /api/ops/datasets`、`/datasets/{id}/versions` | `POST /datasets/{id}/scan`、`.../seal` |
| `#/models` | `GET /api/ops/models` | `POST /models/register`、`.../from-run`、`.../alias` |
| `#/tests` | `GET /api/ops/suites`、`/evaluations` | `POST /suites`、`.../seal`、`POST /evaluations` |
| `#/runs`、`#/train`、`#/jobs`、`#/ops` | 不变，增量显示 `formal/split/inputs` | 不变 |

---

## 6. 模块拆分与重构范围

| 对象 | 现状 | 处置 | 归属阶段 |
|---|---|---|---|
| `tools/ops_db.py`（755 行） | 单文件访问层 | **拆**为 `ledger/` 包：`core / datasets / annotations / experiments / metrics / evaluations / models / jobs`；`ops_db.py` 保留为 re-export 壳（CLI 不变） | P0 |
| `tools/ops_{run,ingest,ingest_train,bundle,stale,model,dataset,metric}.py` | 各自直接写表 | **改**为只调 `ledger/`；CLI 参数保持兼容 | P0–P1 |
| `webapp/backend/main.py`（2412 行） | 单文件 | **拆**为 `routers/{annotate,faces,truth,attr,trackpair,clusterver,mergever,reid,cluster,selector,autoface,media,config}.py`；`main.py` 只做装配 | P2（随写切换逐个迁出） |
| `webapp/backend/workbench_api.py` | 读 `exp_card` | **改**读统一后的 `experiment`；路由迁至 `/api/ops/experiments/*`，旧路由保留转发 | P0–P1 |
| `webapp/backend/e21_label.py`、`selector.py`、`faceauto.py` | 各自落 JSON | 写路径改走 `ops_label`（e21）/ 改为 `audit + job`（见证、采纳，不进 label 表） | P2 |
| `AnnotateView.vue`（1295 行）、`TruthView`、`FacesView`、`E21LabelView`、`AttrAnnotator` | 5 个页面 6 套交互 | **拆**为统一外壳 `AnnotateView.vue` + 渲染器注册表 `renderers/{pair,trackpair,cluster,merge,mask,face,attr,e21,truth}.vue`；每个渲染器 < 300 行 | P2 |
| `ops_ingest.py` 的 8 任务映射 | 硬编码 | **改**为读 `label_schema`；新增 10 类 | P2 |
| `ops_selftest.py` | 既有用例 | **扩**：迁移等价、版本不可变、封存拒绝、归一化往返 | P0–P4 |
| 8 个离线 HTML 标注页 | localStorage | **冻结**：不再新建；已有结果一次性导入后归档 | P2 |
| 12+ 个读 `remote/labels.json` 的评测脚本 | 直读副本 | **不改代码**，但投影必须带哈希；P4 起逐步改为读台账导出 | P2/P4 |
| 497 个未接台账的 `tools/*.py` | 账外 | **不批量改造**；只对进入论文/控制台的结论性脚本（白名单，预计 15–25 个）加 `ops_run` 包装 | P1 起持续 |
| 资产层目录、媒体、权重 | — | **不动** | — |

---

## 7. 分阶段开发计划

| 阶段 | 目标 | 主要改动 | 预估 |
|---|---|---|---|
| **P0 地基** | schema v2 落库，三载体合一，零界面回归 | 加表加列；`ops_migrate_v2.py`（`--dry-run/--apply`，幂等）；`exp_card`→`experiment`；`experiment` 4 行 `done→archived`；`workbench_api` 改读统一表 | 3 天 |
| **P1 实验成为核心** | 任一指标可回溯到数据/模型/代码 | `ops_run` 新参数；`run_input`/`code_version`/`env_version` 采集；`metric_def` 首批登记 + `metric_value`；试验聚合、对比、漂移检测；证据七件套改为系统判定；回填 47 条 run | 5 天 |
| **P2 标注统一** | 18 类判定一个入口、一种写法 | `label_schema` 18 行；10 类未入账任务摄入；写切换（按类灰度）；统一 `#/annotate` 外壳 + 渲染器；`ops_export` → `label_set`；`assignment` 与 kappa 计算 | 8 天 |
| **P3 数据集与模型入口** | 数据有版本、模型有别名 | `dataset_version` + `#/datasets`；`ops_dataset` 支持 `--content-hash` 与排除派生物；`model_alias` + `#/models`；登记控制台与论文实际使用的权重；回填 `run_input` | 5 天 |
| **P4 测试与统计** | 正式结果与开发结果可分，回归可见 | `test_suite` 封存；`ops_eval`；回归对比与告警；共享统计库 `ledger/stats`（bootstrap、确定性 tie-break）替换 16 处各自实现 | 6 天 |
| **P5 远程与资源（按需）** | 远端产出自动回主库 | bundle v2 闭包（含 `run_input/metric_value/evaluation/exp_gate`）；远端 worker 从台账拉 `run` job；`resource` 登记 | 按需 |

**顺序约束**：P0 → P1 必须先于 P2（P2 的导出要登记为 `dataset_version`，依赖 P0 的表与 P1 的 `run_input`）；P3 可与 P2 并行（不触碰写路径）；P4 依赖 P1 与 P3。

**每个阶段的退出条件见 §10。**

---

## 8. 迁移与兼容策略

| 项 | 做法 |
|---|---|
| 备份 | 迁移前 `ops/db_backup/ops_<时间戳>.db`；`ops_migrate_v2.py` 开头校验备份存在 |
| DDL | 全部 `CREATE TABLE IF NOT EXISTS` + 既有 `ADDED_COLUMNS` 机制；`meta.schema_version` → `2`；不 DROP、不 RENAME |
| `exp_card` → `experiment` | 逐字段拷贝（5 行），`id` 不变（`EXP-001…005`），`exp_gate` 表名与列不变；`exp_card` 保留一个发布周期后删除（删除前再备份） |
| `experiment.done → archived` | 仅 4 行；每行写 `audit(action='migrate.v2')`；`ops_ingest_train` 新产出行直接写 `archived` |
| `dataset` | 现有 5 行各生成一条 `dataset_version`（有 hash 的 `status=sealed`，无 hash 的 `draft`）；`dataset.snapshot_hash` 保留为“当前版本”指针，老代码不受影响 |
| `run_input` 回填 | 由 `run.dataset_id / model_id / env_json.code_sha1` 生成；缺失即留空，不猜 |
| 标注写路径 | 开关 `HC_ANNOT_WRITE = legacy | ledger`（环境变量），默认 `legacy`；P2 按**任务类型**逐个切到 `ledger`，每切一类跑 `verify` |
| JSON 投影 | 老接口与评测脚本继续读 JSON；`ops_export` 生成的投影带 `content_hash`，`ops verify-projection` 每日比对 |
| 远程 | 远端 `init_db` 自动建 v2 表；v1 格式 bundle 仍可导入（upsert）；v2 bundle 多出的表在旧端导入时跳过 |
| 前端 | `#/workbench` 保留并重定向到 `#/experiments`；`vite build --emptyOutDir false`；改后端需重启 |
| 回滚 | 恢复备份 + 开关回 `legacy`；新增表可留置不影响旧代码；旧路由未删除 |
| 用户在用 | 控制台正在被使用（11-24 16:06 有操作记录）：每阶段单独发布，发布前重建 dist 并在临时库副本上过 selftest |

---

## 9. 风险点

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | 标注写切换回归（约 17 处调用点、1295 行页面） | 判定丢失或错位 | 按任务类型灰度；每类切换前后 `verify` 零差异；保留一个周期的回退开关 |
| R2 | 归一化丢信息（attr 多字段、clusterver 成员标记、mask 多边形、人脸命名） | 评测口径漂移 | 原值永不改写，`canonical` 只是附加列；`label_schema.normalize_json` 版本化 |
| R3 | 评测脚本读的是 `analysis/truth/remote/labels.json` 副本 | 评测用了旧真值 | 投影带哈希；把“副本哈希 = 台账导出哈希”列入验收；P4 逐步改读 |
| R4 | 现有哈希含 mtime、manifest 混入 `segments/*.jpg` | 重生成截图即“数据变了” | `dataset_version.hash_alg` 记录算法；新增内容哈希选项与排除规则；旧哈希不改写 |
| R5 | 无 git，代码身份只能靠文件哈希 | “同一份代码”不可严格证明 | `code_version.files_json` 存文件清单；UI 明示“文件哈希 ≠ commit” |
| R6 | 封存测试集与“留身份”冲突：现有 holdout 人员不隔离 | 论文口径被误读 | `test_suite.isolation` 必填；`formal` 结果旁始终显示隔离方式 |
| R7 | 台账接入率不升，v2 再次空转 | 投入无回报 | 硬门：正式结果只能由 `ops_eval` 登记；总览显示“账外结论”计数；白名单脚本包装纳入验收 |
| R8 | SQLite 并发与远程两套库 | 锁等待、数据分叉 | 沿用 WAL + busy_timeout；远端只做过程账本，bundle 回传 |
| R9 | 敏感数据（人脸图、`person:姓名`）进入统一库 | 隐私与合规 | 姓名以 `subject` 代号存储，映射表单独保管；导出默认脱敏 |
| R10 | 阶段并行导致互相干扰 | 返工 | 遵守 §7 顺序约束；P2 与 P3 之间只共享只读接口 |
| R11 | 大爆炸式重写 | 长时间不可用 | 所有阶段加法优先；`main.py` 与 `AnnotateView` 随写切换逐个迁出，不一次重写 |

---

## 10. 验收标准

**P0**

1. 迁移前后 `run=47 / label=781 / annotation_item=704 / experiment_run=35` 计数不变；`exp_card` 5 行逐字段等于 `experiment` 对应行。
2. `--dry-run` 输出与 `--apply` 的实际写入差异为 0；重复执行不新增行。
3. 用备份恢复后，`#/workbench`、`#/runs`、`#/train`、`#/annotate` 全部可用（临时库副本实测）。
4. `ops_selftest` 既有 23 个用例全绿（`backlog-plan.md` 记载），新增迁移用例 ≥ 3。

**P1**

5. 任取一条 `metric_value`，页面可依次点到：run → trial（含 `trial_key`、role）→ experiment → `run_input` 中每个输入的 `ref_sha1`。
6. 五张卡：每张要么 ≥ 1 个 trial，要么状态明确为“未执行”；`evidence` 中 logs / metrics / model / data 四项由系统按 `artifact` 与 `run_input` 判定，不依赖手填。
7. 同一 `trial_key` 的两次执行若指标不一致，列表出现“漂移”标记。
8. `metric_def` 中不存在的指标名写入被拒绝（负向用例）。

**P2**

9. `label_schema` 覆盖 18 类任务；`annotated/pending` 总数在总览只有一个来源。
10. 前端对老写接口的调用归零（或全部经适配器且带弃用标记）；`verify` 对 18 类任务零差异。
11. `analysis/truth/remote/labels.json` 与台账导出的 `content_hash` 一致。
12. 统一 `#/annotate` 上每个渲染器 < 300 行；键盘操作与原页面一致（`1/2/3`、`J/K`）。
13. 第二标注人抽 20 条后，页面能显示 kappa 与分歧清单（这是唯一需要人工配合的项）。

**P3**

14. 5 个既有数据集各有版本记录；重扫不改变已封存版本的 `content_hash`。
15. 控制台与论文实际使用的权重（按白名单）100% 有 `model` 记录且 `sha1` 与磁盘一致；`model_alias` 可解析到唯一版本。
16. `formal=1` 的 run 必须引用 `sealed` 的 `dataset_version`（负向用例：引用 `draft` 版本被拒）。

**P4**

17. 封存后篡改测试集任一文件，`ops_eval` 拒绝登记并指出差异文件。
18. 构造一条明显劣化的 run，总览出现回归告警，且告警带区间与基线引用。
19. `ledger/stats` 被 ≥ 5 个评测脚本引用；bootstrap 次数与 seed 在 `metric_value.ci_method` 中可查。

**总体**：v1 §8 留下的验收标准“任取一个历史数字能点到 run → 数据 → 模型 sha”在 P3 结束时闭环；P4 结束时还能回答“这是开发结果还是正式结果、在哪个封存测试集上、隔离方式是什么”。

---

## 附：本次调查未确认的事项

- pair / mirror / mask / truth / trackpair / clusterver / mergever 队列的生成脚本（文件头无注释，`tools/` 内未命中）。
- `trackpair / clusterver / mergever / attr` 在页面中的具体调用行（仅确认 `api.ts` 中的方法定义与 `AnnotateView.vue` 的调用点）。
- 离线页 `purity_35 / purity_8seg / switch_16 / switch_7 / conflict_3` 导出文件的落点。
- `predict.py` 加载的 `intent/model.pt` 与台账登记的 `intent_cc/_e2e_ops2/best.pt` 是否同一权重。
- `tools/eval_full.py`、`tools/e21_cluster_metrics.py` 在 `tools/`、`webapp/backend/`、`src/` 三层内均未找到（记忆中引用的路径已失效或已移动）。
