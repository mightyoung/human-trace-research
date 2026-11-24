# 算法运行 · 标注 · 训练 · 派发 管理系统设计（v1）

> 版本：2026-11-02 ｜ 状态：设计稿，未实施 ｜ 依据：本机代码与数据实测 + 远程服务器只读核查
> 口径声明：本文所有数字均为**实测**（脚本位于 `docs/ops/probe_*.py`，可复跑），推断部分单独标注。

---

## 0. 结论先行

**现状不是"缺一个管理页面"，而是缺一条可追溯的链。**

四件事各自都在跑，但相互之间只有"文件名约定"这一层弱耦合：

| 环节 | 现在的载体 | 能否回答"这批结果从哪来" |
|---|---|---|
| 素材标注 | 4 个 json，2 台机器 | ❌ 只有判定值，没有依据口径 |
| 算法运行 | 手写 shell + nohup，日志散落 | ❌ 无 run_id、无参数快照 |
| 训练/实验 | docs 叙述 + 散装 json/png | ❌ 指标与运行无法互查 |
| 任务派发 | 文档里的角色说明 + 口头指派 | ❌ 无状态、无验收回填 |

**设计主张**：不新建"又一个后台"，而是在现有控制台上加一层**对象台账（ledger）**，把「数据集快照 → 标注批次 → 运行 → 产物 → 指标 → 结论」串成一条带版本的链。页面从"功能容器"改成"对象视图"，人和 agent 共用同一套任务入口。

**三个必须先定的决策**（详见 §7）：数据权威放哪、双写期多长、是否引入双人标注。

---

## 1. 现状实测

### 1.1 标注入口：7 个路由里 3 个是标注页，另有 6 个游离脚本

| 入口 | 路由 / 脚本 | 数据载体 | 规模 | 完成度（实测） |
|---|---|---|---|---|
| 素材标注·候选对 | `#/annotate` 页签① | `backend/data/labels.json` → `pair_decisions` | 75 对 | **33 / 75 = 44%**（CONF 8 / REJ 24 / MODERATE 1） |
| 素材标注·倒影源 | `#/annotate` 页签② | 同上 → `mirror_decisions` | 28 倒影 + 29 源 | **56 / 57 = 98%** |
| 素材标注·玻璃掩膜 | `#/annotate` 页签③ | 同上 → `glass_masks` | 6 机位草稿 | **0 / 6**（`masks.json` 仍标 draft） |
| 人脸库·人卡 | `#/faces` | `backend/data/faces_labels.json` → `persons` | 45 卡 | 45 条判定（含命名） |
| 人脸库·候选对 | `#/faces` | 同上 → `pairs` | 28 对 | **28 / 28 = 100%**（18 CONF / 10 REJ） |
| 真值标注·配对 | `#/truth` | `truth/labels.json` → `pair` | 241 对 | **40 / 241 = 16.6%**（另有 70 条是 S6-b 的判定，被写进了同一段） |
| 真值标注·事件 | `#/truth` | `truth/labels.json` → `event` | 107 事件 | **31 / 107 = 29.0%** |
| S6-b 待裁批 | `#/annotate` 静态载荷 `s6b_pending.json` | 名义写回 `labels.json`，实际落在 `truth/labels.json` | 75 项（16+3+11+5+40） | **70 / 75 = 93%**；余 5 条人脸冲突未判 |
| 游离脚本页 | `_truth_review_page.py`、`build_review_html.py`、`make_knn_review_page.py`、`build_kappa_review.py`、`reflection_audit_review.py`、`gil_split_entities_review.py` | 各自生成 HTML/CSV | 一次性 | 无状态、无回写 |

**待判总量 ≈ 42（候选对）+ 6（掩膜）+ 201（真值对）+ 77（真值事件）+ 5（S6-b）≈ 331 条**，分散在 3 个页面、4 种交互、2 套键名规则。

### 1.2 存储：4 个文件、3 套命名空间、2 个位置

```
远程 ~/human-count/webapp/backend/data/          ← 唯一写入点（服务在此起）
├── labels.json          2.4KB  pair_decisions 33 / mirror_decisions 56 / glass_masks 0
├── faces_labels.json    6.6KB  persons 45 / pairs 28 / updated 10-14 19:49
└── truth/
    ├── state.json      98.5KB  pairs 241 / events 107（只读骨架）
    ├── labels.json      8.8KB  pair 110（其中 70 条键带 PRT_/S6B_ 前缀）/ event 31
    └── labels.csv      25.6KB  导出快照

本地 D:\dev\human-count\webapp\backend\data\       ← 副本，且不完整
├── labels.json          与远程同（10-14 01:47）
├── faces_labels.json    ❌ 不存在
├── clusters_state.json  ❌ 不存在（远程也没有 → 聚类工作台操作从未落盘）
└── truth/labels.json    ❌ 不存在
```

三个命名空间混在同一个 `pair_decisions` 字典里：`318:14|320:3`、`PRT_316-xxx_a`、`S6B_R_xx`。消费端必须"前缀剥离 + 格式校验"，项目此前已因此修过 12 个工具。

### 1.3 运行：无台账，靠 shell 与目录约定

```bash
# tools/run_wave4.sh 的真实形态
PY=~/miniconda3/envs/rfdetr/bin/python
$PY infer_supervision.py --video crosscam/in/305.mp4 --device cuda:0 $BASE > crosscam/305.log 2>&1 &
wait
echo "ALL_WAVE4_DONE $(date '+%H:%M:%S')" > crosscam/wave4_done.txt
```

- 完成信号 = 一个 `*_done.txt` 文件；失败无法区分（没有 exit_code 汇总）。
- 参数只存在脚本里，改一次就丢一次历史；`BASE` 是单个字符串，无法结构化对比。
- 产物按目录约定归位，不记录"这次用了哪个模型 sha、哪份 manifest"。

### 1.4 产物：66 个分析子目录、5 个散落位置、无索引

| 位置 | 文件数 | 最后写入 |
|---|---|---|
| `output_crosscam/analysis/` | 2,168（66 个子目录） | **10-18** |
| `dist/` | 2,757（13 个交付包版本 + viz） | 10-26 |
| `deliverable/` | 1,714 | 10-26 |
| `tmp/` | 6,456 | 11-02 |
| `docs/` | 857 | 11-02 |
| `figs` 图片 | 87 张 | — |

`analysis/` 下 66 个目录里，`cradio_scale_v3/v4/v4_extra/v4c`、`dinov3_scale_v3/v4/v4_extra/v4b_raw` 是典型的"同名迭代留档"，只能靠目录名区分代际——这正是"重跑漂移"（portal_v2 曾漂 1 项）的土壤。

### 1.5 训练/实验：结论在文档里，证据在 json 里，两者不互认

- 训练脚本存在：`train_all.sh`、`train_rfdetr.py`、`train_yolo.py`、`train_selector_cluster.py`。
- 指标存在：`analysis/selector_1012/`、`analysis/intent/`、`analysis/reid_compare/`、`analysis/face/`（499 文件）。
- 但"哪次实验 → 哪个 run → 哪份数据 → 哪张图"只写在 md 叙述中。`docs/` 857 个文件里，实验报告与执行报告混放，无 front-matter、无 ID。
- 结果：`project-status-leader-plan` 里出现"主文/补充表计数相互冲突"，需要人工再审计。

### 1.6 派发：有角色定义，无执行状态

`docs/project-status-leader-plan-2026-10-26.md` 定义了 Leader / engineer / professior / researcher 四类角色与权限边界，但任务状态只存在于文档段落中：没有认领、没有进行中、没有验收回填，也没有"这条结论的证据在哪"的强制字段。

### 1.7 部署：控制台当前**不可访问**

| 检查 | 结果 |
|---|---|
| `ss -ltnp` 端口 8099 | **无监听** |
| 远程 `curl 127.0.0.1:8099/api/health` | 连接失败（000） |
| `webapp/deploy/serve.pid` / `serve.log` | 均不存在 |
| 磁盘上的 dist | 10-14 21:41 的静态包（5.6MB） |
| 本机访问 `:8099` | 403（网关统一拒绝，非服务响应） |
| GPU | 2×L20，显存占用 13.2GB / 1.4GB，利用率 0%（无任务在跑） |

**含义**：所有标注数据只存在于这台机器的磁盘上，服务一停就"看得见、碰不到"。这是把标注存储放在部署机本地文件的直接代价。

---

## 2. 设计目标与不做什么

### 2.1 四个目标（按优先级）

1. **可追溯**：任何一个数字，能一次点击回到「哪次 run / 哪份标注 / 哪个模型 sha / 哪个数据快照」。
2. **可推进**：待办可见、可领取、可验收；人和 agent 同一入口。
3. **可回归**：同一口径下的历史结果自动对比，漂移即告警（针对 portal_v2 那类问题）。
4. **可交接**：状态不依赖某个人的记忆或某个 md 的最新版。

### 2.2 明确不做

| 不做 | 理由 |
|---|---|
| K8s / 分布式调度 | 单机 2 GPU、任务粒度是"分钟到小时"，shell 足够；引入调度器只会增加故障面 |
| 通用 AutoML / 超参搜索 | 当前瓶颈是证据质量，不是搜索空间 |
| 实时视频流监控大屏 | 已有可视化页；实时原型尚未立项 |
| 重写前端框架 | 现有 Vue3 + DESIGN.md 体系可复用，只是加页与改数据源 |
| 迁移大文件 | 视频/图片/权重留在原地，台账只登记 `uri + sha1 + bytes` |

---

## 3. 总体架构

```
┌──────────────────────────────────────────────────────────────┐
│  视图层（现有控制台，+3 页 / 重构 2 页）                        │
│  总览 · 标注 · 运行 · 实验 · 结果 · 派发 · 可视化 · ReID · 人脸 │
├──────────────────────────────────────────────────────────────┤
│  服务层（FastAPI，现有 main.py 扩展）                          │
│  /api/ops/*  统一对象读写 · 队列生成 · 验收校验 · 导出          │
├──────────────────────────────────────────────────────────────┤
│  台账层（SQLite: ops/ops.db，WAL，append-only）                │
│  asset · task · item · label · run · artifact · experiment     │
│  model · job · audit                                           │
├──────────────────────────────────────────────────────────────┤
│  执行层（现有脚本，加薄包装）                                   │
│  ops-run 包装器 → 抓 params/env/git/host/gpu/日志/产物/exit    │
│  现有 tools/*.py 不改，只在入口与出口加钩子                      │
├──────────────────────────────────────────────────────────────┤
│  资产层（原地不动）                                             │
│  media-resource/ · output_crosscam/ · models/ · dist/          │
└──────────────────────────────────────────────────────────────┘
```

**关键取舍**：台账**不存大文件**，只存指针与校验和。这样迁移成本≈0，回退成本≈0（删掉 ops.db 即回到现状）。

**权威单一**：`ops.db` 是唯一写入点；现有 4 个 json 降级为"导出投影"（只读，供老页面与离线包使用）。迁移期允许双写，但 DB 为准，每日自动 diff 校验。

---

## 4. 数据模型

### 4.1 实体关系

```
dataset ──┬─< asset
          ├─< annotation_task ──< annotation_item ──< label ──< adjudication
          └─< run ──< artifact
                 │
model ───────────┘
experiment ──< experiment_run >── run
job ──(refs)──> annotation_task | run | experiment
audit ──(refs)──> 任意对象
```

### 4.2 表定义（核心字段）

```sql
-- 数据快照：一次"确定输入"的指纹
dataset(id PK, name, kind, root_uri, file_count, total_bytes,
        dialect,              -- clean2 / DET_VIEW=clean2 / raw …（口径随行）
        manifest_uri,         -- 文件清单 + sha1
        snapshot_hash,        -- manifest 的 sha1，run 引用它
        created_at, note)

-- 素材：可被标注/渲染的最小单元
asset(id PK, dataset_id FK, kind,   -- image|video|clip|track_crop|frame
      uri, cam, tid, t0, t1, w, h, sha1, bytes, meta_json)

-- 标注任务：一批同质判定
annotation_task(id PK, name, kind,  -- pair|mirror|mask|face_person|face_pair|
                                    -- truth_pair|truth_event|cluster_review
      dataset_id FK, spec_json,     -- 队列生成规则（可复现：seed + 规则版本）
      dialect, mode,                -- single|double|adjudicate
      assignee, budget, due_at, status, created_by, created_at)

-- 标注条目：待判的一条
annotation_item(id PK, task_id FK, asset_refs_json, payload_json,
      stratum,                      -- topk|audit|conflict|boundary|golden
      priority, status,             -- pending|assigned|labeled|reviewed|adjudicated|skipped
      gold_value)                   -- 黄金题（质量抽检）答案，仅服务端可见

-- 判定：可追加，不可覆盖
label(id PK, item_id FK, annotator, value, confidence,
      seconds, ui_version, created_at, supersedes)   -- 改判=新行+supersedes

adjudication(id PK, item_id FK, final_value, by, reason, at)

-- 运行：一次算法执行
run(id PK, name, kind,              -- infer|encode|train|eval|build|export
    entrypoint, params_json, env_json,
    dataset_id FK, model_id FK,     -- 输入与模型（可空）
    host, gpu, git_rev,
    status,                         -- queued|running|succeeded|failed|killed|stale
    started_at, ended_at, exit_code, log_uri, metrics_json, note)

-- 产物：run 的输出，带摘要
artifact(id PK, run_id FK, kind,    -- jsonl|mp4|png|csv|npz|db|pt|onnx
         uri, bytes, sha1, summary_json)

-- 实验：一个科学问题 / 一组对比
experiment(id PK, name, hypothesis, design_json, dataset_id FK,
           status, owner, conclusion, doc_uri, created_at)

experiment_run(experiment_id, run_id, role)   -- baseline|variant|ablation

-- 模型注册
model(id PK, name, version, backbone, weights_uri, sha1,
      license, trained_by_run FK, eval_json, deployable, created_at)

-- 派发：人/agent 的执行单元
job(id PK, kind,                    -- label|run|review|write|fix
    title, goal, inputs_json,
    acceptance_json,                -- 可勾选 + 可自动校验的验收项
    assignee_kind,                  -- human|agent
    assignee, status,               -- backlog|assigned|doing|review|done|blocked
    priority, budget,               -- 时长 / GPU 卡数
    authorization,                  -- 本次允许做什么（不继承跨任务授权）
    created_at, closed_at, evidence_json)

-- 审计
audit(id PK, actor, action, object_type, object_id,
      before_json, after_json, at)
```

### 4.3 三条硬约束

1. **`label` 必须带 `dialect` + `ui_version` + `dataset.snapshot_hash`**（经由 item→task→dataset 链）。理由：项目已发生 SCRFD 坐标系变更导致"历史产物不得混用"，以及 clean2/splits_v5 分叉。口径不进库，就一定会漂。
2. **`run` 结束必须写 `artifact` + `metrics_json`**，否则状态停留在 `running`，由看门狗标 `failed`。理由：现在靠 `*_done.txt`，失败与未启动不可区分。
3. **`job` 关闭必须挂 `evidence`**（文件路径 / run_id / 指标），否则拒绝 `done`。理由：直接落实"声明必须有证据"。

---

## 5. 五个模块的设计

### 5.1 素材标注：从"页面"到"任务 + 队列 + 质量"

**核心变化**：现在每个页签自带一套候选集（`pairs.json` / `mirror.json` / `s6b_pending.json`）。改为**队列生成器**统一产出 item，页面只是 item 的渲染器。

```
队列生成（可复现）            判定（人/agent）           质量（自动）
data snapshot                 item → label             双标率 / kappa
   → 规则(seed+版本)    →      （写 label，不改 item）→  黄金题命中率
   → item[] 入库               仲裁 → adjudication      单条耗时分布
                                                        → 导出 labels_export_<task>_v<n>.jsonl
```

**要点**

- **任务类型**与现有 8 种对齐（pair/mirror/mask/face_person/face_pair/truth_pair/truth_event/cluster_review），一种类型一个渲染器，不再为每批数据新建页面。
- **分层（stratum）显式化**：`topk` / `audit` / `conflict` / `boundary` / `golden`。项目已有"选择偏差"教训（标注子集 AUC=1.0 高估能力），分层必须进库并在统计时按层加权，否则同样的错误会重演。
- **黄金题**：从已判条目中抽 5% 混入队列，作为标注员质量基准；命中率 <85% 时暂停派发该标注员的新任务。
- **改判留痕**：不覆盖，追加新 `label` 行 + `supersedes`。现在 `labels.json` 是直接 `update()`，改判历史全丢。
- **导出即契约**：标注完成后生成带 `snapshot_hash` 的 jsonl，下游 run **必须**引用文件而非"读最新 labels.json"。这是"标注不回写流水线"的正解——不是自动写回，而是**显式导出 + 显式引用**。

### 5.2 算法运行：run 作为第一等公民

**包装器**（新增 `tools/ops_run.py`，不改现有脚本）：

```bash
ops run --name wave4 --kind infer \
        --dataset ds_6cam_synced10 --model m_dino_base \
        --out output_crosscam/analysis/wave4 \
        -- bash tools/run_wave4.sh
```

自动采集：`params_json`（CLI + 环境变量白名单）、`env_json`（python/torch/cuda 版本）、`git_rev`、`host`、`gpu`、`started_at/ended_at`、`exit_code`、`log_uri`、产物扫盘 + sha1。

**状态机**

```
queued ──> running ──┬─> succeeded ──┬─> stale（输入/模型/代码变了但没重跑）
                     ├─> failed      └─> superseded（被新 run 取代）
                     └─> killed
```

`stale` 是这套设计里最有价值的增量：SCRFD 三文件修复后，所有历史检测产物坐标不可比——现在只能靠人记，之后由系统在 run 详情页直接标红。

**运行详情页**要能回答六个问题：用了什么参数 / 吃了哪份数据 / 哪个模型 sha / 产出哪些文件 / 指标多少 / 和上次比变了什么。

### 5.3 训练与实验：实验层与执行层分离

- `experiment` = 科学问题（如 H1 门控、H2 门户、H3 预测；或 exp1–exp8 这类对照）。
- `run` = 一次执行。
- `experiment_run` 显式标注 `baseline / variant / ablation`，对比图只允许在同 `dataset + dialect` 内生成。

**指标 schema 统一**（前端只认 schema，不认实验）：

```json
{"schema":"metrics/v1","run_id":"...","split":"val",
 "items":[{"name":"auc","value":0.922,"n":69,"ci":[0.85,0.98],"note":""}]}
```

强制 `n` 与 `ci`：项目曾出现"标注子集 AUC 1.0"与"69 对 AUC 0.922"，没有 n 与 CI 就无法判断可信度。

**模型注册**：`model` 表登记 sha1 + license + 训练 run + 评估 + `deployable` 开关。现有 `models/` 下 731 个文件、26 个 pt、多套 onnx，靠目录名区分版本，注册表能直接消掉这部分认知成本。

### 5.4 运行结果：从"看目录"到"看对比"

**结果集（result set）** = 一次 run 的 artifact + 指标卡 + 与 baseline 的 diff。

三个必备视图：

1. **指标回归线**：同一 benchmark（如 `frozen_eval_v1` 69 对 / `C0` 186 边）随 run 的变化曲线，超阈值告警。
2. **口径对比表**：clean2 vs 原口径、A/B/H 三档身份臂、v2/v3 真值——同屏并列，禁止跨口径混算。
3. **漂移检测**：同一 run 重跑，产物 sha 与指标不一致即标 `reproducibility: drift`（portal_v2 曾漂 1 项，就是这类问题）。

### 5.5 任务派发：任务卡，不是工单

**为什么不做通用工单**：本项目的"任务"不是"处理一个请求"，而是"产出一个可验收的结论"。所以卡片字段围绕**证据**设计：

| 字段 | 作用 | 例 |
|---|---|---|
| 目标 | 一句话，可判定 | "确认 343 左侧中段是否存在第 2 个出口门" |
| 输入 | 文件/run/dataset 引用 | `run_20261016_scrfd`、`ds_6cam_synced10` |
| 验收 | 可勾选 + 可自动校验 | "补门候选 ≥1 且帧号可复现；`verify_*.cjs` 通过" |
| 预算 | 时长 / GPU / 人工条数 | "≤30min，1×L20，人工 ≤40 条" |
| 授权 | 本次允许改什么 | "只读分析，不改 selector.npz" |
| 证据 | 关闭时必填 | 产物路径 + 指标 + 截图 |

**状态流**：`backlog → assigned → doing → review → done`，另有 `blocked`。`review` 与 `done` 之间强制证据校验。

**自动派生**：
- 某 `annotation_task` 剩余 > 0 且无人认领 → 自动生成 `label` job；
- 某 `experiment` 缺 baseline → 自动生成 `run` job；
- 某 `run` 标 `stale` → 自动生成 `fix/re-run` job。

这样"派发"不是额外负担，而是台账状态的投影。

---

## 6. 信息架构（页面）

现有 7 页保留 5 个，新增 3 个，重构 2 个：

| 页面 | 路由 | 变化 | 主要内容 |
|---|---|---|---|
| **总览** | `#/ops` | 🆕 | 今日待判 / 运行健康 / 实验进度 / 派发队列 四张卡 + 风险清单（stale run、待仲裁、失败 job） |
| **标注** | `#/label` | ♻️ 重构 `#/annotate` | 左：任务列表（类型/进度/认领人）；中：条目工作台（复用现有三页签渲染器）；右：质量面板（双标率/kappa/耗时） |
| **运行** | `#/runs` | 🆕 | 列表（状态色 + 耗时 + GPU + 发起人）→ 详情（参数树 / 产物树 / 指标 / 日志尾 / 血缘 / 对比） |
| **实验** | `#/experiments` | ♻️ 重构 `#/cluster` | 实验卡（假设 / 状态 / 结论）→ 曲线与对比 → 关联 run 列表 → 模型注册 |
| **结果** | `#/results` | 🆕 | benchmark 回归线、口径对比表、漂移列表 |
| **派发** | `#/jobs` | 🆕 | 看板（backlog/doing/review/done）+ 卡片详情（验收清单 + 证据区） |
| 可视化 | `#/` | 保留 | 不变 |
| ReID / 语义 | `#/reid` | 保留 | 增加"该轨迹的标注历史"抽屉 |
| 人脸库 | `#/faces` | 保留 | 判定写入新台账（适配层），UI 不变 |
| 真值标注 | `#/truth` | 保留 | 同上 |
| 算法配置 | `#/config` | 保留 | 参数版本与 run 关联 |

**导航调整**：一级导航从「可视化 / 素材标注 / ReID / 聚类 / 真值 / 人脸 / 配置」改为按**对象**分组：

```
监控：总览 · 可视化
数据：标注 · 数据集
模型：实验 · 运行 · 结果
协作：派发
```

理由：现在的分组是"页面史"（按开发顺序），不是"任务史"（按使用动线）。标注页被拆成 3 个页签 + 2 个独立页，就是因为没有"标注任务"这个统一对象。

---

## 7. 三个待决策（一次一个）

### 决策 1：数据权威放哪？

| 方案 | 做法 | 优点 | 代价 |
|---|---|---|---|
| **A（推荐）共享盘 + 端点服务** | `ops.db` 与标注数据放 `/share/data4/yhh/data/human-count/ops/`，控制台可从任一端启动 | 服务停了数据仍在；本机/远程都能读写；备份简单 | 需保证 NFS 挂载稳定；SQLite over NFS 有锁风险 → 用 WAL + 单写进程 |
| B 维持部署机本地 | 现状，只补备份 | 零改动 | 服务停 = 数据不可达（当前正在发生） |
| C 独立数据库服务 | Postgres/MinIO | 多用户强 | 单机项目过重，运维成本高 |

**推荐 A 的轻量版**：`ops.db` 放本地 `D:\dev\human-count\ops\ops.db`（本机为主），远程只做算力与临时预览；标注数据以本机为准、单向同步到远程供页面读。**理由**：本机已有全部产物（2168 文件），而远程只有 webapp 205MB，本机才是事实上的主库。

### 决策 2：双写期多长？

推荐 **2 周**：适配层把 4 个 json 的读写映射到 `label` 表，双向同步；每日 `ops verify-projection` 比对，差异即告警。2 周后 json 转只读导出。

### 决策 3：是否引入双人标注？

现状全部单标（无 kappa 数据）。真值 241 对里 110 已判，若要报一致性，需对 10–15% 抽双标。这会影响进度，但对论文"双人一致性"章节是刚需（已有 v2/v3 两版口径的教训）。

---

## 8. 分期落地

| 期 | 范围 | 产出 | 验收标准 |
|---|---|---|---|
| **P0**（3–5 天） | ops.db + `ops_run.py` 包装器 + 适配层（4 json ↔ label 表）+ 总览页 + 运行列表/详情 | 能回答"有哪些 run、产物在哪、口径是什么、哪些已 stale" | 用 `run_wave4.sh` 实跑一次，自动生成 run 记录 + 产物 sha1 + 日志链接；`ops verify` 全绿 |
| **P1**（1 周） | 标注页重构（任务/队列/质量/仲裁）+ 真值/人脸/素材统一到 item 模型 + 导出与 run 绑定 | 325 条待判集中入口，进度与质量可查 | 三类标注各判 5 条，落 `label` 表并可导出 jsonl；双写 diff = 0 |
| **P2**（1 周） | 实验看板 + 模型注册 + 结果回归 + 派发看板 | 指标可溯源、漂移可告警、任务可验收 | 任取一个历史数字（如 AUC 0.922）能点到 run → 数据 → 模型 sha |
| **P3**（按需） | 权限与多用户、审计导出、交付包版本绑定 | 对外协作可用 | — |

**先做 P0 的理由**：它不改任何算法、不动 UI 框架，只加一个库和一层包装；即使设计后续调整，P0 的 run 台账也不会白做。

---

## 9. 风险与回退

| 风险 | 表现 | 缓解 |
|---|---|---|
| 双写不一致 | json 与 DB 判定分叉 | 每日 `verify-projection` diff；DB 为准；不一致阻断写入 |
| SQLite 锁 | 并发写 `database is locked` | WAL + 单写进程（现有服务本身就是单进程）；写重试用指数退避 |
| NFS 上的 SQLite | 锁失效导致损坏 | 决策 1A 若采用共享盘，改为"单端写 + 文件级同步"，不直接放 NFS |
| 台账成为负担 | 每次跑都要填参数 | 包装器自动采集，人不填；只有 `note` 选填 |
| 页面增多 | 用户找不到入口 | 一级导航按对象分组（§6），总览页做"今天要做什么"的唯一入口 |

**回退**：删除 `ops/ops.db` 与新增页面路由，恢复 4 个 json 直读，系统回到现状。所有既有产物不动。

---

## 10. 附：证据清单（可复跑）

| 结论 | 证据 |
|---|---|
| 标注规模与完成度 | `docs/ops/probe_landscape.py`；`webapp/backend/data/*.json`；远程 `truth/labels.json` |
| 服务未运行 | 远程 `ss -ltnp` 无 8099；`curl 127.0.0.1:8099/api/health` = 000；无 `serve.pid` |
| 运行无台账 | `tools/run_wave4.sh`（`*_done.txt` 为唯一完成信号） |
| 产物散落 | `find` 统计：output_crosscam 2168 / dist 2757 / deliverable 1714 / tmp 6456 |
| 训练指标散装 | `webapp/dataprep/build_algo_data.py` 输出清单；`analysis/{selector_1012,intent,reid_compare,face}` |
| 命名空间混用 | `labels.json` 中 `318:14\|320:3` 与 `PRT_`/`S6B_` 三类 key 并存；实测 `PRT_` 0 条 |
| 标注不回写流水线 | `webapp/README.md` §六「诚实边界」原文 |

---

## 11. P0 实施记录（2026-11-02 同批完成）

决策已定：**本机为主库，远程只做算力与预览**。P0 在决策后当日实施并通过验收。

### 11.1 产出

| 文件 | 作用 |
|---|---|
| `ops/schema.sql` | 台账 schema v1（13 张表 + 1 视图，WAL） |
| `ops/ops.db` | **权威库**（本机）。备份在 `ops/db_backup/` |
| `ops/imports/` | 从远程拉回的原始标注（来源证据，只读留存） |
| `ops/reports/` | run 产物示例（校验报告、总览快照） |
| `ops/logs/` | 每次 run 的完整日志 |
| `tools/ops_db.py` | 访问层 + CLI（`init` / `overview` / `sql`） |
| `tools/ops_run.py` | 运行包装器（自动采集参数/环境/GPU/产物/退出码） |
| `tools/ops_ingest.py` | 标注适配层（8 任务映射、幂等摄入、投影校验、单条双写） |
| `tools/ops_selftest.py` | 自测（9 用例，全绿） |
| `tools/ops_verify_cmd.py` / `ops_overview_cmd.py` | 两个可被包装的真实任务（演示 + 日常可用） |
| `webapp/backend/ops_api.py` | `/api/ops/*`（可选模块，远程无库时 503） |
| `webapp/frontend/src/views/OpsView.vue` | 总览页 `#/ops` |
| `webapp/frontend/src/views/RunsView.vue` | 运行页 `#/runs` |
| `webapp/frontend/src/views/JobsView.vue` | 派发看板 `#/jobs`（四列 + 状态流转 + 证据校验） |

### 11.2 实施中纠正的两个事实错误

1. **S6-b 不是零判定**。75 项里已判 70 项，但判定写在 `truth/labels.json` 的 `pair` 段（键带 `PRT_`/`S6B_` 前缀），不在名义上的 `labels.json`。v1 设计稿据此写的“0/70”是错的，已修正。
2. **真值配对是 40/241，不是 110/241**。110 条里有 70 条属于 S6-b 批次，混在同一段。适配层按前缀路由后，两个任务的口径才对上。

> 这两条本身就是“为什么需要台账”的最好证据：同一批数据，按文件统计会得出相反结论。

### 11.3 验收

| 项 | 结果 |
|---|---|
| `tools/ops_selftest.py` | **9 / 9 通过**（建表幂等 / run 生命周期 / label append-only / 投影双写 + 隔离 / **反向镜像** / ingest 幂等 / verify 零差异 / **派发状态机** / overview 结构） |
| 标注摄入 | 8 任务 / 633 条目 / 302 判定，与源 json **零差异** |
| 真实 run | 2 条（`verify-projection`、`overview-snapshot`），自动登记参数、GPU、产物 sha1、日志 |
| 前端构建 | `vite build` 通过，dist 5.7MB / 332 文件（含原有 public 资产） |
| 页面 | `#/ops`、`#/runs` 本机 8013 端口实测渲染正常，控制台无报错 |

### 11.4 实施中发现并修复的缺陷

| 缺陷 | 表现 | 修复 |
|---|---|---|
| 幂等索引不含 value | 同一秒内改判触发 `UNIQUE constraint failed` | 索引改为 `(item_id, annotator, value, created_at)`，`init_db` 自动 DROP 旧索引 |
| 投影表在导入期绑定路径 | 测试注入临时文件无效，**写到了真实 labels.json** | `PROJECTION` 改存常量名，`project_item` 运行时 `globals()` 解析；自测加隔离断言 |
| 产物扫描按整目录 | 复用 `--out` 时把上一次产物重复登记 | `scan_artifacts` 增加 `since=t_start`，只收本次运行期间修改的文件 |
| 清除语义不落痕 | 清空判定后 `latest_label` 仍返回旧值 | 清除写入一条 `value=NULL` 的判定行，保持 append-only |

### 11.5 日常用法

```bash
# 1) 初始化（首次或换机）
python tools/ops_db.py init
python tools/ops_ingest.py            # 把现有 json 摄入台账（幂等，可反复跑）
python tools/ops_ingest.py --verify   # 比对台账与 json，应零差异

# 2) 把任何一条命令变成可追溯的 run
python tools/ops_run.py --name wave4 --kind infer \
    --dataset ds_6cam_synced10 --out output_crosscam/analysis/wave4 \
    -- bash tools/run_wave4.sh

# 3) 起控制台看 #/ops 与 #/runs
cd webapp/backend && python -m uvicorn main:app --port 8000
```

**约定**：每次 run 用独立 `--out` 目录；若复用，包装器按 mtime 只登记本次产物，但同名文件会被覆盖——独立目录更稳。

### 11.6 P0 未做（留给 P1/P2）

- 标注页**判读交互**仍走原三页签（台账只做了读写与进度，未换 UI）；
- 老页面的写入已通过 `_ops_mirror` 补进台账（双向补齐），但**投影方向仍是「台账 → json」单条覆盖**，未做全量重建；
- ~~`job` 派发看板~~ 已补做（本轮）：四列看板、状态机、证据校验、按待判量派生；
- `stale` 状态只建了字段与接口，**判定规则**（输入指纹变化）留到 P2；
- 远程部署未更新（远程仍跑 10-14 的 dist；按“本机为主库”，远程只做预览，需单独发布且不含台账）。

---

## 12. 训练过程接入（2026-11-09）

### 12.1 为什么先做训练而不是标注 UI

用户最初要管五件事：算法运行结果、素材标注、训练过程、运行过程、任务派发。P0 之后四项有落点，**训练过程是唯一空白**。而且它不是"以后再说"的空壳——磁盘上已经躺着 **34 次真实训练**（intent 意图分类三批），只存在于日志文本里，没有任何索引。

### 12.2 数据从哪来（可核对）

| 项 | 位置 | 数量 |
|---|---|---|
| 终值指标 | `output_crosscam/analysis/intent/**/metrics_full.json` | 36 个目录（2 个 smoke 用例排除） |
| 超参与曲线 | 同级 `<变体目录>.log`，行格式 `[ep NN] loss=… val_acc=… val_f1=… (Ns)` | **34 个有完整曲线** |
| 实验分组 | 变体目录的父目录（`intent/cc/out`、`intent/cc/out2`、`intent/v2/out_v2`） | 3 组 |

合成结果：**34 个 run（kind=train）/ 3 个 experiment / 2235 个序列点**。

### 12.3 落库方式

- 新增 `run_metric(run_id, step, name, value)`，主键 `(run_id, step, name)`，写用 `ON CONFLICT DO UPDATE`，重复回填不膨胀。
- `run` 复用既有表：`kind='train'`，`params_json` 存 task/id_mode/n_train/n_val/epochs/best_ep/device/torch，`metrics_json` 存 acc/macro_f1/per_cam。
- `started_at` = 指标文件 mtime − 日志中各 epoch 秒数之和；`ended_at` = mtime。**是推算值，不是采集值**，所以 `host='gpu-server'`、`git='—'` 都如实留空而不是编造。
- 幂等：run id 由相对路径 sha1 决定（`run_tr_<变体>_<hash6>`），重复执行只覆盖。

### 12.4 接口与页面

| 层 | 内容 |
|---|---|
| CLI | `tools/ops_ingest_train.py`（`--dry-run` / `--root`），幂等 |
| API | `GET /api/ops/train` → 实验组 + 变体 + 每组曲线；`GET /api/ops/runs/{id}` 增 `series` 与 `experiment` |
| 组件 | `components/TrainCurve.vue`（内联 SVG，纵轴固定 0~1，含 best-epoch 竖线与 loss 细带） |
| 页面 | `#/train` 训练过程：批次 / 变体表（macro_f1 排序 + 迷你曲线）/ 曲线 / 训练配置 / 终值 per-cam |
| 复用 | `#/runs` 选中训练 run 时同组件渲染曲线，可跳「在训练页打开」；列表新增类型筛选（全部/训练/评测） |

### 12.5 验收

| 项 | 结果 |
|---|---|
| `tools/ops_selftest.py` | **11 / 11**（新增：run_metric 覆盖语义、训练回填解析+幂等，均在临时库与临时目录） |
| 回填幂等 | 连跑两次：34 run / 2235 点不变 |
| 实验汇总 | cc·out 最佳 `global_none` 0.472 / cc·out2 `global_none_s123` 0.447 / v2 `base_s43` 0.486（macro_f1，n=1294） |
| 页面 | `#/train`、`#/runs?id=…`（训练 run）本机 8018 实测渲染；控制台无报错 |

### 12.6 已知局限（写清楚，避免误读）

1. **历史回填没有 commit 与配置快照**，只有日志里的超参和指标——能对齐数字，**不能证明可复现**。总览页已把这条作为风险项显示。新训练必须走 `ops run` 包装器采集参数、git、环境。
2. 曲线来自 stdout 文本解析，不是训练脚本主动上报；日志被截断或改动就丢点（36 个目录里 2 个 smoke 用例无曲线，已跳过）。
3. `started_at` 是倒推值，时长（如 26s）只反映 epoch 累计秒数，不含数据加载。
4. 训练页只读，**不支持发起训练**——训练属于算力线，按"本机为主库、远程只做算力"的边界，发起入口留到 P2 与 `job` 派发打通时再做。

---

## 13. 让新训练自带证据（2026-11-09 同日）

### 13.1 问题

§12 回填的是历史：只有日志里的超参和指标，**没有代码身份、没有环境快照**，所以只能对齐数字，不能证明"这一版是怎么跑出来的"。而且训练脚本一直只往 stdout 打 `[ep NN]`，台账读不到——除非事后解析日志。

### 13.2 三处改动

**① 上报口 `tools/ops_metric.py`（三行接入）**

```python
from ops_metric import Reporter
rep = Reporter()                                   # 读 OPS_RUN_ID / OPS_DB
rep.metric(ep, loss=..., val_acc=..., val_f1=...)  # 每个 epoch
rep.close()
```

设计取向是**绝不阻断训练**：
- 没有 `OPS_RUN_ID`（直接 `python tools/intent_model.py`）→ 完全 no-op
- 台账文件不存在 / 表缺失 / 写入报错 → 打一次告警后转 no-op，不抛异常
- 写 `run_metric(run_id, step, name, value)`，同 step 覆盖，幂等

**② 包装器注入 + 代码指纹**

`ops_run.py` 现在给子进程注入 `OPS_RUN_ID / OPS_DB / OPS_ROOT`，并在 `env_json.code` 里记录**执行脚本的 sha1**。本仓没有 `.git`，这是唯一能回答"跑的是哪份代码"的字段。

**③ 指标归一化 `ops_db.normalize_metrics`**

每个训练脚本的 `metrics.json` 长得都不一样（`{test:{...}}` / `{overall:{...}}` / 直接 `{acc}`）。归一化在**写入时**做（`find_metrics` 调用），同时**读取时**再兜一次（`/api/ops/train`、`/runs/{id}`），这样历史 run 不必重写。原始结构完整保留在 `metrics["_raw"]`，不丢信息。

已接入的脚本：`intent_model.py`、`intent_model_cc.py`、`intent_model_social.py`、`selector_classifier.py`（stdout 格式未改，§12 的日志解析仍然兼容）。

### 13.3 用法（新训练的标准姿势）

```bash
python tools/ops_run.py --name intent-v3 --kind train --dataset ds_intent_v3 \
    --out output_crosscam/analysis/intent/v3/global \
    --tag lr=1e-3 --tag d_model=256 -- \
    python tools/intent_model.py --train ... --val ... --test ... \
        --out-dir output_crosscam/analysis/intent/v3/global --epochs 40
```

跑完自动得到：参数快照、环境（python/torch/cuda）、**代码 sha1**、GPU、退出码、产物 sha1、**逐 epoch 曲线**。

### 13.4 验收

| 项 | 结果 |
|---|---|
| `tools/ops_selftest.py` | **13 / 13**（新增：上报口三种路径、代码指纹） |
| 端到端实跑 | 3 次（合成数据 600/150/150，CPU，3–4 epoch），台账记录 run / 12 点曲线 / 3 个产物 / 代码 sha1 / 环境 |
| 确定性 | 同参数同代码跑两次：**12 / 12 个点 maxdiff = 0.0**（seed 生效，且台账能证明这一点） |
| 训练页 | 「未归入批次」兜底组显示 epoch / macro_f1 / acc / best ep / sparkline；训练配置与终值指标不再缺项 |

### 13.5 局限（明确区分已验证与未验证）

1. **端到端用的是合成数据**（`docs/ops/synth_intent_ds.py`），只证明链路通，**不代表真实数据上的训练质量**；真实数据仍在算力线。
2. 远端尚未同步 `ops_metric.py`，也未在 GPU 机验证过；按"本机主库"边界，远端训练需要把 `tools/ops_metric.py` 一起带过去（缺失时脚本自动 no-op，不会崩）。
3. `selector_classifier.py` 每 10 epoch 才打一行，所以它的曲线只有稀疏点，不是每 epoch。
4. 训练页仍**只读**，不能从这里发起训练。

---

## 14. 并行协作期的三个修正（2026-11-09 午后）

背景：算力线同时在跑 e21 切片（ffmpeg 合每日片段）与后续训练，控制台也在被使用。这段时间暴露的都是"多人/多任务共用一套数据"才会出现的问题。

### 14.1 镜像路由缺前缀判断（数据错桶，已修）

**现象**：`ops_ingest --verify` 报 `t_truth_pair` 的 `n_missing_in_json = 12`。

**根因**：`ops_ingest.mirror_truth()`（控制台写入的双写镜像）把所有 `pair` 键都塞进 `t_truth_pair`，**没做 S6-b 前缀判断**；而摄入层 `is_s6b()` 是把 `PRT_`/`S6B_` 路由到 `t_s6b_pending` 的。于是同一条 S6-b 判定在库里存在两份，分属两个任务，按任务统计必然对不上。

**修复与迁移**：
- `mirror_truth` 改为 `task = "t_s6b_pending" if is_s6b(pid) else "t_truth_pair"`；
- 已存在的 12 条错桶条目迁到 `t_s6b_pending`（label 按 `item_id` 改写，遇唯一索引冲突则判为重复删除），迁移前备份 `ops/db_backup/ops.db.bak_pre_taskroute`；
- 迁移后 `t_s6b_pending` **75 / 75**，`t_truth_pair` **226 / 241**，六任务 verify 全部零差异。

> 这条和 §11.2 是同一类错误的两次发作：**同一批判定分散在两个命名空间，按任一维度统计都会错**。前缀判断必须写在"所有写入路径"上，只写在摄入层不够。

### 14.2 out 目录并发占用（新增守卫）

`ops_run` 启动前检查 `--out`：
- 被**进行中**的 run 占用 → 拒绝启动，返回码 **3**，打印占用者 run_id / name / 起始时间；
- 被**已结束**的 run 用过 → 打印提示并继续（产物按 mtime 只登记本次改动）。

理由：产物按目录登记，两个任务写同一目录时文件名会互相覆盖，事后无法判断某个文件属于谁。

### 14.3 外部结果导入（切片/训练都不用改流程）

`ops_ingest_train.py` 现在支持：
- 认 **`metrics.json`**（`intent_model.py` 直接产出）与 `metrics_full.json` 两种；前者必须带 epoch 曲线才认，避免把评测报告当训练；
- `--root <单目录>` 直接导入一次 run；
- `--experiment <名字>` 强制归档；不给且目录不在 `output_crosscam/analysis` 下时**不编造实验**，落到训练页「未归入批次」；
- 同一 `out_dir` 已被别的 run 记录时跳过，不重复入库。

配套交接单：`docs/ops/leader-run-handoff.md`（切片登记 dataset、训练两条接入路线、回传三件套）。

### 14.4 当前标注口径（2026-11-09 10:40 快照）

| 任务 | 完成 |
|---|---|
| 真值·配对 | 226 / 241 |
| 真值·事件 | 107 / 107 |
| S6-b 待裁 | 75 / 75 |
| 倒影 / 人脸卡 / 人脸对 | 56/56 · 45/45 · 28/28 |
| 跨机候选对 | 44 / 75 |
| 玻璃掩膜 | 0 / 6 |
| **合计待判** | **52** |

演练：`tools/ops_selftest.py` **15 / 15**。

---

## 15. 路线 B 落地：远程训练自带证据（2026-11-09 午间）

### 15.1 目标与约束

用户选定路线 B：训练在算力线跑，但**参数、环境、代码身份、逐 epoch 曲线**要自动进台账。约束仍是「本机为主库」——不能让远程机器直接写本机 `ops.db`。

### 15.2 为什么不做「远程直连本机库」

| 方案 | 问题 |
|---|---|
| 远程 NFS/SMB 挂载本机 `ops.db` | SQLite 跨网络文件锁不可靠，WAL 模式在 SMB 上会损坏 |
| 远程 HTTP 写本机 API | 训练机到本机无稳定反向通道；断了会拖慢或卡住训练 |
| **远程本地库 + bundle 回传**（选定） | 各写各的，跑完搬一条记录；断网不影响训练 |

### 15.3 远程部署了什么

`~/human-count/` 下：

| 文件 | 作用 | 与本机是否同源 |
|---|---|---|
| `ops/schema.sql` | 表结构 | sha256 一致 |
| `tools/ops_db.py` | 访问层 | 一致 |
| `tools/ops_run.py` | 运行包装器 | 一致 |
| `tools/ops_metric.py` | 指标上报口 | 一致 |
| `tools/ops_bundle.py` | run 包导出/导入 | 一致 |
| `intent_cc/intent_model_cc.py` | 已加 `Reporter` | 补丁版一致（原件备份 `*.bak_pre_ops_metric`） |
| `intent_v2/intent_model_social.py` | 同上 | 同上 |

上传前先核对：远程两个训练脚本**原件的 sha256 与本机 `.bak_pre_ops_metric` 逐字节相同**，才敢覆盖。覆盖前 `cp -n` 备份，`-n` 保证不覆盖已有备份。

`ops_run.py` 新增注入 `PYTHONPATH=<root>/tools`：远程训练脚本在 `intent_cc/`、`intent_v2/` 下，不在 `tools/` 里，否则 `from ops_metric import Reporter` 会失败并静默降级——那样「接了上报口」等于没接。

### 15.4 bundle：一条 run 的闭包

`tools/ops_bundle.py` 导出 run + run_metric + artifact + experiment/experiment_run + 引用的 dataset/model，导入按主键 upsert。

三个设计点：
1. **artifact 按 run 快照替换**（先 `DELETE WHERE run_id`），否则重复导入产物翻倍；
2. **`--rewrite-out` / `--rewrite-uri`**：远程 `intent_cc/out_x` 与本机 `output_crosscam/analysis/...` 路径不同，导入时可改写，但不改任何文件；
3. **导入写 `audit`**：记来源 host / 曲线点数 / 产物数，事后能回答「这条 run 从哪台机器来的」。

### 15.5 端到端实测（真实数据，非合成）

远程 `intent_cc` 的 1012 真值 npz，3 epoch、`--device cpu`（不占 GPU）：

| 项 | 结果 |
|---|---|
| run | `run_20261109_115354_36cfa3` · succeeded · 7.5s |
| host / GPU | `r4900g3-UniServer-R4900-G3` · L20 ×2 |
| 代码指纹 | `intent_cc/intent_model_cc.py` · `dc9903d60066` |
| 环境 | py3.11.15 · torch2.11.0+cu130 |
| 曲线 | 9 点（3 epoch × loss / val_acc / val_f1） |
| 指标 | acc 0.7017 · macro_f1 0.3658 · n 1294 · per-cam 305/318/320/323 |
| 产物 | best.pt 144.6KB · metrics_full.json 8.4KB，均带 sha1 |
| 本机 | bundle 导入后训练页 `#/train` 与运行页 `#/runs` 均可见，曲线与指标齐全 |

> 这是**链路验证**，3 epoch / d-model 64，不是训练质量结论；run 的 `note` 已标注。

### 15.6 这轮抓到的两个真缺陷

**① `find_metrics` 不认 `metrics_full.json`**

`ops_run` 原本只找 `metrics.json` / `summary.json`，而 `intent_model_cc.py`、`intent_model_social.py` 产出的是 **`metrics_full.json`**。结果：run 记下来了、曲线也有，但 `metrics_json` 是 **null**，训练页的 macro_f1 / acc / 终值指标全空。

第一次端到端跑出来的就是这条空指标记录（`run_20261109_115003_00f410`，已删除）。修法：`find_metrics` 增加该文件名，且**优先**取它（比 `metrics.json` 多 per_cam / per_known / hist）。

**② `normalize_metrics` 不认 `{"rep": {...}}` 包裹**

这两个脚本的指标外面还套了一层 `rep`：

```json
{"rep": {"config": {...}, "overall": {"acc": ..., "macro_f1": ..., "confusion": ...},
         "per_cam": {...}, "per_known": {...}}, "hist": [...], "args": {...}}
```

原归一化只看顶层 `test` / `overall` / `metrics` / `eval`，对这种结构返回原样 → 前端取不到字段。修法：识别 `rep` 包裹后 unwrap，但 **`_raw` 仍存整包**（含 `hist` 与 `args`，那才是完整证据）。

**③ best epoch 无处可取**

`rep` 结构不带 `best_ep`。前端改为从 `val_f1` 曲线取峰值点（`peak()`），最佳 val_f1 同样回退到曲线峰值。这样任何只上报曲线的训练都能显示「最佳 epoch」，不必依赖脚本额外写字段。

### 15.7 验收

| 项 | 结果 |
|---|---|
| `tools/ops_selftest.py` | **17 / 17**（新增：bundle 闭包+幂等+拒绝 running+路径改写+目录导出；指标归一三形态 + `_raw` 保留整包） |
| 远程/本机文件一致 | 5 个工具 + 2 个训练脚本 sha256 全部一致 |
| 远程实跑 | 1 次真实数据短训练，台账 9 点曲线 / 2 产物 / 代码指纹 / 环境齐全 |
| 回传 | export → get → import 三步，本机库记录与远程逐字段一致 |
| 页面 | `#/train?run=…`、`#/runs?id=…` 本机 8022 实测 |

### 15.8 边界与后续

- **正式训练尚未跑**：leader 的 e21 切片仍在进行，GPU 未被我的验证占用（用了 `--device cpu`）。
- 远程 `ops.db` 会随每次 run 增长，属正常；它只是过程账本，权威仍在 `ops/ops.db`。定期 `ops_bundles` 回传后可清理。
- 权重文件不搬回本机，artifact 只记 uri + bytes + sha1；需要部署时按 uri 单独取。
- bundle 目前一次导一条 run；批量导出靠 shell 循环（交接单 §2.4 给了脚本）。

---

## 16. 派发卡片接上队列（2026-11-09 午后）

### 16.1 问题（用户提问触发的一次真实核查）

用户问「任务派发派发的是什么任务」。查台账后：**5 张卡全部是 `kind=label`（人工判读标注队列），全部创建于 11-02 17:50 的同一次派生，此后零改动，`assignee` 全为空**。

更要紧的是数字：

| 卡片 | 卡上写着 | 实际待判 | 当时的真实状态 |
|---|---|---|---|
| 真值·跨机实体配对 | 201 条 | 15 | doing |
| 真值·出入口事件 | 77 条 | 0 | doing（早做完，没关） |
| 候选对 318×320 | 42 条 | 0 | doing（早做完，没关） |
| 玻璃区域掩膜 | 6 条 | 3 | doing |
| S6-b 待裁决批 | 5 条 | 0 | review |

三张已清零的卡既没关也没消失。根因是 `derive_label_jobs` 的幂等条件是 `status != 'done'`，而关闭必须人挂证据——没人关，卡就永远停在那里，数字也永远停在派生那一刻。

**结论：叫「派发」，实际是静态待办清单。**

### 16.2 改法

**① 数字实时，卡片不再存快照**

`list_jobs()` 对 `kind=label` 的卡附加 `live`（来自 `v_task_progress` 现算）：`{pending, items, done, pct, name}` 与 `drained`。派生时也不再往 `goal` 里写数字：

```diff
- goal=f"清空该任务待判队列（当前 {t['pending']} 条）"
+ goal="清空该队列的待判条目（实时进度见卡片）"
```

已有 5 张卡的旧文案做了一次性迁移（`inputs.pending` 也移除），写 `audit`。

**② 清零的卡自动进「待验收」，但绝不自动关闭**

`sync_label_jobs()` 只处理 `status='doing'` 且 `pending==0` 的卡，推进到 `review`。理由：

- 卡是人在做、队列也空了 → 该来验收了；
- `backlog` / `assigned` 不动——还没开始做的卡直接进待验收没有意义；
- **不自动 `done`**：关闭仍要人点一次。自动关闭会把「没人管」洗成「已完成」，比留着一张卡更危险。

**③ 关闭必须有事实，不是必须手填**

`update_job_status()` 在 `kind='label'` 且带 `task_id` 时：

- `pending > 0` → 抛 `ValueError`，报「队列还有 N 条未判完，不能关闭」；
- `pending == 0` 且未提供证据 → **自动生成** `{"auto":"queue_drained","task_id","items","done","at"}`。

这比原来「必须手写一个路径」更硬：**证据来自队列事实，不是人的自述**。前端的证据输入框对这类卡改为「可留空」。

**④ `doing → review` 不拦**

一度想把「提交验收」也要求队列清零。没这么做：验收环节本就是给人判断的，可能接受部分成果；只有 `done` 是终态，必须符合目标。测试中我误把这条当缺陷，实际行为是对的。

### 16.3 接口与页面

| 层 | 变化 |
|---|---|
| `ops_db.list_jobs(conn, status=None, live=True)` | 附加 `live` / `drained` |
| `ops_db.sync_label_jobs(conn)` | 清零的 doing 卡 → review，返回变化列表 |
| `ops_db.derive_label_jobs()` | goal / acceptance / budget 去掉写死数字 |
| `ops_db.update_job_status()` | label 卡 done 前校验队列 |
| `POST /api/ops/jobs/sync` | 新增，返回 `{ok, moved[], n}` |
| `#/jobs` | 卡片显示 `done/items` + 进度条 + 「剩 N 条」/「队列已清零」；顶部加「同步状态」 |

### 16.4 验收

| 项 | 结果 |
|---|---|
| `tools/ops_selftest.py` | **18 / 18**（新增：live 随判读实时变化、sync 只推清零卡、pending>0 拒绝关闭、清零后自动取证） |
| 主库实跑 | sync 推进 2 张（事件 107/107、候选对 75/75）；`review→done` 对 pending=15 的卡被拒 |
| 页面 | `#/jobs` 本机 8023 实测：进行中 2 张带剩余数，待验收 3 张标「队列已清零」 |

### 16.5 仍然没做的

- **`assignee` 仍为空**：卡片没有「派给谁」这一步，等于「列出来等人认领」。真正派发需要人到「认领」按钮，或按人分配——这属于流程约定，不是代码能单方面决定的。
- **另外两类派生仍未实现**：`experiment` 缺 baseline → `run` job、`run` 标 `stale` → `fix` job（后者依赖 P2 的 stale 判定规则）。
- 卡片的 `priority` 仍取派生时的待判数（现在不会过期到误导，因为界面以 live 为准，但排序值仍是旧的）。若要按实时待判排序，需要在读取时算；本轮没做，因为 `priority` 同时承载人工调整意图，混用会互相覆盖。

---

## 17. 剩余开发计划

P0 之后的缺口、做法与验收已单独成文：**`docs/ops/backlog-plan.md`**（2026-11-09）。

范围不含派发（认领/指派/另两类 job 派生，按用户决定不做）。清单按「不做会出什么错」排序：

| 组 | 内容 | 核心问题 |
|---|---|---|
| A 血缘闭环 | 数据集登记 → 模型注册 → `stale` 判定 | 41 条 run 全 `succeeded`，换了检测器也不会有任何标记；`model` 表 0 条，指标无法回答"哪个权重" |
| B 标注统一 | 标注页接管 `annotation_item` → 导出与 run 绑定 | 三个页面 1581 行各写 json，台账靠镜像；已因此出过两次同类事故（§11.2、§14.1） |
| C 质量与回归 | 变体对比 → 实验假设/结论 → 双标 kappa | 只能看单个变体，看不出"比基线差在哪"；`label` 表支持多标注员但实测无第二人 |
| D 部署 | 远程 dist 发布 | 远程仍是 10-14 版 |

建议顺序：A 组一次做完（互相依赖）→ B 组三步走（读切换 / 导出 / 写切换）→ C 组。A 与 B 不建议并行。

---

## 18. A 组落地：血缘闭环（2026-11-09 下午）

对应 `backlog-plan.md` 的 A2 → A3 → A1。目标只有一个：**让任何一个指标能点到「哪份数据、哪个权重、哪版代码」**。

### 18.1 A2 数据集登记

新增 `tools/ops_dataset.py`：

```bash
python tools/ops_dataset.py scan --id ds_6cam_synced10 --name "..." \
    --root media-resource/multi-cemra/synced-10min --dialect clean2
python tools/ops_dataset.py ls / show <id>
```

`snapshot_hash = sha1(按路径排序的 "rel|size|mtime_ns" 行)`，同时写 `ops/manifests/<id>.jsonl`（完整文件清单）。取 mtime 是故意的：重新编码、覆盖替换都会变；只想比规模时用 `--no-mtime`（算法标识随之不同）。

**登记结果**

| dataset | 文件 | 大小 | hash（前 16） |
|---|---|---|---|
| `ds_6cam_synced10` | 7 | 0.57 GB | `fa64a340d6519469` |
| `ds_1012_29h` | 15 | 6.33 GB | `d1d6bf3bbfdc1d2f` |

### 18.2 途中发现的两个真问题

**① 台账里原有的 `snapshot_hash` 不可复算**

`ds_6cam_synced10` 原值 `9f5ea637…` 在整个仓库里**找不到任何生成代码**——是当初手填的。手填的 hash 无法验证，也就无法用于判定"数据变没变"。

处理：新增 `dataset.hash_alg` 列区分算法。扫到"已有 hash 但算法不同"时**拒绝写入**，必须 `--force` 明确换代际：

```
ds_6cam_synced10 已有 snapshot_hash（算法 legacy/手填），本次要写入 rel-size-mtime-v1。
换算法后与历史 run 不可比，确认换代际请加 --force。
```

已 force 一次（旧值不可复算、且当时没有任何 run 用它算过指纹），并写 audit 留痕。

**② `ops_run --dataset` 会把数据集登记抹掉**

`upsert_dataset` 原来无条件写全部字段，`ops_run` 自动登记时 `snapshot_hash` 传 NULL，于是：

```
scan 之后：hash=fa64a340…  file_count=7
ops_run 之后：hash=None     file_count=0   ← 被自动登记清空
```

修法：`upsert_dataset` 只更新**显式传入且非 None** 的字段，且已存在时不覆盖 `name`（否则中文名会被 id 顶掉）。这个 bug 只有在"先 scan 再跑 run"的顺序下才暴露，属于典型的顺序依赖缺陷。

### 18.3 A3 模型注册

新增 `tools/ops_model.py`：

```bash
python tools/ops_model.py register --path models/face/scrfd_2.5g_bnkps.onnx \
    --name scrfd-2.5g --backbone SCRFD --license InsightFace
python tools/ops_model.py from-run <run_id> --name intent-cc-e2e
python tools/ops_model.py ls / show <id>
```

`id = model_<slug>_<sha1[:6]>`，`version = sha1[:8]`；`deployable` 默认 0（**注册 ≠ 可上线**）。

已注册 8 个现成权重：

| model | 大小 | backbone |
|---|---|---|
| `model_scrfd_2_5g_141b9c` | 3.3 MB | SCRFD |
| `model_yunet_2023mar_acbe4b` | 0.2 MB | YuNet |
| `model_w600k_mbf_589d6d` | 13.6 MB | MobileFaceNet |
| `model_w600k_r50_ae27d8` | 174.4 MB | ResNet50 |
| `model_minifasnet_v2_546d20` | 1.7 MB | MiniFASNet |
| `model_osnet_msmt17_b6906f` | 17.3 MB | OSNet-x1.0 |
| `model_osnet_ain_msmt17_c3f0e1` | 17.3 MB | OSNet-AIN |
| `model_osnet_imagenet_cfbe61` | 10.9 MB | OSNet-x1.0 |

`from-run` 还会把 `run.model_id` 反向填上，形成 `run → model → trained_by_run → run` 的双向链。

### 18.4 A1 stale 判定

`ops_run` 现在写 `input_hash` + `input_breakdown`：

```
input = dataset.snapshot_hash + model.sha1 + env.code.sha1
```

只纳入会让结果失效的三样。**超参不纳入**——它由 run 自己记录，不会事后变化。任一分量缺失就不写总 hash（不假装算得准）。

`tools/ops_stale.py check`：

- 默认只比 **数据 + 权重**；`--include-code` 才把执行脚本纳入（代码改动不等于结果失效，且太频繁）
- 差异写明是哪个分量：`数据快照变了（fa64a340d6 → 0000000000）`
- **历史 run 无指纹直接跳过，不追溯猜测**（41 条全跳过）
- `revert <run_id>`：退回 succeeded 并按当前输入刷新指纹——不刷新的话下次 check 会立刻再标一次

### 18.5 验收

| 项 | 结果 |
|---|---|
| `tools/ops_selftest.py` | **21 / 21**（新增：hash 可复算/幂等/双算法区分；模型注册幂等 + from-run 双向挂接；stale 未变不标/变则标并指明分量/无指纹跳过/revert 可退） |
| 实跑 | `rescan-6cam` run：data 指纹 `835ae982…`，代码指纹 `ops_dataset.py · 580c03b28c5a` |
| stale 实测 | 临时改 dataset hash → 精确标 1 条并写明「数据快照变了」；revert 后不再复标；测试数据已清理（run 回到 41 条） |
| 页面 | `#/runs?id=…` 显示「数据集 name · hash (N 文件 · 算法)」「模型 name · sha (backbone)」「输入指纹」 |

### 18.6 仍未闭环的一处

`ds_intent_1012_cc`（远程训练用的数据）**本机没有实体**，扫不出 hash，页面如实显示「无 hash（0 文件）」。

要补上需要：远程 `ops_dataset.py scan` → `ops_bundle` 带 dataset 一起导出（bundle 已包含 dataset）。这项工作并入了 B 组之后的远程例行流程，不单独做。

## 19. B 组第一步落地：标注页读切换到台账（2026-11-09 下午）

### 19.1 起点：一个必须先修的证据缺口

计划里 B1 分三步（读切换 → 写切换 → 下线老接口）。动手前先核对了「台账能不能独立支撑页面渲染」，结论是**不能**：

| 页面 | 老结构 | 台账 `value` | 丢失 |
|---|---|---|---|
| 真值配对 / S6-b | `{v, c, n}` | `'0'` / `'1'` | 置信度 `c`、备注 `n` |
| 真值事件 | `{pid, fp}` | `'pid:28'` 或 `'fp'` | 两字段并存时丢一半 |
| 人脸人卡 | `{verdict, name, at}` | `'person'` | 姓名 |
| 素材配对 / 倒影 | 字符串 | 同 | 无 |
| 玻璃掩膜 | 多边形数组 | 同（JSON 文本） | 无 |

即：**`value` 只适合跨任务比较，不适合承载证据**。若直接切读，页面上所有置信度高亮、备注、姓名会一起消失。所以先补第 0 步。

### 19.2 第 0 步：`label.data_json`（证据列）

- `label` 表加 `data_json TEXT`：存**原始判定对象**，`value` 退回"主值"角色。
- `set_label(..., data=...)`：同 annotator 同 value 但 data 变化 → **不新增行**，只 UPDATE 该行的 `data_json`（补证据 ≠ 改判，不能污染 append-only 的判定序列）。
- `label_data(row)`：老行（`data_json IS NULL`）退回解析 `value`，不抛。
- `tools/ops_backfill_labeldata.py`（一次性，默认预演，`--apply` 才写）：
  - 源 = 4 个 json（labels / truth.labels / truth.seed / faces_labels），按 item_id 归位；
  - 实测：**补证据 615 条 + 补 seed 15 条，`skip_no_item = 0`**——这也顺带证明了「8 个 task 完整覆盖三页所有条目」；
  - 幂等：第二次跑 `same=630 / updated=0`；
  - 写前备份 `ops/db_backup/ops.db.bak_pre_labeldata_1447`。

★ **seed 的口径**：`truth/seed_labels.json` 的 19 条是页面预填（比 labels.json 更早的判定），页面 `_truth_merged` 一直把它当已判显示。回填把它们写成 `annotator='seed'` 的 label 行，并在 `queue` 里用 `seeded` / `by_source` 区分来源——**不假装是真人复核**。

### 19.3 第 1 步：`/api/ops/queue`

```
GET /api/ops/queue?task_id=t_truth_pair,t_truth_event,t_s6b_pending
→ { tasks: [{id,name,kind,dialect,note,spec,items,done,pending,by_source,seeded}],
    items: { task_id: [{id,key,stratum,status,payload,label:{value,data,by_whom,at,confidence}}] } }
```

设计取舍：

- **一次取多个 task**：三页各发 1 个请求即可，避免 N 次往返；
- `payload` 原样返回（候选的几何/分数/缩略图定位全在里面），前端不需要第二个数据源；
- `label.data` 与 `label.value` 同时给：前者渲染，后者比较；
- `by_source` 让"多少条是 seed / 多少条是真人"可见。

`ops_api.py` 里同时给 `/api/ops/label` 加了 `data` 字段（写切换时页面可直接传原始对象，不再压平）。

### 19.4 第 2 步：三个页面的读切换

| 页面 | 改动 | 数据来源边界 |
|---|---|---|
| `TruthView`（552 行） | `load()` 改读 queue（3 task），`pairs/events` 由 `payload` 展开、`labels` 由 `label.data` 还原；互斥冲突改为前端现算 | **条目 + 判定都来自台账**（台账覆盖 241/107/75，与 state.json 一一对应） |
| `AnnotateView`（552 行） | `store.ts` 的 `fetchLabels()` 改为读 queue（3 task），失败回落 `api.labels()` | 候选来自算法产出（pairs.json / mirror.json），**判定来自台账** |
| `FacesView`（503 行） | `loadLab()` 改读 queue（2 task），失败回落 `api.facesLabels()` | 人卡/融合等展示数据仍在 `face_data.json`，**判定来自台账** |

写在 `store.ts` 里的原则（注释也留了）：**判定唯一来源 = 台账；候选列表是展示数据**。两者条目数不等时页面上直接看得出来（分子分母不同源），属于可观测的不一致，不静默。

顺带修的口径 bug：

1. **`state.stats` 曾把 S6-b 键算进真值配对**：`labeled=316 > total=241`（`_truth_merged` 的 merged 里混着 75 条 `S6B_`）。读切换后页面用 `pairs` 现算，`15/212/14` 与 `/api/truth/metrics` 完全一致。
2. **`ingest_truth` 的 stratum 用 `rank < k_topk` 反推**：rank 从 1 起，把第 30 名划进 audit（30→29）。改为直接取 `payload.role`，实测修 200 条，恢复 **topk 30 / audit 20 / rest 191**。
3. `verify()` 只覆盖 6 个 task（缺 `t_mask_glass` / `t_truth_event`），且真值配对的 json 侧不认 seed → 恒报 15 条差异。已补齐到 **8 个 task 零差异**。

### 19.5 验收（本机 8025 实测）

| 项 | 结果 |
|---|---|
| `ops_selftest.py` | **23 / 23**（新增：判定证据往返——同值补证据不新增行 / 改判挂 supersedes / 老行兜底；镜像证据无损——`{v,c,n}`、`{pid,fp}`、人脸 `name`） |
| `ops_ingest --verify` | **8/8 任务零差异**（`mis_db=0 miss_json=0 conflict=0`），含 `t_mask_glass`、`t_truth_event` |
| 页面数字 | 真值配对 **241/241**、事件 **107/107**、S6-b **75/75**（分组 16/16·3/3·11/11·5/5·40/40）；素材候选 **75/75**；人脸 **45/45 人 · 28/28 对** —— 与台账 `v_task_progress` 一致 |
| 端到端往返 | 经 `/api/truth/labels` 写一条 `{v:0,c:'low',n:'B1 端到端自测'}` → json 与台账双更新、`label` 仍 1 行（同值补证据未新增）、queue 返回完整 `data`；写回原值后 `verify` 仍零差异 |
| 截图 | `docs/ops/shots/b1-truth-pair.png` / `b1-truth-event.png` / `b1-truth-s6b.png` / `b1-annotate.png` / `b1-faces.png` |

### 19.6 已知边界（未做 / 不假装做了）

1. **写入仍走老接口**（各页 `saveLabels` / `truthSave` / `faceLabel`）。台账由 `_ops_mirror` 同步，方向是 json → 台账。这是 B1②「写切换」的范围，下批做。
2. **候选集仍是算法产出**：pairs.json / mirror.json / face_data.json 是展示数据，不由台账生成；台账不覆盖「轨迹状态里非 keep 的其余候选」（AnnotateView 的倒影 tab 会合并 `trackStatus`）。
3. **远程预览**：远程无 `ops/ops.db` → `queue` 返回 503 → 前端 catch 回落老 json。远端 dist 仍是 10-14 版，未发布。
4. `notes`（自由备注表）台账侧无对应概念，切读后恒空；页面未渲染它，暂无影响。

## 20. 前端展示体检（2026-11-09 傍晚）

### 20.1 检查范围

11 个页面全部人工过一遍（本机 8025，1600×1000 截图归档 `docs/ops/shots/audit/`），
接口侧核对了 `/api/cams` `/api/tracks/*` `/api/ops/*` `/api/media-list` 的返回与体量。

### 20.2 核心问题：可视化只有前 10 分钟（数据范围，不是播放器 bug）

| 事实 | 证据 |
|---|---|
| 会话窗口固定 600s | `webapp/backend/data/cams.json`：`session.window_sec=600`、`base_wall=08:55:00` |
| 6 路媒体都是 10min/540p | `data/media/*.mp4`：14901~14998 帧 @25fps = 9.93~10.00 min，960×540 |
| 检测产物也只有 10min | `output_crosscam/{cam}_detections.jsonl` 每路 2981~3005 行 = 600s @ stride 5 |
| ★ **磁盘上已有 35 分钟版** | `output_crosscam/long35/{cam}_long35_detections.jsonl` 每路 10480~10517 行 = 35min |
| ★ **两者是同一素材** | 305：前 3000 行（=10min）与 10min 版**逐位一致 3000/3000，0 不一致** |
| 视频源也在本地 | `media-resource/multi-cemra/long-35min/*_long35.mp4` 6 路 35min 高清，共 1.87 GB |

结论：不是"只能看一段"，而是**只导出了这一段的前 10 分钟**；后 25 分钟的检测结果已经在盘上。

扩展缺口（按代价排序）：

1. `webapp/dataprep/build_web_data.py` 硬编码 `output_crosscam/{cam}_detections.jsonl` 与 `window_sec=600` → 需加 `--det-dir/--window-sec/--base-wall` 参数；
2. 6 路 35min 未转 web 版（本机**未装 ffmpeg**；chrome 直读 2560×1440 原片会卡）；
3. 实体图 `analysis/stitch_clean2`、接力链 `analysis/relay_clean2` **只有 10min 口径** → 35min 段没有跨机分析（viz 左侧接力链与"倒影降权"档会失效）；
4. `tracks/*.json` 现共 1.8 MB（10min），35min 约 6.3 MB，首屏需分片/懒加载。

三个方案：

| 方案 | 内容 | 代价 | 取舍 |
|---|---|---|---|
| **A 延长片段（推荐）** | 视频+检测+轨迹扩到 35min；实体/接力链在 0–10min 段正常、之后降级隐藏 | 装 ffmpeg 转码 6×35min（约 20–40 min CPU）+ build 脚本参数化 + tracks 分片 | 能用满 35 分钟连续看人流转机位；跨机分析只覆盖前 10min |
| **B 全链路重跑** | long35 重跑 cleanset→stitch_clean2→relay→C0 | 需算力机数小时；产出一套与现行 C0 v9 并行的口径 | 35min 也有接力链/倒影判定，但要重新校准基线与阈值 |
| **C 不扩** | 10min 作为固定样例，另以表格/指标展示 1012（29h）与 e21（3 天） | 最低 | 可视化永远只有 10min |

### 20.3 已修（本轮，均为展示层，无数据口径改动）

| # | 问题 | 现象 | 修法 |
|---|---|---|---|
| 1 | 素材标注首屏空白 | 默认筛选「待决定」+ 75/75 已判完 → 打开是空列表，像坏了 | 挂载后检测无待判则自动切「全部」 |
| 2 | 训练页缩略曲线无信息 | 变体表 val_f1 缩略图固定 0~1 纵轴 + 18px 高，0.45→0.49 的变化不足 1px → 12 个变体全是同一条平线 | 改**自身 min-max 缩放**（span<0.005 时画平线避免放大噪声），加 `title` 说明区间 |
| 3 | 运行页指标裸 JSON | `{"items":[{n,name,note,value}…]}` 整块 dump，要人工解析 | 结构化表：`name → value`+中文说明，原始 JSON 折叠在 `<details>` |
| 4 | 聚类方法对比表重复行 | `mutual-knn k=5` 的 thr 0.3/0.4/0.5/0.6/0.7 五行数值完全相同，看着像 bug | 按指标合并为一行，参数显示 `thr=0.3–0.7`（**显式说明阈值不敏感**，这本身是结论）；12 行 → 4 行 |
| 5 | 总览 5 张 KPI 高度不齐 | 前 3 张有进度条更高，第 5 张只有文字更矮 | `.kpis{align-items:stretch}` + `.kpi{display:flex}` + `.panel-b{flex:1}` |
| 6 | 派发噪音常驻 | 风险栏「5 张任务卡未关闭」+ KPI「未关闭派发任务」；派发已决定不纳入本期 | KPI 换成「数据集/权重」血缘数字；风险栏撤掉派发条目（`#/jobs` 页面保留） |

验收：构建通过（652 modules），6 张页面截图逐项复核（`docs/ops/shots/audit/{annotate,train,runs,cluster,ops}.png`）。

### 20.4 记录但未修（可接受 / 待定）

| 项 | 判断 |
|---|---|
| 时间双套显示（轴 `00:26` vs 墙钟 `08:54:55`） | 相对时间与同步基准都需要，暂不动；若要改，在轴上标 `+mm:ss` |
| 配置页混淆矩阵右侧空白 | 纯排版，信息无缺失 |
| 可视化右侧「从这里开始」引导 | 未选中目标时的占位，选中即换成旅程/检查器/小地图，设计如此 |
| ReID 页初始空白 | 需要输入描述或点左侧头线，属交互入口而非缺陷 |

## 21. 方案 B 落地：35min 全链路重跑（2026-11-09 夜）

### 21.1 决策与做法

§20.2 给了三个方案，用户选 **B（全链路重跑）** 并要求"用现在的最新算法模型"。
执行口径：**编码器换成 exp8 验证过的 OSNet-x1.0 MSMT17**（冻结集 AUC 0.922 vs DINOv3-B CLS 0.628），
其余算法（同机位拼接、实体图、时间门、mutual-best 链构造）**沿用生产参数**，只换数据根 —— 保证"同一算法、更长窗口"。

跑批细节、命令、耗时、坑：`docs/ops/long35-rebuild.md`（含 11 节，本轮新增 §8~§11）。

### 21.2 结果

| 项 | 10min | 35min |
|---|---|---|
| 轨迹 | 541 | **1779**（3.29x） |
| 实体 | 411 | **1271**（3.09x） |
| 跨机边 | 44 | **135**（3.07x） |
| 接力链 | 29 | **88**（多跳 37） |
| 网页 tracks | 1.8 MB | 7.3 MB |
| 媒体 | 103 MB | 200 MB（960×540，`+faststart`） |

倍率贴近时长比 3.5x，没有出现"越拼越碎"的失控。343（电梯间）增幅最小（1.78x）——人流稀疏，长窗口把同一个人拼成更长实体。

### 21.3 ★ 一条不能含糊的负结果

10min 的 44 条接力边在 35min 的 135 条里只命中 **16 条（36%）**。
逐条追查后确认**不是 35min 链路错**：mutual-best 是全局唯一匹配，实体被拼长之后竞争关系变了；
时间门（60s）也会因为实体首末时间变化而失效。

→ 结论：**候选级可比（前 10min track id 逐位一致），实体级不可比**。
要报 35min 的精度，必须补 35min 的人工标注——本轮没有，所以只报结构指标。

这也是一条方法论记录：**"同一算法换个窗口"不能直接拿旧边集当 ground truth 去算召回**。

### 21.4 阈值

OSNet 的 cos 分布（p50 0.540 / p90 0.678 / p99 0.815）与 cradio 完全不同，不能沿用 0.90。
扫 0.55~0.90 看 mutual 边数，在 0.60~0.65 饱和，取 **0.65**（135 边）。
**这是结构合理性选择，不是精度校准**——写进 run 的 note 里，避免后人误当校准值。

### 21.5 切换与回滚

```
python tools/long35_switch.py snapshot 10min    # 105MB 快照
python tools/long35_switch.py use long35        # 切 35min
python tools/long35_switch.py use snap_10min    # 回滚
```

★ **判定文件（labels / mirror / pairs）不随片段切换**。理由：判定属于"人"，不属于"哪一段"；
且已核 6 路前 10min 的 track id 与 10min 版逐位一致，所以既有 75 对判定在 35min 里仍指向同一个人。
切换后实测全部保留。

### 21.6 两个展示层修复

1. **首帧黑屏**：`currentTime` 已是 0 时再赋 0 是 no-op → 浏览器不解码，面板全黑。
   素材一大（每路 23~42 MB）必现。修法：目标 < 0.02s 时推一帧（0.04s），强制走一次 seek。
2. **深链接 `#/?t=1200`**：本为验收用（定位到第 20 分钟抽帧比对），顺带成了分享单帧的能力。

### 21.7 验收证据

- 时间对齐：t=1200s 抽帧，305 OSD 09:14:50（页面 09:14:55，offset −5）、318 OSD 09:15:18（页面 09:15:20，offset +20）
- 框对齐：305 9 框 / 318 5 框 / 323 4 框，均落在人身上
- 台账：回填 4 条 run（encode/stitch/relay/webpack），含产物 sha1 与指标；被 SIGTERM 的试跑 run 保留 failed 事实并在 note 里指向替代批
- 截图：`docs/ops/shots/audit/viz-35min3.png`、`viz-35min-t1200b.png`、`frame_{305,318,323}_t1200.png`、`{annotate,truth,ops}-35min.png`

## 22. 排查：「两人一重合就丢身份 id」（2026-11-09 夜）

### 22.1 现象校正

不是"人消失"，是 **ID 换了**：同一个人的框还在，但 `track_id` 变了，于是被拆成两个实体，
页面上表现为"选中的人走到别人旁边，身份就没了"。

### 22.2 一条完整证据链（305，f=4415→4440）

| 帧 | raw→keep（removed） | 关键框高 | 发生了什么 |
|---|---|---|---|
| 4405 | 8→6 (2) | t40 h=470 | 正常 |
| 4415 | 8→6 (2) | t40 h=**520** | 旁边坐着的人被"吞"进框 |
| 4420 | 7→7 (0) | t40 h=**623** | 框已合并，NMS 不再判"包含" |
| 4425 | 7→7 (0) | t40 h=**631** | 最后一帧 |
| 4435 | 8→7 (1) | **t46** h=**862** | **换号**：t40 消失，t46 接手（同一个大框） |
| 4440 | 9→6 (3) | t46 h=835 | 另两个框被 NMS 删（1 帧后恢复） |

证据图：`docs/ops/shots/audit/ids_drop_evidence.png`（红=t40，绿=t46）。

### 22.3 机制：四层各自"合理"，叠起来丢人

1. **检测**：DINO 在两人靠近时输出"把两人一起框住"的大框（h 451→520→623→862，+91%）。
2. **Containment-NMS（0.85, keep=larger）**：合并框出现后不再有"包含"关系，救不了；
   反而在 4440 删掉 3 个框（`removed=3`）。
3. **ByteTrack（minimum_matching_threshold=0.8）**：h 631 的框与 h 862 的框 IoU=**0.71** < 0.8 → 匹配失败 → 开新轨。
4. **拼接（cos≥0.80 且 pe_adj≤0.08）**：外观 cos=**0.9123**（很强），但位置外推误差 pe_adj=**0.1602** > 0.08 → 拒。
   而 pe 大**恰恰是因为框本身错了**（合并框让"末位置"偏了），不是人走远了。

### 22.4 量化：每层还有多少空间

305 的 367 轨有 **221 个断点**；其中 **71 个**能在 8s 内同位置（IoU≥0.5）找到后继：

| 分类 | 数量 | 占比 | 含义 |
|---|---|---|---|
| 已接上 | 34 | 48% | 现状判据能处理 |
| **cos < 0.80 被拒** | 24 | 34% | 外观本身不像（遮挡/姿态），**拼接层无解** |
| **pe > 0.08 被拒** | 13 | 18% | 位置外推超限，**可救** |

放宽实验（`tools/exp_stitch_relax.py`，只改判据不重跑算法）：

| 方案 | 305 链接 | 305 实体 | 323 | 318 |
|---|---|---|---|---|
| S0 现状 | 117 | 227 | 159 / 350 | 130 / 287 |
| S1 现状 + 同帧共现硬约束 | 117 | 227 | 152 / 356（**砍掉 11 条误连**） | 130 / 287 |
| S2 +短 gap 放宽（cos≥.90, gap≤20, pe≤.25, scale∈[.8,1.25]） | 121 (+4) | 223 | 152 | 130 |
| S3 +短 gap 放宽（.88/.30/30/.75–1.3） | 122 (+5) | 222 | 152 | 130 |

→ **拼接层上限就是 +4~5 条（约 6%）**。mutual-best 的唯一性约束把更多候选吃掉了。

★ 附带发现（真实缺陷）：**现状没有"同帧共现"检查**，323 有 **11 条边把"同一帧同时出现"的两个 track 连在一起**——那必是两个人。加这条硬约束是纯收益。

### 22.5 三个可选项

| 方案 | 做什么 | 成本 | 收益 |
|---|---|---|---|
| **A 拼接层收紧+放宽**（推荐先做） | 加同帧共现硬约束；短 gap 放宽 pe；scale 兜底 | 改 1 个文件 + 重跑拼接/实体/链/打包 ≈ 5 分钟 | 修掉 323 的 11 条误连；305 多接 4~5 个换号 |
| **B 页面标注"可能同人"** | 把 (cos≥0.90, gap≤0.8s, 同位置) 的相邻实体标成同一色/虚线 | 纯前端 + 一个 json ≈ 1 小时 | 用户能自己看出"这是换号不是消失"，不改数据口径 |
| **C 重跑检测治本** | 加"合并框拒收"（框高突增 >30% 且宽高比异常 → 不作用于跟踪）+ ByteTrack 匹配阈值按 stride 重设 | DINO 重跑 6×35min ≈ **30 小时 CPU**（或上算力机） | 断点从源头减少，预估能消掉 22.2 这类切换的大半 |

**A+B 可以立刻做，C 要排算力。** 三者不冲突。

### 22.6 为什么不能只调 ByteTrack 阈值

试过用现有每帧框重放跟踪（`tools/exp_retrack.py`）：把 `minimum_matching_threshold` 从 0.8 降到 0.6/0.5，
track 数从 367 暴涨到 3123/5990 —— **完全失控**。且 thr=0.8 的重放本身也复现不了原结果（317 vs 367），
说明本机 supervision 0.30.9 的 ByteTrack 已被废弃，行为与产出时的版本不同。
→ **「只调阈值」这条路走不通，必须在检测器那一版环境里重跑。**

### 22.7 A 方案已实施（2026-11-09 23:20）

改动（`tools/rebuild_clean2_entities.py`，生效路径；`tools/track_stitching.py` 同步以便工具一致）：

1. **同帧共现硬约束**（`--no-covis-guard` 可关，默认开）
   `load_spans(..., want_frames=True)` 额外返回每轨帧集合；`clean2_links` 里
   `frames[a] & frames[b]` 非空 → 直接排除。这是物理约束，不是阈值。
2. **短 gap 放宽**（`--relax-gap 20 --relax-cos 0.90 --relax-pe 0.25 --relax-scale 0.80 1.25`，默认关）
   覆盖「合并框让末位置偏了」的场景；`scale`（框高比）兜住"一大一小两个人"。
3. manifest 记录 `link_criteria.{covis_guard, relax}` 与每机位 `covis_dropped`/`relax_added`，可追溯。

**不需要重跑嵌入或 CSV**（cos/pe/gap/scale 都在 `stitch_*.csv` 缓存里），只重算 mutual-best，6 路 4 秒。

结果：

| cam | links | 实体 | covis 排除候选 | relax 新增候选 |
|---|---|---|---|---|
| 305 | 117 → **121** | 250 → **246** | 44 | 8 |
| 316 | 10 → 10 | 140 → 140 | 13 | 0 |
| 318 | 129 → 129 | 338 → 338 | 56 | 0 |
| 320 | 68 → 68 | 111 → 111 | 9 | 0 |
| 323 | 159 → **152** | 350 → **356** | 132 | 0 |
| 343 | 18 → 18 | 82 → 82 | 17 | 0 |
| 合计 | 501 → 498 | 1271 → **1273** | 271 | 8 |

下游同步：事件 5739 对候选 → 134 边 / 89 链（原 135 / 88）→ 重新打包 → `use long35` 切换。

两个可复核的案例：

* **305 的核心 bug 修好了**：改前 `40` 与 `46` 是两个实体（E026 / E030），
  改后 **E026 = [40, 46, 80]** —— 同一个人跨越两次框合并的完整轨迹。
* **323 拆掉 7 个误连实体**（350→356）。例：`[61,62]` 在 f=4420/4425 两帧同时在场，
  两框紧邻（x 873–926 与 922–970），必是两人。证据图 `docs/ops/shots/audit/covis_323_pair.png`。

★ 边际：A 只值 **+4 条合并 / −7 条误连**。拼接层不是主战场，剩下的要靠 C（重跑检测）。

## 23. B 方案落地：把「换号」标出来（2026-11-09 夜）

### 23.1 定位：标注，不是判定

§22 查明「两人重合丢 id」的机制是**换号**（框还在、track_id 变了），但拼接层能救的只有 +4~5 条（§22.7）。
要根治得重跑检测（C 方案，约 30h CPU）。B 方案走另一条路：**不改实体图，只在页面上把「这段可能是同一个人」标出来**，
让人能分辨「他走了又回来」和「他换了 id」。

边界写死在代码注释与接口 docstring 里：`suspects.json` 不参与实体图、接力链、C0 或任何指标；
关掉开关后页面与之前完全一致。

### 23.2 判据

| 项 | 值 | 为什么 |
|---|---|---|
| 外观 `cos` | ≥ 0.85（strong ≥ 0.88） | OSNet 的 p90=0.678 / p99=0.815，0.85 已是高分位 |
| 时间间隙 | ≤ 4.0s（strong ≤ 2.0s） | 人不会在 4 秒内凭空出现在别处 |
| 脚点距离 | ≤ 1.5×框高（strong ≤ 0.8） | 按框高归一化，避免远近景混用同一像素阈值 |
| **同帧共现** | 必须为 0 | 物理约束：同一帧同时出现 ⇒ 必是两人 |
| 最短段 | ≥ 5 观测（≈1s） | 更短的多半是误检碎片（只框到半个身子） |
| 链式合并 | 合并前校验两组两两无共现 | 见 23.4① |

方向也卡了：只收「a 段结束后 b 段才开始」（`b_first_frame > a_last_frame`），
所以单条 link 天然不可能共现——共现只可能由**链式传递**引入。

### 23.3 数据流

```
output_crosscam/long35/stitch_long35/stitch_{cam}.csv   （cos / pe_adj / gap / scale 缓存）
        + cleanset_v2/{cam}_detections.jsonl            （帧集合、首末框）
        + net_tracks_{cam}.json                         （tid → eid）
                    ↓  build_web_data.py: build_suspects()
        webapp/backend/data_long35/suspects.json        （19.7 KB）
                    ↓  /api/suspects
        store.suspectGid["305:140"] = "G305-05"
                    ↓
        画布：#id ≈ 虚线框        检查器：「疑似同一人 · N 段」可跳转
```

构建参数新增 `--stitch-dir`（默认同 `--entities`）与 `--no-suspects`；
`tools/long35_switch.py` 的 `PACK_FILES` 收录 `suspects.json`，且**源包没有时会删掉旧文件**——
否则切回 10min 包时残留的 35min track id 会指向错误的人。

### 23.4 开发中撞到的两个真问题（都是"会把两个人标成一个人"）

**① 链式传递把不同的人串起来。**
初版用并查集按 "A→B、B→C" 连通，结果 305 出现 6 段链，里面 `#222`(1449.8–1460.4) 与 `#224`(1460.8–1520.8) 首尾相接看着没问题，
但链上更远的 `#224` 与 `#232` **有 1 帧共现**——这是两个人。
修法：union 前检查两组的帧集合两两是否有交集，有则拒绝合并（宁可多分一组）。落库字段 `n_rejected_by_overlap`。

**② 2 帧的"幽灵段"是链条的胶水。**
305 的 `#231` 只有 2 个观测（f37975–37980），与 `#224` 共现 2 帧，却把 `#224` 和 `#232` 连了起来。
这类段是遮挡瞬间的误检框，标出来只会误导。加 `min_obs=5` 直接过滤。

两条修完：候选 86 → **66 段 / 53 组**，组内共现 **0**，时间倒挂 **0**。

### 23.5 页面表现

| 位置 | 表现 |
|---|---|
| 画布 | 属于候选组的框画**虚线**（`[7,4]`），标签加后缀 `≈`（strong）/ `?`（weak） |
| 检查器 | 新增「疑似同一人 · N 段」：整组列表 + 当前段高亮 + 点任一段 `seek(t0+0.2)` 并选中；底部给证据 `cos / 间隙 / 脚点差` |
| 工具条 | 「同人候选 53」开关（默认开），关掉后虚线、后缀、检查器块一起消失 |
| 深链接 | `#/?t=1005&sel=305:140` 直达某秒并选中某轨迹（验收时加的，顺带能分享具体某个人） |

### 23.6 验收（本机 8025，全部实测）

| 项 | 结果 |
|---|---|
| `tools/check_suspects.py` | **通过**：53 组 / 66 段；组内两两共现 0、时间倒挂 0、实体全部存在、计数一致 |
| 反向测试 | 人为把某组 tids 改成同一条 → 脚本报 `同帧共现 23 帧（应为两人）` 并退出码 1 |
| 画布 | t=1005 单机位：#140 框为虚线、标签 `#140 ≈`；关开关后变实线、后缀消失 |
| 检查器 | 305#140 → 「疑似同一人 · 4 段」：812.4–1020.0 / 1020.4–1449.4 / 1449.8–1460.4 / 1460.8–1520.2 |
| 跳转 | 点第 2 段：`16:45 → 17:00`，视频 `currentTime=1020.6`（= t0 1020.4 + 0.2），当前段高亮切到第 2 段 |
| 回归 | `ops_selftest` 23/23；前端 652 modules 构建通过 |

截图：`docs/ops/shots/audit/b2-inspector.png`（含检查器全景）、`b2-inspector-zoom.png`（候选块放大）、
`b2-dashed.png` / `b2-dashed-zoom.png`（`#140 ≈` 虚线）、`b2-off.png`（关闭后）、`b2-single.png`（单机位）。

### 23.7 顺手补的验收工具

`tools/cdp_shot.mjs`：用 CDP（`Page.captureScreenshot`）截图，替代 `chrome --screenshot`。
三个理由都是实测撞上的：① 旧 headless 不解码视频 → 六路全黑；② `--headless=new` 能出画但 **canvas 叠加层被丢掉**，检测框看不见；
③ `--virtual-time-budget` 快进时钟，长视频 seek 到中段取不到帧。
现在会顺带回传诊断（video readyState / canvas 非透明像素占比 / 容器与 stage 尺寸），一眼能分清"数据没画"和"截图没截到"。

### 23.8 已知边界

1. **不合并实体**。虚线只是提示；接力链、C0、实体数都不受影响。要真合并仍需 C 方案（重跑检测）。
2. **只在同机位内**。跨机位的换号由接力链负责，两者不交叉。
3. **弱候选可能有误报**。`?`（weak）档共 25 组，判据是 cos 0.85~0.88 或 gap 2~4s；页面用不同后缀区分，不做同等强调。
4. **远程预览无此功能**：远程数据包没有 `suspects.json`，`/api/suspects` 返回空对象，画布退回实线（不报错）。

### 23.9 展示层修复：六路画面从 199×112 到 322×181

**问题**（体检时量出来的）：1600×1000 下每格 `.cam-body` 是 488×112，16:9 的 stage 只能取 **199×112**，
每格约 60% 宽度是黑边——画面像邮票。

两处改动，都只动布局：

1. **甘特默认折叠**（`TimeController.openTable` 默认 `false`）。折叠后腾出约 105px 高度，
   并用一条 **24px 的「在场人数」柱状**替代 6 行甘特：120 个时间桶，柱高 = 该时段六路在场实体数。
   点/拖这条也能定位（实测点 50% 处 → `currentTime=1050` = 2100×0.5，精确）。
2. **容器查询阈值 1000 → 900**。1600 视口下中列实际约 990px，**差 10px** 就掉回 2 列 × 3 行；
   2×3 的格子是 488×147（3.3:1），3×2 的格子是 325×183（≈16:9）。

| 状态 | `.cam-body` | stage | 相对最初 |
|---|---|---|---|
| 改前（甘特展开 + 2×3） | 488×112 | 199×112 | — |
| 只折叠甘特（2×3） | 488×147 | 261×147 | +72% 面积 |
| **折叠甘特 + 3×2（当前）** | **322×237** | **322×181** | **+161% 面积，且 16:9 贴合** |
| 单机位（按 `1`） | 986×466 | 828×466 | 改前 654×368 |

★ 3×2 下展开甘特画面**不变小**（466→466 高，181 高受宽度限制），说明两种状态都稳。

**踩到的坑**：`<svg viewBox="0 0 120 20" preserveAspectRatio="none">` 若不给元素本身 `width/height`，
浏览器按默认 300×150 渲染，被 24px 的容器裁掉后只剩最高的两根柱露头——看着像"数据只有 2 个点"。
实际 120 根柱全在（DOM 查得到高度 2.44~18）。加 `.dens-track svg { width:100%; height:100% }` 解决。

截图：`docs/ops/shots/audit/b3-collapsed.png`（六路 + 人数柱）、`b3-density-zoom.png`（柱状放大）、
`b3-single.png`（单机位 828×466，含 `#140 ≈` 虚线）。
