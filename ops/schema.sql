-- human-count 算法中台台账 · schema v1
-- 权威库：ops/ops.db（本机为主库；远程只做算力与预览）
-- 原则：append-only；大文件只登记 uri + sha1；口径随行（dialect 进每一层）

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS meta (
  k TEXT PRIMARY KEY,
  v TEXT
);

-- ---------------------------------------------------------------- 数据域
CREATE TABLE IF NOT EXISTS dataset (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  kind          TEXT,                 -- video_set | crop_set | mixed
  root_uri      TEXT,
  file_count    INTEGER DEFAULT 0,
  total_bytes   INTEGER DEFAULT 0,
  dialect       TEXT,                 -- clean2 | raw | ...（口径随行）
  manifest_uri  TEXT,
  snapshot_hash TEXT,                 -- manifest 的 sha1；run 引用它
  hash_alg      TEXT,                 -- 算法标识（recompute 前必须一致）
  created_at    TEXT NOT NULL,
  note          TEXT
);

CREATE TABLE IF NOT EXISTS asset (
  id         TEXT PRIMARY KEY,
  dataset_id TEXT REFERENCES dataset(id),
  kind       TEXT,                    -- image|video|clip|track_crop|frame
  uri        TEXT,
  cam        TEXT,
  tid        INTEGER,
  t0         REAL,
  t1         REAL,
  w          INTEGER,
  h          INTEGER,
  sha1       TEXT,
  bytes      INTEGER,
  meta_json  TEXT
);
CREATE INDEX IF NOT EXISTS ix_asset_dataset ON asset(dataset_id);
CREATE INDEX IF NOT EXISTS ix_asset_cam ON asset(cam, tid);

CREATE TABLE IF NOT EXISTS annotation_task (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  kind       TEXT NOT NULL,           -- pair|mirror|mask|face_person|face_pair|truth_pair|truth_event|s6b
  dataset_id TEXT REFERENCES dataset(id),
  spec_json  TEXT,                    -- 队列生成规则（seed + 规则版本）
  dialect    TEXT,
  mode       TEXT DEFAULT 'single',   -- single|double|adjudicate
  assignee   TEXT,
  budget     INTEGER,
  due_at     TEXT,
  status     TEXT DEFAULT 'open',     -- open|done|paused
  created_by TEXT,
  created_at TEXT NOT NULL,
  note       TEXT
);

CREATE TABLE IF NOT EXISTS annotation_item (
  id             TEXT PRIMARY KEY,    -- {kind}:{原键}
  task_id        TEXT NOT NULL REFERENCES annotation_task(id),
  asset_refs_json TEXT,
  payload_json   TEXT,
  stratum        TEXT DEFAULT 'topk', -- topk|audit|conflict|boundary|golden
  priority       INTEGER DEFAULT 0,
  status         TEXT DEFAULT 'pending',
  gold_value     TEXT,
  created_at     TEXT
);
CREATE INDEX IF NOT EXISTS ix_item_task ON annotation_item(task_id, status);

CREATE TABLE IF NOT EXISTS label (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id    TEXT NOT NULL REFERENCES annotation_item(id),
  annotator  TEXT NOT NULL,
  value      TEXT,                    -- 归一化的判定主值（跨任务可比：0/1/2、CONF/REJ、pid:28…）
  confidence REAL,
  seconds    REAL,
  ui_version TEXT,
  created_at TEXT NOT NULL,
  supersedes INTEGER REFERENCES label(id),
  note       TEXT,                    -- 来源标记（imported/mirror/cleared），**不是**用户备注
  data_json  TEXT                     -- ★ 原始判定对象：{v,c,n} / {pid,fp} / {verdict,name} / 多边形
);
-- data_json 是「证据」：value 只保主值，置信度/备注/双字段并存靠它无损往返。
-- 迁移前的行 data_json 为 NULL，可用 tools/ops_backfill_labeldata.py 从原始 json 补齐。
CREATE INDEX IF NOT EXISTS ix_label_item ON label(item_id, id DESC);
-- 幂等兜底：同 item+作者+值+时刻不重复。含 value 是为了允许「同一秒内改判」不被吞掉。
CREATE UNIQUE INDEX IF NOT EXISTS ux_label_ver ON label(item_id, annotator, value, created_at);

CREATE TABLE IF NOT EXISTS adjudication (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id     TEXT NOT NULL REFERENCES annotation_item(id),
  final_value TEXT,
  by_whom     TEXT,
  reason      TEXT,
  at          TEXT NOT NULL
);

-- ---------------------------------------------------------------- 模型域
CREATE TABLE IF NOT EXISTS model (
  id             TEXT PRIMARY KEY,
  name           TEXT NOT NULL,
  version        TEXT,
  backbone       TEXT,
  weights_uri    TEXT,
  sha1           TEXT,
  license        TEXT,
  trained_by_run TEXT,                -- 软引用 run.id（避免与 run.model_id 形成循环外键）
  eval_json      TEXT,
  deployable     INTEGER DEFAULT 0,
  created_at     TEXT
);

CREATE TABLE IF NOT EXISTS run (
  id           TEXT PRIMARY KEY,
  name         TEXT,
  kind         TEXT,                  -- infer|encode|train|eval|build|export|other
  entrypoint   TEXT,
  params_json  TEXT,
  env_json     TEXT,
  dataset_id   TEXT REFERENCES dataset(id),
  model_id     TEXT REFERENCES model(id),
  host         TEXT,
  gpu          TEXT,
  git_rev      TEXT,
  status       TEXT NOT NULL,         -- queued|running|succeeded|failed|killed|stale|superseded
  started_at   TEXT,
  ended_at     TEXT,
  exit_code    INTEGER,
  log_uri      TEXT,
  metrics_json TEXT,
  out_dir      TEXT,
  input_hash      TEXT,               -- 输入指纹：dataset+model+code，stale 判定用
  input_breakdown TEXT,               -- 指纹各分量，出问题时能说清哪个变了
  card_id      TEXT,                  -- 执行准入：挂在哪张 exp_card 下（不设外键：exp_card 在后文定义）
  card_role    TEXT,                  -- baseline|variant|ablation|explore
  note         TEXT,
  created_at   TEXT
);
CREATE INDEX IF NOT EXISTS ix_run_started ON run(started_at DESC);
CREATE INDEX IF NOT EXISTS ix_run_status ON run(status);

CREATE TABLE IF NOT EXISTS artifact (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id       TEXT NOT NULL REFERENCES run(id),
  kind         TEXT,
  uri          TEXT NOT NULL,
  bytes        INTEGER,
  sha1         TEXT,
  summary_json TEXT,
  at           TEXT
);
CREATE INDEX IF NOT EXISTS ix_artifact_run ON artifact(run_id);

CREATE TABLE IF NOT EXISTS experiment (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  hypothesis  TEXT,
  design_json TEXT,
  dataset_id  TEXT REFERENCES dataset(id),
  status      TEXT DEFAULT 'planned', -- planned|doing|done|blocked
  owner       TEXT,
  conclusion  TEXT,
  doc_uri     TEXT,
  created_at  TEXT
);

CREATE TABLE IF NOT EXISTS experiment_run (
  experiment_id TEXT NOT NULL REFERENCES experiment(id),
  run_id        TEXT NOT NULL REFERENCES run(id),
  role          TEXT DEFAULT 'variant', -- baseline|variant|ablation
  PRIMARY KEY (experiment_id, run_id)
);

-- ---------------------------------------------------------------- 调度域
CREATE TABLE IF NOT EXISTS job (
  id            TEXT PRIMARY KEY,
  kind          TEXT,                 -- label|run|review|write|fix
  title         TEXT NOT NULL,
  goal          TEXT,
  inputs_json   TEXT,
  acceptance_json TEXT,
  assignee_kind TEXT DEFAULT 'human', -- human|agent
  assignee      TEXT,
  status        TEXT DEFAULT 'backlog',
  priority      INTEGER DEFAULT 0,
  budget        TEXT,
  authorization TEXT,
  created_at    TEXT,
  closed_at     TEXT,
  evidence_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_job_status ON job(status, priority DESC);

CREATE TABLE IF NOT EXISTS audit (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  actor       TEXT,
  action      TEXT,
  object_type TEXT,
  object_id   TEXT,
  before_json TEXT,
  after_json  TEXT,
  at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_audit_obj ON audit(object_type, object_id, id DESC);


-- ---------------------------------------------------------------- 训练曲线
-- 每个 (run, step, name) 一条；训练脚本按 epoch 写入，评测脚本可按样本块写入。
CREATE TABLE IF NOT EXISTS run_metric (
  run_id TEXT NOT NULL REFERENCES run(id),
  step   INTEGER NOT NULL,
  name   TEXT NOT NULL,
  value  REAL,
  PRIMARY KEY (run_id, step, name)
);
CREATE INDEX IF NOT EXISTS ix_run_metric ON run_metric(run_id, name, step);

-- ---------------------------------------------------------------- 视图
CREATE VIEW IF NOT EXISTS v_task_progress AS
SELECT t.id            AS task_id,
       t.name          AS name,
       t.kind          AS kind,
       COUNT(i.id)     AS items,
       SUM(CASE WHEN i.status IN ('labeled','reviewed','adjudicated') THEN 1 ELSE 0 END) AS done,
       SUM(CASE WHEN i.status = 'pending' THEN 1 ELSE 0 END) AS pending
FROM annotation_task t
LEFT JOIN annotation_item i ON i.task_id = t.id
GROUP BY t.id;

-- ---------------------------------------------------------------- 科研工作台（实验卡）
-- 与 experiment 表分工：experiment 由 ops_ingest 自动汇总训练变体（无假设/无审批）；
-- exp_card 是「人立项 · AI 执行 · 四关审批」的研究卡，两者并存、互不覆写。
CREATE TABLE IF NOT EXISTS exp_card (
  id            TEXT PRIMARY KEY,     -- EXP-001
  title         TEXT NOT NULL,
  question      TEXT,                 -- 一个主要问题
  hypothesis    TEXT,                 -- 可证伪假设
  baseline_json TEXT,                 -- {"text":..., "runs":[]}
  candidate_json TEXT,                -- {"text":..., "runs":[]}
  metric_json   TEXT,                 -- {"primary":..., "secondary":[], "failure":...}
  budget_json   TEXT,                 -- {"gpu_h":..., "human_min":..., "data":...}
  roles_json    TEXT,                 -- {"human":[...], "agent":[...]}
  evidence_json TEXT,                 -- {"items":[{"k","label","done","note"}]}
  runs_json     TEXT,                 -- [{"run_id","role"}] role=baseline|variant|ablation
  state         TEXT DEFAULT 'draft', -- 状态机见 workbench_api.STATE_FLOW
  risk          TEXT,                 -- 口径风险/前置条件（人写）
  source        TEXT,                 -- 出处（文件名 + 日期）
  created_at    TEXT,
  updated_at    TEXT
);

-- 每个关口可重复记录（append-only）：最新一条即当前决定
CREATE TABLE IF NOT EXISTS exp_gate (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  card_id  TEXT NOT NULL REFERENCES exp_card(id),
  gate     TEXT NOT NULL,             -- A 立项 | B 数据与真值 | C 边界变化 | D 成果验收
  decision TEXT NOT NULL,             -- approve | return | hold
  note     TEXT,
  actor    TEXT,
  at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_exp_gate ON exp_gate(card_id, gate, id DESC);

-- ---------------------------------------------------------------- 论文数字台账（v2.1，V1）
-- 权威是 docs/paper/EVIDENCE.md；本表是索引，不接管写入。见 docs/ledger-v2-design-2026-11-24.md §5。
-- 注：原 paper_claim 已升格为通用的 claim（见文件末尾「通用流程层」）。
-- 老库由 tools/ledger_migrate.py 迁移后保留为 paper_claim_v1 备份，读写改走 claim；
-- 读取侧沿用视图 v_paper_claim（paper_tbiom 适配器投影，见 ops/views/paper_tbiom.sql）。

-- 文件级版本：path + 内容 sha256。逐条素材（cam/tid/t0/t1）在 asset 表，两者不同粒度。
CREATE TABLE IF NOT EXISTS asset_version (
  id            TEXT PRIMARY KEY,        -- av_<sha1(path|sha256)[:10]>
  path          TEXT NOT NULL,
  sha256        TEXT,
  bytes         INTEGER,
  mtime         INTEGER,
  kind          TEXT,                    -- media|weights|labels|derived|script|doc|external_dataset
  hash_alg      TEXT,                    -- sha256|size-mtime
  weak          INTEGER DEFAULT 0,
  first_seen_at TEXT NOT NULL,
  note          TEXT
);
CREATE INDEX IF NOT EXISTS ix_asset_version_path ON asset_version(path);

-- ---------------------------------------------------------------- 真值与评测协议（v2.1，V2）
-- label_set：一批人工判定的真值。条数等计数由 ops_claims truth 从文件实测，不抄文档。
CREATE TABLE IF NOT EXISTS label_set (
  id            TEXT PRIMARY KEY,        -- ls_<sha1(name|sha256)[:10]>
  name          TEXT NOT NULL,           -- T1|T2|judged_merges|audited_groups|gallery_double
  sampling      TEXT NOT NULL,           -- 抽样方式（T1 先验抽样 / T2 错误触发）
  domain        TEXT,                    -- D1|D2|D3
  uri           TEXT NOT NULL,
  sha256        TEXT,
  n_items       INTEGER,
  n_same        INTEGER,
  n_diff        INTEGER,
  n_unlabeled   INTEGER,
  annotator     TEXT,
  reviewer      TEXT,
  rule_json     TEXT,                    -- 计数规则与实测分布（含各口径变体）
  note          TEXT,
  created_at    TEXT NOT NULL,
  UNIQUE(name, sha256)
);

-- eval_protocol：一次正式评测的协议与冻结包。status=recorded 表示"当时就是这么算的"。
CREATE TABLE IF NOT EXISTS eval_protocol (
  id                 TEXT PRIMARY KEY,   -- ep_<sha1(name|manifest_sha256)[:10]>
  name               TEXT NOT NULL,
  status             TEXT,               -- recorded|sealed
  sampling           TEXT,
  identity_isolation TEXT,               -- none|time|identity
  params_json        TEXT,
  manifest_uri       TEXT,
  manifest_sha256    TEXT,
  file_count         INTEGER,
  verify_json        TEXT,               -- {checked, ok, missing, mismatch}
  note               TEXT,
  created_at         TEXT NOT NULL
);

-- code_version：一次复算用到的代码/配方快照（内容 sha256 列表 → sha1）。V3 起。
CREATE TABLE IF NOT EXISTS code_version (
  id          TEXT PRIMARY KEY,          -- cv_<sha1(files_json)[:10]>
  sha1        TEXT,                      -- files_json 的 sha1
  files_json  TEXT,                      -- [{path, sha256}]
  as_is       INTEGER DEFAULT 1,         -- 是否原样使用（未针对性调参）
  created_at  TEXT NOT NULL
);

-- eval_run：一次复算 = 一行。约束：verdict=match 只能由 exit_code=0 且带输入/输出哈希的 run 写入。
CREATE TABLE IF NOT EXISTS eval_run (
  id              TEXT PRIMARY KEY,      -- er_<sha1(claim|spec|inputs)[:10]>
  claim_ids_json  TEXT,                  -- ["pc_..."]；一条 run 可服务多条 claim
  script          TEXT,                  -- 执行器
  script_sha1     TEXT,
  as_is           INTEGER DEFAULT 1,
  args_json       TEXT,                  -- 该条配方（checks 原文）
  cwd             TEXT,
  started_at      TEXT,
  ended_at        TEXT,
  exit_code       INTEGER,               -- 0=复算跑通
  code_version_id TEXT,
  inputs_json     TEXT,                  -- [{path, sha256}]
  outputs_json    TEXT,                  -- {recomputed, verdict}
  env_json        TEXT,                  -- {python, tool_sha256}
  created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_eval_run_cv ON eval_run(code_version_id);

-- ================================================================ 通用流程层（K0，2026-11-24）
-- 见 docs/workbench-flow-model-2026-11-24.md §6。这一层不出现任何项目名词：
-- 项目相关的字段一律进 ext_json，由适配器自己解释（内核不读）。
-- 迁移：tools/ledger_migrate.py（paper_claim → claim + claim_anchor，老表改名 paper_claim_v1）。

-- study：一次研究/项目。工作台的第一层容器，experiment 是它的执行记录。
CREATE TABLE IF NOT EXISTS study (
  id          TEXT PRIMARY KEY,          -- st_<slug>
  name        TEXT NOT NULL,
  question    TEXT,                      -- 这项研究要回答的总问题
  owner       TEXT,
  status      TEXT DEFAULT 'active',     -- active|paused|closed
  started_at  TEXT,
  closed_at   TEXT,
  note        TEXT,
  created_at  TEXT NOT NULL
);

-- claim：一条可核验的陈述。论文表格里的一个数、报告里的一句话、看板上一个指标，都是 claim。
-- 守恒：C1 能回答「属于哪张实验卡」（card_id 空则显式 orphan）、C2 能回答「依据哪次 run」。
CREATE TABLE IF NOT EXISTS claim (
  id              TEXT PRIMARY KEY,      -- cl_<sha1(study|locator|value)[:10]>；迁移沿用 pc_* 不破坏引用
  study_id        TEXT REFERENCES study(id),
  card_id         TEXT REFERENCES exp_card(id),   -- 空 = orphan，报告里单列
  locator         TEXT,                  -- 人读定位符，格式由适配器定：E01-004 / R3-Table2-1
  kind            TEXT,                  -- metric|fact|comparison|constraint
  statement       TEXT,                  -- 一句话陈述（人读）
  value_text      TEXT,                  -- 被核验的值（原样字符串）
  unit            TEXT,
  stratum         TEXT,                  -- 分层标签，取值由适配器定（原 domain）
  evidence_grade  TEXT,                  -- data|doc|none（原 source_level）
  verdict         TEXT DEFAULT 'pending',-- pending|match|mismatch|unverifiable|retired
  reason          TEXT,
  recomputed_text TEXT,
  eval_run_id     TEXT REFERENCES eval_run(id),
  provenance_json TEXT,                  -- 6 要件逐项（通用：来源/脚本/步骤/复算/分母/域）
  ext_json        TEXT,                  -- 适配器私有字段，内核不解释
  card_state      TEXT,                  -- NULL=待归属；'n/a'=豁免（说明/日期/总览类）；归属本身在 claim_card
  created_at      TEXT NOT NULL,
  updated_at      TEXT
);
CREATE INDEX IF NOT EXISTS ix_claim_study ON claim(study_id, verdict);
CREATE INDEX IF NOT EXISTS ix_claim_card ON claim(card_id);
CREATE INDEX IF NOT EXISTS ix_claim_loc ON claim(study_id, locator);

-- claim_anchor：一条 claim 可以有多个出处锚点（正文一处、补充材料一处、看板一处）。
-- type 决定 ref_json 的结构，由适配器的 resolver 解释；内核只保证「能定位、能记哈希」。
CREATE TABLE IF NOT EXISTS claim_anchor (
  id            TEXT PRIMARY KEY,        -- ca_<sha1(claim|type|ref)[:10]>
  claim_id      TEXT NOT NULL REFERENCES claim(id),
  type          TEXT NOT NULL,           -- tex_table|md_heading|csv_cell|json_path|url|file_line
  ref_json      TEXT NOT NULL,           -- {file,table,row,key} / {file,line} / {url,selector}
  resolver      TEXT,                    -- 适配器名，如 paper_tbiom
  resolved_sha1 TEXT,                    -- 锚点文件在登记时的 sha1 前缀（变了就能发现）
  created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_anchor_claim ON claim_anchor(claim_id);
CREATE INDEX IF NOT EXISTS ix_anchor_file ON claim_anchor(type, resolver);

-- deliverable：交付物。论文/报告/看板/数据集发布都是它，论文只是 kind='paper'。
CREATE TABLE IF NOT EXISTS deliverable (
  id            TEXT PRIMARY KEY,        -- dl_<sha1(study|name|version)[:10]>
  study_id      TEXT REFERENCES study(id),
  name          TEXT NOT NULL,
  kind          TEXT,                    -- paper|report|dataset|dashboard
  version       TEXT,                    -- r2 / r3 / 2026-11-24
  path          TEXT,
  sha256        TEXT,
  manifest_uri  TEXT,
  status        TEXT DEFAULT 'draft',    -- draft|submitted|published|archived
  created_at    TEXT NOT NULL
);

-- claim_card：claim 与实验卡的多对多归属（K0 增补，Q3/Q9）。
-- role=primary 至多一个（部分唯一索引保证）；C1 的分子只数 primary。
-- source 记归属是怎么来的：section（按章节继承）/ table（按 tex 表继承）/ manual（人工判定）。
CREATE TABLE IF NOT EXISTS claim_card (
  id         TEXT PRIMARY KEY,          -- cc_<sha1(claim|card|role)[:10]>
  claim_id   TEXT NOT NULL REFERENCES claim(id),
  card_id    TEXT NOT NULL REFERENCES exp_card(id),
  role       TEXT NOT NULL DEFAULT 'primary',   -- primary|supporting
  source     TEXT,                      -- section|table|manual
  note       TEXT,
  created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_claim_card_primary
  ON claim_card(claim_id) WHERE role='primary';
CREATE INDEX IF NOT EXISTS ix_claim_card_card ON claim_card(card_id, role);

-- deliverable_ref：交付物里哪一处引用了哪条 claim。投稿前查「多少句有底」就查这张表。
-- anchor_id（K0 增补，Q12）：能挂到具体锚点就挂，location 降为展示文本；
-- C4 报告分「锚定引用」与「纯文本引用」两列。
CREATE TABLE IF NOT EXISTS deliverable_ref (
  id             TEXT PRIMARY KEY,       -- dr_<sha1(deliverable|claim|location)[:10]>
  deliverable_id TEXT NOT NULL REFERENCES deliverable(id),
  claim_id       TEXT NOT NULL REFERENCES claim(id),
  location       TEXT,                   -- 第 5 页表 2 第 1 行 / §4.2 / 看板:身份完整度
  anchor_id      TEXT REFERENCES claim_anchor(id),
  created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_dref_dl ON deliverable_ref(deliverable_id);
CREATE INDEX IF NOT EXISTS ix_dref_claim ON deliverable_ref(claim_id);
