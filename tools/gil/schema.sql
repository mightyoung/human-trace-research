-- GIL (Global Identity Layer) 统一存储 schema
-- 设计原则：
--   1) 一个 SQLite 文件 = 全部结构化真相（元数据/图/状态/标注），事务一致、可 diff、可备份
--   2) 向量不入 SQLite（实测 sqlite-vec 比 numpy 慢 11–57 倍），走 gil_vectors/ 分块矩阵
--   3) 行号即 ID：obs.id 一经分配永不复用；删除用 tombstone（保住版本链可复现）
--   4) 四问各有专属表：遮挡接续=frag/edge(stitch)、全局身份=gperson/membership、
--      去重与责任=visit/owner、镜像倒影=obs.flags（打 flag 不删框）
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS gil_meta (
  k TEXT PRIMARY KEY,
  v TEXT
);

CREATE TABLE IF NOT EXISTS camera (
  cam      TEXT PRIMARY KEY,
  w        INTEGER DEFAULT 0,
  h        INTEGER DEFAULT 0,
  fps      REAL DEFAULT 25.0,
  t_offset REAL DEFAULT 0.0,   -- 相对基准时钟的秒偏移（OSD 标定）
  role     TEXT,               -- overlap | portal | plain
  meta     TEXT
);

-- 观测：最细粒度单元（一条 track 的一段连续出现）
CREATE TABLE IF NOT EXISTS obs (
  id        INTEGER PRIMARY KEY,   -- == 向量矩阵行号，append-only
  key       TEXT UNIQUE NOT NULL,  -- f"{date}|{cam}|{tid}"
  cam       TEXT NOT NULL,
  date      TEXT NOT NULL,
  tid       INTEGER NOT NULL,
  t_first   REAL, t_last REAL,
  u         REAL, v REAL, h_med REAL,
  n_frames  INTEGER,
  quality   REAL,                  -- 0..1 输入质量（清晰度/可见性）
  flags     TEXT,                  -- JSON {"reflection":0.8,...}：打标不删框
  frag_id   INTEGER,
  entity_id INTEGER,
  tombstone INTEGER DEFAULT 0,
  src       TEXT
);
CREATE INDEX IF NOT EXISTS ix_obs_cam_t  ON obs(cam, t_first);
CREATE INDEX IF NOT EXISTS ix_obs_entity ON obs(entity_id);
CREATE INDEX IF NOT EXISTS ix_obs_cam_tid ON obs(cam, tid);

-- 同机位片段（Q1：遮挡断链后的接续单位）
CREATE TABLE IF NOT EXISTS frag (
  id      INTEGER PRIMARY KEY,
  cam     TEXT NOT NULL,
  t0 REAL, t1 REAL,
  obs_ids TEXT NOT NULL,           -- JSON array（升序）
  src     TEXT
);
CREATE INDEX IF NOT EXISTS ix_frag_cam_t ON frag(cam, t0);

-- 同机位实体（= 现 net_tracks 口径，由 frag 拼接）
CREATE TABLE IF NOT EXISTS entity (
  id     INTEGER PRIMARY KEY,
  cam    TEXT NOT NULL,
  eid    TEXT NOT NULL,
  t0 REAL, t1 REAL,
  n_frag INTEGER, n_obs INTEGER,
  src    TEXT,
  UNIQUE(cam, eid)
);
CREATE INDEX IF NOT EXISTS ix_entity_cam_t ON entity(cam, t0);

-- ★ 全局人（Q2/Q3：跨机位同一人的唯一身份）
CREATE TABLE IF NOT EXISTS gperson (
  id       INTEGER PRIMARY KEY,
  label    TEXT UNIQUE,            -- g0001
  state    TEXT DEFAULT 'candidate',-- candidate|active|audit|blocked|rejected|confirmed
  n_visit  INTEGER DEFAULT 0,
  first_t  REAL, last_t REAL,
  note     TEXT
);

-- 归属：节点 → 全局人（层次聚类产物，允许同一节点多候选）
CREATE TABLE IF NOT EXISTS membership (
  id         INTEGER PRIMARY KEY,
  gperson_id INTEGER NOT NULL REFERENCES gperson(id) ON DELETE CASCADE,
  node_kind  TEXT NOT NULL,        -- obs|frag|entity
  node_id    INTEGER NOT NULL,
  score      REAL,                 -- 外观相似
  prior      REAL,                 -- 门户/时间先验
  decision   TEXT,                 -- auto|weak|human
  src        TEXT,
  UNIQUE(node_kind, node_id, gperson_id)
);
CREATE INDEX IF NOT EXISTS ix_mem_g    ON membership(gperson_id);
CREATE INDEX IF NOT EXISTS ix_mem_node ON membership(node_kind, node_id);

-- 边：接续/转移（kind 区分同机位与跨机位，状态机统一）
CREATE TABLE IF NOT EXISTS edge (
  id       INTEGER PRIMARY KEY,
  kind     TEXT NOT NULL,          -- stitch(同机位) | relay(跨机位) | order(时间相邻)
  a_kind TEXT, a_id INTEGER,
  b_kind TEXT, b_id INTEGER,
  dt REAL, z REAL, score REAL,
  status   TEXT,                   -- active|audit|blocked|rejected
  tier     TEXT,
  evidence TEXT,                   -- JSON（sim/h/端点/来源文件）
  UNIQUE(kind, a_kind, a_id, b_kind, b_id)
);
CREATE INDEX IF NOT EXISTS ix_edge_status ON edge(kind, status);
CREATE INDEX IF NOT EXISTS ix_edge_a ON edge(a_kind, a_id);
CREATE INDEX IF NOT EXISTS ix_edge_b ON edge(b_kind, b_id);

-- 门户通行事件（进出画面）
CREATE TABLE IF NOT EXISTS visit (
  id         INTEGER PRIMARY KEY,
  cam        TEXT NOT NULL,
  kind       TEXT,                 -- entry|exit
  t          REAL,
  u REAL, v REAL, dir REAL,
  edge_margin REAL,
  obs_id     INTEGER,
  entity_id  INTEGER,
  gperson_id INTEGER REFERENCES gperson(id)
);
CREATE INDEX IF NOT EXISTS ix_visit_cam_t ON visit(cam, t);
CREATE INDEX IF NOT EXISTS ix_visit_g ON visit(gperson_id);

-- ★ 责任时间线（Q3：任一时刻单机位负责一目标）
CREATE TABLE IF NOT EXISTS owner (
  id         INTEGER PRIMARY KEY,
  t0 REAL, t1 REAL,
  cam        TEXT NOT NULL,
  gperson_id INTEGER REFERENCES gperson(id),
  visit_id   INTEGER,
  utility    REAL,
  reason     TEXT                  -- visibility|handoff|fallback
);
CREATE INDEX IF NOT EXISTS ix_owner_t ON owner(t0, t1);
CREATE INDEX IF NOT EXISTS ix_owner_cam ON owner(cam, t0);

-- 统一人工真值（替代散落的 csv/json）
CREATE TABLE IF NOT EXISTS annotation (
  id          INTEGER PRIMARY KEY,
  target_kind TEXT NOT NULL,       -- pair|event|obs|gperson|owner
  target_id   TEXT NOT NULL,
  value       TEXT,
  confidence  TEXT,
  note        TEXT,
  author      TEXT,
  ts          TEXT,
  UNIQUE(target_kind, target_id)
);

-- 时间互斥 / 禁止同人（约束）
CREATE TABLE IF NOT EXISTS mutex (
  a_kind TEXT, a_id INTEGER,
  b_kind TEXT, b_id INTEGER,
  reason TEXT,
  UNIQUE(a_kind, a_id, b_kind, b_id)
);
