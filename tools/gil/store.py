# -*- coding: utf-8 -*-
"""GIL 统一存储（Global Identity Layer store）。

一个 store 目录 = 一份**可复现的**身份真相：
    gil.sqlite         全部结构化真相（obs/frag/entity/gperson/membership/edge/visit/owner/annotation）
    gil_vectors/*.npz  分块向量（按机位分块；f16 落盘、f32 常驻）
    manifest.json      版本/维度/行数/文件 sha1 —— 版本链与复现依据

设计取舍（均有实测/文献依据，见 docs/gil-storage-design.md）：
  * 结构化真相放 SQLite：事务一致（一次改造原子提交）、可 diff、单文件备份、
    关系+时间查询天然（"t 时刻谁可见"），且远程已有 SQLite 3.51.2。
  * 向量**不放** SQLite：实测 sqlite-vec 比 numpy 精确扫描慢 11–57 倍；
    本项目 10^3–10^5 规模下 numpy 暴力检索 0.1–17 ms，强过滤下 0.03 ms。
  * 数据视图（DET_VIEW）× 版本（v5/v6/…）用 gil_meta 显式记录，防口径漂移。
"""
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

from .vectors import VectorBank, DEFAULT_DIM

SCHEMA = Path(__file__).parent / "schema.sql"
CAMS = ["305", "316", "318", "320", "323", "343"]
# OSD 标定：相对基准时钟的秒偏移
TIME_OFFS = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}


def _sha16(path, limit=None):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit and h.digest() and f.tell() > limit:
                break
    return h.hexdigest()[:16]


class GILStore:
    """GIL 统一存储读写入口。"""

    def __init__(self, root, dim=DEFAULT_DIM, create=True):
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "gil.sqlite"
        self.vec_root = self.root / "gil_vectors"
        self.dim = dim
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        if create:
            self.con.executescript(SCHEMA.read_text(encoding="utf-8"))
            self.con.commit()
        self.vb = None

    # ---------------- 元信息 ----------------
    def meta_set(self, **kv):
        self.con.executemany("INSERT OR REPLACE INTO gil_meta(k,v) VALUES(?,?)",
                             [(k, json.dumps(v, ensure_ascii=False)) for k, v in kv.items()])
        self.con.commit()

    def meta_get(self, k, default=None):
        r = self.con.execute("SELECT v FROM gil_meta WHERE k=?", (k,)).fetchone()
        return json.loads(r["v"]) if r else default

    # ---------------- 相机 ----------------
    def upsert_camera(self, cam, w=0, h=0, fps=25.0, role=None, meta=None):
        t_off = TIME_OFFS.get(cam, 0.0)
        self.con.execute(
            "INSERT INTO camera(cam,w,h,fps,t_offset,role,meta) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(cam) DO UPDATE SET w=excluded.w,h=excluded.h,fps=excluded.fps,"
            "t_offset=excluded.t_offset,role=COALESCE(excluded.role,camera.role),meta=excluded.meta",
            (cam, w, h, fps, t_off, role, json.dumps(meta, ensure_ascii=False) if meta else None))

    # ---------------- 观测 ----------------
    def add_obs(self, key, cam, date, tid, t_first=None, t_last=None, u=None, v=None,
                h_med=None, n_frames=None, quality=None, flags=None, src=None):
        cur = self.con.execute(
            "INSERT OR IGNORE INTO obs(key,cam,date,tid,t_first,t_last,u,v,h_med,n_frames,quality,flags,src)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, cam, date, int(tid), t_first, t_last, u, v, h_med, n_frames,
             quality, json.dumps(flags, ensure_ascii=False) if flags else None, src))
        if cur.lastrowid:
            return int(cur.lastrowid)
        r = self.con.execute("SELECT id FROM obs WHERE key=?", (key,)).fetchone()
        return int(r["id"]) if r else None

    def obs_row(self, obs_id):
        return self.con.execute("SELECT * FROM obs WHERE id=?", (obs_id,)).fetchone()

    def obs_id_of(self, cam, tid):
        r = self.con.execute("SELECT id FROM obs WHERE cam=? AND tid=?", (cam, int(tid))).fetchone()
        return int(r["id"]) if r else None

    # ---------------- 片段 / 实体 ----------------
    def add_frag(self, cam, obs_ids, t0=None, t1=None, src=None):
        cur = self.con.execute("INSERT INTO frag(cam,t0,t1,obs_ids,src) VALUES(?,?,?,?,?)",
                               (cam, t0, t1, json.dumps(sorted(int(x) for x in obs_ids)), src))
        fid = int(cur.lastrowid)
        self.con.executemany("UPDATE obs SET frag_id=? WHERE id=?",
                             [(fid, int(x)) for x in obs_ids])
        return fid

    def add_entity(self, cam, eid, obs_ids, t0=None, t1=None, n_frag=None, src=None):
        cur = self.con.execute(
            "INSERT OR IGNORE INTO entity(cam,eid,t0,t1,n_frag,n_obs,src) VALUES(?,?,?,?,?,?,?)",
            (cam, eid, t0, t1, n_frag, len(obs_ids), src))
        r = self.con.execute("SELECT id FROM entity WHERE cam=? AND eid=?", (cam, eid)).fetchone()
        eid_i = int(r["id"])
        self.con.executemany("UPDATE obs SET entity_id=? WHERE id=?",
                             [(eid_i, int(x)) for x in obs_ids])
        return eid_i

    def entity_of(self, cam, eid):
        return self.con.execute("SELECT * FROM entity WHERE cam=? AND eid=?", (cam, eid)).fetchone()

    # ---------------- 全局人 ----------------
    def add_gperson(self, label=None, state="candidate", note=None):
        if label is None:
            nxt = (self.con.execute("SELECT COALESCE(MAX(id),0)+1 FROM gperson").fetchone()[0])
            label = f"g{nxt:04d}"
        cur = self.con.execute("INSERT OR IGNORE INTO gperson(label,state,note) VALUES(?,?,?)",
                               (label, state, note))
        r = self.con.execute("SELECT id FROM gperson WHERE label=?", (label,)).fetchone()
        return int(r["id"])

    def add_membership(self, gperson_id, node_kind, node_id, score=None, prior=None,
                       decision="auto", src=None):
        self.con.execute(
            "INSERT OR REPLACE INTO membership(gperson_id,node_kind,node_id,score,prior,decision,src)"
            " VALUES(?,?,?,?,?,?,?)",
            (gperson_id, node_kind, int(node_id), score, prior, decision, src))

    def memberships_of(self, node_kind, node_id):
        return self.con.execute(
            "SELECT m.*, g.label g_label, g.state g_state FROM membership m "
            "JOIN gperson g ON g.id=m.gperson_id WHERE m.node_kind=? AND m.node_id=? "
            "ORDER BY COALESCE(m.score,0)+COALESCE(m.prior,0) DESC",
            (node_kind, int(node_id))).fetchall()

    def gperson_timeline(self, gid):
        """该全局人的全部观测（按时间排序）——身份卡统一后的时间线。"""
        return self.con.execute(
            "SELECT o.* FROM membership m JOIN obs o ON o.id=m.node_id "
            "WHERE m.node_kind='obs' AND m.gperson_id=? ORDER BY o.t_first", (gid,)).fetchall()

    # ---------------- 边 ----------------
    def add_edge(self, kind, a_kind, a_id, b_kind, b_id, dt=None, z=None, score=None,
                 status="active", tier=None, evidence=None):
        self.con.execute(
            "INSERT OR REPLACE INTO edge(kind,a_kind,a_id,b_kind,b_id,dt,z,score,status,tier,evidence)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (kind, a_kind, int(a_id), b_kind, int(b_id), dt, z, score, status, tier,
             json.dumps(evidence, ensure_ascii=False) if evidence else None))

    def edges(self, kind=None, status=None):
        sql, p = "SELECT * FROM edge WHERE 1=1", []
        if kind:
            sql += " AND kind=?"; p.append(kind)
        if status:
            sql += " AND status=?"; p.append(status)
        return self.con.execute(sql, p).fetchall()

    # ---------------- 访问 / 责任 ----------------
    def add_visit(self, cam, kind, t, u=None, v=None, dir=None, edge_margin=None,
                  obs_id=None, entity_id=None, gperson_id=None):
        cur = self.con.execute(
            "INSERT INTO visit(cam,kind,t,u,v,dir,edge_margin,obs_id,entity_id,gperson_id)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (cam, kind, t, u, v, dir, edge_margin, obs_id, entity_id, gperson_id))
        return int(cur.lastrowid)

    def add_owner(self, t0, t1, cam, gperson_id=None, visit_id=None, utility=None, reason=None):
        cur = self.con.execute(
            "INSERT INTO owner(t0,t1,cam,gperson_id,visit_id,utility,reason) VALUES(?,?,?,?,?,?,?)",
            (t0, t1, cam, gperson_id, visit_id, utility, reason))
        return int(cur.lastrowid)

    def owners_at(self, t):
        """t 时刻的 owner 列表（Q3 去重视图的数据源）。"""
        return self.con.execute(
            "SELECT * FROM owner WHERE t0<=? AND t1>=? ORDER BY utility DESC", (t, t)).fetchall()

    # ---------------- 标注 ----------------
    def add_annotation(self, target_kind, target_id, value, confidence=None, note=None,
                       author=None, ts=None):
        self.con.execute(
            "INSERT OR REPLACE INTO annotation(target_kind,target_id,value,confidence,note,author,ts)"
            " VALUES(?,?,?,?,?,?,?)",
            (target_kind, str(target_id), str(value) if value is not None else None,
             confidence, note, author, ts or time.strftime("%Y-%m-%d %H:%M:%S")))

    def annotations(self, kind=None, only_decided=False):
        sql, p = "SELECT * FROM annotation WHERE 1=1", []
        if kind:
            sql += " AND target_kind=?"; p.append(kind)
        if only_decided:
            sql += " AND value IN ('1','0')"
        return self.con.execute(sql, p).fetchall()

    # ---------------- 向量 ----------------
    def vectors(self, load=False):
        if self.vb is None:
            self.vb = VectorBank(self.vec_root, self.dim)
        # ★ load=True 必须**每次**生效：若块已在磁盘但未入缓存（如本进程刚 flush 过），
        #   旧实现只在首次创建时 load → 静默返回空缓存（曾致冒烟 n()=0）
        if load and not self.vb._cache:
            self.vb.load()
        return self.vb

    def search_similar(self, obs_id, cross_cam=True, t_window=None, topk=10, **kw):
        """给定观测，找相似观测。默认排除同机位（跨机位候选），可选时间窗。"""
        vb = self.vectors(load=True)
        o = self.obs_row(obs_id)
        if o is None:
            return []
        q = None
        b = o["cam"]
        E, ids, id2row = vb._cache.get(b, (None, None, None))
        if E is None or int(obs_id) not in id2row:
            return []
        q = E[id2row[int(obs_id)]]
        blocks = None
        if cross_cam and len(vb._cache) > 1:
            blocks = [x for x in vb._cache if x != b]
        allow = None
        if t_window is not None:
            rows = self.con.execute(
                "SELECT id FROM obs WHERE t_first BETWEEN ? AND ? AND tombstone=0",
                (t_window[0], t_window[1])).fetchall()
            allow = [int(r["id"]) for r in rows]
        return vb.search(q, blocks=blocks, allow_ids=allow, exclude_ids=[obs_id], topk=topk, **kw)

    # ---------------- 统计 / 版本 ----------------
    def stats(self):
        c = self.con
        # 先把缓存页落盘，否则 db_mb 会读到 0（SQLite 延迟写）
        try:
            c.commit()
            c.execute("PRAGMA wal_checkpoint(FULL)")
        except Exception:  # noqa: BLE001
            pass

        def n(t):
            return c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        by_cam = {r[0]: r[1] for r in c.execute(
            "SELECT cam,COUNT(*) FROM obs WHERE tombstone=0 GROUP BY cam")}
        return {"obs": n("obs"), "obs_alive": c.execute(
            "SELECT COUNT(*) FROM obs WHERE tombstone=0").fetchone()[0],
            "frag": n("frag"), "entity": n("entity"), "gperson": n("gperson"),
            "membership": n("membership"), "edge": n("edge"), "visit": n("visit"),
            "owner": n("owner"), "annotation": n("annotation"),
            "by_cam": by_cam, "db_mb": round(self.db_path.stat().st_size / 1e6, 2)}

    def snapshot(self, extra=None):
        """写 manifest.json（版本链）：文件 sha1 + 各表行数。"""
        m = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
             "dim": self.dim, "stats": self.stats(), "files": {}}
        for p in ([self.db_path] + sorted(self.vec_root.glob("*.npz"))):
            m["files"][p.name] = {"size": p.stat().st_size, "sha1_16": _sha16(p)}
        vb = self.vectors()
        m["vector_blocks"] = vb.stats()
        m["meta"] = {r["k"]: json.loads(r["v"]) for r in
                     self.con.execute("SELECT k,v FROM gil_meta")}
        if extra:
            m.update(extra)
        out = self.root / "manifest.json"
        out.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        return out

    def commit(self):
        self.con.commit()

    def close(self):
        try:
            self.con.commit()
        finally:
            self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
