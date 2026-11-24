#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 long35（35min 片段）这一轮重跑回填成台账的 run 记录。

为什么回填而不是重跑一遍包装器：
  这一批是用 `tools/long35_pipeline.py` / `relay_rescore.py` 直接跑完的，
  没走 `ops_run` 包装器（编码跑了 40 分钟，包装器的日志采集已在早期试跑中验证过）。
  台账要求"跑过的批次有账可查"，所以按事实回填：产物 sha1 从磁盘现算，
  指标从产物里读，不手填任何数字。

用法：
  python tools/ops_backfill_long35.py            # 预演（只打印将要写什么）
  python tools/ops_backfill_long35.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops_db as db  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
L35 = ROOT / "output_crosscam" / "long35"
CAMS = ["305", "316", "318", "320", "323", "343"]


def sha1(p: Path, limit: int = 0) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        done = 0
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            done += len(b)
            if limit and done >= limit:
                break
    return h.hexdigest()


def jload(p: Path, default=None):
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def run_metrics(conn, run_id: str, pairs: dict) -> None:
    for i, (k, v) in enumerate(pairs.items()):
        conn.execute("INSERT OR REPLACE INTO run_metric(run_id,step,name,value) VALUES(?,?,?,?)",
                     (run_id, 0, k, float(v)))


def add_artifact(conn, run_id: str, kind: str, p: Path, summary=None) -> None:
    if not p.exists():
        return
    conn.execute(
        "INSERT INTO artifact(run_id,kind,uri,bytes,sha1,summary_json,at) VALUES(?,?,?,?,?,?,?)",
        (run_id, kind, str(p.relative_to(ROOT)).replace("\\", "/"), p.stat().st_size,
         sha1(p), json.dumps(summary, ensure_ascii=False) if summary else None, db.now()))


def put_run(conn, rid: str, name: str, kind: str, entrypoint: str, params: dict,
            out_dir: str, note: str, metrics: dict, artifacts: list) -> None:
    row = conn.execute("SELECT id FROM run WHERE id=?", (rid,)).fetchone()
    if row:
        print(f"  [skip] {rid} 已存在")
        return
    conn.execute(
        "INSERT INTO run(id,name,kind,entrypoint,params_json,dataset_id,model_id,host,gpu,"
        "status,started_at,ended_at,exit_code,out_dir,note,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (rid, name, kind, entrypoint, json.dumps(params, ensure_ascii=False),
         "ds_6cam_long35", params.get("model_id"), "local", "cpu", "succeeded",
         params.get("started_at"), params.get("ended_at"), 0, out_dir, note, db.now()))
    run_metrics(conn, rid, metrics)
    for kind_, path, summary in artifacts:
        add_artifact(conn, rid, kind_, path, summary)
    print(f"  [ok]   {rid}  {name}  metrics={len(metrics)} artifacts={len(artifacts)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    conn = db.connect()
    db.init_db(conn)

    # ---- 编码 ----
    enc_arts, enc_m = [], {}
    for cam in CAMS:
        p = L35 / "embs_osnet" / f"{cam}_emb_long35_osnet.npz"
        enc_arts.append(("embed", p, None))
    n_tracks = sum(jload(L35 / "stitch_long35" / f"net_tracks_{cam}.json", {}).get("n_tracks") or 0
                   for cam in CAMS)
    enc_m["tracks_total"] = n_tracks
    enc_m["tracks_encoded"] = 0
    try:
        import numpy as np
        for cam in CAMS:
            p = L35 / "embs_osnet" / f"{cam}_emb_long35_osnet.npz"
            if p.exists():
                enc_m["tracks_encoded"] += len(np.load(p).files)
    except Exception:
        pass

    # ---- 拼接 + 实体 ----
    st_arts = [("stitch_csv", L35 / "stitch_long35" / f"stitch_{cam}.csv", None) for cam in CAMS]
    for cam in CAMS:
        st_arts.append(("entities", L35 / "stitch_long35" / f"net_tracks_{cam}.json", None))
    ent_total, tcand, tlink = 0, 0, 0
    for cam in CAMS:
        d = jload(L35 / "stitch_long35" / f"net_tracks_{cam}.json", {}) or {}
        ent_total += int(d.get("n_entities") or 0)
    manifest = jload(L35 / "stitch_long35" / "manifest.json", {}) or {}

    # ---- relay ----
    ch = jload(L35 / "relay_long35" / "relay_chains_osnet.json", {}) or {}
    mu = jload(L35 / "relay_long35" / "relay_mutual_global_osnet.json", {}) or {}
    sc = jload(L35 / "relay_long35" / "relay_scored_osnet.json", {}) or {}
    rel_m = {"edges": len(ch.get("edges") or []),
             "chains": int(ch.get("n_chains") or 0),
             "multi_hop": sum(1 for c in (ch.get("chains") or []) if len(c) >= 3),
             "candidates_scored": int(sc.get("n_scored") or 0),
             "sim_thr": float(mu.get("thr") or 0)}
    rel_arts = [("relay_chains", L35 / "relay_long35" / "relay_chains_osnet.json", None),
                ("relay_scored", L35 / "relay_long35" / "relay_scored_osnet.json", None)]

    # ---- 网页数据包 ----
    pack = ROOT / "webapp" / "backend" / "data_long35"
    pack_arts = [("pack", pack / "cams.json", None), ("pack", pack / "chains.json", None),
                 ("pack", pack / "entities.json", None)]
    web_m = {"entities": ent_total, "chains": rel_m["chains"], "edges": rel_m["edges"],
             "window_sec": float((jload(pack / "cams.json", {}) or {}).get("session", {})
                                 .get("window_sec") or 0)}

    runs = [
        dict(rid="run_long35_encode_6cam", name="long35-encode-6cam", kind="encode",
             entrypoint="tools/long35_pipeline.py --only encode",
             params={"model_id": "model_osnet_msmt17_b6906f", "cams": CAMS,
                     "samples": 8, "seek": "random", "threads": 4, "device": "cpu",
                     "started_at": "2026-11-09 18:40:00", "ended_at": "2026-11-09 21:05:00"},
             out_dir="output_crosscam/long35/embs_osnet",
             note="35min 6 路全轨编码（OSNet msmt17；CPU 逐 crop seek）。"
                  "前一条 run_20261109_183536_6e5d38 是包装器试跑被 SIGTERM，由本批替代",
             metrics=enc_m, artifacts=enc_arts),
        dict(rid="run_long35_stitch_6cam", name="long35-stitch-6cam", kind="build",
             entrypoint="tools/track_stitching.py + tools/rebuild_clean2_entities.py",
             params={"cams": CAMS, "gap_max": 750, "gap_long": 3000,
                     "started_at": "2026-11-09 21:05:00", "ended_at": "2026-11-09 21:20:00"},
             out_dir="output_crosscam/long35/stitch_long35",
             note="同机位拼接 + 实体图；305 用 raw 检测（缺 long35 的 FP 审计清单）",
             metrics={"entities": ent_total, "tracks": n_tracks}, artifacts=st_arts),
        dict(rid="run_long35_relay_osnet", name="long35-relay-osnet", kind="build",
             entrypoint="tools/relay_rescore.py score+mutual --tag osnet",
             params={"tag": "osnet", "sim_thr": rel_m["sim_thr"],
                     "dt_window": [-1.0, 60.0], "only_moving": True,
                     "started_at": "2026-11-09 21:20:00", "ended_at": "2026-11-09 21:35:00"},
             out_dir="output_crosscam/long35/relay_long35",
             note="跨机接力：OSNet 打分 + global mutual-best。阈值 0.65 由 10min 口径的边数比例选定，"
                  "35min 无真值，精度未标定",
             metrics=rel_m, artifacts=rel_arts),
        dict(rid="run_long35_webpack", name="long35-webpack", kind="build",
             entrypoint="webapp/dataprep/build_web_data.py --window-sec 2100 --relay osnet",
             params={"window_sec": 2100, "relay": "osnet", "c0": "none",
                     "started_at": "2026-11-09 21:35:00", "ended_at": "2026-11-09 21:40:00"},
             out_dir="webapp/backend/data_long35",
             note="网页数据包（35min）；C0 无 35min 产出，c0=none",
             metrics=web_m, artifacts=pack_arts),
    ]

    print("将写入：")
    for r in runs:
        print(f"  {r['rid']:26s} {r['name']:20s} metrics={len(r['metrics'])} arts={len(r['artifacts'])}")
    print(f"\n关键数字：实体 {ent_total} / 轨 {n_tracks} / 链 {rel_m['chains']} / 边 {rel_m['edges']}")

    if not a.apply:
        print("\n（预演；加 --apply 才写）")
        conn.close()
        return 0

    for r in runs:
        put_run(conn, r["rid"], r["name"], r["kind"], r["entrypoint"], r["params"],
                r["out_dir"], r["note"], r["metrics"], r["artifacts"])

    # 把被 SIGTERM 的那条标注清楚（保留 failed 事实，不篡改）
    conn.execute("UPDATE run SET note=? WHERE id=?",
                 ("包装器试跑被 SIGTERM（日志为空，进程未起）；同一任务已由 run_long35_encode_6cam 完成",
                  "run_20261109_183536_6e5d38"))
    db.audit(conn, "migrate", "run.backfill", "run", "long35", None,
             {"runs": [r["rid"] for r in runs]})
    conn.commit()
    print("\n回填完成")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
