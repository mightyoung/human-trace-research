# -*- coding: utf-8 -*-
"""把现有资产导入 GIL 统一存储（一次性迁移 + 口径对账）。

导入源：
  向量/观测 : analysis/global_reid/store_cradio_v4（v2 代际格式，经 CURRENT.json 解析；
              旧 flat 库需先 python tools/store_admin.py migrate-in-place）   (539 轨)
  实体      : analysis/stitch_clean2/net_tracks_{cam}.json                    (411 实体)
  拼接边    : 由实体的多成员轨推出（kind=stitch）
  接力边    : analysis/relay_v2/edges_v2.json (182) + relay_clean2/relay_chains_cradio_v5.json
  真值      : analysis/truth/remote/labels.json + webapp/backend/data/truth/seed_labels.json

用法：
  python tools/gil_migrate.py [--out output_crosscam/analysis/gil/store_v1]
                              [--reid <store_dir>] [--entities stitch_clean2|stitch]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gil import GILStore, CAMS  # noqa: E402
from global_reid_store import GlobalReIDStore  # noqa: E402

ROOT = HERE.parent


def load_reid(store_dir):
    """读回轨道行（key 升序）与嵌入矩阵（行序一一对应）。

    v2 代际格式：经 CURRENT.json 解析当前代并做全量校验（旧 flat 库需先
    store_admin migrate-in-place 迁移）。返回 rows（dict 列表，含 key/cam/date/
    tid/t_first/t_last/u/v/h_med）与 emb（f32）。
    """
    st = GlobalReIDStore.open(store_dir)
    keys = sorted(st.rows)
    emb = (np.stack([st.rows[k] for k in keys]).astype(np.float32)
           if keys else np.empty((0, st.dim), dtype=np.float32))
    rows = [{"key": k, **st.meta[k]} for k in keys]
    return rows, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output_crosscam/analysis/gil/store_v1")
    ap.add_argument("--reid", default="output_crosscam/analysis/global_reid/store_cradio_v4")
    ap.add_argument("--entities", default="output_crosscam/analysis/stitch_clean2")
    ap.add_argument("--relay-v2", default="output_crosscam/analysis/relay_v2/edges_v2.json")
    ap.add_argument("--relay-v5", default="output_crosscam/analysis/relay_clean2/relay_chains_cradio_v5.json")
    ap.add_argument("--relay-gate",
                    default="output_crosscam/analysis/relay_v2/edges_v2_gated.json",
                    help="合并后的 relay 边集；只取其 channel==gate 的 14 条作为独立 kind")
    ap.add_argument("--truth", default="output_crosscam/analysis/truth/remote/labels.json")
    ap.add_argument("--seed", default="webapp/backend/data/truth/seed_labels.json")
    ap.add_argument("--extra-emb", default="output_crosscam/analysis/cradio_scale_v4_extra",
                    help="补缺嵌入源（clean2 新增轨；必须与主库同空间 = cradio_scale_v4_extra）")
    ap.add_argument("--det-dir", default="output_crosscam/cleanset_v2",
                    help="补缺 obs 的元数据来源（该视图的 detections.jsonl）")
    ap.add_argument("--keep", action="store_true",
                    help="保留既有 store（默认**整库重建**：向量块是 append 语义，"
                         "重复迁移会把块翻倍 → 默认先清空，保证可复现）")
    a = ap.parse_args()

    out = ROOT / a.out
    if not a.keep and out.exists():
        import shutil
        shutil.rmtree(out)
        print(f"[gil] 清理旧 store（默认整库重建，--keep 可保留）: {out}")
    st = GILStore(out)
    print(f"[gil] store = {out}")

    # ---- 相机 ----
    RES = {"305": (2560, 1440), "316": (2560, 1440), "318": (1920, 1080),
           "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}
    for c in CAMS:
        st.upsert_camera(c, *RES[c], fps=25.0)
    st.con.commit()

    # ---- 向量 + obs ----
    reid_dir = ROOT / a.reid
    rows, emb = load_reid(reid_dir)
    vb = st.vectors()
    by_block = {c: {"ids": [], "embs": []} for c in CAMS}
    k2id = {}
    for i, r in enumerate(rows):
        cam = r["cam"]
        oid = st.add_obs(r["key"], cam, r["date"], r["tid"],
                         t_first=r["t_first"], t_last=r["t_last"],
                         u=r["u"], v=r["v"], h_med=r["h_med"],
                         src=f"migrate:store_cradio_v4")
        k2id[r["key"]] = oid
        if cam in by_block:
            by_block[cam]["ids"].append(oid)
            by_block[cam]["embs"].append(emb[i])
    for cam, d in by_block.items():
        if d["ids"]:
            vb.add(cam, d["ids"], np.stack(d["embs"]))
    vb.flush()
    st.con.commit()
    print(f"[gil] obs={len(rows)} 向量块={ {k: len(v['ids']) for k, v in by_block.items() if v['ids']} }")

    (cam_tid2obs := {})
    for r in st.con.execute("SELECT id,cam,tid FROM obs"):
        cam_tid2obs[(r["cam"], r["tid"])] = r["id"]

    DATE = rows[0]["date"]

    # ---- 补缺：clean2 视图新增、ReID 主库(clean v1 口径)未覆盖的轨 ----
    # 根因：store_cradio_v4 在 clean(v1) 上编码（539 轨），实体图在 clean2 上重建（541 成员轨）；
    # clean2 多保留的 7 条轨（305|12 316|32 318|53/82/98/135 323|193）无嵌入。
    # 补源 = cradio_scale_v4_extra/（与主库逐位同源 v4 空间，已重新编码；**不可**用 cradio_scale(v5)）。
    extra_dir = ROOT / a.extra_emb
    ent_json = {}
    needed = set()
    for cam in CAMS:
        p = ROOT / a.entities / f"net_tracks_{cam}.json"
        if p.exists():
            ent_json[cam] = json.loads(p.read_text(encoding="utf-8"))
            for e in ent_json[cam]["entities"]:
                for t in e["tracks"]:
                    needed.add((cam, int(t)))
    miss = sorted(x for x in needed if x not in cam_tid2obs)
    n_extra = 0
    if miss:
        if not extra_dir.exists():
            print(f"[gil] !! 缺 {len(miss)} 条轨且无补充源 {extra_dir} —— 接力边将不完整")
        else:
            from global_reid_store import build_meta_from_detections  # noqa: E402
            ex_cache, meta_cache = {}, {}
            by_cam = {}
            for c, t in miss:
                by_cam.setdefault(c, []).append(t)
            for cam, tids in sorted(by_cam.items()):
                if cam not in ex_cache:
                    ex = {}
                    for p in sorted(extra_dir.glob(f"{cam}*.npz")):
                        z = np.load(p)
                        for k in z.files:
                            mm = re.search(r"(\d+)$", k)
                            if mm:
                                ex[int(mm.group(1))] = z[k].astype(np.float32)
                    ex_cache[cam] = ex
                    det = ROOT / a.det_dir / f"{cam}_detections.jsonl"
                    meta_cache[cam] = build_meta_from_detections(cam, str(det)) if det.exists() else {}
                ex, md = ex_cache[cam], meta_cache[cam]
                ids, embs = [], []
                for t in tids:
                    if t not in ex:
                        print(f"       [warn] 补充源缺 {cam}|{t}")
                        continue
                    m = md.get(t, {})
                    key = f"{DATE}|{cam}|{t}"
                    oid = st.add_obs(key, cam, DATE, t,
                                     t_first=m.get("t_first"), t_last=m.get("t_last"),
                                     u=m.get("u"), v=m.get("v"), h_med=m.get("h_med"),
                                     src="extra:cradio_scale_v4_extra")
                    cam_tid2obs[(cam, t)] = oid
                    k2id[key] = oid
                    ids.append(oid)
                    embs.append(ex[t])
                    n_extra += 1
                if ids:
                    vb.add(cam, ids, np.stack(embs))
            vb.flush()
            st.con.commit()
            print(f"[gil] 补缺 obs={n_extra}（clean2 新增轨；源 {extra_dir.name}）"
                  f" 剩余未匹配={len(miss) - n_extra}")

    # ---- 实体（同机位）+ 片段 + 拼接边 ----
    ent_dir = ROOT / a.entities
    n_ent = n_stitch = n_frag = miss_tid = 0
    for cam in CAMS:
        p = ent_dir / f"net_tracks_{cam}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for e in d["entities"]:
            obs_ids = [cam_tid2obs[(cam, int(t))] for t in e["tracks"] if (cam, int(t)) in cam_tid2obs]
            miss_tid += len(e["tracks"]) - len(obs_ids)
            if not obs_ids:
                continue
            os_ = [st.obs_row(i) for i in obs_ids]
            t0 = min(x["t_first"] for x in os_ if x["t_first"] is not None) if os_ else None
            t1 = max(x["t_last"] for x in os_ if x["t_last"] is not None) if os_ else None
            st.add_entity(cam, e["eid"], obs_ids, t0, t1,
                          n_frag=e.get("n_frag"), src=f"migrate:{a.entities}")
            n_ent += 1
            # frag = 同机位的一段连续出现（= 一个成员轨）；多 frag 的实体即"遮挡后已接续"的实例
            for oid in obs_ids:
                r = st.obs_row(oid)
                st.add_frag(cam, [oid], r["t_first"], r["t_last"],
                            src=f"migrate:{a.entities}")
                n_frag += 1
            # 同机位拼接边（成员轨两两按时间相邻）
            order = sorted(obs_ids, key=lambda i: (st.obs_row(i)["t_first"] or 0))
            for u, v in zip(order, order[1:]):
                st.add_edge("stitch", "obs", u, "obs", v,
                            dt=(st.obs_row(v)["t_first"] or 0) - (st.obs_row(u)["t_last"] or 0),
                            status="active", tier="intra", evidence={"src": "entity_members"})
                n_stitch += 1
    st.con.commit()
    print(f"[gil] entity={n_ent} frag={n_frag} 同机位拼接边={n_stitch} 未匹配 tid={miss_tid}")

    # ---- 接力边 v2 / v5 ----
    def ent_id(x):
        cam, eid = x.split(":", 1)
        r = st.entity_of(cam, eid)
        return int(r["id"]) if r else None

    n_v2 = n_v5 = n_gate = 0
    skipped = []
    pv2 = ROOT / a.relay_v2
    if pv2.exists():
        d = json.loads(pv2.read_text(encoding="utf-8"))
        for e in d["edges"]:
            aid, bid = ent_id(e["a"]), ent_id(e["b"])
            if aid is None or bid is None:
                skipped.append({"src": "relay_v2", "a": e["a"], "b": e["b"],
                                "missing": [x for x, i in ((e["a"], aid), (e["b"], bid)) if i is None],
                                "reason": "端点实体不在 entity 表（成员轨缺 ReID 嵌入）"})
                continue
            st.add_edge("relay", "entity", aid, "entity", bid, dt=e.get("dt"),
                        score=e.get("score"), status=e.get("status", "active"),
                        tier=e.get("tier"), evidence={"src": "relay_v2", "dir": e.get("dir"),
                                                      "sim": e.get("sim"), "type": e.get("edge_type"),
                                                      "w_dt": e.get("w_dt")})
            n_v2 += 1
    pv5 = ROOT / a.relay_v5
    if pv5.exists():
        d = json.loads(pv5.read_text(encoding="utf-8"))
        for e in d.get("edges", []):
            aid, bid = ent_id(e["a"]), ent_id(e["b"])
            if aid is None or bid is None:
                skipped.append({"src": "relay_clean2/v5", "a": e["a"], "b": e["b"],
                                "missing": [x for x, i in ((e["a"], aid), (e["b"], bid)) if i is None],
                                "reason": "端点实体不在 entity 表（成员轨缺 ReID 嵌入）"})
                continue
            st.add_edge("relay_v5", "entity", aid, "entity", bid,
                        dt=e.get("dt"), score=e.get("score") or e.get("sim"),
                        status="active", tier=e.get("tier"),
                        evidence={"src": "relay_clean2/v5"})
            n_v5 += 1
    # ---- 门户驱动通道（gate）：独立 kind，不并入 relay（保 v2 对账）----
    pg = ROOT / a.relay_gate
    if pg.exists():
        d = json.loads(pg.read_text(encoding="utf-8"))
        for e in d.get("edges", []):
            if e.get("channel") != "gate":
                continue
            aid, bid = ent_id(e["a"]), ent_id(e["b"])
            if aid is None or bid is None:
                skipped.append({"src": "gate", "a": e["a"], "b": e["b"],
                                "missing": [x for x, i in ((e["a"], aid), (e["b"], bid)) if i is None],
                                "reason": "端点实体不在 entity 表"})
                continue
            st.add_edge("relay_gate", "entity", aid, "entity", bid,
                        dt=e.get("dt"), score=e.get("score"), status="gate", tier="gate",
                        evidence={"src": "gil/gate_candidates", "dir": e.get("dir"),
                                  "sim": e.get("sim"), "w_dt": e.get("w_dt")})
            n_gate += 1
    st.con.commit()
    # ★ 零静默：被跳过的边必须落盘可查，不得悄悄丢弃
    sk_path = out / "skipped_edges.json"
    sk_path.write_text(json.dumps(skipped, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gil] relay v2 边={n_v2}  v5 边={n_v5}  gate 边={n_gate}  "
          f"跳过(端点实体缺失)={len(skipped)} -> {sk_path.name}")
    for x in skipped:
        print(f"       skip {x['src']}: {x['a']} -> {x['b']}  缺 {x['missing']}")

    # ---- 真值标注 ----
    n_ann = 0
    for path, tag in ((ROOT / a.seed, "seed"), (ROOT / a.truth, "user")):
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8")) or {}
        pair = d.get("pair", d) if isinstance(d, dict) else {}
        for pid, v in pair.items():
            if not isinstance(v, dict):
                continue
            val = v.get("v")
            if val is None:
                continue
            st.add_annotation("pair", pid, {0: "0", 1: "1", 2: "?"}.get(int(val), str(val)),
                              confidence=v.get("c"), note=v.get("note"),
                              author=tag)
            n_ann += 1
        ev = d.get("event", {}) if isinstance(d, dict) else {}
        for eid, v in (ev or {}).items():
            if isinstance(v, dict) and v.get("v") is not None:
                st.add_annotation("event", eid, str(v.get("v")),
                                  confidence=v.get("c"), author=tag)
    st.con.commit()
    print(f"[gil] annotation={n_ann}")

    st.meta_set(view=os.environ.get("DET_VIEW", "clean2"),
                entities_dir=a.entities,
                model_version="cradio_gemma2_bridge_v4",
                dim=st.dim,
                note="GIL 统一存储 v1（迁移自既有资产；向量分块、结构化真相入 SQLite）")
    mf = st.snapshot()
    # ★ 不变量：向量总数 == obs 存活数（防重复迁移导致向量块翻倍）
    _vb = st.vectors(load=True)
    _nvec = sum(_vb.n(b) for b in _vb.blocks())
    _s = st.stats()
    assert _nvec == _s["obs_alive"], f"向量总数 {_nvec} != obs {_s['obs_alive']}（疑似重复迁移）"
    print("[gil] stats =", json.dumps(st.stats(), ensure_ascii=False))
    print(f"[gil] 不变量 OK：向量总数 {_nvec} == obs {_s['obs_alive']}")
    print("[gil] manifest ->", mf)
    st.close()


if __name__ == "__main__":
    main()
