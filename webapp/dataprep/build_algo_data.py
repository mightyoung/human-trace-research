#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""算法工作台数据准备：全局 ReID 库 / 跨镜头聚类 / 意图模型 / 离线训练指标。

产物（webapp/backend/data/）：
  reid/store_emb.npy        539x768 float16 全局轨迹嵌入（C-RADIOxGemma2 桥接 v4）
  reid/store_meta.json      与嵌入同序的轨迹元信息（含代表帧，供裁剪缩略图）
  reid/clusters.json        539 轨 mutual-kNN 聚类结果（成员/质心/未分配）
  reid/report.json          聚类的离线训练与评估指标（来自 selector_1012）
  reid/entity_index.json    (cam,tid) -> 跨机接力链 / 同机实体 反查
  intent/params.json        意图模型推理参数（mu/sd/窗长/类别）
  intent/model.pt           意图模型权重（BERT 范式 Transformer 编码器）
  assets/*.png|json         训练曲线/混淆矩阵/相似度分布等原图与日志
运行：python webapp/dataprep/build_algo_data.py
"""
import json
import os
import shutil
import sqlite3

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "output_crosscam", "analysis")
DATA = os.path.join(ROOT, "webapp", "backend", "data")
STORE = os.path.join(A, "global_reid", "store_cradio_v4")
SEL = os.path.join(A, "selector_1012")
INTENT = os.path.join(A, "intent", "model_1012")


def l2(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


def mutual_knn_cluster(E, k=5, thr=0.30):
    """对称 mutual-kNN 图连通分量聚类（无 scipy 依赖，并查集实现）。"""
    n = len(E)
    S = E @ E.T
    np.fill_diagonal(S, -1.0)
    knn = np.argsort(-S, axis=1)[:, :k]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in knn[i]:
            if S[i, j] >= thr and i in knn[j]:
                union(int(i), int(j))
    lab = np.array([find(i) for i in range(n)])
    # 紧凑编号 + 按规模排序
    uniq, counts = np.unique(lab, return_counts=True)
    order = np.argsort(-counts)
    remap = {int(u): r for r, u in enumerate(uniq[order])}
    return np.array([remap[int(x)] for x in lab]), S, knn


def rep_frame(tracks_json, tid):
    pts = tracks_json["tracks"].get(str(tid)) or []
    if not pts:
        return None, None
    p = pts[len(pts) // 2]
    return int(p[0]), [round(float(x), 1) for x in p[1:5]]


def main():
    os.makedirs(os.path.join(DATA, "reid"), exist_ok=True)
    os.makedirs(os.path.join(DATA, "intent"), exist_ok=True)
    assets = os.path.join(DATA, "assets")
    os.makedirs(assets, exist_ok=True)

    # ---------- 1. 全局 ReID 库 ----------
    E = np.load(os.path.join(STORE, "emb.npy")).astype(np.float32)
    con = sqlite3.connect(os.path.join(STORE, "meta.sqlite"))
    rows = con.execute("SELECT key,cam,date,tid,t_first,t_last,h_med,u,v FROM tracks").fetchall()
    con.close()
    assert len(rows) == len(E), (len(rows), len(E))
    En = l2(E)
    np.save(os.path.join(DATA, "reid", "store_emb.npy"), En.astype(np.float16))

    tracks_cache = {}
    for cam in sorted(set(r[1] for r in rows)):
        with open(os.path.join(DATA, "tracks", f"{cam}.json"), encoding="utf-8") as f:
            tracks_cache[cam] = json.load(f)

    meta = []
    for i, (key, cam, date, tid, t0, t1, h_med, u, v) in enumerate(rows):
        frep, box = (None, None)
        tj = tracks_cache.get(cam)
        if tj:
            f_ob, box = rep_frame(tj, tid)
            fps = tj.get("fps") or 25.0
            f_ob = None if f_ob is None else f_ob
            f_rep = f_ob
        meta.append({
            "i": i, "key": key, "cam": cam, "date": date, "tid": int(tid),
            "t0": round(float(t0), 2), "t1": round(float(t1), 2),
            "h_med": round(float(h_med), 1),
            "u": round(float(u), 1), "v": round(float(v), 1),
            "f_rep": f_rep
        })
    with open(os.path.join(DATA, "reid", "store_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"n": len(meta), "tracks": meta,
                   "version": json.load(open(os.path.join(STORE, "version.json")))}, f,
                  ensure_ascii=False)

    # ---------- 2. 聚类（与论文口径一致：mutual-kNN） ----------
    lab, S, knn = mutual_knn_cluster(En, k=3, thr=0.95)
    clusters = {}
    for i, c in enumerate(lab):
        clusters.setdefault(int(c), []).append(i)
    clist = []
    for c, mem in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        idx = np.array(mem)
        centroid = l2(En[idx].mean(0))
        # 代表成员 = 与质心最相似
        rep = int(idx[np.argmax(En[idx] @ centroid)])
        clist.append({
            "cid": c, "label": "", "size": len(mem), "members": [int(x) for x in idx],
            "rep": rep, "centroid_cos": round(float(np.mean(En[idx] @ centroid)), 4)
        })
    n_big = sum(1 for c in clist if c["size"] >= 2)
    n_single = len(clist) - n_big
    with open(os.path.join(DATA, "reid", "clusters.json"), "w", encoding="utf-8") as f:
        json.dump({"method": "mutual-knn", "k": 3, "thr": 0.95, "n": len(meta),
                   "n_clusters": len(clist), "n_multi": n_big, "n_singleton": n_single,
                   "clusters": clist}, f, ensure_ascii=False)

    # ---------- 3. 反查索引（跨机链 / 同机实体） ----------
    chains = json.load(open(os.path.join(DATA, "chains.json"), encoding="utf-8"))
    entities = json.load(open(os.path.join(DATA, "entities.json"), encoding="utf-8"))
    track_ent, ent_chain, track_chain = {}, {}, {}
    for cam, e in entities.items():
        for eid, ent in (e.get("entities") or {}).items():
            for tid in ent.get("tracks", []):
                track_ent[f"{cam}:{tid}"] = eid
    for ch in chains.get("chains", []):
        for n in ch["nodes"]:
            ent_chain[f"{n['cam']}:{n['eid']}"] = ch["id"]
            for tid in (entities.get(n["cam"], {}).get("entities", {}).get(n["eid"], {}).get("tracks") or []):
                track_chain[f"{n['cam']}:{tid}"] = ch["id"]
    with open(os.path.join(DATA, "reid", "entity_index.json"), "w", encoding="utf-8") as f:
        json.dump({"track_ent": track_ent, "ent_chain": ent_chain, "track_chain": track_chain}, f,
                  ensure_ascii=False)

    # ---------- 4. 意图模型 ----------
    dst_model = os.path.join(DATA, "intent", "model.pt")
    if os.path.exists(os.path.join(INTENT, "best.pt")):
        shutil.copy2(os.path.join(INTENT, "best.pt"), dst_model)
    mu = sd = None
    tr = os.path.join(A, "intent", "1012_train.npz")
    if os.path.exists(tr):
        z = np.load(tr, allow_pickle=True)
        X = z["X"].astype(np.float32)
        mu = X.reshape(-1, X.shape[-1]).mean(0).tolist()
        sd = (X.reshape(-1, X.shape[-1]).std(0) + 1e-6).tolist()
    params = {
        "classes": ["stop", "forward", "backward", "turn_left", "turn_right"],
        "L_in": 12, "L_fut": 5, "win_stride": 2, "stride": 5, "fps": 25.0,
        "d_model": 256, "layers": 4, "heads": 4, "ff": 1024, "n_feat": 7,
        "mu": mu, "sd": sd,
        "note": "意图=未来 5 步(1s)运动类别；特征 [u,v,du,dv,w,h,DT]，按机位真实分辨率归一化"
    }
    with open(os.path.join(DATA, "intent", "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)

    # ---------- 5. 离线训练指标 ----------
    rep = {}
    for name in ["cluster_report.json", "eval_test.json", "clf_test.json", "crossseg.json"]:
        p = os.path.join(SEL, name)
        if os.path.exists(p):
            rep[name.replace(".json", "")] = json.load(open(p, encoding="utf-8"))
    mp = os.path.join(INTENT, "metrics.json")
    if os.path.exists(mp):
        rep["intent_metrics"] = json.load(open(mp, encoding="utf-8"))
    rep["intent_train_log"] = ""
    lp = os.path.join(INTENT, "train.log")
    if os.path.exists(lp):
        rep["intent_train_log"] = open(lp, encoding="utf-8").read()
    # 数组降采样，避免 JSON 过大
    if "intent_metrics" in rep and isinstance(rep["intent_metrics"].get("hist"), list):
        rep["intent_metrics"]["hist"] = rep["intent_metrics"]["hist"][:]
    with open(os.path.join(DATA, "reid", "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False)

    for src, dst in [(os.path.join(SEL, "sim_dist.png"), "cluster_sim_dist.png"),
                     (os.path.join(INTENT, "confusion_test.png"), "intent_confusion.png"),
                     (os.path.join(INTENT, "traj_viz_test.jpg"), "intent_traj_viz.jpg")]:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(assets, dst))

    print(f"[reid] {len(meta)} tracks -> store_emb.npy {E.shape}")
    print(f"[cluster] {len(clist)} 簇（>=2: {n_big}, 单例: {n_single}）")
    print(f"[cluster] 最大 5 簇: {[c['size'] for c in clist[:5]]}")
    print(f"[intent] model.pt {'OK' if os.path.exists(dst_model) else 'MISSING'}, mu/sd {'OK' if mu else 'MISSING'}")
    print(f"[out] {DATA}")


if __name__ == "__main__":
    main()
