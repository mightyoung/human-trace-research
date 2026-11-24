#!/usr/bin/env python
"""选择器干净口径对照评估（实体零交集）：OSNet vs DINOv3-B。

污染纪律（2026-10-17 用户指正后固化）：
  - 训练集 = splits_v5.train（304 实体 / 349 轨）
  - 测试集 = splits_v5.test（48 实体 / 98 轨，318+320）
  - 已核验：两者 tid 与实体交集均为 0（见 report 的 pollution 段）
  - frozen_eval_v1 与 train 有 81/109 实体重叠 → **不得**用于本脚本的评估

流程：
  1. 读 6 路轨迹嵌入（键 `v{cam}_{tid}`）
  2. train 向量做 mutual-kNN 聚类（k=5，thr 网格）→ 簇原型库
  3. test 评估：
     - POS30（8 对确认同人）/ NEG30（22 对否决，318×320）
     - 余弦 AUC（pos vs neg）
     - 选择器同簇率（同簇=判同人）
     - pairs_same（61 对同机同人）同簇率
输出：output_crosscam_local/selector_clean/{encoder}.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "output_crosscam/analysis/splits_v5.json"

CAMS = ["305", "316", "318", "320", "323", "343"]
POS30 = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
NEG30 = [(14, 1), (131, 21), (61, 14), (50, 11), (132, 21), (3, 1), (7, 1), (146, 21),
         (127, 19), (123, 19), (57, 11), (131, 23), (124, 18), (131, 24), (3, 4),
         (126, 18), (43, 11), (10, 1), (131, 28), (98, 26), (142, 26), (53, 11)]

EMB = {
    "osnet": "output_crosscam_local/selector_clean/emb_{split}_osnet.npz",
    "d3b": "output_crosscam_local/selector_clean/emb_{split}_d3b.npz",
}


def norm(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


def load_split(enc, split):
    """读 emb_{split}_{enc}.npz，键 {cam}_{tid}。"""
    p = ROOT / EMB[enc].format(split=split)
    if not p.exists():
        return {}
    d = np.load(p)
    out = {}
    for k in d.files:
        cam, tid = k.rsplit("_", 1)
        out[(cam, int(tid))] = d[k].astype(np.float32)
    return out


def mutual_knn(V, k=5, thr=0.90):
    S = V @ V.T
    np.fill_diagonal(S, -1.0)
    kk = min(k, len(V) - 1)
    idx = np.argsort(-S, axis=1)[:, :kk]
    A = np.zeros(S.shape, dtype=bool)
    rows = np.arange(len(V))[:, None]
    A[rows, idx] = True
    M = A & A.T & (S >= thr)
    n_comp, lab = connected_components(csr_matrix(M), directed=False)
    return lab, n_comp


def auc_mw(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]; i = 0
    while i < len(sv):
        j = i + 1
        while j < len(sv) and sv[j] == sv[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="osnet,d3b")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--thrs", default="0.86,0.88,0.90,0.92,0.94")
    ap.add_argument("--out-dir", default="output_crosscam_local/selector_clean")
    args = ap.parse_args()

    splits = json.load(open(SPLITS, encoding="utf-8"))
    tr = splits["train"]["tracks"]
    te = splits["test"]["tracks"]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    thrs = [float(x) for x in args.thrs.split(",")]

    report = {"pollution": {}, "encoders": {}}

    # ---- 污染核验（写入报告）----
    ent = {}
    for cam in CAMS:
        p = ROOT / f"output_crosscam/analysis/stitch_clean2/net_tracks_{cam}.json"
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            ent[cam] = {int(t): x["eid"] for x in d["entities"] for t in x["tracks"]}
    tr_e = {ent[c][int(t)] for c, ts in tr.items() for t in ts if c in ent and int(t) in ent[c]}
    te_e = {ent[c][int(t)] for c, ts in te.items() for t in ts if c in ent and int(t) in ent[c]}
    tr_t = {(c, int(t)) for c, ts in tr.items() for t in ts}
    te_t = {(c, int(t)) for c, ts in te.items() for t in ts}
    report["pollution"] = {
        "train_tracks": len(tr_t), "test_tracks": len(te_t),
        "train_entities": len(tr_e), "test_entities": len(te_e),
        "track_overlap": len(tr_t & te_t), "entity_overlap": len(tr_e & te_e),
        "note": "splits_v5 内部实体零交集；frozen_eval_v1 与 train 81/109 实体重叠，"
                "故本评估不使用 frozen_eval。",
    }

    encs = args.encoders.split(",")
    # 预加载 + 取公共轨迹（两编码器裁剪门槛不同，必须同集合对比）
    embs = {e: (load_split(e, "train"), load_split(e, "test")) for e in encs}
    common_tr = set.intersection(*[set(v[0].keys()) for v in embs.values()])
    common_te = set.intersection(*[set(v[1].keys()) for v in embs.values()])
    print(f"[common] train={len(common_tr)} test={len(common_te)}", flush=True)

    for enc in encs:
        emb_tr, emb_te = embs[enc]
        # train 向量（公共键）
        Xtr, keys_tr = [], []
        for (cam, tid) in sorted(common_tr):
            v = emb_tr.get((cam, tid))
            if v is not None:
                Xtr.append(v); keys_tr.append((cam, tid))
        Xtr = norm(np.asarray(Xtr, np.float32))
        # test 向量（公共键）
        Xte, keys_te = [], []
        for (cam, tid) in sorted(common_te):
            v = emb_te.get((cam, tid))
            if v is not None:
                Xte.append(v); keys_te.append((cam, tid))
        Xte = norm(np.asarray(Xte, np.float32))
        idx = {k: i for i, k in enumerate(keys_te)}

        res = {"n_train": len(Xtr), "n_test": len(Xte), "grid": {}}
        for thr in thrs:
            lab, n_comp = mutual_knn(Xtr, args.k, thr)
            protos = []
            ids = sorted(set(lab))
            for c in ids:
                m = Xtr[lab == c].mean(0)
                protos.append(norm(m))
            P = np.asarray(protos, np.float32)

            def pred(k):
                i = idx.get(k)
                if i is None:
                    return None, None
                S = Xte[i] @ P.T
                j = int(S.argmax())
                return ids[j], float(S[j])

            # cos AUC（直接用配对向量）
            def cos_pair(a, b):
                ia, ib = idx.get(a), idx.get(b)
                if ia is None or ib is None:
                    return None
                return float(np.dot(Xte[ia], Xte[ib]))

            pos = [c for c in (cos_pair(("318", a), ("320", b)) for a, b in POS30) if c is not None]
            neg = [c for c in (cos_pair(("318", a), ("320", b)) for a, b in NEG30) if c is not None]
            auc = auc_mw(pos, neg)

            # 选择器同簇判定
            pos_same = 0; pos_cov = 0
            for a, b in POS30:
                ca, sa = pred(("318", a)); cb, sb = pred(("320", b))
                if ca is None or cb is None:
                    continue
                pos_cov += 1
                if ca == cb:
                    pos_same += 1
            neg_same = 0; neg_cov = 0
            for a, b in NEG30:
                ca, sa = pred(("318", a)); cb, sb = pred(("320", b))
                if ca is None or cb is None:
                    continue
                neg_cov += 1
                if ca == cb:
                    neg_same += 1
            same_hit = 0; same_cov = 0
            for cam, a, b in splits["test"]["pairs_same"]:
                ca, _ = pred((cam, a)); cb, _ = pred((cam, b))
                if ca is None or cb is None:
                    continue
                same_cov += 1
                if ca == cb:
                    same_hit += 1

            res["grid"][f"thr={thr}"] = {
                "n_clusters": int(n_comp),
                "n_protos": len(P),
                "cos_auc_pos30_vs_neg30": round(auc, 4) if auc is not None else None,
                "pos30_same_cluster": f"{pos_same}/{pos_cov}",
                "neg30_same_cluster": f"{neg_same}/{neg_cov}",
                "pairs_same_hit": f"{same_hit}/{same_cov}",
                "cos_pos_median": round(float(np.median(pos)), 4) if pos else None,
                "cos_neg_median": round(float(np.median(neg)), 4) if neg else None,
            }
        report["encoders"][enc] = res
        print(f"[{enc}] train={len(Xtr)} test={len(Xte)}")
        for k, v in res["grid"].items():
            print(f'  {k}: clusters={v["n_clusters"]} AUC={v["cos_auc_pos30_vs_neg30"]} '
                  f'pos_same={v["pos30_same_cluster"]} neg_same={v["neg30_same_cluster"]} '
                  f'same_hit={v["pairs_same_hit"]}')

    op = out_dir / "selector_clean_eval.json"
    op.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {op}")


if __name__ == "__main__":
    main()
