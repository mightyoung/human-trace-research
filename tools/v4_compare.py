# -*- coding: utf-8 -*-
"""R4.1 v4（clean 视图）vs v3（raw）对照评估。

协议 A（318×320 判读对）：
  v3: 8 正 + 22 负（30 对）
  v4: 8 正 + 20 负（NEG30 的 (98,26)/(53,11) 因 318#98/#53 为清洗剔除的倒影轨而不可评估）
  指标：AUC / margin / pos_min / neg_max + 逐对 sim 变化表
检索：8 查询（318 侧）→ gallery 其余 5 路 / 仅 320 库 → rank 对照

用法：
  python tools/v4_compare.py                 # 本地
  python tools/v4_compare.py --emb-v4 output_crosscam/analysis/cradio_scale_v4
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ["318", "320", "305", "316", "323", "343"]
POS30 = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
NEG30 = [(14, 1), (131, 21), (61, 14), (50, 11), (132, 21), (3, 1), (7, 1), (146, 21),
         (127, 19), (123, 19), (57, 11), (131, 23), (124, 18), (131, 24), (3, 4),
         (126, 18), (43, 11), (10, 1), (131, 28), (98, 26), (142, 26), (53, 11)]
NEG30_DROPPED = [(98, 26), (53, 11)]   # 清洗剔除（318#98 / 318#53 为倒影轨）


def key_to_tid(k):
    num = ""
    for ch in reversed(k):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def load_embs(d, pat="{cam}_emb_scale.npz"):
    out = {}
    for cam in CAMS:
        p = os.path.join(d, pat.format(cam=cam))
        if not os.path.exists(p):
            print(f"[warn] 缺 {p}")
            continue
        z = np.load(p)
        out[cam] = {key_to_tid(k): z[k].astype(np.float32) for k in z.files}
    return out


def sim(embs, c1, t1, c2, t2):
    if c1 not in embs or c2 not in embs or t1 not in embs[c1] or t2 not in embs[c2]:
        return None
    a, b = embs[c1][t1], embs[c2][t2]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def auc(pv, nv):
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pv for n in nv]))


def proto_a(embs):
    pv, nv, rows = [], [], []
    for a, b in POS30:
        s = sim(embs, "318", a, "320", b)
        pv.append(s); rows.append(("pos", a, b, s))
    for a, b in NEG30:
        s = sim(embs, "318", a, "320", b)
        if s is not None:
            nv.append(s)
        rows.append(("neg", a, b, s))
    r = {"n_pos": len([x for x in pv if x is not None]),
         "n_neg": len(nv),
         "auc": round(auc([x for x in pv if x is not None], nv), 4),
         "pos_min": round(min([x for x in pv if x is not None]), 4),
         "neg_max": round(max(nv), 4)}
    r["margin"] = round(r["pos_min"] - r["neg_max"], 4)
    return r, rows


def retrieval(embs, query_cam="318"):
    gal = [c for c in CAMS if c != query_cam and c in embs]
    keys, vecs = [], []
    for c in gal:
        for tid, v in embs[c].items():
            keys.append((c, tid)); vecs.append(v)
    E = np.stack(vecs); E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    rk_all, rk_320 = [], []
    for a, b in POS30:
        if query_cam not in embs or a not in embs[query_cam]:
            continue
        q = embs[query_cam][a]
        q = q / (np.linalg.norm(q) + 1e-9)
        s = E @ q
        order = np.argsort(-s)[:600]
        hit = next((i + 1 for i, j in enumerate(order) if keys[j] == ("320", b)), None)
        rk_all.append(hit)
        if "320" in embs and b in embs["320"]:
            s3 = np.array([embs["320"][t] for t in sorted(embs["320"])])
            s3n = s3 / (np.linalg.norm(s3, axis=1, keepdims=True) + 1e-9)
            tids = sorted(embs["320"])
            o3 = np.argsort(-(s3n @ q))
            hit3 = next((i + 1 for i, j in enumerate(o3) if tids[j] == b), None)
            rk_320.append(hit3)
    return {"all_rank1": sum(1 for r in rk_all if r == 1), "n": len(rk_all),
            "all_ranks": rk_all,
            "avg_all": round(float(np.mean([r for r in rk_all if r])), 2) if rk_all else None,
            "only320_rank1": sum(1 for r in rk_320 if r == 1), "only320_ranks": rk_320,
            "avg_320": round(float(np.mean([r for r in rk_320 if r])), 2) if rk_320 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-v3", default=os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale_v3"))
    ap.add_argument("--emb-v4", default=os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale_v4"))
    ap.add_argument("--emb-pat", default="{cam}_emb_scale.npz",
                    help="npz 文件名模式（如 d3b: {cam}_emb_d3b.npz）")
    ap.add_argument("--single", action="store_true",
                    help="单模型模式：只评估 emb-v4 侧（用于 d3b 等备选桥）")
    ap.add_argument("--out", default=os.path.join(ROOT, "output_crosscam", "analysis", "bridge_scale",
                                                  "eval_v4_vs_v3.json"))
    args = ap.parse_args()

    if args.single:
        print("[load] single ..."); e = load_embs(args.emb_v4, args.emb_pat)
        pa, rows = proto_a(e)
        rt = retrieval(e)
        out = {"single": {"protoA": pa,
                          "rows": [{"kind": k, "a": a, "b": b, "sim": s} for k, a, b, s in rows],
                          "retrieval": rt}}
        print(f"\n[single] 协议A: AUC={pa['auc']} margin={pa['margin']:+.4f} "
              f"(pos {pa['n_pos']} / neg {pa['n_neg']})")
        print(f"[single] 检索: 全库 rank1 {rt['all_rank1']}/{rt['n']} avg {rt['avg_all']} | "
              f"320库 rank1 {rt['only320_rank1']} avg {rt['avg_320']}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n[out] {args.out}")
        return

    print("[load] v3 ..."); e3 = load_embs(args.emb_v3, args.emb_pat)
    print("[load] v4 ..."); e4 = load_embs(args.emb_v4, args.emb_pat)
    out = {}
    for name, e in [("v3_raw", e3), ("v4_clean", e4)]:
        pa, rows = proto_a(e)
        rt = retrieval(e)
        out[name] = {"protoA": pa, "rows": [{"kind": k, "a": a, "b": b, "sim": s}
                                            for k, a, b, s in rows], "retrieval": rt}
        print(f"\n[{name}] 协议A: AUC={pa['auc']} margin={pa['margin']:+.4f} "
              f"(pos {pa['n_pos']} / neg {pa['n_neg']})")
        print(f"[{name}] 检索: 全库 rank1 {rt['all_rank1']}/{rt['n']} avg {rt['avg_all']} | "
              f"320库 rank1 {rt['only320_rank1']} avg {rt['avg_320']}")

    # 逐对变化表
    print("\n===== 逐对 sim 变化（v3 → v4，仅列有变化/被剔除） =====")
    m3 = {(r["kind"], r["a"], r["b"]): r["sim"] for r in out["v3_raw"]["rows"]}
    m4 = {(r["kind"], r["a"], r["b"]): r["sim"] for r in out["v4_clean"]["rows"]}
    n_chg = 0
    for k in sorted(m3):
        s3v, s4v = m3[k], m4.get(k)
        if s4v is None:
            print(f"  [剔除] {k[0]} ({k[1]},{k[2]})  v3={s3v:.4f} → v4=不可评估")
            n_chg += 1
        elif abs(s3v - s4v) > 0.005:
            print(f"  [{k[0]}] ({k[1]},{k[2]})  {s3v:.4f} → {s4v:.4f}  (Δ{s4v-s3v:+.4f})")
            n_chg += 1
    if not n_chg:
        print("  （无显著变化）")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
