# -*- coding: utf-8 -*-
"""EVAL-4：预测器验收改造——"下一门户头"（跟踪层口径）+ 三臂对照。

臂：
  A0 无预测：只用 zone-pair 历史转移频率（边际先验）
  A1 运动学基线：末段速度方向 → 门方向匹配
  A2 学习预测：10min 实体轨迹训练的 softmax 门分类器（per-cam）

评测：
  L1（大样本、迁移）：在 long35 六路轨迹上评估"出画门"预测 top1/top3（A2 vs A1 vs A0 marginal）
  L2（验收、小样本）：55 判 pos（13）+ 4 组 winner 的 a 端 → 预测下一门户（cam,gate）
      zone 分数 = Σ_g p(g|traj_a) × T[(cam_a,g)→(cam_b,gn)]（T 从 scored_v2，剔除被评对本对）
      指标：下一门户 top-k 命中率 + 下一机位 top-k（描述）+ dt 预测（zone 的 dt 中位数）MAE。
      分类 acc/mF1 仅作副指标（L1 top1 即分类精度）。

纪律：训练集 = 10min 实体（**剔除被评对的 a 端实体**，防自证）；不写任何生产文件。
输出：analysis/eval4/next_portal.json
"""
import json
import os
import csv
import sys
from collections import Counter, defaultdict

import numpy as np

os.environ.setdefault("DET_VIEW", "clean2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from crosscam_relay import load_entities  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
C0 = os.path.join(ANA, "c0", "c0_baseline_v7.json")
CAND = os.path.join(ANA, "relay_v2", "scored_v2.json")
LONG35 = os.path.join(ANA, "topology", "long35_data")
OUT = os.path.join(ANA, "eval4")
LABELS_JSON = os.path.join(ANA, "truth", "remote", "labels.json")
LABELED = os.path.join(ANA, "truth", "pair_candidates_labeled.csv")
VERDICTS = os.path.join(ANA, "truth", "_review", "conflict_verdicts.json")
CAMS = ["305", "316", "318", "320", "323", "343"]
RES_H = {"316": 1440, "305": 1440, "318": 1080, "320": 1440, "323": 1080, "343": 1440}
T0 = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}
GATE_R = 0.15
ZONE_R = 0.25


def parse_pair(s):
    cam, eid = s.split(":", 1)
    return cam, eid


def load_gates():
    d = json.load(open(C0, encoding="utf-8"))
    out = {}
    for cam, lst in d["gates"].items():
        out[cam] = dict(exit=[g["center"] for g in lst if g.get("kind") == "exit"],
                        entry=[g["center"] for g in lst if g.get("kind") == "entry"])
    return out


def dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def nearest(pt, gates, r):
    best, bi = None, None
    for i, g in enumerate(gates):
        dd = dist(pt, g)
        if best is None or dd < best:
            best, bi = dd, i
    return bi if (best is not None and best <= r) else None


def feats(u, v, h, npts):
    """轨迹特征（用轨迹尾部）：位置/速度/位移/尺度。"""
    k = min(6, len(u))
    du = (u[-1] - u[-k]) / max(k - 1, 1)
    dv = (v[-1] - v[-k]) / max(k - 1, 1)
    du_tot, dv_tot = u[-1] - u[0], v[-1] - v[0]
    sp = (du ** 2 + dv ** 2) ** 0.5
    return [u[-1], v[-1], du, dv, du_tot, dv_tot, sp, h[-1] / 1000.0, np.log1p(npts)]


def split_traj(u, v, h):
    n = len(u)
    m = max(4, int(n * 0.5))
    return u[:m], v[:m], h[:m]


def train_softmax(X, y, K, iters=500, lr=0.5, l2=1e-3):
    N, D = X.shape
    W, b = np.zeros((D, K)), np.zeros(K)
    Y = np.zeros((N, K))
    Y[np.arange(N), y] = 1
    for _ in range(iters):
        Z = X @ W + b
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        G = (P - Y) / N
        W -= lr * (X.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def main():
    os.makedirs(OUT, exist_ok=True)
    gates = load_gates()
    cand = json.load(open(CAND, encoding="utf-8"))["candidates"]

    # ---- 被评对（pos + winners + neg 描述）----
    def parse_pid(pid):
        left, right = pid.split("__")
        la, lb = left.split("_"), right.split("_")
        return (la[0], la[2]), (lb[0], lb[2])

    judged = {}
    for r in csv.DictReader(open(LABELED, encoding="utf-8-sig")):
        v = (r.get("human_same_person") or "").strip()
        if v in ("1", "0", "?"):
            judged[r["pair_id"]] = v
    lj = json.load(open(LABELS_JSON, encoding="utf-8"))
    for pid, rec in lj.get("pair", {}).items():
        # ★ 命名空间过滤：S6B_P_/PRT_ 前缀剥离后并入；实体级键（S6B_B_/S6B_R_ 等）跳过
        p2 = pid
        for pref in ("S6B_P_", "PRT_"):
            if p2.startswith(pref):
                p2 = p2[len(pref):]
                break
        if "__" not in p2:
            continue
        la, lb = p2.split("__", 1)[0].split("_"), p2.split("__", 1)[1].split("_")
        if len(la) < 3 or len(lb) < 3:
            continue
        judged[p2] = {1: "1", 0: "0", 2: "?"}.get(rec.get("v"), "?")
    groups = json.load(open(VERDICTS, encoding="utf-8"))["groups"]
    winner_pids = []
    for gk, gv in groups.items():
        for c in gv.get("candidates", []):
            if c.get("b") == gv["winner"]:
                winner_pids.append((c["pair_id"], getattr(main, "_g", gk)))
                break
    winner_pids = [w[0] for w in winner_pids]

    test_pids = [p for p, v in judged.items() if v == "1"] + winner_pids
    test_a = {parse_pid(p)[0] for p in test_pids}   # 待剔除的 a 端实体

    # ---- 训练集：10min 实体（剔除被评 a 端）----
    samples = defaultdict(lambda: dict(X=[], y=[]))
    ent_store = {}
    for cam in CAMS:
        ents = load_entities(cam)
        ent_store[cam] = ents
        gates_e = gates.get(cam, {}).get("exit", [])
        for eid, (fr, u, v, h) in ents.items():
            key = (cam, eid)
            if key in test_a or len(u) < 8:
                continue
            hu, hv, hh = split_traj(u, v, h)
            lab = nearest((u[-1], v[-1]), gates_e, GATE_R)
            lab = int(lab) if lab is not None else -1
            samples[cam]["X"].append(feats(hu, hv, hh, len(fr)))
            samples[cam]["y"].append(lab)
    print("[train] per-cam n:", {c: len(samples[c]["y"]) for c in CAMS}, flush=True)

    # ---- 训练 per-cam 模型 ----
    models = {}
    for cam in CAMS:
        Xs = np.asarray(samples[cam]["X"], np.float64)
        ys = np.asarray(samples[cam]["y"], np.int64)
        gates_e = gates.get(cam, {}).get("exit", [])
        K = len(gates_e) + 1   # gates + none
        if len(Xs) < 20:
            models[cam] = None
            continue
        mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
        W, b = train_softmax((Xs - mu) / sd, ys, K)
        freq = np.array([(ys == k).sum() for k in range(K)], float)
        freq /= freq.sum()
        models[cam] = dict(W=W, b=b, mu=mu, sd=sd, K=K, freq=freq, n=len(ys))
    print("[train] models:", {c: (models[c]["n"] if models[c] else 0) for c in CAMS}, flush=True)

    # ---- L1：long35 迁移评估 ----
    keep = json.load(open(os.path.join(LONG35, "splits_v4.json"), encoding="utf-8"))
    keep = {cam: set(keep["train"]["tracks"][cam]) for cam in CAMS}
    l1 = {}
    for cam in CAMS:
        W = {"316": 2560, "305": 2560, "318": 1920, "320": 2560, "323": 1920, "343": 2560}[cam]
        Hh = RES_H[cam]
        acc2 = {}
        with open(os.path.join(LONG35, f"{cam}_detections.jsonl"), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                for tid, box in zip(d["track_ids"], d["boxes"]):
                    if tid is None or int(tid) < 0 or int(tid) not in keep[cam]:
                        continue
                    x1, y1, x2, y2 = box
                    acc2.setdefault(int(tid), []).append(
                        (d["frame"], (x1 + x2) / 2 / W, y2 / Hh, y2 - y1))
        rows = []
        gates_e = gates.get(cam, {}).get("exit", [])
        for tid, lst in acc2.items():
            lst.sort()
            fr = np.array([x[0] for x in lst], float)
            u = np.array([x[1] for x in lst])
            v = np.array([x[2] for x in lst])
            h = np.array([x[3] for x in lst])
            if len(u) < 8:
                continue
            hu, hv, hh = split_traj(u, v, h)
            ytrue = nearest((u[-1], v[-1]), gates_e, GATE_R)
            if ytrue is None or models.get(cam) is None:
                rows.append((None, None, None))
                continue
            f = np.asarray(feats(hu, hv, hh, len(fr)))[None, :]
            mm = models[cam]
            P = softmax(((f - mm["mu"]) / mm["sd"]) @ mm["W"] + mm["b"])[0]
            p_gates = P[:len(gates_e)]
            # A1 运动学
            k = min(6, len(hu))
            du = (hu[-1] - hu[-k]) / max(k - 1, 1)
            dv = (hv[-1] - hv[-k]) / max(k - 1, 1)
            kscores = []
            for g in gates_e:
                vec = np.array([g[0] - hu[-1], g[1] - hv[-1]])
                nv = np.linalg.norm(vec) + 1e-9
                dirv = np.array([du, dv])
                nd = np.linalg.norm(dirv) + 1e-9
                cos = float(dirv @ vec / (nd * nv))
                kscores.append(cos - 0.3 * float(nv))
            kscores = np.array(kscores)
            # A0 边际
            freq_g = models[cam]["freq"][:len(gates_e)]
            lr_rank = np.argsort(-p_gates)
            kr_rank = np.argsort(-kscores)
            fr_rank = np.argsort(-freq_g)
            rows.append((int(ytrue), lr_rank.tolist(), kr_rank.tolist(), fr_rank.tolist()))
        rows = [r for r in rows if r[0] is not None and r[1]]
        if rows:
            t1 = sum(r[0] == r[1][0] for r in rows) / len(rows)
            t3 = sum(r[0] in r[1][:3] for r in rows) / len(rows)
            k1 = sum(r[0] == r[2][0] for r in rows) / len(rows)
            k3 = sum(r[0] in r[2][:3] for r in rows) / len(rows)
            u1 = sum(r[0] == r[3][0] for r in rows) / len(rows)
            u3 = sum(r[0] in r[3][:3] for r in rows) / len(rows)
            l1[cam] = dict(n=len(rows), lr_top1=round(t1, 4), lr_top3=round(t3, 4),
                           kin_top1=round(k1, 4), kin_top3=round(k3, 4),
                           marg_top1=round(u1, 4), marg_top3=round(u3, 4))
            print(f"[L1 {cam}] n={len(rows)} learned {t1:.3f}/{t3:.3f} "
                  f"kin {k1:.3f}/{k3:.3f} marg {u1:.3f}/{u3:.3f}", flush=True)

    # ---- 转移表 T（zone-pair→计数；剔除被评对本对）----
    c2zone = {}
    for c in cand:
        ca = c["a"].split(":")[0]
        cb = c["b"].split(":")[0]
        ge = nearest(c["a_end"], gates[ca]["exit"], ZONE_R)
        gn = nearest(c["b_start"], gates[cb]["entry"], ZONE_R)
        c2zone[(c["a"], c["b"])] = (ca, ge, cb, gn)
    test_pairs = {parse_pid(p) for p in test_pids}
    T = defaultdict(Counter)
    dt_zone_cam = defaultdict(list)
    for key, (ca, ge, cb, gn) in c2zone.items():
        if key in test_pairs:
            continue
        T[(ca, ge)][(cb, gn)] += 1
        if c["dt"] is not None:
            dt_zone_cam[(ca, cb, gn)].append(c["dt"])

    # ---- L2：判对回放 ----
    def l2_eval(pids, tag):
        res = dict(n=0, missing=0, lr_zone_top={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
                   kin_zone_top={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
                   marg_zone_top={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
                   lr_cam_top={1: 0, 2: 0, 3: 0}, kin_cam_top={1: 0, 2: 0, 3: 0},
                   marg_cam_top={1: 0, 2: 0, 3: 0},
                   dt_err=[], details=[])
        for pid in pids:
            (ca, ea), (cb, eb) = parse_pid(pid)
            ent = ent_store.get(ca, {}).get(ea)
            if ent is None or models.get(ca) is None:
                res.setdefault("missing", 0)
                res["missing"] += 1
                continue
            fr, u, v, h = ent
            if len(u) < 8:
                res["missing"] += 1
                continue
            hu, hv, hh = split_traj(u, v, h)
            gates_e = gates[ca]["exit"]
            b_ent = ent_store.get(cb, {}).get(eb)
            gn_true = None
            if b_ent is not None:
                gn_true = nearest((b_ent[1][0], b_ent[2][0]), gates[cb]["entry"], ZONE_R)
            # arms
            mm = models[ca]
            f = np.asarray(feats(hu, hv, hh, len(fr)))[None, :]
            P = softmax(((f - mm["mu"]) / mm["sd"]) @ mm["W"] + mm["b"])[0]
            p_g = P[:len(gates_e)]
            k = min(6, len(hu))
            du = (hu[-1] - hu[-k]) / max(k - 1, 1)
            dv = (hv[-1] - hv[-k]) / max(k - 1, 1)
            ks = []
            for g in gates_e:
                vec = np.array([g[0] - hu[-1], g[1] - hv[-1]])
                nv = np.linalg.norm(vec) + 1e-9
                dirv = np.array([du, dv])
                nd = np.linalg.norm(dirv) + 1e-9
                ks.append(float(dirv @ vec / (nd * nv)) - 0.3 * float(nv))
            p_kin = softmax(np.array(ks) * 3)
            p_marg = mm["freq"][:len(gates_e)] / max(mm["freq"][:len(gates_e)].sum(), 1e-9)
            zone_scores = {}
            for arm, p in (("lr", p_g), ("kin", p_kin), ("marg", p_marg)):
                zz = defaultdict(float)
                for gi, pg in enumerate(p):
                    for (cb2, gn2), cnt in T.get((ca, gi), {}).items():
                        tot = sum(T[(ca, gi)].values())
                        zz[(cb2, gn2)] += pg * cnt / tot
                zone_scores[arm] = zz
            if gn_true is not None:
                res["n"] += 1
                zt = (cb, gn_true)
                for arm in ("lr", "kin", "marg"):
                    ranks = sorted(zone_scores[arm].items(), key=lambda x: -x[1])
                    keys = [k2 for k2, _ in ranks]
                    for kk in (1, 2, 3, 4, 5, 6):
                        if zt in keys[:kk]:
                            res[f"{arm}_zone_top"][kk] += 1
                    # cam level
                    cam_zz = defaultdict(float)
                    for (cb2, gn2), s in zone_scores[arm].items():
                        cam_zz[cb2] += s
                    ckeys = [k2 for k2, _ in sorted(cam_zz.items(), key=lambda x: -x[1])]
                    for kk in (1, 2, 3):
                        if cb in ckeys[:kk]:
                            res[f"{arm}_cam_top"][kk] += 1
                if (gn_true is not None and b_ent is not None and len(ent[0]) and len(b_ent[0])):
                    dz = dt_zone_cam.get((ca, cb, gn_true), [])
                    if dz:
                        dt_actual = (b_ent[0][0] - ent[0][-1]) / 25.0 + (T0[cb] - T0[ca])
                        res["dt_err"].append(round(abs(float(np.median(dz)) - dt_actual), 2))
                res["details"].append(dict(pid=pid, zone_true=str(zt),
                                           lr_top=[k2 for k2, _ in sorted(zone_scores['lr'].items(), key=lambda x: -x[1])][:3]))
        return res

    pos = [p for p, v in judged.items() if v == "1"]
    neg = [p for p, v in judged.items() if v == "0"]
    l2_pos = l2_eval(pos, "pos")
    l2_win = l2_eval(winner_pids, "win")
    l2_neg = l2_eval(neg, "neg")
    # v2 active 边（47 条）作为代理锚点（非真值；检验门户先验与已采纳边的一致性）
    edges = json.load(open(os.path.join(ANA, "relay_v2", "edges_v2_gated.json"),
                           encoding="utf-8"))["edges"]
    def to_pid(a, b):
        return a.replace(":", "_E_") + "__" + b.replace(":", "_E_")
    active_pids = [to_pid(e["a"], e["b"]) for e in edges if e["status"] == "active"]
    l2_active = l2_eval(active_pids, "active")

    out = dict(version="eval4/1",
               note="训练=10min 实体（剔除被评 a 端）；L1 迁移到 long35；L2 判对回放；"
                    "dt 预测留作描述（小样本）",
               train_n={c: (models[c]["n"] if models[c] else 0) for c in CAMS},
               L1_long35=l1, L2_pos=l2_pos, L2_winners=l2_win, L2_neg=l2_neg,
               L2_active47=l2_active)
    with open(os.path.join(OUT, "next_portal.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print("[L2 pos] n:", l2_pos.get("n"), "lr zone top1/3/4/6:",
          l2_pos["lr_zone_top"][1], l2_pos["lr_zone_top"][3], l2_pos["lr_zone_top"][4],
          l2_pos["lr_zone_top"][6],
          "| kin:", l2_pos["kin_zone_top"][1], l2_pos["kin_zone_top"][3],
          l2_pos["kin_zone_top"][4], l2_pos["kin_zone_top"][6])
    print("[L2 active47] n:", l2_active.get("n"), "lr zone top1/3/5:",
          l2_active["lr_zone_top"][1], l2_active["lr_zone_top"][3], l2_active["lr_zone_top"][5],
          "| cam top1/2/3:", l2_active["lr_cam_top"][1], l2_active["lr_cam_top"][2],
          l2_active["lr_cam_top"][3])
    print("[L2 win] n:", l2_win.get("n"), "lr/kin zone top3:", l2_win["lr_zone_top"][3],
          l2_win["kin_zone_top"][3])
    print("saved", os.path.join(OUT, "next_portal.json"))


if __name__ == "__main__":
    main()
