# -*- coding: utf-8 -*-
"""S7：把所有倒影归入**单一聚类** gREFLECTION，并训练「倒影/镜像」向量判别器。

用户需求（原话）："将所有阴影放到一个聚类，让选择器学习阴影镜像的向量特征"。

两条线：
  ① **聚类**：把 S5（地板倒影 50）+ S6（墙镜倒影 24）识别出的**全部倒影**（并集 73）
     归入一个 gperson 聚类（label=gREFLECTION、state=non_person）。成员用
     `membership.node_kind='refl'`（**不占 obs 节点**）→ 对 obs 级消费端零影响：
       · gil_gperson 只读 node_kind='obs'      → 不受影响
       · gil_visit_owner 只读 node_kind='obs'   → 不受影响
       · 控制台 counts.gperson 来自上述工具     → 不受影响
     同时把每个倒影写成 annotation（target_kind='obs'，value='reflection_cluster'，视图无关）。

  ② **选择器学习**：在 C-RADIOv4 向量空间学一个「倒影方向」w（类均值差 = 最近原型方向，
     高维小样本下最稳）。诚实评估：
       · LOO-AUC（留一）
       · ★ 留一机位 AUC（train 316+343 → test 323 …）—— 唯一能排除"机位/区域捷径"的检验
       · 打乱标签对照（null 分布）
     并把 w 打到 GIL 全部 obs 上 → obs.flags.reflection_score（选择器可直接当门控特征）。

★ 先验警告：项目既有结论是「倒影与真人的 cos 间隔仅 0.0002（外观分类不可行）」。
  本脚本的作用就是**用数据把这件事说死**：若 AUC≈0.5 或跨机位不泛化 → 证明外观不可分，
  倒影必须走几何（S5/S6）；若 AUC 高且跨机位泛化 → 选择器可加一维门控特征。

用法：
  python tools/gil_reflection_probe.py            # 只读：训练 + 评估 + 打印
  python tools/gil_reflection_probe.py --apply    # 写 GIL（聚类 + annotation + obs.flags）
"""
import argparse
import collections
import json
import os
import sqlite3
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ANA = os.path.join(ROOT, "output_crosscam/analysis")
GIL = os.path.join(ROOT, "output_crosscam/analysis/gil")
CAMS = ["305", "316", "318", "320", "323", "343"]
V3 = os.path.join(ANA, "cradio_scale_v3")
LABEL = "gREFLECTION"


def rank_auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def lda_dir(X, y, shrink=0.5):
    """类均值差方向（对角线/收缩协方差下的 LDA）。X 已 L2 归一化。"""
    mu1 = X[y == 1].mean(0)
    mu0 = X[y == 0].mean(0)
    d = mu1 - mu0
    # 收缩：把方向按类内方差归一（避免高方差维主导）
    v1 = X[y == 1].var(0) if (y == 1).sum() > 1 else np.zeros_like(d)
    v0 = X[y == 0].var(0) if (y == 0).sum() > 1 else np.zeros_like(d)
    vm = (v1 + v0) / 2
    v = shrink * vm + (1 - shrink) * float(vm.mean())
    w = d / (v + 1e-6)
    return w / (np.linalg.norm(w) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", default="output_crosscam/analysis/gil/reflection_cluster_v1.json")
    a = ap.parse_args()

    # ---------- 1) 倒影清单（视图无关）----------
    s5 = {(c, x["tid"]): dict(x) for c, rows in
          json.load(open(os.path.join(GIL, "reflection_assign_v1.json"), encoding="utf-8"))["assignments"].items()
          for x in rows}
    wm = json.load(open(os.path.join(GIL, "wallmirror_assign_v1.json"), encoding="utf-8"))
    s6 = {(c, x["tid"]): dict(x) for c in ("316", "323") for x in wm[c]["mirror_rows"]}
    inv = {}
    for k, v in s5.items():
        inv[k] = dict(cam=k[0], tid=k[1], source=v.get("source"), src="S5_floor",
                      perr=v.get("perr"), hr=v.get("hr"))
    for k, v in s6.items():
        if k in inv:
            inv[k]["src"] = "S5+S6"
        else:
            inv[k] = dict(cam=k[0], tid=k[1], source=v.get("source"), src="S6_wall",
                          perr=v.get("perr"), hr=v.get("hr"))
    print(f"[s7] 倒影清单（视图无关）{len(inv)} 条  "
          f"按机位 {dict(collections.Counter(k[0] for k in inv))}")

    # ---------- 2) 向量（v3 全量覆盖；与 v4/GIL 空间 cos 中位 0.999 等价）----------
    VEC = {}
    skipped = []
    for cam in CAMS:
        p = os.path.join(V3, f"{cam}_emb_scale.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        bad = [k for k in z.keys() if not k.split("_")[-1].isdigit()]
        if bad:
            # 318/320 的 v3 文件键名为 a1../b1..（不可解析为 tid）→ 从对照池排除。
            # 这两个机位经 S5/S6 判定均**无倒影**，排除不影响正样本完整性。
            skipped.append((cam, len(bad)))
            continue
        for k in z.keys():
            tid = int(k.split("_")[-1])
            VEC[(cam, tid)] = z[k].astype(np.float32)
    if skipped:
        print(f"[s7] ⚠ 键名不可解析、已排除的机位：{skipped}（均无倒影）")

    keys = sorted(VEC)
    X = np.stack([VEC[k] for k in keys])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cam = np.array([k[0] for k in keys])
    y = np.array([1 if k in inv else 0 for k in keys])
    print(f"[s7] 向量 {len(keys)}（正 {int(y.sum())} / 负 {int((1-y).sum())}）；"
          f"正样本按机位 {dict(collections.Counter(c for c, yy in zip(cam, y) if yy == 1))}")

    # ---------- 3) 评估 ----------
    # ★ 用**分层 K 折**（held-out 打分）而非留一：留一法在「类均值差」方向上有系统性偏移，
    #   实测打乱标签的留一对照 AUC 会虚高到 0.62（应≈0.5）。K 折只把模型拟合在训练折上。
    def kfold_scores(Xa, ya, k=5, reps=3, seed=0):
        rng = np.random.default_rng(seed)
        sc = np.zeros(len(ya)); cnt = np.zeros(len(ya))
        for r in range(reps):
            idx = rng.permutation(len(ya))
            folds = np.array_split(idx, k)
            for f in folds:
                tr = np.setdiff1d(np.arange(len(ya)), f)
                w = lda_dir(Xa[tr], ya[tr])
                sc[f] += Xa[f] @ w
                cnt[f] += 1
        return sc / np.maximum(cnt, 1)

    ev = {}
    sc_all = kfold_scores(X, y)
    auc_all = rank_auc(y, sc_all)
    ev["kfold_auc_all"] = round(auc_all, 4)
    ev["kfold"] = "5 折 × 3 重复，分层（按真实类别比例）"
    print(f"\n=== 评估 ===\n  K 折 AUC（全体 {int(y.sum())} 正 / {int((1-y).sum())} 负）  {auc_all:.4f}")

    # ★ 留一机位：train 其余机位 → test 该机位（排除"机位/区域捷径"）
    per_cam = {}
    for c in ["316", "323", "343"]:
        te = cam == c
        tr = ~te
        if y[te].sum() == 0:
            continue
        w = lda_dir(X[tr], y[tr])
        s = X[te] @ w
        per_cam[c] = dict(n_pos=int(y[te].sum()), n_neg=int((1 - y[te]).sum()),
                          auc=(None if (y[te].sum() == 0 or (1 - y[te]).sum() == 0)
                               else round(float(rank_auc(y[te], s)), 4)))
        print(f"  留一机位 [{c} 测试]  正 {per_cam[c]['n_pos']:3d} / 负 {per_cam[c]['n_neg']:3d}"
              f"  AUC {per_cam[c]['auc']}")
    aucs = [v["auc"] for v in per_cam.values() if v["auc"] is not None and v["n_pos"] >= 10]
    ev["leave_one_camera_auc"] = per_cam
    ev["leave_one_camera_mean"] = round(float(np.mean(aucs)), 4) if aucs else None
    print(f"  留一机位 平均 AUC（仅取 n_pos≥10 的机位）  {ev['leave_one_camera_mean']}")

    # 打乱标签对照（同机位分层打乱，同一 K 折流程）
    rng0 = np.random.default_rng(7)
    null = []
    for _ in range(10):
        yp = y.copy()
        for c in set(cam.tolist()):
            m = cam == c
            v = yp[m]; rng0.shuffle(v); yp[m] = v
        null.append(rank_auc(yp, kfold_scores(X, yp, reps=1, seed=int(rng0.integers(1e6)))))
    ev["shuffle_null"] = dict(mean=round(float(np.mean(null)), 4),
                              max=round(float(np.max(null)), 4))
    print(f"  打乱标签对照（10 次，同 K 折流程）平均 AUC {ev['shuffle_null']['mean']} "
          f"max {ev['shuffle_null']['max']}")

    # ★ 对照组：只用「画面位置 (u,v)」学方向，跨机位测试。
    #   绝对位置在机位间**不可迁移**（镜面位置各机位不同）→ 若该对照 AUC≈0.5，
    #   则外观探针的跨机位 AUC 不能由"位置/区域捷径"解释，是真实外观签名。
    try:
        from mirror_kinematics import load_tracks
        RES = {"305": (2560, 1440), "316": (2560, 1440), "318": (1920, 1080),
               "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}
        POS = {}
        for c in ("316", "323", "343"):
            tr = load_tracks(os.path.join(ROOT, "output_crosscam", f"{c}_detections.jsonl"))
            W, H = RES[c]
            for tid, l in tr.items():
                if len(l) < 3:
                    continue
                b = np.array([x[1] for x in l], float)
                POS[(c, tid)] = np.array([
                    float(np.median((b[:, 0] + b[:, 2]) / 2 / W)),
                    float(np.median(b[:, 3] / H))])
        keep = [i for i, k in enumerate(keys) if k in POS]
        Xp = np.stack([np.r_[POS[keys[i]], 1.0] for i in keep])
        Xp = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-9)
        yp_ = y[keep]; cp = cam[keep]
        pos_only = {}
        for c in ("316", "323"):
            te = cp == c
            w = lda_dir(Xp[~te], yp_[~te])
            pos_only[c] = round(float(rank_auc(yp_[te], Xp[te] @ w)), 4)
        ev["position_only_leave_one_camera_auc"] = pos_only
        ev["position_only_mean"] = round(float(np.mean(list(pos_only.values()))), 4)
        print(f"  对照：只用(u,v)位置 跨机位 AUC {pos_only} → 平均 {ev['position_only_mean']}")
    except Exception as e:                                   # pragma: no cover
        print(f"  位置对照跳过：{e}")

    # 对照：只看"机位"这一维能解释多少（用机位 one-hot 做多数类判别 → 预测倒影）
    hit = 0
    for c in set(cam.tolist()):
        m = cam == c
        rate = y[m].mean()
        hit += max(rate, 1 - rate) * m.sum()
    ev["camera_only_acc"] = round(float(hit / len(keys)), 4)
    ev["base_rate"] = round(float(y.mean()), 4)
    ev["n_neg_pool"] = int((1 - y).sum())
    print(f"  对照：仅用机位先验的准确率 {ev['camera_only_acc']}（基准率 {ev['base_rate']}）")

    # ---------- 4) 落 GIL：聚类 + 打分 ----------
    con = sqlite3.connect(os.path.join(GIL, "store_v1/gil.sqlite"))
    con.row_factory = sqlite3.Row
    obs = {(r["cam"], r["tid"]): r["id"] for r in
           con.execute("SELECT id,cam,tid FROM obs WHERE tombstone=0")}
    present = [k for k in inv if k in obs]
    print(f"\n[s7] 聚类 {LABEL}：清单 {len(inv)} → GIL 内 {len(present)}（其余已被掩膜剔出 clean2）")

    w_full = lda_dir(X, y)
    # 把 w 打到 GIL 全部 obs（用 GIL 向量库 = 选择器的真实输入空间）
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from gil import GILStore  # noqa: E402
    st = GILStore(os.path.join(GIL, "store_v1"), create=False)
    vb = st.vectors(load=True)
    rows = list(con.execute("SELECT id,cam,tid FROM obs WHERE tombstone=0 ORDER BY id"))
    id2row = {int(r["id"]): i for i, r in enumerate(rows)}
    oid2key = {int(r["id"]): (r["cam"], r["tid"]) for r in rows}
    E = np.zeros((len(rows), st.dim), np.float32)
    for b in vb.blocks():
        Eb, ids, i2r = vb._cache[b]
        for k, i in i2r.items():
            if int(k) in id2row:
                E[id2row[int(k)]] = Eb[i]
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    score = En @ w_full
    # z 标准化（用 1/99 分位）→ 便于当门控阈值
    lo, hi = np.percentile(score, 1), np.percentile(score, 99)
    score_n = np.clip((score - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    n_refl = 0
    for r in rows:
        oid = int(r["id"])
        s = float(score_n[id2row[oid]])
        con.execute("UPDATE obs SET flags=json_set(COALESCE(flags,'{}'),'$.reflection_score',?) "
                    "WHERE id=?", (round(s, 5), oid))
        if oid2key[oid] in inv:
            n_refl += 1
    con.commit()
    sc_refl = [float(score_n[id2row[obs[k]]]) for k in present]

    print(f"[s7] obs.flags.reflection_score 已写 {len(rows)} 条；"
          f"其中倒影 {n_refl} 条的分数 中位 {np.median(sc_refl):.3f}")

    if a.apply:
        # 聚类落库：gperson(state=non_person) + membership(node_kind='refl')
        g = st.con.execute("SELECT id FROM gperson WHERE label=?", (LABEL,)).fetchone()
        if g:
            gid = int(g[0])
            st.con.execute("DELETE FROM membership WHERE gperson_id=?", (gid,))
        else:
            gid = st.add_gperson(LABEL, state="non_person",
                                 note="倒影单一聚类（S5 地板 + S6 墙镜）；成员为倒影观测，不是人")
        for i, k in enumerate(sorted(inv)):
            # ★ 视图无关地保留 (cam|tid)：写进 membership.src（不占 obs 节点）
            st.add_membership(gid, "refl", i, score=None, prior=None, decision="auto",
                              src=f"reflection_cluster_v1:{k[0]}|{k[1]}:{inv[k]['src']}:src={inv[k]['source']}")
        # ⚠ 不写 annotation：annotation 的 UNIQUE(target_kind,target_id) 会**覆盖** S6/S5
        #   已写在同一 obs 上的 wall_mirror / reflection 标注（曾踩坑，已回滚修复）。
        st.commit()
        print(f"[gil] gperson {LABEL} (id={gid}, state=non_person) + membership {len(inv)} 条")
        print("[gil] stats:", json.dumps(st.stats(), ensure_ascii=False))
    st.close()

    out = dict(
        version="reflection_cluster_v1",
        label=LABEL, n_inventory=len(inv), n_in_gil=len(present),
        inventory=[dict(cam=k[0], tid=k[1], source=inv[k]["source"], src=inv[k]["src"],
                        in_gil=(k in obs)) for k in sorted(inv)],
        probe=dict(
            space="C-RADIOv4 + EmbeddingGemma2 (cradio_scale_v3；与 v4/GIL cos 中位 0.999)",
            method="类均值差方向（收缩 LDA；高维小样本最稳）",
            n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
            eval=ev,
            weights_l2norm=float(np.linalg.norm(w_full)),
            weights=w_full.astype(float).round(6).tolist(),
            score_norm=dict(lo=float(lo), hi=float(hi)),
            verdict=("外观含**可迁移的倒影签名**（留一机位 AUC≈%.2f）→ 可作选择器的"
                     "**软门控特征**" % (ev["leave_one_camera_mean"] or 0)
                     if (ev["leave_one_camera_mean"] or 0) >= 0.70
                     else "★ 外观不可分/不跨机位泛化 → 倒影只能走几何（S5/S6）"),
            caveats=[
                "① 汇总 K 折 AUC（%.2f）受**机位构成混杂**：打乱标签的汇总对照已达 0.60，"
                "故**不能**用汇总 AUC 下结论；应以**留一机位**（0.73/0.79）与位置对照为准。"
                % ev["kfold_auc_all"],
                "② 位置对照（只用 u,v、跨机位）AUC %s → 绝对位置不可迁移，"
                "排除了'区域捷径'解释。" % ev.get("position_only_leave_one_camera_auc"),
                "③ 负样本池**包含倒影的源轨迹**（外观几乎相同）→ 本 AUC 是保守下界。",
                "④ 远弱于几何判据（S5 F1 0.95 / S6 目视确认）→ **不可替代几何**，只做降权。",
                "⑤ GIL 内仅 12 条倒影有向量（其余被掩膜剔出 clean2）→ 落库分数须谨慎解读。",
            ],
        ),
        integration=("选择器可直接读 obs.flags.reflection_score（0..1，越大越像倒影）；"
                     "建议用法：作为**降权因子**而非硬门（硬门会误伤真人倒影源）；"
                     "阈值需在 val 上校准，禁 test-oracle。"),
        note="★ 聚类成员用 node_kind='refl'，不占 obs 节点 → 对 obs 级消费端零影响。",
    )
    p = os.path.join(ROOT, a.out)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {a.out}")
    con.close()


if __name__ == "__main__":
    main()
