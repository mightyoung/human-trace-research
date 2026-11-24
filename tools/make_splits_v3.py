# -*- coding: utf-8 -*-
"""构建 训练/验证/测试 三集划分（实体粒度，防数据泄漏）。

规则：
- 测试集 test：全部人工判读评估对涉及的**实体**（跨机位 30 对协议 + labels_v2 跨机位 + 同机位链接对）
  实体的全部成员轨迹入 test（同一人的其它碎片也不得进训练/验证）。
- 验证集 val：实体图中不含任一测试轨的实体，按比例抽样（seed=42）。
  用途：① 接力 sim 阈值校准；② 无监督度量超参选择；③ 桥接泛化体检。
  val 正对 = 实体内部跨片段对（同人、跨时间）；val 负对 = 跨实体同机位硬负（d3b 嵌入 top 相似，模拟最难假阳性）。
- 训练集 train：其余全部轨迹（桥接 W 拟合 + 无监督度量训练）。

输出：
  output_crosscam/analysis/splits_{SPLITS_TAG}.json   （完整划分 + 冻结哈希）
  output_crosscam/analysis/{SPLITS_TAG}_holdout_server.json （服务器跳过表：test ∪ val）

口径（2026-10-14 起）：
  - 实体图 = **规范目录 `analysis/stitch_clean2`**（411 实体、悬空 0），可用 NET_DIR 覆盖；
  - 检测视图 = `DET_VIEW`（默认 raw；clean2 实验须显式 DET_VIEW=clean2）；
  - 两者不配套会直接中止（防"口径分裂"，见 docs/clean2-rebuild-report.md）。
  - 默认 SPLITS_TAG 已由 v3 改为 **v5**，避免误覆盖冻结的 splits_v3。
  - 冻结哈希：`frozen_sha1_16` 只覆盖评估对（口径无关，应与历史一致）；
    `frozen_test_tracks_sha1_16` / `frozen_splits_sha1_16` 随口径变化，用于新口径冻结。
"""
import json
import os
import hashlib
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from det_view import (local_det, describe,             # noqa: E402
                      net_tracks_path, net_dir, net_consistency, current_view)

SPLITS_TAG = os.environ.get("SPLITS_TAG", "v5")   # 输出文件名 splits_{tag}.json
OUT = os.path.join(ROOT, "output_crosscam", "analysis")
CAMS = ["318", "320", "305", "316", "323", "343"]
SEED = 42

# 评估标签文件：实体图目录优先，缺失则回退 stitch/（标签与实体图口径无关）
def labels_path():
    return next((p for p in (os.path.join(net_dir(), "labels_v2.json"),
                             os.path.join(OUT, "stitch", "labels_v2.json"))
                 if os.path.exists(p)), os.path.join(OUT, "stitch", "labels_v2.json"))

# ---- 评估对（测试集）----
POS30 = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
NEG30 = [(14, 1), (131, 21), (61, 14), (50, 11), (132, 21), (3, 1), (7, 1), (146, 21),
         (127, 19), (123, 19), (57, 11), (131, 23), (124, 18), (131, 24), (3, 4),
         (126, 18), (43, 11), (10, 1), (131, 28), (98, 26), (142, 26), (53, 11)]


def load_all_tracks():
    t = {}
    for cam in CAMS:
        tids = set()
        with open(local_det(cam), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                for x in (r.get("track_ids") or []):
                    if x is not None and x >= 0:
                        tids.add(int(x))
        t[cam] = tids
    return t


def load_entities(all_tracks):
    """实体图（规范口径 = stitch_clean2）→ cam -> [(eid, tracks)]。
    数据视图过滤：实体引用的轨若不在当前视图轨集（如被清洗剔除），从实体中移除；
    全轨被剔除的实体整体剔除（记录数量）。"""
    ents = {}
    n_drop = {}
    for cam in CAMS:
        d = json.load(open(net_tracks_path(cam), encoding="utf-8"))
        lst, drop = [], 0
        for e in d["entities"]:
            trs = set(int(x) for x in e["tracks"]) & all_tracks[cam]
            if trs:
                lst.append((e["eid"], trs))
            else:
                drop += 1
        ents[cam] = lst
        n_drop[cam] = drop
    if any(n_drop.values()):
        print(f"[view-filter] 实体图中全轨被剔除的实体: "
              f"{ {c: n for c, n in n_drop.items() if n} }")
    return ents


def load_d3b(cam):
    p = os.path.join(OUT, "relay", "remote_emb", f"reid_emb_{cam}_d3b.npz")
    if not os.path.exists(p):
        return {}
    d = np.load(p)

    def k2t(k):
        num = ""
        for ch in reversed(k):
            if ch.isdigit():
                num = ch + num
            else:
                break
        return int(num)
    return {k2t(k): d[k] for k in d.files}


def main():
    rng = np.random.RandomState(SEED)
    ok, msg = net_consistency()
    print(msg)
    if not ok:
        raise SystemExit(f"[中止] {msg}")
    all_tracks = load_all_tracks()
    ents = load_entities(all_tracks)

    # ---------- 1. 评估轨 → 测试实体 ----------
    eval_tracks = {c: set() for c in CAMS}
    for a, b in POS30 + NEG30:
        eval_tracks["318"].add(a)
        eval_tracks["320"].add(b)
    eval_dropped = {}

    # 数据视图过滤：评估轨若在当前视图不存在（如被清洗的倒影轨），剔除并记录
    for cam in CAMS:
        miss = eval_tracks[cam] - all_tracks[cam]
        if miss:
            eval_dropped[cam] = sorted(miss)
        eval_tracks[cam] &= all_tracks[cam]
    if eval_dropped:
        print(f"[view-filter] 评估轨被剔除（当前视图不存在）: {eval_dropped}")

    lv = json.load(open(labels_path(), encoding="utf-8"))
    same_pos = [(str(p["cam"]), int(p["a"]), int(p["b"])) for p in lv["same_camera"]["positives"]]
    same_neg = [(str(p["cam"]), int(p["a"]), int(p["b"])) for p in lv["same_camera"]["negatives"]]
    cross_pos = [("318", int(a), "320", int(b)) for a, b in lv["cross_camera"]["positives"]]
    cross_neg = [("318", int(a), "320", int(b)) for a, b in lv["cross_camera"]["negatives"]]
    for c, a, b in same_pos + same_neg:
        eval_tracks[c].add(a)
        eval_tracks[c].add(b)
    for c1, a, c2, b in cross_pos + cross_neg:
        eval_tracks[c1].add(a)
        eval_tracks[c2].add(b)
    for cam in CAMS:
        miss = eval_tracks[cam] - all_tracks[cam]
        if miss:
            eval_dropped.setdefault(cam, [])
            eval_dropped[cam] = sorted(set(eval_dropped[cam]) | set(miss))
        eval_tracks[cam] &= all_tracks[cam]

    test_tracks = {c: set() for c in CAMS}
    test_ent_ids = {c: set() for c in CAMS}
    for cam in CAMS:
        for eid, trs in ents[cam]:
            if trs & eval_tracks[cam]:
                test_tracks[cam] |= trs
                test_ent_ids[cam].add(eid)
        # 评估轨即使不在任何实体（孤儿）也进测试
        test_tracks[cam] |= eval_tracks[cam]

    # ---------- 2. val 抽样 ----------
    # 设计：多片段实体优先（val 正对来源），各机位轮转取样至链接数达标；
    # 再从 singleton 补少量（冒充者池）。318/320 训练轨已少且其多片段实体全在
    # 测试集内，不设 val。
    VAL_CAMS = ["305", "316", "323", "343"]
    LINK_TARGET = 60
    SINGLE_PER_CAM = 8

    val_ent_ids = {c: set() for c in CAMS}
    val_tracks = {c: set() for c in CAMS}
    pools = {}
    for cam in VAL_CAMS:
        cand = [(eid, set(trs)) for eid, trs in ents[cam]
                if len(trs) >= 2 and not (trs & test_tracks[cam])]
        cand.sort()
        idx = rng.permutation(len(cand))
        pools[cam] = [cand[i] for i in idx]
    n_links = 0
    ptr = {cam: 0 for cam in VAL_CAMS}
    active = list(VAL_CAMS)
    while n_links < LINK_TARGET and active:
        for cam in list(active):
            if ptr[cam] >= len(pools[cam]):
                active.remove(cam)
                continue
            eid, trs = pools[cam][ptr[cam]]
            ptr[cam] += 1
            val_ent_ids[cam].add(eid)
            val_tracks[cam] |= trs
            n_links += len(trs) * (len(trs) - 1) // 2
            if n_links >= LINK_TARGET:
                break
    for cam in VAL_CAMS:
        singles = [(eid, set(trs)) for eid, trs in ents[cam]
                   if len(trs) == 1 and not (trs & test_tracks[cam])]
        singles.sort()
        idx = rng.permutation(len(singles))
        for i in idx[:SINGLE_PER_CAM]:
            eid, trs = singles[i]
            val_ent_ids[cam].add(eid)
            val_tracks[cam] |= trs
    print(f"[val] 多片段优先+singleton | 实体 {sum(len(v) for v in val_ent_ids.values())} | "
          f"轨迹 {sum(len(v) for v in val_tracks.values())} | 内部链接对 {n_links}")

    # ---------- 3. val 对（正 = 实体内部；负 = 硬负挖掘）----------
    val_pos_pairs = []
    for cam in CAMS:
        for eid, trs in ents[cam]:
            if eid not in val_ent_ids[cam]:
                continue
            trs = sorted(trs)
            for i in range(len(trs)):
                for j in range(i + 1, len(trs)):
                    val_pos_pairs.append([cam, trs[i], trs[j]])

    # 负对构造（两类合并）：
    #  A. 共现负对（铁定不同人）：同机位两轨同帧共现 >= 5 帧（tracker 抖动冗余）
    #  B. 硬负挖掘：跨实体 top-sim（可能含漏链/反光噪声，用稳健分位数处理）
    ent_of_all = {}
    for cam in CAMS:
        for eid, trs in ents[cam]:
            for t in trs:
                ent_of_all[(cam, t)] = eid
    spans = {}
    for cam in CAMS:
        t = {}
        with open(local_det(cam), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                for x in (r.get("track_ids") or []):
                    if x is not None and x >= 0:
                        x = int(x)
                        if x not in t:
                            t[x] = [r["frame"], r["frame"]]
                        else:
                            t[x][1] = r["frame"]
        spans[cam] = t
    co_pairs = []
    for cam in VAL_CAMS:
        vt = sorted(x for x in val_tracks[cam] if x in spans[cam])
        for i in range(len(vt)):
            sa = spans[cam][vt[i]]
            ea = ent_of_all.get((cam, vt[i]))
            for j in range(i + 1, len(vt)):
                sb = spans[cam][vt[j]]
                if ea == ent_of_all.get((cam, vt[j])):
                    continue   # 同实体邻帧是正例，不是负对
                if min(sa[1], sb[1]) - max(sa[0], sb[0]) >= 5:
                    co_pairs.append([cam, vt[i], vt[j]])

    val_neg_pairs, seen = [], set()
    for cam in CAMS:
        e = load_d3b(cam)
        vt = sorted(t for t in val_tracks[cam] if t in e)
        if len(vt) < 4:
            continue
        E = np.stack([e[t] for t in vt]).astype(np.float32)
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        S = E @ E.T
        np.fill_diagonal(S, -1)
        for i, ta in enumerate(vt):
            cnt = 0
            for j in np.argsort(-S[i]):
                tb = vt[int(j)]
                if ent_of_all.get((cam, ta)) == ent_of_all.get((cam, tb)):
                    continue
                key = (cam, min(ta, tb), max(ta, tb))
                if key in seen:
                    continue
                seen.add(key)
                val_neg_pairs.append([cam, key[1], key[2]])
                cnt += 1
                if cnt >= 4:
                    break
    co_set = {(c, a, b) for c, a, b in co_pairs}
    mined_only = sorted(r for r in val_neg_pairs if (r[0], r[1], r[2]) not in co_set)
    val_neg_co = sorted(co_pairs)
    val_neg_pairs = val_neg_co + mined_only
    print(f"[val] 负对: 共现(铁定) {len(val_neg_co)} + 硬负补充 {len(mined_only)} "
          f"= {len(val_neg_pairs)}")

    # ---------- 4. train = 全部 - test - val ----------
    train_tracks = {c: sorted(all_tracks[c] - test_tracks[c] - val_tracks[c]) for c in CAMS}

    # ---------- 5. 输出 ----------
    def sset(x):
        return sorted(int(t) for t in x)

    frozen_items = sorted(
        [("A", a, b) for a, b in POS30 + NEG30] +
        [("S", c, a, b) for c, a, b in same_pos + same_neg] +
        [("X", c1, a, c2, b) for c1, a, c2, b in cross_pos + cross_neg])
    frozen_sha = hashlib.sha1(json.dumps(frozen_items, sort_keys=True).encode()).hexdigest()[:16]

    def ssha(obj):
        return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]

    # 口径相关的冻结哈希：随实体图/视图变化而变化（frozen_sha 只覆盖评估对，口径无关）
    test_sha = ssha({c: sorted(int(t) for t in test_tracks[c]) for c in CAMS})
    split_sha = ssha({
        "test": {c: sorted(int(t) for t in test_tracks[c]) for c in CAMS},
        "val": {c: sorted(int(t) for t in val_tracks[c]) for c in CAMS},
        "train": {c: sorted(train_tracks[c]) for c in CAMS},
        "test_ents": {c: sorted(test_ent_ids[c]) for c in CAMS},
        "val_ents": {c: sorted(val_ent_ids[c]) for c in CAMS},
    })

    def md5f(p):
        try:
            return hashlib.md5(open(p, "rb").read()).hexdigest()
        except OSError:
            return None

    nd = net_dir()
    out = {
        "meta": {
            "note": "实体粒度三集划分；测试对在 2026-10-12 判读后冻结，禁止用于训练/校准",
            "seed": SEED,
            "val_design": "multi-first（305/316/323/343；link_target=60；+8 singletons/cam）；"
                          "负对=共现铁定(>=5帧)+硬负挖掘；校准用稳健分位数",
            "frozen_sha1_16": frozen_sha,
            "frozen_test_tracks_sha1_16": test_sha,
            "frozen_splits_sha1_16": split_sha,
            "view": describe(),
            "det_view": current_view(),
            "net_dir": os.path.relpath(nd, ROOT).replace("\\", "/"),
            "net_md5": {c: md5f(net_tracks_path(c)) for c in CAMS},
            "labels_src": os.path.relpath(labels_path(), ROOT).replace("\\", "/"),
            "eval_tracks_dropped_by_view": {k: v for k, v in eval_dropped.items() if v},
        },
        "test": {
            "pairs_protocol30": [[a, b] for a, b in POS30 + NEG30],
            "pairs_cross": [[c1, a, c2, b] for c1, a, c2, b in cross_pos + cross_neg],
            "pairs_same": [[c, a, b] for c, a, b in same_pos + same_neg],
            "tracks": {c: sset(test_tracks[c]) for c in CAMS},
            "entities": {c: sorted(test_ent_ids[c]) for c in CAMS},
        },
        "val": {
            "tracks": {c: sset(val_tracks[c]) for c in CAMS},
            "entities": {c: sorted(val_ent_ids[c]) for c in CAMS},
            "pos_pairs": val_pos_pairs,
            "neg_pairs": val_neg_pairs,
            "neg_pairs_co": val_neg_co,
        },
        "train": {"tracks": train_tracks,
                  "count": sum(len(v) for v in train_tracks.values())},
    }
    json.dump(out, open(os.path.join(OUT, f"splits_{SPLITS_TAG}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 服务器跳过表（旧格式）
    hold = {c: sset(test_tracks[c] | val_tracks[c]) for c in CAMS}
    json.dump(hold, open(os.path.join(OUT, f"{SPLITS_TAG}_holdout_server.json"), "w"),
              ensure_ascii=False, indent=1)

    # ---------- 6. 汇总打印 ----------
    print()
    print(describe())
    print(f"[splits] tag={SPLITS_TAG}")
    print(f"{'cam':>4} {'total':>6} {'test':>5} {'val':>5} {'train':>6}  {'test_ents':>9} {'val_ents':>8}")
    for c in CAMS:
        print(f"{c:>4} {len(all_tracks[c]):>6} {len(test_tracks[c]):>5} "
              f"{len(val_tracks[c]):>5} {len(train_tracks[c]):>6}  "
              f"{len(test_ent_ids[c]):>9} {len(val_ent_ids[c]):>8}")
    tot = {k: sum(len(v) for v in [test_tracks, val_tracks, train_tracks][i].values())
           for i, k in enumerate(["test", "val", "train"])}
    print(f"合计: total={sum(len(v) for v in all_tracks.values())} "
          f"test={tot['test']} val={tot['val']} train={tot['train']}")
    print(f"val 对: pos={len(val_pos_pairs)} neg={len(val_neg_pairs)}")
    print(f"冻结哈希(测试对): {frozen_sha}")
    print(f"冻结哈希(test 轨集): {test_sha}")
    print(f"冻结哈希(test+val+train): {split_sha}")
    print(f"实体图: {os.path.relpath(nd, ROOT)}")


if __name__ == "__main__":
    main()
