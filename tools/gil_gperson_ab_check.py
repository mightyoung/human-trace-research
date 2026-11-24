# -*- coding: utf-8 -*-
"""gperson 各版本 A/B 逐判对对账（只读）：找出 TP 波动对与来源。"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("DET_VIEW", "clean2")
from gil import GILStore  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
STORE = ROOT + "/output_crosscam/analysis/gil/store_v1"

# 版本 → node2gid 文件（v2 = 冻结 T_same=0.975 + s3 传导 + 倒影排除 + cap12 的最终候选）
VERSION_FILES = {
    "base": "nodes_base.json",
    "cap8": "nodes_cap8.json",
    "cap10": "nodes_cap10.json",
    "cap12": "nodes_cap12.json",
    "comboRC": "nodes_combo2.json",      # 重校准 T_same=0.960（漂移版，仅对照）
    "v2": "gperson_v2.json",             # 冻结 0.975 最终候选
}
VERSIONS = list(VERSION_FILES)


def main():
    st = GILStore(STORE, create=False)
    # entity → obs keys
    obs_key = {}
    for r in st.con.execute("SELECT id,cam,tid FROM obs WHERE tombstone=0"):
        obs_key[int(r["id"])] = f"{r['cam']}|{int(r['tid'])}"
    ent2keys = defaultdict(list)
    for r in st.con.execute("SELECT id,entity_id FROM obs WHERE entity_id IS NOT NULL"):
        k = obs_key.get(int(r["id"]))
        if k:
            ent2keys[int(r["entity_id"])].append(k)
    eid2keyname = {}
    for r in st.con.execute("SELECT id,cam,eid FROM entity"):
        eid2keyname[int(r["id"])] = f"{r['cam']}_E_{r['eid']}"

    # 判读
    tr = {}
    for src in (os.path.join(ROOT, "webapp/backend/data/truth/seed_labels.json"),
                os.path.join(ANA, "truth/remote/labels.json")):
        if os.path.exists(src):
            d = json.load(open(src, encoding="utf-8")) or {}
            tr.update(d.get("pair", d) if isinstance(d, dict) else {})

    # 各版本 node2gid
    vers = {}
    for v in VERSIONS:
        p = os.path.join(ANA, "gil", VERSION_FILES[v])
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            vers[v] = d.get("node2gid") or {}
            print(f"[{v}] n_gperson={d['n_gperson']} node2gid={len(vers[v])}")

    def pred(v, keyname):
        """entity keyname → 预测 gid（多数决）或 None"""
        # 找 entity
        for eid, kn in eid2keyname.items():
            if kn == keyname:
                keys = ent2keys.get(eid)
                if not keys:
                    return None
                g = vers[v]
                votes = [g[k] for k in keys if k in g]
                if not votes:
                    return None
                return Counter(votes).most_common(1)[0][0]
        return None

    rows = []
    for pid, val in sorted(tr.items()):
        if not isinstance(val, dict) or val.get("v") not in (0, 1):
            continue
        # 命名空间防御：S6B_P_/PRT_ 前缀剥离后计入；非法格式键跳过
        p2 = pid
        for pref in ("S6B_P_", "PRT_"):
            if p2.startswith(pref):
                p2 = p2[len(pref):]
                break
        if p2.count("__") != 1 or p2.count("_E_") != 2:
            continue
        pa, pb = p2.split("__")
        preds = {}
        for v in VERSIONS:
            ga, gb = pred(v, pa), pred(v, pb)
            if ga is None or gb is None:
                preds[v] = None
            else:
                preds[v] = int(ga == gb)
        rows.append((pid, val["v"], preds))

    print("\n== 逐判对预测（v=真值；.-无归属）==")
    hdr = "  ".join(f"{v:>6}" for v in VERSIONS)
    print(f"{'pair':<44} v | {hdr}")
    changed = []
    for pid, v, preds in rows:
        cells = "  ".join(
            ("   .  " if preds[x] is None else ("  SAME" if preds[x] == 1 else "  DIFF"))
            for x in VERSIONS)
        mark = ""
        # 找"某版本与 base 不同"的对
        base = preds["base"]
        diffs = [x for x in VERSIONS if preds[x] != base]
        if diffs and preds["v2"] is not None and base is not None:
            mark = "  ← " + ",".join(diffs)
            changed.append((pid, v, preds))
        print(f"{pid:<44} {v} | {cells}{mark}")

    print("\n== 变化对（相对 base）==")
    for pid, v, preds in changed:
        print(f"  {pid} 真值={v}  " + " ".join(f"{x}={preds[x]}" for x in VERSIONS))

    # 汇总 TP/FP/FN（同 gperson 工具口径：只有两个 entity 都有归属才计入）
    print("\n== 汇总（可比口径）==")
    for v in VERSIONS:
        tp = fp = fn = unk = 0
        for pid, tv, preds in rows:
            if preds[v] is None:
                unk += 1
                continue
            if tv == 1:
                tp += preds[v]; fn += 1 - preds[v]
            else:
                fp += preds[v]
        P = tp / (tp + fp) if tp + fp else None
        R = tp / (tp + fn) if tp + fn else None
        print(f"  [{v}] TP={tp} FP={fp} FN={fn} unk={unk} P={P} R={round(R,4) if R else R}")


if __name__ == "__main__":
    main()
