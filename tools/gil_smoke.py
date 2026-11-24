# -*- coding: utf-8 -*-
"""GIL 统一存储冒烟：完整性 / 检索正确性与性能 / 关系查询 / 版本快照。

用法： python tools/gil_smoke.py [--store output_crosscam/analysis/gil/store_v1]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gil import GILStore  # noqa: E402

ROOT = HERE.parent
OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  [{'OK ' if cond else '!! '}] {name} {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="output_crosscam/analysis/gil/store_v1")
    a = ap.parse_args()
    st = GILStore(ROOT / a.store)

    print("=" * 74)
    print("1. 完整性")
    s = st.stats()
    print("   stats:", json.dumps(s, ensure_ascii=False))
    check("obs>0", s["obs"] > 0, f"obs={s['obs']}")
    check("entity>0", s["entity"] > 0, f"entity={s['entity']}")
    check("edge>0", s["edge"] > 0, f"edge={s['edge']}")

    # 行数一致性：obs 数 == 向量块总和
    vb = st.vectors(load=True)
    nvec = vb.n()
    check("obs 数 == 向量总数", nvec == s["obs_alive"], f"{nvec} vs {s['obs_alive']}")

    # 向量 id 必须都能在 obs 找到
    allv = np.concatenate([vb._cache[b][1] for b in vb.blocks()])
    ids_obs = {r["id"] for r in st.con.execute("SELECT id FROM obs")}
    check("向量 id 全在 obs 中", set(allv.tolist()) <= ids_obs,
          f"缺 {len(set(allv.tolist()) - ids_obs)}")

    # 悬空边审计（如实报告，不隐藏）
    dang = []
    for e in st.edges(kind="relay"):
        for k, i in (("a", e["a_id"]), ("b", e["b_id"])):
            if not st.con.execute("SELECT 1 FROM entity WHERE id=?", (i,)).fetchone():
                dang.append((e["kind"], i))
    for e in st.edges(kind="relay_v5"):
        for i in (e["a_id"], e["b_id"]):
            if not st.con.execute("SELECT 1 FROM entity WHERE id=?", (i,)).fetchone():
                dang.append(("relay_v5", i))
    print(f"   [注] 接力边悬空端点 {len(dang)} 个（实体成员轨缺 ReID 嵌入 → 数据缺口，非存储缺陷）")

    print("=" * 74)
    print("2. 检索正确性 vs 裸 numpy（同源对照：两边都用 GIL 向量库）")
    blocks = vb.blocks()
    E = np.concatenate([vb._cache[b][0] for b in blocks])
    ids_all = np.concatenate([vb._cache[b][1] for b in blocks])
    rowof = {int(i): k for k, i in enumerate(ids_all)}
    cam_of = {r["id"]: r["cam"] for r in st.con.execute("SELECT id,cam FROM obs")}
    cams_arr = np.array([cam_of[int(i)] for i in ids_all])
    obs_sel = [r["id"] for r in st.con.execute("SELECT id FROM obs ORDER BY id")][:50]
    hit = 0
    for oid in obs_sel[:20]:
        res = st.search_similar(oid, cross_cam=True, topk=10)
        q = E[rowof[oid]]
        s = E @ q
        s[cams_arr == cam_of[oid]] = -9        # 屏蔽同机位
        s[rowof[oid]] = -9                     # 屏蔽自身
        ref = set(ids_all[np.argsort(-s)[:10]].tolist())
        got = {r["id"] for r in res}
        hit += len(ref & got)
    check("跨机位 top-10 与裸 numpy 一致", hit == 20 * 10, f"命中 {hit}/200")

    print("=" * 74)
    print("3. 检索性能（本地 Windows / 539 规模——只作功能验证，性能以远程 48 核实测为准）")
    oid = obs_sel[0]
    row = st.obs_row(oid)

    def timeit(fn, reps=100):
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            t.append((time.perf_counter() - t0) * 1000)
        t.sort()
        return t[len(t) // 2], t[int(len(t) * 0.95)]

    p50, p95 = timeit(lambda: st.search_similar(oid, cross_cam=True, topk=10))
    print(f"   Q1 跨机位 top-10           : P50 {p50:.3f} / P95 {p95:.3f} ms")
    p50, p95 = timeit(lambda: st.search_similar(oid, cross_cam=True, topk=10,
                                                t_window=(row["t_first"] - 3, row["t_first"] + 3)))
    print(f"   Q3 跨机位 + 时间窗 ±3s     : P50 {p50:.3f} / P95 {p95:.3f} ms")

    print("=" * 74)
    print("4. 关系查询（GIL 的价值：不是纯向量，而是带约束的身份查询）")
    # 造一个最小 gperson 验证时间线/owner 通路
    gid = st.add_gperson("smoke_g1")
    two = list(st.con.execute("SELECT id,cam,t_first,t_last FROM obs ORDER BY t_first LIMIT 2"))
    for r in two:
        st.add_membership(gid, "obs", r["id"], score=0.97, prior=0.5, decision="auto")
    st.add_owner(two[0]["t_first"], two[1]["t_last"], two[0]["cam"], gid, reason="visibility")
    st.commit()
    tl = st.gperson_timeline(gid)
    check("gperson 时间线可查", len(tl) == 2, f"{len(tl)} 条")
    # ★ 只看自己造的 gperson：owner 表在生产下非空，不能用"恰好 1 行"断言
    own = [r for r in st.owners_at((two[0]["t_first"] + two[1]["t_last"]) / 2)
           if int(r["gperson_id"]) == gid]
    check("owner@t 可查", len(own) == 1, f"本 gperson {len(own)} 条")
    # 跨机位身份卡：同一 gperson 覆盖多个机位？
    cams = {r["cam"] for r in tl}
    print(f"   gperson {gid} 覆盖机位: {cams}")
    # 清理冒烟数据
    st.con.execute("DELETE FROM membership WHERE gperson_id=?", (gid,))
    st.con.execute("DELETE FROM owner WHERE gperson_id=?", (gid,))
    st.con.execute("DELETE FROM gperson WHERE id=?", (gid,))
    st.commit()

    print("=" * 74)
    print("5. 版本快照（可复现）")
    mf = st.snapshot()
    m = json.loads(Path(mf).read_text(encoding="utf-8"))
    check("manifest 含 sha1", all("sha1_16" in v for v in m["files"].values()),
          f"{len(m['files'])} 个文件")
    print("   files:", {k: v["sha1_16"] for k, v in m["files"].items()})
    print("   meta:", json.dumps(m.get("meta", {}), ensure_ascii=False)[:200])

    print("=" * 74)
    print(f"结果: OK {len(OK)} / FAIL {len(BAD)}")
    if BAD:
        print("失败项:", BAD)
    st.close()
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
