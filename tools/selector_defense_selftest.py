#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A3 selftest：selector 边界防御补丁的回归验证。

覆盖（docs/commit-015/PHASE_A_PLAN.md §4.3）：
  S1 K=1 库 + margin → 不崩（降级为纯阈值判定 + warning）
  S2 稀疏簇 ID（空簇）→ 不崩、空簇永不命中、load 警告
  S3 非有限输入（v / refl / sims_all）→ ValueError
  S4 正常路径回归 → 与修复前基线逐位一致（sel_baseline2.npz）
  S5 调用形态冒烟（cams / loose / prec / refl 门 / 单样本）

用法: python tools/selector_defense_selftest.py
"""
import os
import sys
import tempfile
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from selector import IdentitySelector  # noqa: E402

RESULTS = []
NPZ = os.path.join(ROOT, "output_crosscam", "analysis", "selector_1012", "selector_v2.npz")


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def t_s1():
    rng = np.random.RandomState(7)
    protos = rng.randn(3, 768).astype(np.float32)
    # ★ K=1 场景：3 个子原型全属同一簇（n_clusters=1）
    sel = IdentitySelector(protos, thr_unknown=0.3,
                           proto_cluster=np.array([0, 0, 0]), margin_prec=0.01)
    V = (sel.protos[:1] + rng.randn(2, 768).astype(np.float32) * 0.01).astype(np.float32)
    warn_ok = True
    # 两路触发 K=1 margin 分支：显式数值 / prec（库带推荐 margin_prec=0.01）
    for kw in (dict(margin=0.1), dict(margin="prec")):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ids, sims = sel.predict(V, **kw)
            warn_ok &= any("margin 条件已跳过" in str(x.message) for x in w)
    ids2, _ = sel.predict(V)                       # 纯阈值判定
    same = np.array_equal(np.asarray(ids), np.asarray(ids2))
    check("S1 K=1 + margin 不崩且=纯阈值判定", same and warn_ok,
          f"ids={np.asarray(ids).tolist()} warn={warn_ok}")


def t_s2(tmp):
    rng = np.random.RandomState(8)
    protos = rng.randn(4, 768).astype(np.float32)
    pc = np.array([0, 2, 2, 3])                    # 簇 1 为空（稀疏 ID）
    sel = IdentitySelector(protos, thr_unknown=0.3, proto_cluster=pc)
    V = protos.copy()
    ids, sims = sel.predict(V)
    S = sel.sims_all(V)
    col1_inf = bool(np.isneginf(S[:, 1]).all())
    no_cluster1 = bool((np.asarray(ids) != 1).all())
    # save/load 触发健全性警告
    p = os.path.join(tmp, "sparse.npz")
    sel.save(p)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sel2 = IdentitySelector.load(p)
        has_warn = any("空簇" in str(x.message) for x in w)
    ids2, _ = sel2.predict(V)
    check("S2a 稀疏簇不崩/空簇不命中", no_cluster1 and col1_inf and len(ids) == 4,
          f"ids={np.asarray(ids).tolist()} 空簇列=-inf:{col1_inf}")
    check("S2b load 触发空簇警告", has_warn, f"warn={has_warn}")
    check("S2c save→load 后预测一致", np.array_equal(np.asarray(ids), np.asarray(ids2)))


def t_s3():
    sel = IdentitySelector.load(NPZ)
    ok = 0
    V_nan = np.full((1, 768), np.nan, np.float32)
    try:
        sel.predict(V_nan); print("      predict(nan) 未报错！")
    except ValueError:
        ok += 1
    V_inf = np.zeros((1, 768), np.float32); V_inf[0, 0] = np.inf
    try:
        sel.predict(V_inf); print("      predict(inf) 未报错！")
    except ValueError:
        ok += 1
    try:
        sel.predict(np.ones((2, 768), np.float32), refl=[np.nan, 0.1]); print("      refl(nan) 未报错！")
    except ValueError:
        ok += 1
    try:
        sel.sims_all(V_nan); print("      sims_all(nan) 未报错！")
    except ValueError:
        ok += 1
    check("S3 非有限输入全部 ValueError", ok == 4, f"{ok}/4")


def t_s4():
    base_p = os.path.join(os.environ.get("TEMP", tempfile.gettempdir()), "sel_baseline2.npz")
    if not os.path.exists(base_p):
        check("S4 基线对比", False, f"基线文件不存在: {base_p}（先跑基线脚本）")
        return
    base = np.load(base_p, allow_pickle=False)
    sel = IdentitySelector.load(NPZ)
    rng = np.random.RandomState(42)
    cams = (["305", "316", "318", "320", "323", "343"] * 9)[:50]
    VA = rng.randn(50, 768).astype(np.float32)
    VB = (sel.protos[:50] + rng.randn(50, 768).astype(np.float32) * 0.004).astype(np.float32)
    n_ok = n_tot = 0
    for tag, V in (("A", VA), ("B", VB)):
        for name, kw in (("def", {}), ("loose", dict(loose=True)),
                         ("prec", dict(margin="prec")), ("marg", dict(margin=0.05))):
            ids, sims = sel.predict(V, cams=cams, **kw)
            n_tot += 1
            n_ok += bool(np.array_equal(np.asarray(ids), base[f"{tag}_{name}_ids"])
                         and np.array_equal(np.asarray(sims), base[f"{tag}_{name}_sims"]))
        ids, sims = sel.predict(V[:10], cams=cams[:10], refl=np.linspace(0.0, 1.0, 10))
        n_tot += 1
        n_ok += bool(np.array_equal(np.asarray(ids), base[f"{tag}_refl_ids"])
                     and np.array_equal(np.asarray(sims), base[f"{tag}_refl_sims"]))
        S = sel.sims_all(V)
        n_tot += 1
        n_ok += bool(np.array_equal(S, base[f"{tag}_S"]))
    ids1, sims1 = sel.predict(VB[0], cams=["305"])
    n_tot += 1
    n_ok += bool(np.array_equal(np.array([ids1]), base["single_ids"])
                 and np.array_equal(np.array([sims1]), base["single_sims"]))
    check("S4 正常路径与修复前基线逐位一致", n_ok == n_tot, f"{n_ok}/{n_tot} 组")


def main():
    print("=== A3 selector 防御 selftest ===")
    tmp = tempfile.mkdtemp(prefix="a3_selftest_")
    try:
        t_s1(); t_s2(tmp); t_s3(); t_s4()
        # S5：既有 npz 各调用形态冒烟（无异常）
        sel = IdentitySelector.load(NPZ)
        rng = np.random.RandomState(1)
        V = rng.randn(20, 768).astype(np.float32)
        sel.predict(V, cams=["323"] * 20)
        sel.predict(V, loose=True)
        sel.predict(V, margin=0.01)
        sel.predict(V, refl=np.zeros(20))
        sel.predict(V, refl=np.zeros(20), refl_mode="hard", refl_tau=0.9)
        sel.sims_all(V)
        check("S5 既有调用形态冒烟", True, "全绿")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print(f"\n=== 结果：{n_pass}/{len(RESULTS)} 通过，{n_fail} 失败 ===")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
