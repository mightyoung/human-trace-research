# -*- coding: utf-8 -*-
"""v2 代际库自检套件（故障注入 + 一致性回归 + 迁移 + 真实库冒烟）。

运行：
  python tools/global_reid_store_selftest.py

覆盖（T1–T20）：
  T1  空库全流程（save/load/search/stats）
  T2  往返一致性（extra 回读；入库不归一 → 读取逐位一致）；二代发布
  T3  模型版本护栏（strict raise / strict=False 告警放行）
  T4  维度护栏
  T5  CURRENT 损坏 → fail-closed
  T6  manifest 篡改 → fail-closed
  T7  emb.npy 篡改 → fail-closed
  T8  meta.sqlite 篡改 → fail-closed
  T9  故障注入 after_vectors：发布中止、旧代可读、stage 已清理
  T10 故障注入 before_publish：CURRENT 不切换、旧代可读
  T11 stale writer 拒绝 + 重载后重存
  T12 apply_and_save 整批原子性（非法项 / 发布失败内存回滚）
  T13 prepare_rebuild 重建（清空语义 + parent 链 + 旧代可回退）
  T14 旧库守卫 + migrate-in-place（ack 必填 / 幂等 / 归档 / 逐位一致）
  T15 强约束先过滤后粗筛
  T16 MRL 覆盖键失效保护（前缀仅内存派生）
  T17 add 校验矩阵（NaN / 零范数 / 维度 / tid / 时序 / extra）
  T18 topk=0 / 空库检索
  T19 store_admin rollback（expected_current 防护 + 数据回退）
  T20 真实库冒烟（5 库；未迁移则跳过）
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from global_reid_store import GlobalReIDStore, read_pointer          # noqa: E402
from store_admin import (inspect as admin_inspect,                    # noqa: E402
                         migrate_in_place, read_legacy_flat,
                         rollback as admin_rollback)

TMP = tempfile.mkdtemp(prefix="grs_v2_st_")
D = "2026-10-12"
RESULTS = []


def _mk(name):
    p = os.path.join(TMP, name)
    os.makedirs(p, exist_ok=True)
    return p


def _put(st, k, v, cam="c", tid=1, extra=None, **kw):
    st.add(k, np.asarray(v, dtype=np.float32), cam=cam, date=D, tid=tid,
           extra=extra, **kw)


def _inject(st, phase):
    def hook(p):
        if p == phase:
            raise OSError(f"simulated disk failure @ {phase}")
    st._fault_hook = hook


def _no_stage_leftover(root):
    gdir = os.path.join(root, "generations")
    if not os.path.isdir(gdir):
        return True
    return not any(n.startswith(".stage-") for n in os.listdir(gdir))


def _synth_legacy(root, n=5, dim=4, mrl=2, model="vleg"):
    os.makedirs(root, exist_ok=True)
    rs = np.random.RandomState(0)
    E = rs.randn(n, dim).astype(np.float16)
    E /= np.linalg.norm(E.astype(np.float32), axis=1, keepdims=True)
    np.save(os.path.join(root, "emb.npy"), E)
    np.save(os.path.join(root, "emb_mrl.npy"), E[:, :mrl])
    con = sqlite3.connect(os.path.join(root, "meta.sqlite"))
    con.execute("CREATE TABLE tracks(key TEXT PRIMARY KEY, cam TEXT, date TEXT, "
                "tid INTEGER, t_first REAL, t_last REAL, h_med REAL, u REAL, "
                "v REAL, model TEXT, extra TEXT)")
    keys = []
    for i in range(n):
        k = f"{D}|c|{i}"
        keys.append(k)
        con.execute("INSERT INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (k, "c", D, i, float(i), float(i) + 1, 10.0, 1.0, 2.0, model, "null"))
    con.commit()
    con.close()
    with open(os.path.join(root, "version.json"), "w", encoding="utf-8") as f:
        json.dump({"model_version": model, "dim": dim, "mrl_dim": mrl, "n": n}, f)
    return E, keys


# ---------------- 测试 ----------------
def t1_empty():
    root = _mk("t1")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    assert st.search([1, 0, 0, 0]) == []
    assert st.stats()["n"] == 0
    assert st.save() == 0
    assert read_pointer(root) is not None
    assert GlobalReIDStore(root, "v1", 4, mrl_dim=2).load() == 0


def t2_roundtrip():
    root = _mk("t2")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [2, 0, 0, 0], extra={"derived_from": ["t-2"], "q": 0.8})
    _put(st, "b", [0, 1, 0, 0])
    n = st.save()
    assert n == 2
    r = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    assert r.load() == 2
    assert r.meta["a"]["extra"] == {"derived_from": ["t-2"], "q": 0.8}
    # 入库不归一：读取应逐位等于输入（f16 可精确表示）
    assert np.array_equal(r.rows["a"], np.array([2, 0, 0, 0], np.float32))
    # 二代发布
    r.add("c", np.array([0, 0, 1, 0], np.float32), cam="c", date=D, tid=3)
    assert r.save() == 3
    assert GlobalReIDStore(root, "v1", 4, mrl_dim=2).load() == 3


def t3_version_guard():
    root = _mk("t3")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [1, 0, 0, 0])
    st.save()
    try:
        GlobalReIDStore(root, "v2", 4, mrl_dim=2).load()
        raise AssertionError("版本不符未 raise")
    except ValueError as e:
        assert "model_version" in str(e)
    assert GlobalReIDStore(root, "v2", 4, mrl_dim=2).load(strict=False) == 1


def t4_dim_guard():
    root = _mk("t4")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [1, 0, 0, 0])
    st.save()
    try:
        GlobalReIDStore(root, "v1", 8, mrl_dim=4).load()
        raise AssertionError("维度不符未 raise")
    except ValueError as e:
        assert "dim" in str(e)


def t5_corrupt_current():
    root = _mk("t5")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [1, 0, 0, 0])
    st.save()
    with open(os.path.join(root, "CURRENT.json"), "w", encoding="utf-8") as f:
        f.write("{broken json")
    try:
        GlobalReIDStore(root, "v1", 4, mrl_dim=2).load()
        raise AssertionError("CURRENT 损坏未 raise")
    except ValueError as e:
        assert "CURRENT" in str(e)


def _first_gen(root):
    return read_pointer(root)["generation"]


def t6_manifest_tamper():
    root = _mk("t6")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [1, 0, 0, 0])
    st.save()
    mp = os.path.join(root, "generations", _first_gen(root), "manifest.json")
    with open(mp, "ab") as f:
        f.write(b" ")
    try:
        GlobalReIDStore(root, "v1", 4, mrl_dim=2).load()
        raise AssertionError("manifest 篡改未 raise")
    except ValueError as e:
        assert "manifest" in str(e)


def t7_emb_tamper():
    root = _mk("t7")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [1, 0, 0, 0])
    st.save()
    p = os.path.join(root, "generations", _first_gen(root), "emb.npy")
    b = bytearray(open(p, "rb").read())
    b[-1] ^= 0xFF
    open(p, "wb").write(bytes(b))
    try:
        GlobalReIDStore(root, "v1", 4, mrl_dim=2).load()
        raise AssertionError("emb 篡改未 raise")
    except ValueError as e:
        assert "校验失败" in str(e)


def t8_sqlite_tamper():
    root = _mk("t8")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [1, 0, 0, 0])
    st.save()
    p = os.path.join(root, "generations", _first_gen(root), "meta.sqlite")
    with open(p, "ab") as f:
        f.write(b"\x00")
    try:
        GlobalReIDStore(root, "v1", 4, mrl_dim=2).load()
        raise AssertionError("sqlite 篡改未 raise")
    except ValueError as e:
        assert "校验失败" in str(e)


def t9_fault_after_vectors():
    root = _mk("t9")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [1, 0, 0, 0])
    st.save()
    gen0 = _first_gen(root)
    st.add("b", np.array([0, 1, 0, 0], np.float32), cam="c", date=D, tid=2)
    _inject(st, "after_vectors")
    try:
        st.save()
        raise AssertionError("故障注入未中止 save")
    except OSError:
        pass
    assert _first_gen(root) == gen0, "CURRENT 不应变化"
    assert _no_stage_leftover(root), "stage 应被清理"
    r = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    assert r.load() == 1 and set(r.rows) == {"a"}


def t10_fault_before_publish():
    root = _mk("t10")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [1, 0, 0, 0])
    st.save()
    gen0 = _first_gen(root)
    st.add("b", np.array([0, 1, 0, 0], np.float32), cam="c", date=D, tid=2)
    _inject(st, "before_publish")
    try:
        st.save()
        raise AssertionError("故障注入未中止 save")
    except OSError:
        pass
    assert _first_gen(root) == gen0, "CURRENT 不应切换"
    r = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    assert r.load() == 1 and set(r.rows) == {"a"}


def t11_stale_writer():
    root = _mk("t11")
    a = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(a, "a", [1, 0, 0, 0])
    a.save()
    b = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    b.load()
    a.add("b", np.array([0, 1, 0, 0], np.float32), cam="c", date=D, tid=2)
    a.save()
    b.add("c", np.array([0, 0, 1, 0], np.float32), cam="c", date=D, tid=3)
    try:
        b.save()
        raise AssertionError("stale writer 未拒绝")
    except RuntimeError as e:
        assert "stale" in str(e)
    b.load()
    b.add("c", np.array([0, 0, 1, 0], np.float32), cam="c", date=D, tid=3)
    assert b.save() == 3


def t12_apply_and_save():
    root = _mk("t12")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [1, 0, 0, 0])
    st.save()
    try:
        st.apply_and_save([dict(key="b", emb=[0, 1, 0, 0], cam="c", date=D, tid=2),
                           dict(key="bad", emb=[0, 0, 0, 0], cam="c", date=D, tid=3)])
        raise AssertionError("整批含非法项未 raise")
    except ValueError:
        pass
    assert set(st.rows) == {"a"}
    _inject(st, "before_publish")
    try:
        st.apply_and_save([dict(key="b", emb=[0, 1, 0, 0], cam="c", date=D, tid=2)])
        raise AssertionError("发布失败未 raise")
    except OSError:
        pass
    assert set(st.rows) == {"a"}, "发布失败后内存应回滚"
    st._fault_hook = lambda p: None
    assert st.apply_and_save([dict(key="b", emb=[0, 1, 0, 0], cam="c", date=D, tid=2)]) == 2


def t13_prepare_rebuild():
    root = _mk("t13")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [1, 0, 0, 0])
    st.save()
    gen1 = _first_gen(root)
    st.add("b", np.array([0, 1, 0, 0], np.float32), cam="c", date=D, tid=2)
    st.save()
    gen2 = _first_gen(root)
    r = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    r.prepare_rebuild()
    assert set(r.rows) == set(), "prepare_rebuild 应清空内存"
    _put(r, "c", [0, 0, 1, 0], tid=3)
    assert r.save() == 1
    gen3 = _first_gen(root)
    # parent 链
    with open(os.path.join(root, "generations", gen3, "manifest.json"),
              encoding="utf-8") as f:
        m3 = json.load(f)
    assert m3["parent"] == gen2
    assert GlobalReIDStore(root, "v1", 4, mrl_dim=2).load() == 1
    # 旧代仍可完整读取（回退路径）
    from store_admin import snapshot
    rd, _, _ = snapshot(root, gen1)
    assert set(rd.rows) == {"a"}


def t14_legacy_guard_and_migrate():
    root = _mk("t14")
    E, keys = _synth_legacy(root)
    try:
        GlobalReIDStore(root, "vleg", 4, mrl_dim=2).load()
        raise AssertionError("旧库 load 未拒绝")
    except ValueError as e:
        assert "migrate-in-place" in str(e)
    try:
        GlobalReIDStore(root, "vleg", 4, mrl_dim=2).save()
        raise AssertionError("旧库 save 未拒绝")
    except ValueError as e:
        assert "migrate-in-place" in str(e)
    try:
        migrate_in_place(root, acknowledge_unverified=False)
        raise AssertionError("未确认的迁移未拒绝")
    except ValueError:
        pass
    res = migrate_in_place(root, acknowledge_unverified=True)
    assert res["n"] == 5 and not res["already_migrated"]
    assert set(res["archived_files"]) >= {"emb.npy", "meta.sqlite", "version.json"}
    st = GlobalReIDStore.open(root)
    srt = sorted(st.rows)
    assert srt == sorted(keys)
    for i, k in enumerate(keys):
        assert np.array_equal(st.rows[k], E[i].astype(np.float32)), f"{k} 数据不一致"
    assert st.stats()["provenance"]["kind"] == "legacy_import_unverified"
    assert migrate_in_place(root, acknowledge_unverified=True)["already_migrated"]
    # 旧档目录可读（对账用）
    leg = read_legacy_flat(os.path.join(root, "_legacy_flat"))
    assert leg["version"]["n"] == 5


def t15_strong_filter():
    root = _mk("t15")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    for i in range(70):
        _put(st, f"b{i}", [1, 0, 0, 0], cam="deny", tid=i)
    _put(st, "allow", [0.9, 0.1, 0, 0], cam="allow", tid=999)
    st.save()
    hits = st.search([1, 0, 0, 0], topk=1, cams=["allow"])
    assert hits and hits[0]["key"] == "allow"


def t16_mrl_overwrite():
    root = _mk("t16")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [0, 1, 0, 0], tid=1)
    _put(st, "b", [0.9, 0.1, 0, 0], tid=2)
    st.save()
    st.add("a", np.array([1, 0, 0, 0], np.float32), cam="c", date=D, tid=1)   # 覆盖键
    st.save()
    hits = st.search([1, 0, 0, 0], topk=1, stage="two", nprobe=1)
    assert hits[0]["key"] == "a", "前缀应随内存同代重算（旧前缀会漏选）"


def t17_add_validation():
    root = _mk("t17")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    bads = [dict(emb=[np.nan, 0, 0, 0]), dict(emb=[0, 0, 0, 0]),
            dict(emb=[1, 2]), dict(emb=[1, 0, 0, 0], tid=1.5),
            dict(emb=[1, 0, 0, 0], t_first=5, t_last=3),
            dict(emb=[1, 0, 0, 0], extra=object())]
    for i, bad in enumerate(bads):
        item = dict(key=f"x{i}", cam="c", date=D, tid=1)
        item.update(bad)
        try:
            st.add(**item)
            raise AssertionError(f"非法项未 raise：{bad}")
        except (ValueError, TypeError):
            pass
    assert len(st.rows) == 0
    try:
        st.add_many([dict(key="ok", emb=[1, 0, 0, 0], cam="c", date=D, tid=1),
                     dict(key="bad", emb=[0, 0, 0, 0], cam="c", date=D, tid=2)])
        raise AssertionError("add_many 含非法项未 raise")
    except ValueError:
        pass
    assert len(st.rows) == 0, "add_many 不得部分落库"


def t18_edges():
    root = _mk("t18")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [1, 0, 0, 0])
    assert st.search([1, 0, 0, 0], topk=0) == []
    empty = GlobalReIDStore(_mk("t18b"), "v1", 4, mrl_dim=2)
    assert empty.search([1, 0, 0, 0]) == []
    # 无 MRL 库：stage="two" 应优雅降级单级（历史缺陷：None 下标崩溃）
    st2 = GlobalReIDStore(_mk("t18c"), "v1", 4, mrl_dim=None)
    _put(st2, "a", [1, 0, 0, 0])
    st2.save()
    assert st2.search([1, 0, 0, 0], stage="two")[0]["key"] == "a"
    try:
        st2.search([1, 0, 0, 0], stage="bad")
        raise AssertionError("stage 非法未 raise")
    except ValueError:
        pass


def t19_rollback():
    root = _mk("t19")
    st = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    _put(st, "a", [1, 0, 0, 0])
    st.save()
    gen1 = _first_gen(root)
    st.add("b", np.array([0, 1, 0, 0], np.float32), cam="c", date=D, tid=2)
    st.save()
    gen2 = _first_gen(root)
    try:
        admin_rollback(root, gen1, expected_current=gen1)   # 预期值错误
        raise AssertionError("expected_current 不符未拒绝")
    except RuntimeError as e:
        assert "拒绝陈旧回退" in str(e)
    res = admin_rollback(root, gen1, expected_current=gen2)
    assert res["published"] and _first_gen(root) == gen1
    r = GlobalReIDStore(root, "v1", 4, mrl_dim=2)
    assert r.load() == 1 and set(r.rows) == {"a"}
    info = admin_inspect(root, list_generations=True)
    assert {g["generation"] for g in info["generations"]} == {gen1, gen2}


def t20_real_stores():
    base = os.path.join(ROOT, "output_crosscam", "analysis", "global_reid")
    names = ["store_cradio", "store_cradio_v3", "store_cradio_v4",
             "store_d3b", "store_long35"]
    done, skipped = [], []
    for name in names:
        root = os.path.join(base, name)
        if not os.path.exists(os.path.join(root, "CURRENT.json")):
            skipped.append(name)
            continue
        st = GlobalReIDStore.open(root)
        s = st.stats()
        assert s["n"] > 0 and st.search(st.rows[sorted(st.rows)[0]], topk=1)
        done.append(f"{name}:{s['n']}")
    if skipped:
        return f"已迁移 {len(done)}（{' '.join(done)}）；跳过未迁移 {skipped}"
    return " ".join(done)


TESTS = [
    ("T1  空库全流程", t1_empty),
    ("T2  往返一致性 + 二代", t2_roundtrip),
    ("T3  模型版本护栏", t3_version_guard),
    ("T4  维度护栏", t4_dim_guard),
    ("T5  CURRENT 损坏", t5_corrupt_current),
    ("T6  manifest 篡改", t6_manifest_tamper),
    ("T7  emb 篡改", t7_emb_tamper),
    ("T8  sqlite 篡改", t8_sqlite_tamper),
    ("T9  故障注入 after_vectors", t9_fault_after_vectors),
    ("T10 故障注入 before_publish", t10_fault_before_publish),
    ("T11 stale writer", t11_stale_writer),
    ("T12 apply_and_save 原子性", t12_apply_and_save),
    ("T13 prepare_rebuild 重建", t13_prepare_rebuild),
    ("T14 旧库守卫 + 迁移", t14_legacy_guard_and_migrate),
    ("T15 强约束先过滤", t15_strong_filter),
    ("T16 MRL 覆盖失效保护", t16_mrl_overwrite),
    ("T17 add 校验矩阵", t17_add_validation),
    ("T18 topk=0 / 空库", t18_edges),
    ("T19 rollback + 防护", t19_rollback),
    ("T20 真实库冒烟", t20_real_stores),
]


def main():
    passed = failed = 0
    for name, fn in TESTS:
        try:
            note = fn()
            passed += 1
            print(f"[PASS] {name}" + (f" —— {note}" if note else ""))
        except Exception as e:
            failed += 1
            RESULTS.append((name, e))
            print(f"[FAIL] {name} —— {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n== v2 selftest: {passed}/{passed + failed} 通过 ==")
    if failed == 0:
        shutil.rmtree(TMP, ignore_errors=True)
    else:
        print(f"（失败现场保留：{TMP}）")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
