#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ops 台账 · 自测（P0 验收）

覆盖：建表幂等 / run 生命周期 / label append-only / 投影双写 / ingest 幂等 /
verify 零差异 / overview 结构。全部在临时库上跑，不动主库与真实标注文件。

用法：python tools/ops_selftest.py
退出码 0 = 全绿。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops_db as db          # noqa: E402
import ops_ingest as ingest  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, str(detail)))
        except AssertionError as e:
            RESULTS.append((name, False, str(e)))
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return deco


TMP = os.path.join(db.OPS_DIR, "_selftest_tmp")
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP, exist_ok=True)
DB = os.path.join(TMP, "test.db")
DB2 = os.path.join(TMP, "ingest.db")


# ---------------------------------------------------------------- 用例
@check("建表幂等（两次 init 表结构一致）")
def _():
    c = db.connect(DB)
    a = db.init_db(c)
    b = db.init_db(c)
    assert a.keys() == b.keys(), "表集合不一致"
    assert all(b[k] == 0 for k in ("run", "label", "artifact")), f"新库应为空: {b}"
    c.close()
    return f"{len(a)} 张表"


@check("run 生命周期：running → artifact → succeeded")
def _():
    c = db.connect(DB)
    rid = db.create_run(c, "t-run", "eval", entrypoint="echo hi",
                        params={"cmd": ["echo", "hi"], "k": "v"}, env={"python": "3.x"},
                        host="h", gpu="cpu", status="running")
    row = c.execute("SELECT status FROM run WHERE id=?", (rid,)).fetchone()
    assert row["status"] == "running", "初始状态应为 running"
    p = os.path.join(TMP, "a.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"ok": 1}, f)
    assert db.add_artifact(c, rid, p) is not None, "首次登记应成功"
    assert db.add_artifact(c, rid, p) is None, "重复登记应被忽略（幂等）"
    db.finish_run(c, rid, "succeeded", 0, metrics={"items": []})
    d = db.get_run(c, rid)
    assert d["status"] == "succeeded" and d["exit_code"] == 0
    assert len(d["artifacts"]) == 1 and d["artifacts"][0]["sha1"]
    c.close()
    return f"run={rid} artifacts=1"


@check("label append-only：同值去重 / 改判留痕 / 清除语义")
def _():
    c = db.connect(DB)
    db.upsert_task(c, "t_x", "测试任务", "pair")
    db.upsert_item(c, "t_x:k1", "t_x", payload={"a": 1})
    i1 = db.set_label(c, "t_x:k1", "CONF", "u1")
    i2 = db.set_label(c, "t_x:k1", "CONF", "u1")
    assert i1 == i2, "同 annotator 同值不应新增"
    i3 = db.set_label(c, "t_x:k1", "REJ", "u1")
    assert i3 != i1, "改判应新增一行"
    row = c.execute("SELECT supersedes FROM label WHERE id=?", (i3,)).fetchone()
    assert row["supersedes"] == i1, "改判应指向上一行"
    n = c.execute("SELECT COUNT(*) FROM label WHERE item_id='t_x:k1'").fetchone()[0]
    assert n == 2, f"应有 2 条判定，实际 {n}"
    st = c.execute("SELECT status FROM annotation_item WHERE id='t_x:k1'").fetchone()[0]
    assert st == "labeled", f"条目状态应为 labeled，实际 {st}"
    c.close()
    return "去重/改判/状态均符合"


@check("投影双写：写回 json 且内容正确（且不碰真实文件）")
def _():
    tmp_json = os.path.join(TMP, "labels.json")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "pair_decisions": {}}, f)
    real = ingest.LABELS
    real_snap = None
    if os.path.isfile(real):
        with open(real, encoding="utf-8") as f:
            real_snap = f.read()
    ingest.LABELS = tmp_json
    try:
        r = ingest.project_item("t_pair_318x320:318:1|320:2", "CONF")
        assert r["ok"], f"投影失败: {r}"
        with open(tmp_json, encoding="utf-8") as f:
            d = json.load(f)
        assert d["pair_decisions"]["318:1|320:2"] == "CONF", "写回值不符"
        ingest.project_item("t_pair_318x320:318:1|320:2", None)
        with open(tmp_json, encoding="utf-8") as f:
            d2 = json.load(f)
        assert "318:1|320:2" not in d2["pair_decisions"], "清除后应删除键"
    finally:
        ingest.LABELS = real
    if real_snap is not None:
        with open(real, encoding="utf-8") as f:
            assert f.read() == real_snap, "★ 隔离失效：投影写到了真实文件"
    return "写入 / 清除正确，真实文件未改动"


@check("反向镜像：老页面写入落台账（结构化值不丢类型）")
def _():
    old = db.DB_PATH
    db.DB_PATH = DB
    try:
        ingest.mirror_labels({"pair_decisions": {"318:9|320:9": "CONF"},
                              "glass_masks": {"305": [[0.1, 0.2], [0.3, 0.4]]}},
                             annotator="t")
        ingest.mirror_truth({"pair": {"p1": {"v": 1}}}, annotator="t")
        ingest.mirror_face_pair("g1@g2", "CONF", annotator="t")
        ingest.mirror_face_person("g1", {"verdict": "person", "name": "张三"}, annotator="t")
    finally:
        db.DB_PATH = old
    c = db.connect(DB)
    rows = {r["item_id"]: r["value"] for r in
            c.execute("SELECT item_id, value FROM label WHERE annotator='t'")}
    c.close()
    assert rows.get("t_pair_318x320:318:9|320:9") == "CONF", "pair 未落库"
    assert rows.get("t_mask_glass:305") == "[[0.1, 0.2], [0.3, 0.4]]", \
        f"mask 编码不符: {rows.get('t_mask_glass:305')}"
    assert rows.get("t_truth_pair:p1") == "1", "truth 未落库"
    assert rows.get("t_face_pair:g1@g2") == "CONF", "face pair 未落库"
    assert rows.get("t_face_person:g1") == "person", "face person 未落库"

    # 投影回 json 时必须还原为 list，不能留成字符串
    tmp = os.path.join(TMP, "mirror_labels.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "glass_masks": {}}, f)
    keep = ingest.LABELS
    ingest.LABELS = tmp
    try:
        ingest.project_item("t_mask_glass:305", rows["t_mask_glass:305"])
        with open(tmp, encoding="utf-8") as f:
            back = json.load(f)["glass_masks"]["305"]
        assert isinstance(back, list) and back[0] == [0.1, 0.2], f"类型未还原: {back!r}"
    finally:
        ingest.LABELS = keep
    return f"{len(rows)} 条判定，结构化值 JSON 往返无损"


@check("ingest 幂等（临时库跑两次，label 数不变）")
def _():
    c = db.connect(DB2)
    db.init_db(c)
    ingest.ingest(c)
    n1 = c.execute("SELECT COUNT(*) FROM label").fetchone()[0]
    n2i = c.execute("SELECT COUNT(*) FROM annotation_item").fetchone()[0]
    ingest.ingest(c)
    n2 = c.execute("SELECT COUNT(*) FROM label").fetchone()[0]
    n2j = c.execute("SELECT COUNT(*) FROM annotation_item").fetchone()[0]
    c.close()
    assert n1 == n2, f"重复摄入改变了 label 数：{n1} → {n2}"
    assert n2i == n2j, f"条目数变了：{n2i} → {n2j}"
    assert n2 > 0, "未摄入任何判定"
    return f"label={n2} items={n2j} 两次一致"


@check("主库 verify 零差异（台账与 json 一致）")
def _():
    c = db.connect()
    rep = ingest.verify(c)
    c.close()
    bad = {k: v for k, v in rep.items()
           if v["n_missing_in_db"] or v["n_missing_in_json"] or v["n_conflict"]}
    assert not bad, f"存在差异: {json.dumps(bad, ensure_ascii=False)[:300]}"
    total = sum(v["db_labeled"] for v in rep.values())
    return f"{len(rep)} 个任务 / {total} 条已判，全部一致"


@check("派发状态机：越级流转被拒 / 关闭需证据")
def _():
    c = db.connect(DB)
    jid = db.create_job(c, title="自测任务", kind="review")
    try:
        db.update_job_status(c, jid, "done")
        raise AssertionError("backlog→done 应被拒绝")
    except ValueError as e:
        assert "非法流转" in str(e), f"错误信息不符: {e}"
    for s in ("assigned", "doing", "review"):
        db.update_job_status(c, jid, s)
    try:
        db.update_job_status(c, jid, "done")
        raise AssertionError("无证据应被拒绝")
    except ValueError as e:
        assert "证据" in str(e), f"错误信息不符: {e}"
    j = db.update_job_status(c, jid, "done", evidence={"files": ["docs/x.md"]})
    assert j["status"] == "done" and j["closed_at"], "关闭后应记录时间"
    # 造一个有待判条目的任务，验证派生逻辑
    db.upsert_task(c, "t_derive", "派生测试任务", "label")
    db.upsert_item(c, "t_derive:a", "t_derive", status="pending")
    db.upsert_item(c, "t_derive:b", "t_derive", status="pending")
    made = db.derive_label_jobs(c)
    made2 = db.derive_label_jobs(c)
    c.close()
    assert len(made) == 1, f"应派生 1 张，实际 {len(made)}"
    assert made2 == [], f"二次派生应跳过，实际 {len(made2)}"
    return f"流转校验通过；派生 1 张，二次派生 0 张（幂等）"


@check("训练曲线：epoch 序列写入与读取")
def _():
    c = db.connect(DB)
    rid = db.create_run(c, "t-train", "train", status="succeeded")
    for ep in (1, 2, 3):
        db.set_metric(c, rid, ep, "val_f1", 0.1 * ep)
    db.set_metric(c, rid, 2, "val_f1", 0.99)      # 同 step 覆盖
    s = db.get_series(c, rid)
    c.close()
    assert list(s) == ["val_f1"], f"序列名不符: {list(s)}"
    assert len(s["val_f1"]) == 3, f"覆盖不应新增行: {len(s['val_f1'])}"
    assert s["val_f1"][1]["value"] == 0.99, "同 step 应覆盖"
    return f"{len(s['val_f1'])} 点，覆盖生效"


@check("训练回填：日志解析 + 幂等 + 不碰真实输出目录")
def _():
    import ops_ingest_train as tr
    root = os.path.join(TMP, "trainsrc")
    d = os.path.join(root, "expA", "variant_x")
    os.makedirs(d, exist_ok=True)
    log = d + ".log"
    with open(log, "w", encoding="utf-8") as f:
        f.write("[env] torch=2.0 device=cuda task=cams:['305'] id_mode=none\n")
        f.write("[data] train=100 val=20 test=30\n")
        f.write("[model] params=1.2M\n")
        for ep in range(1, 5):
            f.write(f"[ep {ep:02d}] loss={1.0 / ep:.4f} val_acc=0.5 val_f1={0.2 * ep:.4f} (2s)\n")
    with open(os.path.join(d, "metrics_full.json"), "w", encoding="utf-8") as f:
        json.dump({"config": {"n_train": 100, "best_ep": 3, "best_val_f1": 0.6},
                   "overall": {"acc": 0.7, "macro_f1": 0.6, "n": 30},
                   "per_cam": {"305": {"acc": 0.68}}}, f)

    c = db.connect(DB2)
    items = tr.collect(root)
    assert len(items) == 1, f"应发现 1 个训练目录，实际 {len(items)}"
    assert len(items[0]["parsed"]["curve"]) == 4, "应解析出 4 个 epoch"
    tr.ingest(c, items)
    n1 = c.execute("SELECT COUNT(*) FROM run_metric").fetchone()[0]
    r1 = c.execute("SELECT COUNT(*) FROM run").fetchone()[0]
    tr.ingest(c, items)                       # 幂等
    n2 = c.execute("SELECT COUNT(*) FROM run_metric").fetchone()[0]
    r2 = c.execute("SELECT COUNT(*) FROM run").fetchone()[0]
    row = c.execute("SELECT params_json, metrics_json FROM run WHERE kind='train'").fetchone()
    c.close()
    p = json.loads(row["params_json"])
    m = json.loads(row["metrics_json"])
    assert (n1, r1) == (n2, r2), f"重复回填变化: run {r1}→{r2}, metric {n1}→{n2}"
    assert p["n_train"] == 100 and m["macro_f1"] == 0.6, f"解析错: {p} / {m}"
    return f"1 变体 / {n1} 序列点，二次回填无变化"


@check("指标上报口：有 run_id 才写，缺台账不抛异常")
def _():
    import ops_metric
    # 1) 没有 OPS_RUN_ID → 完全 no-op
    os.environ.pop("OPS_RUN_ID", None)
    r0 = ops_metric.Reporter(quiet=True)
    assert not r0.enabled, "无 run_id 时不应启用"
    r0.metric(1, val_f1=0.5)                      # 不应抛异常

    # 2) 有 run_id + 可用库 → 写入
    c = db.connect(DB)
    rid = db.create_run(c, "t-reporter", "train", status="running")
    r1 = ops_metric.Reporter(run_id=rid, db_path=DB, quiet=True)
    assert r1.enabled, "应启用"
    for ep in (1, 2):
        r1.metric(ep, loss=1.0 / ep, val_acc=0.5 + ep / 100)
    r1.close()
    s = db.get_series(c, rid)
    n = c.execute("SELECT COUNT(*) FROM run_metric WHERE run_id=?", (rid,)).fetchone()[0]
    c.close()
    assert n == 4, f"应写 4 个点，实际 {n}"
    assert set(s) == {"loss", "val_acc"}, f"指标名不符: {set(s)}"

    # 3) 库路径不存在 → 降级不抛
    r2 = ops_metric.Reporter(run_id="x", db_path=os.path.join(TMP, "nope.db"), quiet=True)
    assert not r2.enabled, "缺库应降级"
    r2.metric(1, loss=0.1)
    return f"no-op / 4 点写入 / 缺库降级 三种路径均正常"


@check("代码指纹：脚本 sha1 可算，非 py 参数返回空")
def _():
    import ops_run
    fp = ops_run.code_fingerprint(["python", "tools/intent_model.py", "--epochs", "4"])
    assert fp and len(fp["sha1"]) == 40, f"指纹异常: {fp}"
    assert fp["path"].endswith("intent_model.py"), fp["path"]
    assert ops_run.code_fingerprint(["bash", "tools/run_wave4.sh"]) is None, "非 py 应返回 None"
    return f"{fp['path']} · {fp['sha1'][:12]} · {fp['bytes']}B"


@check("单目录导入：认 metrics.json（无 experiment 时落到未归组）")
def _():
    import ops_ingest_train as tr
    root = os.path.join(TMP, "single")
    d = os.path.join(root, "run_x")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "train.log"), "w", encoding="utf-8") as f:
        f.write("[env] torch=2.0 device=cpu\n[data] train=80 val=20 test=40\n")
        for ep in range(1, 4):
            f.write(f"[ep {ep:02d}] loss={0.9 / ep:.4f} val_acc=0.6 val_f1={0.3 * ep:.4f} (1s)\n")
    with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"test": {"acc": 0.72, "macro_f1": 0.68,
                            "confusion": [[5, 0], [0, 5]]},
                   "val_best": {"ep": 3, "macro_f1": 0.66},
                   "config": {"epochs": 3}}, f)

    c = db.connect(DB2)
    items = tr.collect(root)
    assert len(items) == 1, f"应发现 1 个训练目录，实际 {len(items)}"
    assert items[0]["metrics_name"] == "metrics.json", items[0]["metrics_name"]
    r = tr.ingest(c, items)                      # 不传 experiment
    row = c.execute("SELECT id, metrics_json, params_json, out_dir FROM run "
                    "WHERE name='run_x'").fetchone()
    n_exp = c.execute("SELECT COUNT(*) FROM experiment_run").fetchone()[0]
    c.close()
    m = json.loads(row["metrics_json"])
    p = json.loads(row["params_json"])
    assert r["runs"] == 1 and r["loose"] == 1, f"应 1 条未归组，实际 {r}"
    assert m["acc"] == 0.72 and m["n"] == 10, f"指标未归一: {m}"
    assert p["best_ep"] == 3, f"best_ep 未从 val_best 取: {p}"
    assert n_exp == 0, "无边界的单目录导入不应编造 experiment"
    return f"1 run / acc={m['acc']} / n={m['n']} / best_ep={p['best_ep']} / 未归组"


@check("镜像路由：S6-b 前缀不落进真值配对桶")
def _():
    db3 = os.path.join(TMP, "route.db")
    c = db.connect(db3)
    db.init_db(c)
    c.close()
    old = db.DB_PATH
    db.DB_PATH = db3
    try:
        # _apply 只写库、不投影文件，所以在临时库上跑是安全的
        ingest.mirror_truth({"pair": {"PRT_TEST_A__PRT_TEST_B": {"v": 1},
                                      "S6B_M_x__y": {"v": 0},
                                      "316_E_316-E001__343_E_343-E002": {"v": 1}}})
    finally:
        db.DB_PATH = old
    c = db.connect(db3)
    rows = {r["id"]: r["task_id"] for r in
            c.execute("SELECT id, task_id FROM annotation_item")}
    c.close()
    assert rows.get("t_s6b_pending:PRT_TEST_A__PRT_TEST_B") == "t_s6b_pending", \
        f"PRT_ 应落 S6-b 桶: {rows}"
    assert rows.get("t_s6b_pending:S6B_M_x__y") == "t_s6b_pending", "S6B_ 应落 S6-b 桶"
    assert rows.get("t_truth_pair:316_E_316-E001__343_E_343-E002") == "t_truth_pair", \
        "普通配对应落真值桶"
    bad = [k for k in rows if k.startswith("t_truth_pair:") and
           (k.startswith("t_truth_pair:PRT_") or k.startswith("t_truth_pair:S6B_"))]
    assert not bad, f"真值桶里混进了 S6-b 键: {bad}"
    return "3 条判定按前缀各归其桶"


@check("判定证据往返：data_json 保置信度/备注，同值补证据不新增行")
def _():
    path = os.path.join(TMP, "labeldata.db")
    if os.path.exists(path):
        os.remove(path)
    c = db.connect(path)
    db.init_db(c)
    db.upsert_task(c, "t_x", "测试任务", "pair", created_by="selftest")
    db.upsert_item(c, "t_x:a|b", "t_x", payload={"k": 1})
    iid = "t_x:a|b"
    n0 = db.set_label(c, iid, "1", "human", data={"v": 1, "c": "high", "n": "看清了"})
    got = db.jload(db.latest_label(c, iid)["data_json"])
    assert got == {"v": 1, "c": "high", "n": "看清了"}, f"证据未落库: {got}"
    # 同值 + 证据变化（改置信度）→ 不新增行，只补该行（补证据≠改判）
    n1 = db.set_label(c, iid, "1", "human", data={"v": 1, "c": "low", "n": ""})
    cnt = c.execute("SELECT COUNT(*) FROM label WHERE item_id=?", (iid,)).fetchone()[0]
    assert n1 == n0 and cnt == 1, f"同值补证据不该新增行: id={n1} n={cnt}"
    assert db.jload(db.latest_label(c, iid)["data_json"])["c"] == "low", "证据未更新"
    # 改判 → 新行 + supersedes 指向旧行
    n2 = db.set_label(c, iid, "0", "human", data={"v": 0})
    row = db.latest_label(c, iid)
    assert n2 != n0 and int(row["supersedes"]) == n0, "改判应新增行并挂 supersedes"
    # 老行（data_json 为空）读回时退回 value，不抛
    c.execute("UPDATE label SET data_json=NULL WHERE id=?", (n2,))
    assert db.label_data(db.latest_label(c, iid)) == "0", "缺 data_json 应退回 value"
    c.close()
    return "证据写入 / 同值补证据不新增 / 改判挂 supersedes / 老行兜底"


@check("镜像证据无损：{v,c,n} 与 {pid,fp} 不压平")
def _():
    """镜像只写 value 会丢 c/n 与 pid+fp 并存；data_json 必须原样保住。"""
    db4 = os.path.join(TMP, "mirror_ev.db")
    if os.path.exists(db4):
        os.remove(db4)
    c = db.connect(db4)
    db.init_db(c)
    c.close()
    old = db.DB_PATH
    db.DB_PATH = db4
    try:
        ingest.mirror_truth({"pair": {"316_E_316-E001__343_E_343-E002":
                                      {"v": 1, "c": "mid", "n": "侧脸但同人"}},
                             "event": {"316_E_316-E009": {"pid": "7", "fp": True}}})
        ingest.mirror_face_person("g0001", {"verdict": "person", "name": "张三",
                                            "at": "2026-01-01 00:00:00"})
    finally:
        db.DB_PATH = old
    c = db.connect(db4)

    def ev(iid):
        return db.jload(db.latest_label(c, iid)["data_json"])

    pair_ev = ev("t_truth_pair:316_E_316-E001__343_E_343-E002")
    event_ev = ev("t_truth_event:316_E_316-E009")
    face_ev = ev("t_face_person:g0001")
    val = c.execute("SELECT value FROM label WHERE item_id='t_truth_event:316_E_316-E009'"
                    ).fetchone()[0]
    c.close()
    assert pair_ev["c"] == "mid" and pair_ev["n"] == "侧脸但同人", f"置信/备注丢失: {pair_ev}"
    assert event_ev == {"pid": "7", "fp": True}, f"pid+fp 并存丢失: {event_ev}"
    assert face_ev["name"] == "张三", f"人脸姓名丢失: {face_ev}"
    # 主值单值语义：fp 优先（整条事件作废比编号更强），pid 只存在证据里
    assert val == "fp", f"主值应为 fp（fp 优先是既有口径）: {val}"
    return "置信度/备注 / pid+fp 并存 / 人脸姓名 全部无损；主值 fp 优先"


@check("bundle 导出/导入：闭包完整、幂等、拒绝 running")
def _():
    import ops_bundle as bundle
    # 源库：造一条带曲线/产物/实验的 run
    src_path = os.path.join(TMP, "bundle_src.db")
    if os.path.exists(src_path):
        os.remove(src_path)
    c = db.connect(src_path)
    db.init_db(c)
    rid = db.create_run(c, "bundle-src", "train", out_dir="remote/out/x", status="succeeded")
    db.set_metric(c, rid, 1, "val_f1", 0.31)
    db.set_metric(c, rid, 2, "val_f1", 0.44)
    c.execute("INSERT INTO artifact(run_id, kind, uri, bytes, sha1, at) "
              "VALUES(?,'metrics','remote/out/x/metrics.json',123,'deadbeef',?)",
              (rid, db.now()))
    c.execute("INSERT INTO experiment(id,name,status,created_at) VALUES('exp_t','测试批次','done',?)",
              (db.now(),))
    c.execute("INSERT INTO experiment_run(experiment_id,run_id,role) VALUES('exp_t',?,'variant')",
              (rid,))
    c.commit()

    # running 状态应被拒
    running_id = db.create_run(c, "bundle-running", "train", out_dir="remote/out/y",
                              status="running")
    try:
        bundle.export_bundle(c, running_id)
        raise AssertionError("running 状态应被拒绝导出")
    except SystemExit:
        pass
    b = bundle.export_bundle(c, rid, force=False)
    b = json.loads(json.dumps(b))          # 走一遍序列化，模拟文件传输
    c.close()

    # 目标库：导入两次
    dst_path = os.path.join(TMP, "bundle_dst.db")
    if os.path.exists(dst_path):
        os.remove(dst_path)
    d = db.connect(dst_path)
    db.init_db(d)
    bundle.import_bundle(d, b)
    bundle.import_bundle(d, b)
    n_run = d.execute("SELECT COUNT(*) FROM run").fetchone()[0]
    n_met = d.execute("SELECT COUNT(*) FROM run_metric").fetchone()[0]
    n_art = d.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
    f1 = d.execute("SELECT value FROM run_metric WHERE run_id=? AND step=2 AND name='val_f1'",
                   (rid,)).fetchone()[0]
    out_dir = d.execute("SELECT out_dir FROM run WHERE id=?", (rid,)).fetchone()[0]
    d.close()

    assert n_run == 1, f"run 应只有 1 条，实际 {n_run}"
    assert n_met == 2, f"metric 应为 2 点（幂等不翻倍），实际 {n_met}"
    assert n_art == 1, f"artifact 应为 1（不翻倍），实际 {n_art}"
    assert out_dir == "remote/out/x", f"out_dir 不应被改：{out_dir}"
    assert abs(f1 - 0.44) < 1e-9, f"曲线值丢失：{f1}"

    # 路径改写
    dst2 = os.path.join(TMP, "bundle_dst2.db")
    if os.path.exists(dst2):
        os.remove(dst2)
    d = db.connect(dst2)
    db.init_db(d)
    bundle.import_bundle(d, b, out_pairs=[("remote/out", "analysis/v3")],
                         uri_pairs=[("remote/", "output_crosscam/")])
    got = d.execute("SELECT out_dir, (SELECT uri FROM artifact WHERE run_id=run.id) u "
                    "FROM run WHERE id=?", (rid,)).fetchone()
    d.close()
    assert got[0] == "analysis/v3/x", f"out_dir 改写失败：{got[0]}"
    assert got[1] == "output_crosscam/out/x/metrics.json", f"uri 改写失败：{got[1]}"

    # CLI 层：--out 传目录时自动命名 <run_id>.json（交接单里的批量导出用法）
    import subprocess
    out_dir = os.path.join(TMP, "bundle_cli")
    os.makedirs(out_dir, exist_ok=True)
    r = subprocess.run([sys.executable, os.path.join(db.ROOT, "tools", "ops_bundle.py"),
                        "--db", src_path, "export", "--run", rid, "--out", out_dir + os.sep],
                       capture_output=True, text=True, encoding="utf-8")
    made = os.path.join(out_dir, f"{rid}.json")
    assert r.returncode == 0 and os.path.isfile(made), f"目录导出失败：{r.stdout} {r.stderr}"
    return f"{rid} 闭包 metric={n_met} artifact={n_art}；幂等 + 改写 + 目录导出均通过"


@check("指标归一：rep 包裹 / test / 顶层三种形态都能取到 acc")
def _():
    # intent_model_cc / social 的 metrics_full.json
    rep_form = {"rep": {"overall": {"acc": 0.70, "macro_f1": 0.36, "confusion": [[3, 1], [2, 4]]},
                        "config": {"epochs": 3},
                        "per_cam": {"305": {"acc": 0.62}, "318": {"acc": 0.64}}},
                "hist": [{"ep": 1}], "args": {"epochs": 3}}
    n = db.normalize_metrics(rep_form)
    assert abs(n["acc"] - 0.70) < 1e-9, f"rep 形态 acc 未取到: {n}"
    assert abs(n["macro_f1"] - 0.36) < 1e-9, f"rep 形态 macro_f1 未取到: {n}"
    assert n.get("n") == 10, f"应由 confusion 求和得 n=10，实际 {n.get('n')}"
    assert n.get("per_cam") == {"305": 0.62, "318": 0.64}, f"per_cam 未取到: {n.get('per_cam')}"
    assert list(n["_raw"]) == ["rep", "hist", "args"], "_raw 应保留整包而非 unwrap 后的 rep"

    # intent_model.py 的 {test: {...}, val_best: {...}}
    test_form = {"test": {"acc": 0.5, "macro_f1": 0.4}, "val_best": {"ep": 7, "macro_f1": 0.45}}
    n2 = db.normalize_metrics(test_form)
    assert abs(n2["acc"] - 0.5) < 1e-9 and n2.get("best_ep") == 7, f"test 形态失败: {n2}"

    # 已是标准形态 → 原样返回，不套 _raw（否则历史会二次包裹）
    flat = {"acc": 0.6, "macro_f1": 0.5}
    n3 = db.normalize_metrics(flat)
    assert "_raw" not in n3, "标准形态不应被包裹"

    # 幂等：已归一的再归一，值不变
    assert db.normalize_metrics(n)["acc"] == n["acc"], "二次归一改变了值"
    return "rep / test / 顶层三形态；_raw 保留整包；幂等"


@check("派发卡：进度实时、清零进待验收、关闭须队列清零")
def _():
    p = os.path.join(TMP, "jobsync.db")
    if os.path.exists(p):
        os.remove(p)
    c = db.connect(p)
    db.init_db(c)

    # 造一个 3 条目的队列，先判 2 条
    c.execute("INSERT INTO annotation_task(id,name,kind,created_at) "
              "VALUES('t_demo','演示队列','pair',?)", (db.now(),))
    for i in range(3):
        c.execute("INSERT INTO annotation_item(id,task_id,status,created_at) "
                  "VALUES(?,?,?,?)", (f"t_demo:k{i}", "t_demo", "pending", db.now()))
    c.commit()

    made = db.derive_label_jobs(c)
    assert len(made) == 1, f"应派生 1 张卡，实际 {len(made)}"
    goal = c.execute("SELECT goal FROM job WHERE id=?", (made[0],)).fetchone()[0]
    assert "3 条" not in goal, f"卡上不应写死待判数（会过期）：{goal}"

    j = db.list_jobs(c)[0]
    assert j["live"]["pending"] == 3 and j["drained"] is False, f"live 初始不对: {j['live']}"

    # 判完 2 条 → live 应立刻反映（不需要重新派生）
    for i in range(2):
        c.execute("UPDATE annotation_item SET status='labeled' WHERE id=?", (f"t_demo:k{i}",))
        db.set_label(c, f"t_demo:k{i}", "X", "u")
    c.commit()
    j = db.list_jobs(c)[0]
    assert j["live"]["pending"] == 1 and j["live"]["done"] == 2, f"live 未实时更新: {j['live']}"

    db.update_job_status(c, j["id"], "assigned")
    db.update_job_status(c, j["id"], "doing")
    assert db.sync_label_jobs(c) == [], "还有待判，不该推进"

    # 判完最后一条 → sync 推进到 review
    c.execute("UPDATE annotation_item SET status='labeled' WHERE id='t_demo:k2'")
    db.set_label(c, "t_demo:k2", "X", "u")
    c.commit()
    moved = db.sync_label_jobs(c)
    assert len(moved) == 1 and moved[0]["to"] == "review", f"清零未推进: {moved}"

    # 未清零的卡不能关闭
    c.execute("UPDATE annotation_item SET status='pending' WHERE id='t_demo:k0'")
    c.commit()
    try:
        db.update_job_status(c, j["id"], "done")
        raise AssertionError("待判 1 条时不该允许关闭")
    except ValueError as e:
        assert "未判完" in str(e), f"报错文案不对: {e}"

    # 清零后关闭：证据由事实生成，不需要人手填
    c.execute("UPDATE annotation_item SET status='labeled' WHERE id='t_demo:k0'")
    c.commit()
    done = db.update_job_status(c, j["id"], "done")
    ev = done["evidence"] or {}
    assert ev.get("auto") == "queue_drained", f"应自动生成证据: {ev}"
    assert ev.get("items") == 3 and ev.get("done") == 3, f"证据数字不对: {ev}"
    assert done["closed_at"], "关闭应记录时间"
    c.close()
    return f"live 实时 / sync 推进 / 关闭拦截 / 自动取证 均通过"


@check("数据集登记：hash 可复算、幂等、换算法要拦")
def _():
    import ops_dataset as ds
    root = os.path.join(TMP, "ds_root")
    os.makedirs(os.path.join(root, "sub"), exist_ok=True)
    for i, body in enumerate([b"aaa", b"bbbb", b"ccccc"]):
        with open(os.path.join(root, "sub" if i == 2 else "", f"f{i}.bin"), "wb") as f:
            f.write(body)

    files = ds.walk_files(root)
    assert len(files) == 3, f"应扫到 3 个文件，实际 {len(files)}"
    h1 = ds.snapshot_hash(files)
    h2 = ds.snapshot_hash(ds.walk_files(root))
    assert h1 == h2, "同目录两次扫描 hash 应一致（可复算）"

    # 内容变化 → hash 变
    with open(os.path.join(root, "f0.bin"), "wb") as f:
        f.write(b"zzz")
    h3 = ds.snapshot_hash(ds.walk_files(root))
    assert h3 != h1, "内容变了 hash 必须变"

    # no-mtime 口径独立
    h4 = ds.snapshot_hash(ds.walk_files(root), use_mtime=False)
    assert h4 != h3, "两种算法应给出不同 hash（避免混用）"
    return f"{len(files)} 文件；可复算 / 变化敏感 / 双算法区分"


@check("模型注册：权重入库 + 训练产物提升 + 幂等")
def _():
    import ops_model as om
    p = os.path.join(TMP, "w.onnx")
    with open(p, "wb") as f:
        f.write(b"fake-weights" * 100)

    conn = db.connect(os.path.join(TMP, "model.db"))
    db.init_db(conn)
    r1 = om.register(conn, p, "demo-net", "DemoBackbone", "MIT")
    r2 = om.register(conn, p, "demo-net", "DemoBackbone", "MIT")
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM model").fetchone()[0]
    assert n == 1, f"重复注册应幂等，实际 {n} 条"
    assert not r1["existed"] and r2["existed"], "existed 标记不对"
    assert r1["sha1"] == r2["sha1"] and len(r1["sha1"]) == 40, "sha1 不稳定"

    # from-run：产物提升为模型，并反挂 run.model_id
    rid = db.create_run(conn, "demo-train", "train", status="succeeded")
    conn.execute("INSERT INTO artifact(run_id, kind, uri, bytes, sha1, at) "
                 "VALUES(?, 'pt', 'out/best.pt', 1024, ?, ?)",
                 (rid, "a" * 40, db.now()))
    conn.commit()
    m = om.from_run(conn, rid, name="demo-trained")
    conn.commit()
    assert conn.execute("SELECT model_id FROM run WHERE id=?", (rid,)).fetchone()[0] == m["id"], \
        "run.model_id 未回填"
    assert conn.execute("SELECT trained_by_run FROM model WHERE id=?",
                        (m["id"],)).fetchone()[0] == rid, "trained_by_run 未写"
    conn.close()
    return f"{r1['id']}；注册幂等；from-run 双向挂接"


@check("stale 判定：输入变了才标、revert 可退、缺指纹不猜")
def _():
    import ops_stale as st
    p = os.path.join(TMP, "stale.db")
    if os.path.exists(p):
        os.remove(p)
    c = db.connect(p)
    db.init_db(c)

    code = os.path.join(TMP, "entry.py")
    with open(code, "w", encoding="utf-8") as f:
        f.write("print(1)\n")
    code_sha1 = st.sha1_file(code)

    c.execute("INSERT INTO dataset(id,name,snapshot_hash,hash_alg,created_at) "
              "VALUES('d1','D1','h1','rel-size-mtime-v1',?)", (db.now(),))
    c.execute("INSERT INTO model(id,name,version,sha1,created_at) "
              "VALUES('m1','M1','v1','msha1',?)", (db.now(),))
    h, parts = db.input_fingerprint(c, "d1", "m1", code_sha1)

    has = db.create_run(c, "with-fp", "infer", dataset_id="d1", model_id="m1",
                        status="succeeded")
    c.execute("UPDATE run SET input_hash=?, input_breakdown=? WHERE id=?",
              (h, db.jdump(parts), has))
    no_fp = db.create_run(c, "no-fp", "infer", status="succeeded")
    c.commit()

    # 输入未变 → 不标
    class A:  # 模拟 argparse
        db = p
        dry_run = False
        include_code = False
        kind = ""
    st.cmd_check(A())
    assert c.execute("SELECT status FROM run WHERE id=?", (has,)).fetchone()[0] == "succeeded", \
        "输入未变时不该标 stale"

    # 数据变了 → 标 stale 且 note 说明哪个分量
    c.execute("UPDATE dataset SET snapshot_hash='h2' WHERE id='d1'")
    c.commit()
    st.cmd_check(A())
    row = c.execute("SELECT status, note FROM run WHERE id=?", (has,)).fetchone()
    assert row["status"] == "stale", f"数据变了应标 stale，实际 {row['status']}"
    assert "数据快照" in row["note"], f"note 应指明分量: {row['note']}"
    assert c.execute("SELECT status FROM run WHERE id=?", (no_fp,)).fetchone()[0] == "succeeded", \
        "无指纹的 run 不应被判"

    # revert：退回并按当前输入刷新（否则下次 check 会立刻再标）
    st.cmd_revert(type("B", (), {"db": p, "run_id": has, "note": "确认过"}))
    assert c.execute("SELECT status FROM run WHERE id=?", (has,)).fetchone()[0] == "succeeded"
    st.cmd_check(A())
    assert c.execute("SELECT status FROM run WHERE id=?", (has,)).fetchone()[0] == "succeeded", \
        "revert 后不该再次被判 stale"
    c.close()
    return "未变不标 / 变则标并说明 / 无指纹跳过 / revert 可退"


@check("overview 结构完整")
def _():
    c = db.connect()
    ov = db.overview(c)
    c.close()
    for k in ("annotation", "runs", "jobs", "counts", "authority"):
        assert k in ov, f"缺少字段 {k}"
    assert ov["annotation"]["items"] == ov["annotation"]["done"] + ov["annotation"]["pending"] + \
        sum(1 for _ in []), "done+pending 应等于 items（无其它状态时）"
    assert ov["authority"] == "local", "权威口径应为 local"
    return f"tasks={ov['annotation']['tasks']} runs={ov['runs']['total']}"


# ---------------------------------------------------------------- 主流程
def main() -> int:
    print("=" * 68)
    print("ops 台账自测")
    print("=" * 68)
    ok = 0
    for name, passed, detail in RESULTS:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        if detail:
            print(f"       {detail}")
        ok += 1 if passed else 0
    print("-" * 68)
    print(f"{ok} / {len(RESULTS)} 通过")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
