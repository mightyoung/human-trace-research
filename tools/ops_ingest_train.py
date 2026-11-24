"""把历史训练产物摄入台账（只读源文件）。

扫描 output_crosscam/**/metrics_full.json，配对同名 .log，解析：
  - 超参：日志头 [env] / [data] / [model]
  - 曲线：[ep NN] loss=… val_acc=… val_f1=… (Ns)
  - 终值：metrics_full.json 的 config / overall
写入 run(kind=train) + run_metric + artifact + experiment/experiment_run。

幂等：run id 由相对路径决定，重复跑只覆盖不新增。

用法：
  python tools/ops_ingest_train.py            # 摄入
  python tools/ops_ingest_train.py --dry-run  # 只打印将要写入的内容
  python tools/ops_ingest_train.py --root DIR # 换扫描根（默认 output_crosscam）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops_db as db  # noqa: E402

ROOT = db.ROOT
DEFAULT_SCAN = os.path.join(ROOT, "output_crosscam")

RE_EP = re.compile(r"^\[ep\s*(\d+)\]\s*loss=([\d.eE+-]+)\s+val_acc=([\d.eE+-]+)"
                   r"(?:\s+val_f1=([\d.eE+-]+))?(?:\s*\((\d+)s\))?")
RE_ENV = re.compile(r"^\[env\]\s*(.+)$")
RE_DATA = re.compile(r"^\[data\]\s*(.+)$")
RE_MODEL = re.compile(r"^\[model\]\s*(.+)$")

# 目录名 → 中文实验名（只影响展示，不改任何数据）
EXPERIMENT_CN = {
    "intent": "意图分类（intent）",
    "intent/cc": "意图分类 · 跨相机条件（intent/cc）",
    "intent/v2": "意图分类 · v2 行为特征（intent/v2）",
}


def slug(s: str) -> str:
    s = s.replace("\\", "/").strip("/")
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s[:60] or "x"


def parse_log(path: str) -> dict:
    env, data, model = {}, {}, {}
    curve = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                m = RE_EP.match(line)
                if m:
                    ep = int(m.group(1))
                    rec = {"step": ep, "loss": float(m.group(2)),
                           "val_acc": float(m.group(3))}
                    if m.group(4):
                        rec["val_f1"] = float(m.group(4))
                    if m.group(5):
                        rec["secs"] = int(m.group(5))
                    curve.append(rec)
                    continue
                if not env:
                    m = RE_ENV.match(line)
                    if m:
                        for kv in m.group(1).split():
                            if "=" in kv:
                                k, v = kv.split("=", 1)
                                env[k] = v
                        continue
                if not data:
                    m = RE_DATA.match(line)
                    if m:
                        for kv in m.group(1).split():
                            if "=" in kv:
                                k, v = kv.split("=", 1)
                                data[k] = v
                        continue
                if not model:
                    m = RE_MODEL.match(line)
                    if m:
                        for kv in m.group(1).split():
                            if "=" in kv:
                                k, v = kv.split("=", 1)
                                model[k] = v
    except OSError:
        pass
    return {"env": env, "data": data, "model": model, "curve": curve}


def find_log(dirpath: str) -> str | None:
    sib = dirpath.rstrip("/\\") + ".log"
    if os.path.isfile(sib):
        return sib
    for fn in sorted(os.listdir(dirpath)):
        if fn.endswith(".log"):
            return os.path.join(dirpath, fn)
    return None


def collect(root: str) -> list[dict]:
    """扫描训练产物目录。

    认两种指标文件名：
      - metrics_full.json  （intent/cc 系的完整评测报告）
      - metrics.json       （intent_model.py 直接产出；**必须**有 epoch 曲线，否则可能是评测产物，跳过）
    """
    items = []
    for dirpath, _dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if base.startswith("_smoke"):
            continue
        name = None
        for cand in ("metrics_full.json", "metrics.json"):
            if cand in filenames:
                name = cand
                break
        if not name:
            continue
        mpath = os.path.join(dirpath, name)
        try:
            d = json.load(open(mpath, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rep = d.get("rep") if isinstance(d.get("rep"), dict) else d
        if not isinstance(rep, dict):
            continue
        cfg = rep.get("config") or {}
        overall = rep.get("overall") or {}
        log = find_log(dirpath)
        parsed = parse_log(log) if log else {"env": {}, "data": {}, "model": {}, "curve": []}
        # metrics.json 只在有 epoch 曲线时才当成训练产物（避免把评测报告当训练）
        if name == "metrics.json" and not parsed["curve"]:
            continue
        rel_dir = os.path.relpath(dirpath, ROOT).replace("\\", "/")
        items.append({
            "rel_dir": rel_dir,
            "name": base,
            "rep": rep, "cfg": cfg, "overall": overall,
            "metrics_path": mpath, "log_path": log, "parsed": parsed,
            "metrics_name": name,
        })
    return items


def experiment_of(rel_dir: str, forced: str | None = None) -> tuple[str, str] | None:
    """按父目录分组：…/intent/cc/out/cam305_none → (exp id, 中文名)

    - 传了 --experiment 就用它
    - 目录在 output_crosscam/analysis 下才自动分组
    - 其它（单目录导入 / 临时目录）返回 None，让 run 落到「未归入批次」而不是编个假实验
    """
    if forced:
        return f"exp_{slug(forced)}", forced
    prefix = "output_crosscam/analysis/"
    if not rel_dir.replace("\\", "/").startswith(prefix):
        return None
    parent = os.path.dirname(rel_dir)
    key = parent[len(prefix):]
    cn = EXPERIMENT_CN.get(key)
    if cn is None:
        cn = key.replace("intent/", "意图 ").replace("/", " · ")
    return f"exp_{slug(key)}", cn


def ingest(conn, items: list[dict], dry: bool = False, experiment: str | None = None) -> dict:
    made, skipped = [], 0
    exp_best: dict[str, tuple[float, str]] = {}

    for it in items:
        cfg, overall, parsed = it["cfg"], it["overall"], it["parsed"]
        n_ep = len(parsed["curve"])
        if n_ep == 0:
            skipped += 1
            continue

        # metrics.json（intent_model.py 等）结构与 metrics_full.json 不同，先归一再取
        norm = db.normalize_metrics(it["rep"]) or {}
        if not isinstance(norm, dict):
            norm = {}
        if norm.get("acc") is not None:
            overall = {"acc": norm.get("acc"), "macro_f1": norm.get("macro_f1"),
                       "n": norm.get("n")}

        h = hashlib.sha1(it["rel_dir"].encode()).hexdigest()[:6]
        rid = f"run_tr_{slug(it['name'])}_{h}"
        # 同一 out_dir 已被「别的」run 记录（例如 ops_run 跑出来的）→ 不重复导入；
        # 同 id 则是本工具自己的记录，允许覆盖刷新（幂等）。
        dup = conn.execute("SELECT id FROM run WHERE out_dir=? AND id<>?",
                           (it["rel_dir"], rid)).fetchone()
        if dup:
            skipped += 1
            continue
        exp_pair = experiment_of(it["rel_dir"], experiment)
        exp_id, exp_name = exp_pair if exp_pair else (None, None)

        # 时间：以指标文件 mtime 为结束，减去日志内各 epoch 秒数之和为开始
        try:
            ended = datetime.fromtimestamp(os.path.getmtime(it["metrics_path"]))
        except OSError:
            ended = datetime.now()
        total_s = sum(c.get("secs", 0) for c in parsed["curve"])
        started = ended - timedelta(seconds=total_s) if total_s else None

        params = {
            "variant": it["name"],
            "epochs": n_ep,
            "task": parsed["env"].get("task") or cfg.get("cams"),
            "id_mode": parsed["env"].get("id_mode") or cfg.get("id_mode"),
            "n_train": _int(parsed["data"].get("train") or cfg.get("n_train")),
            "n_val": _int(parsed["data"].get("val") or cfg.get("n_val")),
            "n_test": _int(parsed["data"].get("test") or overall.get("n")),
            "params": parsed["model"].get("params"),
            "best_ep": cfg.get("best_ep") or norm.get("best_ep"),
            "best_val_f1": _round(cfg.get("best_val_f1") or norm.get("best_val_f1")),
            "device": parsed["env"].get("device"),
            "torch": parsed["env"].get("torch"),
        }
        metrics = {
            "acc": _round(overall.get("acc")),
            "macro_f1": _round(overall.get("macro_f1")),
            "n": overall.get("n"),
            "per_cam": {k: _round((v or {}).get("acc") if isinstance(v, dict) else v)
                        for k, v in (norm.get("per_cam") or {}).items()},
        }
        made.append({"id": rid, "name": it["name"], "exp": exp_id,
                     "n_epochs": n_ep, "acc": metrics["acc"], "macro_f1": metrics["macro_f1"]})
        if exp_id and metrics["macro_f1"] is not None:
            cur = exp_best.get(exp_id)
            if cur is None or metrics["macro_f1"] > cur[0]:
                exp_best[exp_id] = (metrics["macro_f1"], it["name"])

        if dry:
            continue

        conn.execute(
            "INSERT INTO run(id, name, kind, entrypoint, params_json, env_json, host, gpu, "
            "status, started_at, ended_at, exit_code, log_uri, metrics_json, out_dir, note, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET params_json=excluded.params_json, "
            "env_json=excluded.env_json, metrics_json=excluded.metrics_json, "
            "started_at=excluded.started_at, ended_at=excluded.ended_at, log_uri=excluded.log_uri",
            (rid, it["name"], "train", "tools/intent_model*.py",
             db.jdump(params),
             db.jdump({"source": "historical", "rel_dir": it["rel_dir"]}),
             "gpu-server", params.get("device") or "—",
             "succeeded",
             started.strftime("%Y-%m-%d %H:%M:%S") if started else None,
             ended.strftime("%Y-%m-%d %H:%M:%S"), 0,
             db.rel(it["log_path"]) if it["log_path"] else None,
             db.jdump(metrics), it["rel_dir"],
             f"历史回填 · {n_ep} epoch", db.now()))

        for c in parsed["curve"]:
            db.set_metric(conn, rid, c["step"], "loss", c["loss"])
            db.set_metric(conn, rid, c["step"], "val_acc", c["val_acc"])
            if "val_f1" in c:
                db.set_metric(conn, rid, c["step"], "val_f1", c["val_f1"])

        for p, kind in ((it["log_path"], "log"), (it["metrics_path"], "metrics")):
            if p and os.path.isfile(p):
                db.add_artifact(conn, rid, p, kind=kind)

        if exp_id:
            conn.execute(
                "INSERT INTO experiment(id, name, design_json, status, conclusion, created_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
                (exp_id, exp_name,
                 db.jdump({"auto": "ops_ingest_train", "parent": os.path.dirname(it["rel_dir"])}),
                 "done", None, db.now()))
            conn.execute("INSERT OR IGNORE INTO experiment_run(experiment_id, run_id, role) VALUES(?,?,?)",
                         (exp_id, rid, "variant"))

    if not dry:
        for exp_id, (best, name) in exp_best.items():
            conn.execute(
                "UPDATE experiment SET conclusion=? WHERE id=? AND (conclusion IS NULL OR conclusion='')",
                (f"自动汇总：最佳变体 {name}（macro_f1 {best:.3f}）", exp_id))
        conn.commit()

    return {"runs": len(made), "skipped": skipped, "experiments": len(exp_best),
            "exp_ids": sorted(exp_best), "loose": sum(1 for m in made if not m["exp"])}


def _round(v):
    return round(v, 4) if isinstance(v, (int, float)) else None


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_SCAN)
    ap.add_argument("--experiment", default=None,
                    help="强制归入某个实验名（不传则按目录推断，推断不出的落到未归入批次）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.root):
        print(f"扫描根不存在：{a.root}")
        return 2
    items = collect(a.root)
    conn = db.connect()
    try:
        r = ingest(conn, items, dry=a.dry_run, experiment=a.experiment)
        total = conn.execute("SELECT COUNT(*) FROM run_metric").fetchone()[0]
        print(f"{'[dry-run] ' if a.dry_run else ''}训练 run {r['runs']} 条"
              f"（跳过无曲线 {r['skipped']}），实验 {r['experiments']} 组，序列点 {total}")
        if r["loose"]:
            print(f"  另有 {r['loose']} 个 run 未归入批次（训练页「未归入批次」可见）")
        for eid in r["exp_ids"]:
            row = conn.execute(
                "SELECT e.name, (SELECT COUNT(*) FROM experiment_run er WHERE er.experiment_id=e.id) n, "
                "e.conclusion FROM experiment e WHERE e.id=?", (eid,)).fetchone()
            if row:
                print(f"  {row['name']:<28} {row['n']:>3} 变体  {row['conclusion'] or ''}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
