"""盘点 output_crosscam 下的训练产物：metrics_full.json + 同名 .log 的 epoch 曲线。

只读。输出每个训练目录的：指标键、日志中的 epoch 行数、超参首行、checkpoint 是否存在。
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCAN = os.path.join(ROOT, "output_crosscam")

EP = re.compile(r"^\[ep\s*(\d+)\]\s*loss=([\d.]+)\s+val_acc=([\d.]+)(?:\s+val_f1=([\d.]+))?")

rows = []
for dirpath, dirnames, filenames in os.walk(SCAN):
    if "metrics_full.json" not in filenames:
        continue
    mpath = os.path.join(dirpath, "metrics_full.json")
    try:
        d = json.load(open(mpath, encoding="utf-8"))
    except Exception as e:
        rows.append({"dir": os.path.relpath(dirpath, ROOT), "err": str(e)})
        continue

    rep = d.get("rep") or d
    cfg = (rep.get("config") or {}) if isinstance(rep, dict) else {}
    overall = (rep.get("overall") or {}) if isinstance(rep, dict) else {}

    # 训练脚本把日志写在结果目录的同级：<dir>.log
    cand = [os.path.join(dirpath, f) for f in filenames if f.endswith(".log")]
    sib = dirpath.rstrip("/\\") + ".log"
    if os.path.isfile(sib):
        cand.insert(0, sib)
    eps, meta = 0, {}
    log_name = os.path.basename(cand[0]) if cand else None
    if log_name:
        text = open(cand[0], encoding="utf-8", errors="replace").read()
        eps = len([ln for ln in text.splitlines() if EP.match(ln)])
        for ln in text.splitlines()[:20]:
            if ln.startswith("[env]"):
                meta["env"] = ln[6:].strip()
            elif ln.startswith("[data]"):
                meta["data"] = ln[7:].strip()
            elif ln.startswith("[model]"):
                meta["model"] = ln[8:].strip()

    rows.append({
        "dir": os.path.relpath(dirpath, ROOT),
        "keys": sorted(rep.keys())[:8] if isinstance(rep, dict) else type(rep).__name__,
        "n_train": cfg.get("n_train"),
        "best_ep": cfg.get("best_ep"),
        "best_val_f1": round(cfg.get("best_val_f1"), 4) if isinstance(cfg.get("best_val_f1"), (int, float)) else None,
        "acc": round(overall.get("acc"), 4) if isinstance(overall.get("acc"), (int, float)) else None,
        "macro_f1": round(overall.get("macro_f1"), 4) if isinstance(overall.get("macro_f1"), (int, float)) else None,
        "epochs": eps,
        "log": log_name,
        "meta": meta,
        "ckpt": [f for f in filenames if f.endswith((".pt", ".pth", ".npz"))][:2],
    })

print(f"总目录 {len(rows)}")
with_curve = [r for r in rows if r.get("epochs")]
print(f"有 epoch 曲线 {len(with_curve)}")
print()
for r in rows[:14]:
    print(f"{r['dir']:<52} ep={r.get('epochs')} acc={r.get('acc')} f1={r.get('macro_f1')} ckpt={r.get('ckpt')}")
print()
print("--- 无曲线的目录 ---")
for r in rows:
    if not r.get("epochs"):
        print(f"  {r['dir']}  keys={r.get('keys')}")
print()
print("--- env 行样本 ---")
seen = set()
for r in rows:
    e = (r.get("meta") or {}).get("env")
    if e and e not in seen:
        seen.add(e)
        print("  ", e, "| data:", (r.get("meta") or {}).get("data"), "|", r.get("meta", {}).get("model"))
