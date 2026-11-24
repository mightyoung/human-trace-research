#!/usr/bin/env python
"""把控制台「真值标注」页的判定与实时指标拉回本地。

远程：/home/r4900g3/human-count/webapp/backend/data/truth/{labels.json,labels.csv,event_labels.csv}
      /api/truth/metrics（排序准确率 / Brier / 分层估计 / 同预算效率 / relay 精度）
本地：output_crosscam/analysis/truth/remote/

用法： python tools/pull_truth_labels.py
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN = ROOT / "tools" / "remote_run.py"
RDIR = "/home/r4900g3/human-count/webapp/backend/data/truth"
LDIR = ROOT / "output_crosscam" / "analysis" / "truth" / "remote"
LDIR.mkdir(parents=True, exist_ok=True)

FILES = ["labels.json", "labels.csv", "event_labels.csv", "metrics.json"]


def _fmt(x):
    try:
        return "%.3f" % float(x)
    except Exception:  # noqa: BLE001
        return "—"


def rr(*args) -> str:
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    p = subprocess.run([sys.executable, str(RUN), *args], capture_output=True, text=True, env=env)
    return (p.stdout or "") + (p.stderr or "")


# 先把服务端的实时指标落成文件（服务在跑时才可用）
rr(f"curl -s -m 8 http://127.0.0.1:8099/api/truth/metrics -o {RDIR}/metrics.json && "
   f"head -c 200 {RDIR}/metrics.json")

ls = rr(f"ls -la {RDIR}/*.csv {RDIR}/labels.json {RDIR}/metrics.json 2>/dev/null")
print(ls.strip() or "(远程暂无标签文件)")
n = 0
for f in FILES:
    out = rr("--get", f"{RDIR}/{f}", str(LDIR / f))
    if (LDIR / f).exists() and (LDIR / f).stat().st_size > 0:
        print("已取回", f, (LDIR / f).stat().st_size, "B")
        n += 1
    else:
        print("跳过（不存在）", f)
print(f"→ {LDIR}（{n} 个文件）")

# 摘要打印关键指标
try:
    import json
    m = json.loads((LDIR / "metrics.json").read_text(encoding="utf-8"))
    print("\n[指标] 预算 %s/%s · 排序准确率 %s · Brier %s · 估计同人 %s (%s) · "
          "同预算命中 %s vs 随机 %s · relay 精度 %s"
          % (m.get("budget_done"), m.get("budget"),
             _fmt(m.get("rank_acc")), _fmt(m.get("brier")),
             m.get("est_total_same"), _fmt(m.get("prevalence")),
             m.get("arm_found"), m.get("random_expected"),
             _fmt((m.get("relay") or {}).get("precision"))))
except Exception as e:  # noqa: BLE001
    print("[warn] 指标摘要失败：", e)
