#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C 方案策略网格：同一份 raw 原始框上跑多组参数，比 tracks / breaks / switches。

判据（三者必须一起看，否则会自欺）：
  tracks ↑   = 轨迹更碎（坏）
  breaks ↑   = 断点更多（坏）
  switches ↑ = 换号更多（坏）
  n_kept 不变 = 没靠"少检人"换指标（关键反作弊项）
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RAW = sys.argv[1] if len(sys.argv) > 1 else "probe_out/305_raw10.part.jsonl"
CMP = sys.argv[2] if len(sys.argv) > 2 else "output_crosscam/long35/305_long35_detections.jsonl"
PY = sys.executable
OUT = Path("probe_out/grid")
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    ("base", {}),
    ("mm0.7", {"--mm-thr": "0.7"}),
    ("mm0.6", {"--mm-thr": "0.6"}),
    ("mm0.5", {"--mm-thr": "0.5"}),
    ("mm0.4", {"--mm-thr": "0.4"}),
    ("hy0.9", {"--policy": "hybrid", "--hy-score-k": "0.9"}),
    ("hy0.8", {"--policy": "hybrid", "--hy-score-k": "0.8"}),
    ("score", {"--policy": "score"}),
    ("gate1.5shrink", {"--size-gate": "1.5", "--size-act": "shrink"}),
    ("gate1.35shrink", {"--size-gate": "1.35", "--size-act": "shrink"}),
    ("gate1.35drop", {"--size-gate": "1.35", "--size-act": "drop"}),
    ("mm0.6+hy0.9", {"--mm-thr": "0.6", "--policy": "hybrid", "--hy-score-k": "0.9"}),
    ("mm0.6+gate1.4s", {"--mm-thr": "0.6", "--size-gate": "1.4", "--size-act": "shrink"}),
    ("all", {"--mm-thr": "0.6", "--policy": "hybrid", "--hy-score-k": "0.9",
             "--size-gate": "1.4", "--size-act": "shrink"}),
]

rows = []
for name, kw in CASES:
    out = OUT / f"{name}.jsonl"
    cmd = [PY, "tools/retrack_from_raw.py", "--raw", RAW, "--out", str(out), "--compare", CMP]
    for k, v in kw.items():
        cmd += [k, v]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    txt = r.stdout + r.stderr
    d = {"case": name, **kw}
    for ln in txt.splitlines():
        if ln.startswith("[stats]"):
            d["stats"] = ln[len("[stats] "):]
        if ln.startswith("[box]"):
            for tok in ln.split():
                if "=" in tok:
                    key, _, val = tok.partition("=")
                    d[key] = val
        if ln.startswith("[compare] 轨迹划分一致"):
            d["partition_same"] = "是" in ln
    rows.append(d)
    print(f"[{name}] {d.get('stats')}  kept={d.get('NMS后')} gate={d.get('尺寸门修正')}", flush=True)

print()
hdr = f"{'case':16s} {'tracks':>6s} {'breaks':>7s} {'switch':>7s} {'kept':>6s} {'gate':>5s} {'划分同':>6s}"
print(hdr)
print("-" * len(hdr))
for d in rows:
    s = d.get("stats", "{}")
    try:
        st = json.loads(s.replace("'", '"'))
    except Exception:
        st = {}
    print(f"{d['case']:16s} {st.get('tracks', '-'):>6} {st.get('breaks', '-'):>7} "
          f"{st.get('switches', '-'):>7} {d.get('NMS后', '-'):>6} {d.get('尺寸门修正', '-'):>5} "
          f"{'Y' if d.get('partition_same') else '-':>6}")
Path(OUT / "grid_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
