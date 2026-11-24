#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨机位对照：同一份 raw 上跑 base 与 hybrid，看「框数变化率」是否是该机位的指纹。

核心判据：
  框数变化率 = (hy 的 NMS 后框数 − base 的) / base 的
  → 显著 >0 = 该机位存在「低分大框吃掉高分小框」（hybrid 才有用武之地）
  → ≈0      = 该机位没有这个问题，改 NMS 无效

用法: python tools/exp_c_crosscam.py 305:10500 318:800 320:800 323:800 343:800 316:2688
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PY = sys.executable
OUT = Path("probe_out/grid")
OUT.mkdir(parents=True, exist_ok=True)


def run(raw: str, out: str, extra: list[str]) -> dict:
    cmd = [PY, "tools/retrack_from_raw.py", "--raw", raw, "--out", out] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    d: dict = {}
    for ln in (r.stdout + r.stderr).splitlines():
        if ln.startswith("[stats]"):
            try:
                d.update(json.loads(ln[len("[stats] "):].replace("'", '"')))
            except Exception:
                pass
        if ln.startswith("[box]"):
            for tok in ln.split():
                k, _, v = tok.partition("=")
                if k == "NMS后":
                    d["kept"] = int(v)
    return d


results = []
for spec in sys.argv[1:]:
    cam, _, nfr = spec.partition(":")
    raw = f"probe_out/{cam}_raw10.jsonl" if cam != "305" else "probe_out/305_raw10.full.jsonl"
    if cam == "316":
        raw = "probe_out/316_raw10.part.jsonl"
    b = run(raw, str(OUT / f"x_{cam}_base.jsonl"), [])
    h = run(raw, str(OUT / f"x_{cam}_hy8.jsonl"), ["--policy", "hybrid", "--hy-score-k", "0.8"])
    results.append({"cam": cam, "frames": b.get("tracks") is not None and int(nfr), "base": b, "hy": h})
    print(f"[{cam}] base={b} hy={h}", flush=True)

print()
hdr = (f"{'cam':5s} {'帧':>6s} | {'tracks':>13s} {'breaks':>13s} {'switch':>11s} "
       f"{'NMS后框':>16s} {'框变化':>8s}")
print(hdr)
print("-" * len(hdr))
for r in results:
    b, h = r["base"], r["hy"]
    dk = (h.get("kept", 0) - b.get("kept", 0)) / max(1, b.get("kept", 1)) * 100
    print(f"{r['cam']:5s} {r['frames']:>6d} | "
          f"{b.get('tracks','-'):>5}→{h.get('tracks','-'):<6} "
          f"{b.get('breaks','-'):>5}→{h.get('breaks','-'):<6} "
          f"{b.get('switches','-'):>4}→{h.get('switches','-'):<5} "
          f"{b.get('kept',0):>7}→{h.get('kept',0):<7} {dk:>+7.2f}%")
