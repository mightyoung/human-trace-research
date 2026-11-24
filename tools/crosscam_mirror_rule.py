#!/usr/bin/env python
"""P2：跨机一致性镜像判定规则（v1）

物理依据：**镜像与其真身天然同步**——玻璃反光里的"人"与被照者（可能在另一机位画面里）
在时间上几乎同时出现/消失（实测镜像对 dt≈0.6~1.4s，且外观余弦高）；
而**真实跨区转移**必须包含步行时间（通常 ≥3s，视机位间距）。

规则（v1，仅用 时序 dt + 外观 cos；不依赖掩膜）：
  - dt < dt_fast 且 cos ≥ cos_min  → MIRROR_SUSPECT（镜像嫌疑；也可能是极近机位快速转移，需深度/空间复核）
  - dt ≥ dt_slow                    → TRANSITION_CANDIDATE（含步行时间）
  - 其余                             → UNDECIDED

用法：
  python tools/crosscam_mirror_rule.py \
    --events output_crosscam/analysis/masked/transfer_events.json \
    --out output_crosscam/analysis/masked/mirror_rule.json
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt-fast", type=float, default=2.0)
    ap.add_argument("--dt-slow", type=float, default=3.0)
    ap.add_argument("--cos-min", type=float, default=0.6)
    args = ap.parse_args()

    d = json.loads(Path(args.events).read_text(encoding="utf-8"))
    cands = d.get("candidates", [])
    rows = []
    for c in cands:
        if c.get("cos") is None:
            tier = "NO_COS"
        elif c["dt"] < args.dt_fast and c["cos"] >= args.cos_min:
            tier = "MIRROR_SUSPECT"
        elif c["dt"] >= args.dt_slow:
            tier = "TRANSITION_CANDIDATE"
        else:
            tier = "UNDECIDED"
        rows.append({**c, "tier": tier})

    from collections import Counter
    cnt = Counter(r["tier"] for r in rows)
    print(f"[mirror_rule] {args.events}")
    print(f"  规则: dt<{args.dt_fast}s & cos>={args.cos_min} -> MIRROR_SUSPECT; "
          f"dt>={args.dt_slow}s -> TRANSITION_CANDIDATE")
    print(f"  计数: {dict(cnt)}  (共 {len(rows)})")

    def fmt(r):
        dd, aa = r["disp"], r["app"]
        return (f"  [{r['tier']}] {r['dirn']}  {dd['tag']}#{dd['tid']} -> {aa['tag']}#{aa['tid']}  "
                f"dt={r['dt']}s cos={r['cos']}")

    print("\n=== MIRROR_SUSPECT（镜像嫌疑）===")
    for r in sorted([r for r in rows if r["tier"] == "MIRROR_SUSPECT"],
                    key=lambda r: -r["cos"]):
        print(fmt(r))
    print("\n=== TRANSITION_CANDIDATE（含步行时间的真实转移候选）===")
    for r in sorted([r for r in rows if r["tier"] == "TRANSITION_CANDIDATE"],
                    key=lambda r: -r["cos"]):
        print(fmt(r))

    Path(args.out).write_text(json.dumps(
        {"dt_fast": args.dt_fast, "dt_slow": args.dt_slow, "cos_min": args.cos_min,
         "counts": dict(cnt), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
