"""预生成身份选择器的轨迹缩略图（一次性，539 条）。

用法：
    PY=~/.workbuddy/binaries/python/envs/default/Scripts/python.exe
    "$PY" tools/build_selector_thumbs.py            # 只补缺失的
    "$PY" tools/build_selector_thumbs.py --force    # 全量重建

为什么要预生成：每张要 seek 一次原生 1440p 视频（300–500 MB/路），
在接口里现裁会让候选表一次拉 40 张时排队几十秒。落盘后接口只做静态读。
产物：webapp/backend/data/reid/thumbs/{cam}_{tid}.jpg（宽 256）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp" / "backend"))

import selector as selmod  # noqa: E402
import selthumb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sel = selmod.init()
    if not sel.ready:
        print(f"[fail] 选择器未就绪：{sel.err}")
        return 1
    rows = sel.tracks()
    if args.limit:
        rows = rows[: args.limit]
    print(f"[env] {len(rows)} 轨 / {len({r['cam'] for r in rows})} 机位", flush=True)

    ok = skip = 0
    fails: list[str] = []
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        p = selthumb.path_of(r["cam"], r["tid"])
        if p.exists() and not args.force and p.stat().st_size > 512:
            skip += 1
            continue
        try:
            selthumb.ensure(sel, r["cam"], r["tid"], force=args.force)
            ok += 1
        except Exception as e:
            fails.append(f"{r['cam']}#{r['tid']} {type(e).__name__}: {e}")
        if n % 50 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(rows)}  新裁 {ok} 跳过 {skip} 失败 {len(fails)}  "
                  f"{el:.0f}s  ETA {(len(rows) - n) * el / max(1, ok + skip):.0f}s", flush=True)

    el = time.time() - t0
    print(f"[done] 新裁 {ok} / 跳过 {skip} / 失败 {len(fails)}  用时 {el:.0f}s", flush=True)
    for f in fails[:12]:
        print("  ✗", f, flush=True)
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
