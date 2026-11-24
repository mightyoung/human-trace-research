# -*- coding: utf-8 -*-
import json, os, glob, sqlite3

BASE = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "webapp", "backend", "data")


def load(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"__err__": str(e)}


print("=== 数据包规模 ===")
for fn in ["labels.json", "pairs.json", "masks.json", "mirror.json", "gil.json",
           "metrics.json", "cams.json", "chains.json", "entities.json"]:
    p = os.path.join(BASE, fn)
    if not os.path.exists(p):
        print(f"{fn:16s} MISSING")
        continue
    d = load(p)
    if isinstance(d, dict):
        desc = ", ".join(f"{k}={len(v) if isinstance(v, (list, dict)) else v}"
                         for k, v in list(d.items())[:8])
        print(f"{fn:16s} dict({len(d)}) {desc[:180]}")
    elif isinstance(d, list):
        print(f"{fn:16s} list({len(d)})")
    else:
        print(f"{fn:16s} {type(d)}")

print("\n=== labels.json 结构 ===")
lab = load(os.path.join(BASE, "labels.json"))
if isinstance(lab, dict):
    for k, v in lab.items():
        print(" ", k, type(v).__name__, len(v) if isinstance(v, (list, dict)) else v)
    # 抽样
    for k, v in lab.items():
        if isinstance(v, dict) and v:
            kk = list(v.keys())[0]
            print(f"  sample {k}[{kk}] =", json.dumps(v[kk], ensure_ascii=False)[:200])

print("\n=== truth 目录 ===")
for p in sorted(glob.glob(os.path.join(BASE, "truth", "*")))[:30]:
    print(" ", os.path.basename(p), os.path.getsize(p) if os.path.isfile(p) else "(dir)")

print("\n=== 前端 public 静态标注载荷 ===")
pub = os.path.join(os.path.dirname(BASE), "frontend", "public")
for p in sorted(glob.glob(os.path.join(pub, "*.json"))):
    d = load(p)
    n = len(d) if isinstance(d, (list, dict)) else "?"
    print(f"  {os.path.basename(p):28s} {os.path.getsize(p):>9d}B  n={n}")
    if isinstance(d, dict):
        print("     keys:", list(d.keys())[:10])
