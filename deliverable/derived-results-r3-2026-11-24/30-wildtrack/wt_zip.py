"""WILDTRACK zip 的最小读取器（带中央目录缓存）。

背景：这个包是 32 位偏移溢出（>4 GB），EOCD 里的 cd_off 被截断，真实偏移要用
`文件长 - 22 - cd_size` 反推；本地头偏移同样可能少 4 GB，逐个试 PK\\x03\\x04。

用法：
  python tmp/wt/wt_zip.py list [子串]     列出条目
  python tmp/wt/wt_zip.py get <名字...>   取回文件（落到 tmp/wt/data/Wildtrack_dataset/）
"""
import json
import sys
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings()
URL = "https://hf-mirror.com/datasets/sein123/wildtrack/resolve/main/Wildtrack_dataset_full.zip"
N = 6807496358
OUT = Path("tmp/wt/data/Wildtrack_dataset")
CD = Path("tmp/wt/cd.json")
S = requests.Session()
S.verify = False
S.headers["User-Agent"] = "Mozilla/5.0"


def rng(a, b):
    for _ in range(4):
        try:
            r = S.get(URL, headers={"Range": f"bytes={a}-{b}"}, timeout=180)
            if r.status_code in (200, 206):
                return r.content
        except Exception:
            pass
    raise RuntimeError(f"range {a}-{b} failed")


def entries():
    if CD.exists():
        return json.loads(CD.read_text(encoding="utf-8"))
    import struct
    tail = rng(N - 70000, N - 1)
    i = tail.rfind(b"PK\x05\x06")
    _, _, _, _, _, cds, cdo, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    if abs(cdo - (N - 22 - cds)) > 1024:
        cdo = N - 22 - cds
    cd = rng(cdo, cdo + cds - 1)
    out, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        f = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        name = cd[p + 46:p + 46 + f[10]].decode("utf-8", "replace")
        out.append(dict(name=name, method=f[4], cs=f[8], us=f[9], off=f[16]))
        p += 46 + f[10] + f[11] + f[12]
    CD.write_text(json.dumps(out), encoding="utf-8")
    return out


def fetch(e):
    dest = OUT / e["name"]
    if dest.exists() and dest.stat().st_size == e["us"]:
        return "cached"
    hdr = None
    for d in (0, 4294967296):
        o = e["off"] + d
        if o + e["cs"] > N:
            continue
        try:
            h = rng(o, o + 29)
        except Exception:
            continue
        if h[:4] == b"PK\x03\x04":
            e["off"], hdr = o, h
            break
    if hdr is None:
        return "bad-header"
    import struct
    nlen, elen = struct.unpack("<HH", hdr[26:30])
    data = rng(e["off"] + 30 + nlen + elen, e["off"] + 30 + nlen + elen + e["cs"] - 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return f"ok {len(data)}"


def main():
    es = entries()
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        sub = sys.argv[2] if len(sys.argv) > 2 else ""
        hit = [e for e in es if sub in e["name"] and not e["name"].startswith("__MACOSX")]
        print(f"总条目 {len(es)}，匹配 {len(hit)}")
        for e in hit[:60]:
            print(f"  {e['us']:>10}  {e['name']}")
        return
    for name in sys.argv[2:]:
        hit = [e for e in es if e["name"] == name and not e["name"].startswith("__MACOSX")]
        if not hit:
            print(f"未找到 {name}")
            continue
        print(f"{name}: {fetch(hit[0])}")


if __name__ == "__main__":
    main()
