"""打两个 zip：主包（不含拼图）+ 拼图包。"""
import hashlib
import os
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "deliverable", "data-request-v2")
DIST = os.path.join(ROOT, "deliverable", "dist")
STAMP = "2026-10-19"
TOP = f"data-request-v2_{STAMP}"


def add_dir(zf, sub, skip_prefix=None):
    n = 0
    base = os.path.join(SRC, sub) if sub else SRC
    for dp, dn, fn in os.walk(base):
        dn.sort()
        for f in sorted(fn):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, SRC).replace("\\", "/")
            if skip_prefix and rel.startswith(skip_prefix):
                continue
            arc = f"{TOP}/{rel}"
            comp = zipfile.ZIP_STORED if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".zip", ".npz", ".mp4")) else zipfile.ZIP_DEFLATED
            zf.write(p, arc, compress_type=comp)
            n += 1
    return n


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


os.makedirs(DIST, exist_ok=True)
t0 = time.time()

# 主包
p1 = os.path.join(DIST, f"{TOP}.zip")
with zipfile.ZipFile(p1, "w", allowZip64=True) as zf:
    n_main = add_dir(zf, None, skip_prefix="gt/crops/")
print(f"[1/2] {p1}")
print(f"      {n_main} 文件  {os.path.getsize(p1)/1024/1024:.1f} MB  {sha(p1)[:16]}")

# 拼图包
p2 = os.path.join(DIST, f"{TOP}_crops.zip")
with zipfile.ZipFile(p2, "w", allowZip64=True) as zf:
    n_crop = 0
    for dp, dn, fn in os.walk(os.path.join(SRC, "gt", "crops")):
        dn.sort()
        for f in sorted(fn):
            p = os.path.join(dp, f)
            zf.write(p, f"{TOP}/gt/crops/" + os.path.relpath(p, os.path.join(SRC, "gt", "crops")).replace("\\", "/"),
                     compress_type=zipfile.ZIP_STORED)
            n_crop += 1
print(f"[2/2] {p2}")
print(f"      {n_crop} 文件  {os.path.getsize(p2)/1024/1024:.1f} MB  {sha(p2)[:16]}")
print(f"\n耗时 {time.time()-t0:.1f}s")
