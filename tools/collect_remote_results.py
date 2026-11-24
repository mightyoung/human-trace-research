#!/usr/bin/env python3
"""Collect every experiment OUTPUT produced on the compute node.

Run from the repo root:

    python tools/remote_run.py --put tools/collect_remote_results.py /home/r4900g3/_pack_results.py
    python tools/remote_run.py "cd ~ && python3 _pack_results.py --dry --list /tmp/hc_list.txt"
    python tools/remote_run.py "cd ~ && python3 _pack_results.py --tar"
    MSYS_NO_PATHCONV=1 python tools/remote_run.py --get \
        /share/data4/yhh/data/human-count/_export/hc_results.tar remote-results/hc_results.tar

Rule: keep derived artefacts (metrics, track tables, embeddings, logs, plots,
scripts); skip inputs (video, frame sets, detector datasets) and binaries
(model weights, env tarballs, wheel caches).

Usage:
    python3 _pack_results.py --dry      # list + size, write nothing
    python3 _pack_results.py --tar      # write /share/data4/yhh/data/human-count/_export/hc_results.tar
"""
import argparse
import os
import sys

ROOTS = [
    "/share/data4/yhh/data/human-count",
    os.path.expanduser("~/human-count/analysis"),
    os.path.expanduser("~/e20/probe"),
]

# single files that live loose in $HOME and are real outputs
HOME_FILES = [
    "face_emb_gemma_p0c.npz",
    "face_manifest_p0c.json",
    "face_gemma.log",
]

# directories never descended into (inputs / weights / caches)
EXCL_DIR = {
    "models", "dataset", "raw", "__pycache__", ".cache", "C-RADIOv4-SO400M",
    "_p2160", "node_modules", ".git", "venvs", "miniconda3",
    "images", "labels",   # det-bench/yolo: the YOLO training set (an input)
}

# extensions that are inputs or binaries
EXCL_EXT = {
    ".mp4", ".avi", ".mkv", ".mov", ".ts", ".webm",
    ".pt", ".pth", ".pkl", ".tar", ".gz", ".tgz", ".whl", ".zip",
    ".safetensors", ".incomplete", ".pyc", ".lock", ".model", ".onnx",
    ".so", ".bin", ".ckpt",
}

KEEP = {
    ".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".md",
    ".npz", ".npy", ".png", ".jpg", ".jpeg", ".svg", ".pdf",
    ".py", ".sh", ".yaml", ".yml", ".done", ".html", ".ipynb",
}


def want(path):
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    if ext in EXCL_EXT:
        return False
    if ext and ext not in KEEP:
        return False
    if not ext:
        return name in ("done",)
    return True


def walk(root):
    out = []
    if not os.path.exists(root):
        return out
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in EXCL_DIR]
        for fn in fns:
            p = os.path.join(dp, fn)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            if want(p):
                try:
                    out.append((p, os.path.getsize(p)))
                except OSError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--tar", action="store_true")
    ap.add_argument("--list", help="write plain file list here")
    args = ap.parse_args()

    files = []
    for r in ROOTS:
        files += walk(r)
    home = os.path.expanduser("~")
    for f in HOME_FILES:
        p = os.path.join(home, f)
        if os.path.isfile(p):
            files.append((p, os.path.getsize(p)))

    files.sort()
    total = sum(s for _, s in files)

    # summary by top-level bucket
    bucket = {}
    for p, s in files:
        if p.startswith("/share/data4/yhh/data/human-count/"):
            rel = p[len("/share/data4/yhh/data/human-count/"):]
            k = rel.split("/")[0] + ("/" if "/" in rel else " (file)")
        elif p.startswith(home + "/human-count/"):
            k = "~human-count/" + p[len(home + "/human-count/"):].split("/")[0]
        elif p.startswith(home + "/e20/"):
            k = "~e20/" + p[len(home + "/e20/"):].split("/")[0]
        else:
            k = "~ (loose file)"
        a, c = bucket.get(k, (0, 0))
        bucket[k] = (a + s, c + 1)

    print("=== buckets ===")
    for k in sorted(bucket, key=lambda x: -bucket[x][0]):
        a, c = bucket[k]
        print("%9.1f MB  %6d files  %s" % (a / 1048576, c, k))
    print("=== total: %.2f GB in %d files ===" % (total / 1073741824, len(files)))

    ext_tot = {}
    for p, s in files:
        e = os.path.splitext(p)[1].lower() or "(none)"
        a, c = ext_tot.get(e, (0, 0))
        ext_tot[e] = (a + s, c + 1)
    print("=== by extension ===")
    for e in sorted(ext_tot, key=lambda x: -ext_tot[x][0]):
        a, c = ext_tot[e]
        print("%9.1f MB  %6d  %s" % (a / 1048576, c, e))

    if args.list:
        with open(args.list, "w", encoding="utf-8") as fh:
            for p, s in files:
                fh.write("%d\t%s\n" % (s, p))
        print("list ->", args.list)

    if args.tar:
        import tarfile
        outdir = "/share/data4/yhh/data/human-count/_export"
        os.makedirs(outdir, exist_ok=True)
        tpath = os.path.join(outdir, "hc_results.tar")
        n = 0
        with tarfile.open(tpath, "w") as tf:
            for p, _ in files:
                # arcname: keep the tree visible from the archive root
                if p.startswith("/share/data4/yhh/data/human-count/"):
                    arc = "remote/" + p[len("/share/data4/yhh/data/human-count/"):]
                else:
                    arc = "home/" + p[len(home) + 1:]
                tf.add(p, arcname=arc, recursive=False)
                n += 1
                if n % 500 == 0:
                    print("  ... %d" % n, flush=True)
        print("tar ->", tpath, "%.2f GB" % (os.path.getsize(tpath) / 1073741824))


if __name__ == "__main__":
    main()
