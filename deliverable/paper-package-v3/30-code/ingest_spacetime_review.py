"""把时空矛盾复核页导出的判读结果回写成跨机位真值。

用法
  python tools/ingest_spacetime_review.py deliverable/spacetime_review/labels.json

口径（写死在这里，避免每次重新解释）：
  - v=same  → label 1（同人）
  - v=diff  → label 0（异人）
  - v=unsure / 无 v         → 不进真值
  - flag=boundary（交界处走过）→ 不进真值：这类重叠不能当「两个人」的证据
  - flag=mixA / mixB（某一侧轨内混人）→ 不进真值：比较对象本身不纯
  ★ 这 27 条是「算法已合并 + 时空矛盾」的**可疑样本**，只能用来量**合并精度**，
    不能当 AUC / 召回的抽样集（严重偏向负例）。所以单独存一个文件，不并进 gt_long35.json。
"""
import collections
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "deliverable/spacetime_review"
GT = ROOT / "deliverable/gt_long35/gt_long35.json"
OUT = BASE / "gt_spacetime_items.json"
REPORT = BASE / "ingest_report.md"


def main(labels_path):
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    L = labels.get("labels") or {}
    items = json.loads((BASE / "items.json").read_text(encoding="utf-8"))["items"]
    by_id = {it["id"]: it for it in items}

    keep, drop = [], collections.Counter()
    for iid, r in sorted(L.items()):
        it = by_id.get(iid)
        if not it:
            drop["未知条目"] += 1
            continue
        v, flag = r.get("v"), r.get("flag")
        if v not in ("same", "diff"):
            drop["未判/看不清"] += 1
            continue
        if flag in ("boundary", "mixA", "mixB"):
            drop["标记:" + flag] += 1
            continue
        keep.append({
            "a": it["a"], "b": it["b"], "pair": it["pair"],
            "label": 1 if v == "same" else 0,
            "overlap_s": it["ov"], "group": it["group"],
            "source": "spacetime_review", "item": iid,
        })

    OUT.write_text(json.dumps({
        "built": datetime.now().isoformat(timespec="seconds"),
        "from": str(Path(labels_path).name),
        "n_items": len(items), "n_labeled": len(L), "n_usable": len(keep),
        "caveat": "可疑样本（已合并+时空矛盾），只用于合并精度，不可用于 AUC/召回",
        "items": keep,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 报告 ----
    n_same = sum(1 for k in keep if k["label"] == 1)
    n_diff = len(keep) - n_same
    per_pair = collections.defaultdict(lambda: [0, 0])
    for k in keep:
        per_pair[k["pair"]][0] += 1
        per_pair[k["pair"]][1] += k["label"]
    old = json.loads(GT.read_text(encoding="utf-8"))["items"]
    old_pairs = {o["pair"] for o in old}
    lines = [f"# 时空矛盾复核结果（{datetime.now():%Y-%m-%d %H:%M}）", "",
             f"- 队列 {len(items)} 条，已判 {len(L)} 条，可用 {len(keep)} 条"
             f"（剔除：{'；'.join(f'{k} {v}' for k, v in drop.items()) or '无'}）",
             f"- **误合并 {n_diff} / {len(keep)} = "
             f"{n_diff / len(keep) * 100:.1f}%**" if keep else "- 无可用判读",
             "", "| 机位对 | 可判 | 同人 | 误合并 |", "|---|---|---|---|"]
    for p in sorted(per_pair, key=lambda x: -sum(per_pair[x])):
        n, s = per_pair[p]
        lines.append(f"| {p} | {n} | {s} | {n - s} |")
    new_pairs = sorted({k["pair"] for k in keep} - old_pairs)
    lines += ["", f"- 新增覆盖机位对：{', '.join(new_pairs) if new_pairs else '无'}",
              f"- 真值总机位对：{len(old_pairs | {k['pair'] for k in keep})}",
              "", "> 这批是「算法已合并」的样本，只能当精度审计，不能当召回/AUC 的抽样。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n→ {OUT}\n→ {REPORT}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
