# -*- coding: utf-8 -*-
"""全局 ReID 缓存层演示：构建 store（六路全轨迹）→ 跨机位/跨时间检索 → 规模测算。

用法:
  python tools/global_reid_demo.py --models d3b             # 先用现有 DINOv3 嵌入演示
  python tools/global_reid_demo.py --models d3b,cradio      # 拿到新嵌入后对比
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from global_reid_store import GlobalReIDStore, build_meta_from_detections  # noqa: E402
from det_view import local_det  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ["318", "320", "305", "316", "323", "343"]
DATE = "2026-10-12"

MODELS = {
    "d3b": {
        "dir": os.path.join(ROOT, "output_crosscam", "analysis", "relay", "remote_emb"),
        "pat": "reid_emb_{cam}_d3b.npz",
        "version": "dinov3b_v1",
        "store": os.path.join(ROOT, "output_crosscam", "analysis", "global_reid", "store_d3b"),
    },
    "cradio": {
        "dir": os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale"),
        "pat": "{cam}_emb_scale.npz",
        "version": "cradio_gemma2_bridge_v1",
        "store": os.path.join(ROOT, "output_crosscam", "analysis", "global_reid", "store_cradio"),
    },
    # v3 = 严格三集版（W 排除 test∪val 重训，2026-10-13）
    "cradio_v3": {
        "dir": os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale_v3"),
        "pat": "{cam}_emb_scale.npz",
        "version": "cradio_gemma2_bridge_v3",
        "store": os.path.join(ROOT, "output_crosscam", "analysis", "global_reid", "store_cradio_v3"),
    },
    # v4 = clean 视图版（cleanset + W 重训，2026-10-13，默认口径）
    "cradio_v4": {
        "dir": os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale_v4"),
        "pat": "{cam}_emb_scale.npz",
        "version": "cradio_gemma2_bridge_v4",
        "store": os.path.join(ROOT, "output_crosscam", "analysis", "global_reid", "store_cradio_v4"),
        "det_dir": "cleanset",   # 元数据从 cleanset 读取
    },
    "d3bridge_v3": {
        "dir": os.path.join(ROOT, "output_crosscam", "analysis", "dinov3_scale_v3"),
        "pat": "{cam}_emb_d3b.npz",
        "version": "dinov3b_gemma2_bridge_v3",
        "store": os.path.join(ROOT, "output_crosscam", "analysis", "global_reid", "store_d3bridge_v3"),
    },
}

POS_PAIRS = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23),
             (139, 24), (146, 28), (37, 11)]


def build_store(cfg):
    st = GlobalReIDStore(cfg["store"], cfg["version"], 768)
    st.prepare_rebuild()   # v2 代际库：已有发布代时先采用基线再清空（整库重建语义）
    n = 0
    for cam in CAMS:
        p = os.path.join(cfg["dir"], cfg["pat"].format(cam=cam))
        if not os.path.exists(p):
            print(f"  [skip] {cam}: {p} 不存在")
            continue
        meta = build_meta_from_detections(cam, local_det(cam))
        n += st.add_npz(p, cam=cam, date=DATE,
                        meta_fn=lambda tid: meta.get(tid, {}))
    st.save()
    return st, n


def demo(cfg, name):
    print(f"\n{'='*70}\n模型: {name}  ({cfg['version']})\n{'='*70}")
    t0 = time.time()
    st, n = build_store(cfg)
    s = st.stats()
    print(f"[build] {s['n']} 轨 / {s['bytes']/1e6:.2f} MB "
          f"({s['bytes_per_track']:.0f} B/轨) | {time.time()-t0:.1f}s")
    print(f"[build] 按机位: {s['by_cam']}")

    # ---- 1. 跨机位检索（8 对已知配对；两种 gallery：全库 vs 仅 320）----
    res_pairs = []
    for a, b in POS_PAIRS:
        ka = f"{DATE}|318|{a}"
        kb = f"{DATE}|320|{b}"
        if ka not in st.rows or kb not in st.rows:
            res_pairs.append({"pair": [a, b], "rank": None, "rank_320": None})
            continue
        hits = st.search(st.rows[ka], topk=600, exclude_cam="318")
        rank = next((i + 1 for i, h in enumerate(hits) if h["key"] == kb), None)
        hits320 = [h for h in hits if h["cam"] == "320"]
        rank320 = next((i + 1 for i, h in enumerate(hits320) if h["key"] == kb), None)
        res_pairs.append({"pair": [a, b], "rank": rank, "rank_320": rank320,
                          "top1": hits[0]["key"], "top1_score": round(hits[0]["score"], 4),
                          "top5": [h["key"] for h in hits[:5]],
                          "rank_list_scores": [round(h["score"], 4) for h in hits[:6]]})
    ranks = [r["rank"] for r in res_pairs if r["rank"]]
    ranks320 = [r["rank_320"] for r in res_pairs if r.get("rank_320")]
    print(f"\n[跨机位检索] 8 对已知配对（318 查询；gallery=其余 5 机位 545 轨 / 仅 320 29 轨）:")
    for r in res_pairs:
        tag = f"rank={r['rank']}" if r["rank"] else "未进 top600"
        t320 = f"320库 rank={r['rank_320']}" if r.get("rank_320") else "320库 未进"
        ok = "✓" if r.get("top1") == f"{DATE}|320|{r['pair'][1]}" else f"(top1={r.get('top1')})"
        print(f"  318:{r['pair'][0]:>3} → 320:{r['pair'][1]:>3}  全库 {tag} | {t320} {ok}")
    if ranks:
        print(f"  → 全库: rank-1 {sum(1 for r in ranks if r==1)}/8 | 平均 rank {np.mean(ranks):.1f}")
    if ranks320:
        print(f"  → 仅320库: rank-1 {sum(1 for r in ranks320 if r==1)}/8 | "
              f"rank≤3 {sum(1 for r in ranks320 if r<=3)}/8 | 平均 rank {np.mean(ranks320):.2f}")

    # ---- 2. 跨时间检索（白衣人链：318:3 上午出现，查 9 分钟后）----
    fam = [3, 42, 49, 50, 58, 70, 72, 74, 77, 81, 83, 85, 86, 91,
           102, 111, 123, 134, 144, 148, 149]
    k3 = f"{DATE}|318|3"
    if k3 in st.rows:
        t0q = st.meta[k3]["t_first"]
        hits = st.search(st.rows[k3], topk=25, cams=["318"], exclude_key=k3)
        hits = [h for h in hits if h["t_first"] is not None and h["t_first"] > t0q + 30]
        fam_hit = [h["key"] for h in hits[:10] if int(h["key"].split("|")[2]) in fam]
        print(f"\n[跨时间检索] 查询 318:3 (t={t0q:.0f}s) → 只看 >{t0q+30:.0f}s 的同机位轨迹:")
        print(f"  top10 里白衣人链成员命中: {len(fam_hit)} 个 {fam_hit}")
        for h in hits[:5]:
            mark = " ★白衣人链" if int(h["key"].split("|")[2]) in fam else ""
            print(f"    {h['key']}  t={h['t_first']:.0f}s  score={h['score']:.3f}{mark}")

    # ---- 3. 时窗 + 机位组合过滤 ----
    print(f"\n[组合过滤] 318:3 跨机位且 t∈[0,120]s 的 top3:")
    for h in st.search(st.rows[k3], topk=3, exclude_cam="318", t_range=(0, 120)):
        print(f"    {h['key']}  t={h['t_first']:.0f}s  score={h['score']:.3f}")

    # ---- 4. 检索延迟 ----
    keys = list(st.rows.keys())[:100]
    t0 = time.perf_counter()
    for k in keys:
        st.search(st.rows[k], topk=10)
    dt = (time.perf_counter() - t0) / len(keys) * 1000
    print(f"\n[延迟] 平均检索 {dt:.1f} ms（库 {s['n']} 轨；暴力余弦）")

    # ---- 5. 规模测算 ----
    per_hour = s["n"] / (10 / 60)          # 10 分钟 → 每小时新增轨道（6 机位）
    est = {
        "bytes_per_track": s["bytes_per_track"],
        "tracks_per_hour_6cam": round(per_hour),
        "mb_per_day_6cam": round(per_hour * 24 * s["bytes_per_track"] / 1e6, 1),
        "gb_per_year_6cam": round(per_hour * 24 * 365 * s["bytes_per_track"] / 1e9, 2),
        "gb_per_year_50cam": round(per_hour * 24 * 365 * (50 / 6) * s["bytes_per_track"] / 1e9, 2),
    }
    print(f"[规模测算] {json.dumps(est, ensure_ascii=False)}")

    return {"model": name, "n": s["n"], "bytes_per_track": s["bytes_per_track"],
            "pairs": res_pairs, "latency_ms": round(dt, 1), "estimate": est,
            "by_cam": s["by_cam"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="d3b")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "output_crosscam", "analysis", "global_reid", "demo_results.json"))
    args = ap.parse_args()
    all_res = []
    for name in args.models.split(","):
        name = name.strip()
        if name not in MODELS:
            print(f"未知模型 {name}")
            continue
        all_res.append(demo(MODELS[name], name))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=1)
    print(f"\n[out] -> {args.out}")


if __name__ == "__main__":
    main()
