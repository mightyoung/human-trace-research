# -*- coding: utf-8 -*-
"""v4 修正评估（cradio）：恢复被误剔的评估对 —— (98,26)、(53,11)。

背景（2026-10-13 晚复核）：318 的 4 条运动学"倒影"标记（53/82/98/135）经目视+
跨时刻复核判为**假阳性**（真实人员；静坐/行走对），318 无可靠地板倒影证据。
→ v4 据此剔除的 NEG30 两对 (98,26)、(53,11) 应恢复评估。

做法：4 轨（53/82/98/135）用 v4 W 补编码（embed_server，inference-out-extra/318_extra4_emb.npz），
并入 cradio_scale_v4 的 318 嵌入集，重算协议 A：
  - v4c（修正，22 负）vs v4-report（剔除版，20 负）vs v3 raw（22 负，W 为 v3）

用法：python tools/v4_corrected_eval.py（extra npz 已下载到 output_crosscam/analysis/cradio_scale_v4_extra/）
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ["318", "320", "305", "316", "323", "343"]
POS30 = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
NEG30 = [(14, 1), (131, 21), (61, 14), (50, 11), (132, 21), (3, 1), (7, 1), (146, 21),
         (127, 19), (123, 19), (57, 11), (131, 23), (124, 18), (131, 24), (3, 4),
         (126, 18), (43, 11), (10, 1), (131, 28), (98, 26), (142, 26), (53, 11)]


def key_to_tid(k):
    num = ""
    for ch in reversed(k):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def load_embs(d, pat="{cam}_emb_scale.npz"):
    out = {}
    for cam in CAMS:
        p = os.path.join(d, pat.format(cam=cam))
        if not os.path.exists(p):
            continue
        z = np.load(p)
        out[cam] = {key_to_tid(k): z[k].astype(np.float32) for k in z.files}
    return out


def sim(embs, c1, t1, c2, t2):
    if c1 not in embs or c2 not in embs or t1 not in embs[c1] or t2 not in embs[c2]:
        return None
    a, b = embs[c1][t1], embs[c2][t2]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def metrics(embs, negs, tag):
    pv = [sim(embs, "318", a, "320", b) for a, b in POS30]
    pv = [x for x in pv if x is not None]
    nv, rows = [], []
    for a, b in negs:
        s = sim(embs, "318", a, "320", b)
        if s is not None:
            nv.append(s)
        rows.append((a, b, s))
    auc = float(np.mean([(p > n) + 0.5 * (p == n) for p in pv for n in nv]))
    r = {"tag": tag, "n_pos": len(pv), "n_neg": len(nv),
         "auc": round(auc, 4), "pos_min": round(min(pv), 4),
         "neg_max": round(max(nv), 4), "margin": round(min(pv) - max(nv), 4)}
    print(f"[{tag}] 协议A: AUC={r['auc']} margin={r['margin']:+.4f} "
          f"(pos {r['n_pos']} / neg {r['n_neg']}) | pos_min={r['pos_min']} neg_max={r['neg_max']}")
    return r, rows


def main():
    print("[load] cradio_scale_v4 ...")
    e4 = load_embs(os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale_v4"))
    # 补编码 4 轨
    extra_dir = os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale_v4_extra")
    z = np.load(os.path.join(extra_dir, "318_extra4_emb.npz"))
    extra = {key_to_tid(k): z[k].astype(np.float32) for k in z.files}
    print(f"[load] extra 4轨: {sorted(extra.keys())}")

    e4c = {c: dict(v) for c, v in e4.items()}
    e4c["318"].update(extra)

    print("\n[load] cradio_scale_v3 (raw 对照) ...")
    try:
        e3 = load_embs(os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale_v3"))
        have3 = True
    except Exception as ex:
        print("  v3 加载失败:", ex)
        have3 = False

    out = {}
    print()
    r_report, _ = metrics(e4, [p for p in NEG30 if p not in ((98, 26), (53, 11))], "v4-report(20负)")
    r_corr, rows_corr = metrics(e4c, NEG30, "v4c-修正(22负)")
    out["v4_report"] = r_report
    out["v4_corrected"] = r_corr
    if have3:
        r3, _ = metrics(e3, NEG30, "v3-raw(22负)")
        out["v3_raw"] = r3

    # 恢复对的具体数值
    print("\n---- 恢复对 sim（v4c 用 v4 W）----")
    for a, b in ((98, 26), (53, 11)):
        s = sim(e4c, "318", a, "320", b)
        s_report = sim(e4, "318", a, "320", b)
        print(f"  ({a},{b}): v4c={s if s is None else round(s,4)} | v4-report={s_report}")
        out[f"pair_{a}_{b}"] = {"v4c": s, "v4_report": s_report}

    outp = os.path.join(ROOT, "output_crosscam", "analysis", "bridge_scale",
                        "eval_v4_corrected.json")
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {outp}")


if __name__ == "__main__":
    main()
