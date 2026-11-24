# -*- coding: utf-8 -*-
"""v4 修正评估（d3bridge 备选线）：恢复被误剔的评估对 (98,26)/(53,11)。

配套 cradio 版 tools/v4_corrected_eval.py；嵌入取：
  - dinov3_scale_v4/         （clean 重跑，20 负口径）
  - dinov3_scale_v4_extra/   （4 轨补编码，d3b patched 链路，见 d3b_extra_encode_remote.py）
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from v4_corrected_eval import (  # noqa: E402
    POS30, NEG30, key_to_tid, load_embs, metrics, sim)


def main():
    d4 = os.path.join(ROOT, "output_crosscam", "analysis", "dinov3_scale_v4")
    print(f"[load] d3bridge clean: {d4}")
    e4 = load_embs(d4, pat="{cam}_emb_d3b.npz")

    extra_dir = os.path.join(ROOT, "output_crosscam", "analysis", "dinov3_scale_v4_extra")
    z = np.load(os.path.join(extra_dir, "318_extra4_emb_d3b.npz"))
    extra = {key_to_tid(k): z[k].astype(np.float32) for k in z.files}
    print(f"[load] extra 4轨(d3b): {sorted(extra.keys())}")

    e4c = {c: dict(v) for c, v in e4.items()}
    e4c["318"].update(extra)

    out = {}
    print()
    r_report, _ = metrics(e4, [p for p in NEG30 if p not in ((98, 26), (53, 11))],
                          "d3b v4-report(20负)")
    r_corr, _ = metrics(e4c, NEG30, "d3b v4c-修正(22负)")
    out["d3b_v4_report"] = r_report
    out["d3b_v4_corrected"] = r_corr

    print("\n---- 恢复对 sim（d3b v4c）----")
    for a, b in ((98, 26), (53, 11)):
        s = sim(e4c, "318", a, "320", b)
        print(f"  ({a},{b}): d3b v4c = {s if s is None else round(s, 4)}")
        out[f"pair_{a}_{b}"] = s

    # cradio 对照（若已有修正评估）
    cp = os.path.join(ROOT, "output_crosscam", "analysis", "bridge_scale",
                      "eval_v4_corrected.json")
    if os.path.exists(cp):
        c = json.load(open(cp, encoding="utf-8"))
        print(f"\n[cradio 对照] v4c margin = {c['v4_corrected']['margin']:+.4f}")
        out["cradio_ref"] = c["v4_corrected"]

    outp = os.path.join(ROOT, "output_crosscam", "analysis", "bridge_scale",
                        "eval_v4_corrected_d3b.json")
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {outp}")


if __name__ == "__main__":
    main()
