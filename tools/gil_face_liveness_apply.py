#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L0 执行层：把 liveness_rule_v0 的规则判定落到 per-shot / 池(tid) / 人(gperson) 三级 flag。

与 gil_face_liveness.py 的关系：
- liveness_rule_v0.json = 规则"验证分析"层（静态簇 / 对账单）
- 本工具 = "执行"层：对全量 shots 逐条打 flag → 汇总到池与人 → 产出人审 review 分级

纪律：只读全部上游产物，只写本 JSON；不进 GIL 主链、不改 face_shot/face_best。

人工定性（2026-10-14 目检结论，写入 audit_notes，证据图见 debug_assoc/liveness/）：
- 318 R1 簇（窗框区）：窗玻璃/窗框纹理误检（z8_318_*_f*.jpg）；被 nearest 兜底吸收进 4 个真实轨迹
  （318|1/120/128/139）的个别帧；被吸收帧 q≤0.64 全部非 usable，top1 未受影响。
- 323 R2 簇（门区）：真人背影/后脑勺/低头被检为脸（NPD，非"非人"）；"吸收"进 323|1/2/132 等轨迹。
  323|2 的池内 top1（f560, q=0.562）即后脑勺 → 唯一"最佳脸本身不可用"的已知案例（警戒）。
- 305 R2 簇 ×2（墙面/家具纹理）：纹理误检被吸收进 13 个真实轨迹的个别帧；全部非 usable。
- 倒影 4 个 tid（316|32、323|91、323|122、323|80）：top1 即倒影帧；p0c 已从 gperson 层移出
  （gp=None），但池内条目仍保留 → 标注 isolated，池消费端应跳过。

用法：python tools/gil_face_liveness_apply.py --tag p0c --rule-tag v0 --out-tag v0
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DIR = os.path.join(ROOT, "output_crosscam", "analysis", "face")

# ---- 人工定性注记（目检结论 + 证据引用）----
AUDIT_NOTES = {
    "316|32": dict(tier="isolated", text="倒影 tid（玻璃墙反射的真人被检为脸）：top1 即倒影帧 q=0.964；"
                                          "p0c 重链接已将其移出 gperson 层（gp=None），池消费应跳过。证据：crops/tid_316_32_f4760_q0.96.jpg"),
    "323|91": dict(tier="isolated", text="倒影 tid：全 46 帧命中 obs 倒影标注，top1 即倒影帧；已移出人层。"),
    "323|122": dict(tier="isolated", text="倒影 tid：全 23 帧命中倒影标注；已移出人层。"),
    "323|80": dict(tier="isolated", text="倒影 tid（1 帧）：非 usable，已移出人层。"),
    "323|2": dict(tier="warn", text="★ 池内 top1（f560, q=0.562, usable）是后脑勺（NPD 检错过质量门）；"
                                    "142/197 帧为背影/后脑勺低质检出。L1 MiniFAS spoof=0.026（未检出），"
                                    "L0 zone 未命中 top1 → 仅人审可判。建议：人审后剔除或降权该池条目。"
                                    "证据：crops/tid_323_2_f560_q0.56.jpg"),
    "318|1": dict(tier="hint", text="轨迹含 3 帧窗框纹理误检（q≤0.61 非 usable，被吸收）；top1（f40, q=0.765）真人在别位。"),
    "318|120": dict(tier="hint", text="轨迹含 1 帧窗框误检（非 usable）；top1 干净。"),
    "318|128": dict(tier="hint", text="轨迹含 4 帧窗框区检出（q≤0.64 非 usable，含窗边局部模糊）；top1（q=0.892）干净。"),
    "318|139": dict(tier="hint", text="轨迹含 2 帧窗框纹理误检（非 usable）；top1（q=0.875）干净。"),
    "323|1": dict(tier="hint", text="轨迹含 75 帧门区背影/后脑勺低质检出（均非 usable）；top1（f3460, q=0.719，低头侧脸）可用。"),
    "323|132": dict(tier="hint", text="轨迹含 12 帧门区背影低质检出（非 usable）；top1（q=0.737）可用。"),
    "305|40": dict(tier="hint", text="轨迹含 3 帧墙面纹理误检吸收（非 usable）；top1（q=0.708，侧脸）可用。"),
    "305|53": dict(tier="hint", text="轨迹含 1 帧纹理误检吸收（非 usable）；top1（q=0.762，走动模糊）可用。"),
    "305|57": dict(tier="ignore", text="1 帧轨迹且全部 static_r2，非 usable——纯误检短轨迹。"),
    "305|71": dict(tier="ignore", text="17/18 帧 static_r2（含 top1），非 usable——误检轨迹。"),
    "305|76": dict(tier="ignore", text="1/2 帧 static_r2，非 usable。"),
    "318|52": dict(tier="ignore", text="2/4 帧 static_r1（含 top1），非 usable。"),
}
TIER_ORDER = {"warn": 0, "hint": 1, "isolated": 2, "ignore": 3}


def load_json(p, required=True):
    if not os.path.exists(p):
        if required:
            raise SystemExit(f"missing: {p}")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_zones(rule):
    """从 liveness_rule_v0 的静态簇 members（40px bin）重建 zone 表"""
    zones = []
    for rec in rule["static_clusters"]:
        code = "static_r1" if rec["rule"].startswith("R1") else "static_r2"
        for m in rec["members"]:
            zones.append((rec["cam"], m[0] * 40, m[1] * 40, (m[0] + 1) * 40, (m[1] + 1) * 40, code))
    return zones


def in_zone(zones, cam, cx, cy):
    for c, x1, y1, x2, y2, code in zones:
        if c == cam and x1 <= cx < x2 and y1 <= cy < y2:
            return code
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p0c")
    ap.add_argument("--rule-tag", default="v0")
    ap.add_argument("--out-tag", default="v0")
    args = ap.parse_args()

    shots = load_json(os.path.join(FACE_DIR, f"face_shot_{args.tag}.json"))["shots"]
    best = load_json(os.path.join(FACE_DIR, f"face_best_{args.tag}.json"))
    rule = load_json(os.path.join(FACE_DIR, f"liveness_rule_{args.rule_tag}.json"))
    zones = build_zones(rule)
    refl_obs = set(r["obs_id"] for r in rule["reflection_linked"]["items"])
    print(f"[apply] shots={len(shots)} zones={len(zones)} refl_obs={len(refl_obs)}", flush=True)

    # ---- per-shot flag ----
    per_shot = []
    by_tid = defaultdict(list)
    flag_shots_counter = Counter()
    gap_by_cam = Counter()
    for s in shots:
        cx = (s["face"][0] + s["face"][2]) / 2
        cy = (s["face"][1] + s["face"][3]) / 2
        fl = []
        z = in_zone(zones, s["cam"], cx, cy)
        if z:
            fl.append(z)
        if s.get("obs_id") is not None and int(s["obs_id"]) in refl_obs:
            fl.append("reflection")
        if s.get("usable") and s.get("tid") is None:
            fl.append("no_track")
            gap_by_cam[s["cam"]] += 1
        if s.get("tid") is not None:
            by_tid[f"{s['cam']}|{s['tid']}"].append((s, fl))
        if fl:
            for code in fl:
                flag_shots_counter[code] += 1
            per_shot.append(dict(cam=s["cam"], frame=s["frame"], tid=s.get("tid"),
                                 flags=fl, q=round(s["q"], 3), usable=bool(s.get("usable")),
                                 face=[round(float(v), 1) for v in s["face"]]))

    # ---- tid 级汇总 ----
    tid_flags = {}
    for key, lst in by_tid.items():
        flagged = [(s, fl) for s, fl in lst if fl]
        if not flagged:
            continue
        top1 = max(lst, key=lambda t: (t[0]["usable"], t[0]["q"]))[0]
        top1_fl = in_zone(zones, top1["cam"],
                          (top1["face"][0] + top1["face"][2]) / 2,
                          (top1["face"][1] + top1["face"][3]) / 2)
        top1_flag = []
        if top1_fl:
            top1_flag.append(top1_fl)
        if top1.get("obs_id") is not None and int(top1["obs_id"]) in refl_obs:
            top1_flag.append("reflection")
        any_flags = sorted(set(c for _, fl in flagged for c in fl))
        audit = AUDIT_NOTES.get(key)
        if audit is None:
            # 自动分级（无人工注记时）
            if "reflection" in any_flags and top1.get("gperson_id") is None:
                tier = "isolated"
            elif top1_flag and top1.get("usable"):
                tier = "warn"
            elif top1.get("usable"):
                tier = "hint"
            else:
                tier = "ignore"
            audit = dict(tier=tier, text="（自动分级，未目检）")
        # L1 交叉引用（若有）
        tid_flags[key] = dict(
            flags=any_flags,
            n_shots=len(lst), n_flagged=len(flagged),
            top1=dict(frame=top1["frame"], q=round(top1["q"], 3),
                      usable=bool(top1.get("usable")), flags=top1_flag),
            gperson_id=top1.get("gperson_id"), gperson_label=top1.get("gperson_label"),
            tier=audit["tier"], note=audit["text"])

    # ---- gperson 级汇总 ----
    gp_flags = defaultdict(lambda: dict(tids=[], tiers=[]))
    for key, rec in tid_flags.items():
        gp = rec.get("gperson_id")
        if gp is None:
            continue
        gp_flags[int(gp)]["tids"].append(key)
        gp_flags[int(gp)]["tiers"].append(rec["tier"])
    gp_out = {}
    for gp, r in gp_flags.items():
        tiers = sorted(set(r["tiers"]), key=lambda t: TIER_ORDER.get(t, 9))
        gp_out[str(gp)] = dict(tids=sorted(r["tids"]), tier=tiers[0],
                               note="；".join(tid_flags[t]["note"][:60] for t in sorted(r["tids"])))

    # ---- review 分级清单 ----
    def by_tier(t):
        return [k for k, r in tid_flags.items() if r["tier"] == t]
    review = dict(
        warn=sorted(by_tier("warn")),          # 需人审确认（含 323|2）
        hint=sorted(by_tier("hint")),          # 提示（top1 干净，轨迹含低质帧）
        isolated=sorted(by_tier("isolated")),  # 倒影（已移出人层）
        ignore=sorted(by_tier("ignore")),      # 低质误检轨迹（非 usable）
    )

    out = dict(
        meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), tag=args.tag,
                  source=f"liveness_rule_{args.rule_tag}.json",
                  note="L0 执行层：三级 liveness flag（只读标注）。"
                       "tier: warn=需人审(hint=提示/isolated=倒影已移出/ignore=低质误检轨迹)。"
                       "人工定性见 audit_notes（2026-10-14 目检，证据图 debug_assoc/liveness/）"),
        summary=dict(
            n_shots=len(shots), n_flagged=len(per_shot),
            by_rule=dict(flag_shots_counter),
            no_track_by_cam=dict(gap_by_cam),
            n_tids_flagged=len(tid_flags),
            n_tids_usable_pool=sum(1 for r in tid_flags.values() if r["top1"]["usable"]
                                   and r["gperson_id"] is not None),
            n_gperson_flagged=len(gp_out),
            tiers=dict((t, len(v)) for t, v in review.items())),
        review=review,
        tid_flags=tid_flags,
        gperson_flags=gp_out,
        per_shot=per_shot)
    out_path = os.path.join(FACE_DIR, f"face_live_flags_{args.out_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[apply] flagged tids={len(tid_flags)} gperson={len(gp_out)} tiers={out['summary']['tiers']}", flush=True)
    print(f"[apply] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
