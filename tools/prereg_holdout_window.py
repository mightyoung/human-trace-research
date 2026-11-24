"""B2：按确定性规则计算"同日留出"可用时段（v2，2026-10-24 修订）。

输入：docs/material-audit-2026-10-20.json（每卷起始 OSD、时长、已用片段）
输出：候选的「60 min 图库 + 15 min 查询」时段对，以及各机位空闲片段清单。

规则（写进预注册，不可事后改）：
  R1 只用 1012 同日素材，不引入 multi-cemra / MARS / CHIRLA。
  R2 一个窗口必须四机位（305/318/320/323）都有素材覆盖。
  R3 与现有 12 段（4 train + 4 val + 4 test）任意一段重叠即排除，容差 0 秒。
  R4 窗口内卷间缺口 ≤60 s 视为连续（实测 480→540 为 41 s）；反向重叠同样视为连续
     （540→543 实测重叠 4 s，见 docs/holdout-seam-check-2026-10-24.md），>60 s 即断开。
  R5 图库窗口固定 60 min，与主实验一致（主实验：每机位 60 min 图库 + 15 min 查询）。
  R6 查询窗口固定 15 min，与主实验一致；且须在图库结束 ≥60 min 之后开始。
  R7 候选按「跨卷次数」升序、再按「开始时刻」升序，取第一个。

v1→v2 修订理由：v1 取 15 min 图库，与主实验（60 min）不一致，且可答查询很可能
不足 100 条而触发「无法判定」。v2 把图库长度对齐主实验。

运行：python tools/prereg_holdout_window.py
"""
import io
import json

AUDIT = "docs/material-audit-2026-10-20.json"
CAMS = ["305", "318", "320", "323"]
GALLERY_S = 60 * 60
QUERY_S = 15 * 60
MIN_GAP_S = 60 * 60
MAX_HOLE_S = 60


def tsec(s):
    s = str(s).strip()
    if "?" in s or ":" not in s:
        return None  # 起始时刻未验证的卷不参与，不猜
    p = [int(x) for x in s.split(":")]
    while len(p) < 3:
        p.append(0)
    return p[0] * 3600 + p[1] * 60 + p[2]


def fmt(s):
    s = int(round(s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def vid(vol):
    """卷名形如 0100000251400000000.mp4；第 8-10 位是文档沿用的三位卷号（514/529/...）"""
    stem = vol.split(".")[0]
    return stem[8:11]


def load():
    d = json.load(io.open(AUDIT, encoding="utf-8"))
    best = {}
    for v in d["volumes_1012"]:
        key = (v["cam"], v["vol"])
        if key not in best or v["dur_s"] > best[key]["dur_s"]:
            best[key] = v
    vols = {c: [] for c in CAMS}
    for (cam, _), v in best.items():
        t0 = tsec(v.get("osd_readme"))
        if t0 is None or cam not in vols:
            continue
        used = [(t0 + u["ss"], t0 + u["ss"] + u["dur"]) for u in v.get("used", [])]
        vols[cam].append(dict(vol=v["vol"], t0=t0, t1=t0 + v["dur_s"], used=used))
    for cam in vols:
        vols[cam].sort(key=lambda x: x["t0"])
    return vols


def free_slices(vols, cam):
    """按卷切出未被占用的片段 [(a, b, vol)]；每段必落在单卷内。"""
    out = []
    for v in vols[cam]:
        segs = [(v["t0"], v["t1"])]
        for ua, ub in v["used"]:
            nxt = []
            for a, b in segs:
                if ub <= a or ua >= b:
                    nxt.append((a, b))
                    continue
                if ua > a:
                    nxt.append((a, min(ua, b)))
                if ub < b:
                    nxt.append((max(ub, a), b))
            segs = nxt
        out += [(a, b, v["vol"]) for a, b in segs if b - a > 60]
    return sorted(out)


def build_common(vols):
    """四机位共同覆盖，按换卷点切开。返回 [[a, b, {cam: vol}]]。"""
    events = set()
    for cam in CAMS:
        for a, b, _ in free_slices(vols, cam):
            events |= {a, b}
    pts = sorted(events)
    pieces = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        mid = (a + b) / 2
        vols_at, ok = {}, True
        for cam in CAMS:
            hit = [(x, y, v) for x, y, v in free_slices(vols, cam) if x <= mid < y]
            if not hit:
                ok = False
                break
            vols_at[cam] = hit[0][2]
        if ok:
            pieces.append([a, b, vols_at])
    merged = []
    for a, b, vs in pieces:
        if merged and merged[-1][1] == a and merged[-1][2] == vs:
            merged[-1][1] = b
        else:
            merged.append([a, b, vs])
    return merged


def window_stats(merged, start, dur):
    end = start + dur
    inside = [p for p in merged if p[0] < end and p[1] > start]
    if not inside:
        return None
    covered = sum(min(p[1], end) - max(p[0], start) for p in inside)
    hole = 0
    for i in range(len(inside) - 1):
        hole = max(hole, inside[i + 1][0] - inside[i][1])
    if hole > MAX_HOLE_S:
        return None
    if covered < dur - hole - 1:
        return None
    vols_at = {}
    for p in inside:
        for cam, v in p[2].items():
            vols_at.setdefault(cam, []).append(v)
    return {"seams": len(inside) - 1, "hole": hole, "vols": vols_at}


def main():
    vols = load()

    print("== 各机位空闲片段（已扣已用段，每段落在单卷内） ==")
    for cam in CAMS:
        s = free_slices(vols, cam)
        print(f"{cam}: " + " | ".join(f"{fmt(a)}-{fmt(b)}({vid(v)},{int((b-a)//60)}m)" for a, b, v in s))

    merged = build_common(vols)
    print("\n== 四机位共同空闲（按换卷点切开） ==")
    for a, b, vs in merged:
        if b - a >= QUERY_S:
            tag = " ".join(f"{c}={vid(vs[c])}" for c in CAMS)
            print(f"{fmt(a)}-{fmt(b)}  {(b-a)/60:.1f}m  {tag}")

    gal = []
    lo = min(p[0] for p in merged)
    hi = max(p[1] for p in merged)
    t = lo
    while t + GALLERY_S <= hi:
        st = window_stats(merged, t, GALLERY_S)
        if st:
            gal.append((st["seams"], st["hole"], t, st))
        t += 60

    print(f"\n== 60 min 图库候选（{len(gal)} 个，按跨卷数/缺口/时刻排序） ==")
    for seams, hole, t, st in sorted(gal)[:10]:
        print(f"{fmt(t)}-{fmt(t + GALLERY_S)}  跨卷 {seams} 次  缺口 {hole:.0f}s")

    if not gal:
        print("无可用图库窗口：四机位没有任何 60 min 连续空闲段。")
        return

    _, _, g_start, g_st = sorted(gal)[0]

    qc = []
    t = g_start + GALLERY_S + MIN_GAP_S
    while t + QUERY_S <= hi:
        st = window_stats(merged, t, QUERY_S)
        if st:
            qc.append((st["seams"], st["hole"], t, st))
        t += 60

    print(f"\n== 15 min 查询候选（图库结束 ≥{MIN_GAP_S // 60} min 后，{len(qc)} 个） ==")
    for seams, hole, t, st in sorted(qc)[:10]:
        print(f"{fmt(t)}-{fmt(t + QUERY_S)}  跨卷 {seams} 次  缺口 {hole:.0f}s")

    if not qc:
        print("无可用查询窗口。")
        return

    _, _, q_start, _ = sorted(qc)[0]
    print("\n== 选定（R7） ==")
    print(f"图库 {fmt(g_start)}-{fmt(g_start + GALLERY_S)}（{GALLERY_S // 60} min，跨卷 {g_st['seams']} 次，缺口 {g_st['hole']:.0f}s）")
    for cam in CAMS:
        print(f"    {cam}: " + " → ".join(vid(v) for v in g_st["vols"][cam]))
    print(f"查询 {fmt(q_start)}-{fmt(q_start + QUERY_S)}（{QUERY_S // 60} min）")
    print(f"间隔 {(q_start - g_start - GALLERY_S) // 60} min")


if __name__ == "__main__":
    main()
