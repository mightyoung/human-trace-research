# -*- coding: utf-8 -*-
"""构建「真值标注」页数据包：webapp/backend/data/truth/

输入（项目内，真值工作目录）：
  output_crosscam/analysis/truth/_review/pair_machine.csv   （241 对 + 机器外观 cos 建议）
  output_crosscam/analysis/truth/pair_candidates_labeled.csv（已有判定，可缺）
  output_crosscam/analysis/truth/events_all.csv              （107 事件）
  output_crosscam/analysis/truth/thumbs/<event_id>.jpg

输出：
  backend/data/truth/state.json      （pairs / events / meta / select）
  backend/data/truth/thumbs/*.jpg    （缩放到最大宽 220，q=80）

预测式选择（Top-K + audit，借鉴 RSI-Anything 的"同预算下比效率"）：
  pred = cos_appear + λ·ln w_dt(dt)      —— 复用 relay v2 联合分形式（λ=0.01）
  w_dt  = Parzen 过渡时间先验，取自 analysis/relay_v2/dist_v1.json
          （其 fit 集**已剔除全部已人工判定行** → 选择器与人工真值独立，防自证）
  role  : topk(前 K，逐事件度数≤2 以暴露互斥冲突) / audit(随机审计，无偏估计其余分层) / rest
  ★ 审计样本从「未预先判定」的剩余对中随机抽（seed 固定），保证无偏。

注意：**不写 labels.json**（用户判定由后端单独保存在 labels.json，重建数据不覆盖）。
"""
import csv, json, math, os, random, sys
from pathlib import Path

try:
    from PIL import Image
except Exception:
    Image = None

ROOT = Path(__file__).resolve().parent.parent.parent          # 项目根
SRC = ROOT / "output_crosscam" / "analysis" / "truth"
OUT = ROOT / "webapp" / "backend" / "data" / "truth"
TH = OUT / "thumbs"
OUT.mkdir(parents=True, exist_ok=True)
TH.mkdir(parents=True, exist_ok=True)


def rd(p):
    if not p.exists():
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


# ---------- 事件 ----------
seen = {}
for r in rd(SRC / "events_all.csv"):
    seen.setdefault(r["event_id"], r)
events = []
for eid, r in sorted(seen.items(), key=lambda kv: (kv[1]["cam"], f(kv[1]["t"]))):
    events.append({
        "eid": eid, "short": eid.split("_E_")[1], "cam": r["cam"], "kind": r["kind"],
        "edge": r["edge"], "t": f(r["t"]), "u": f(r["u"]), "v": f(r["v"]),
        "n_obs": int(f(r["n_obs"])), "dur_s": f(r["dur_s"]), "score": f(r["score"]),
        "entity": r["entity"],
    })

# ---------- 缩略图 ----------
n_thumb = 0
for e in events:
    s = SRC / "thumbs" / (e["eid"] + ".jpg")
    d = TH / (e["eid"] + ".jpg")
    if not s.exists():
        continue
    if Image is None:
        d.write_bytes(s.read_bytes())
    else:
        im = Image.open(s).convert("RGB")
        if im.width > 220:
            im = im.resize((220, max(1, round(im.height * 220 / im.width))), Image.LANCZOS)
        im.save(d, "JPEG", quality=80)
    n_thumb += 1

# ---------- 配对 ----------
# relay v2 状态（另一路产物，仅作展示/对照，不参与选择）
v2 = {r["pair_id"]: (r.get("in_relay_v2") or "").strip()
      for r in rd(SRC / "pair_candidates.csv")}
n_v2 = sum(1 for x in v2.values() if x)

pairs = []
for r in rd(SRC / "_review" / "pair_machine.csv"):
    a, b = r["event_a"], r["event_b"]
    pairs.append({
        "pair_id": r["pair_id"], "a": a, "b": b,
        "sa": a.split("_E_")[1], "sb": b.split("_E_")[1],
        "ta": f(r["t_a"]), "tb": f(r["t_b"]), "dt": f(r["dt"]),
        "relay": int(f(r["in_relay_v5"])), "cos": f(r["cos_appear"], -9.0),
        "machine": r["machine"], "relay_v2": v2.get(r["pair_id"], ""),
    })

# ---------- 预测式选择：联合分 pred = cos + λ·ln w_dt ----------
LAM = 0.01
TOPK_K = 30
AUDIT_K = 20
SEED = 42
DIST = ROOT / "output_crosscam" / "analysis" / "relay_v2" / "dist_v1.json"


def _w_dt_fn():
    """Parzen 过渡时间密度（0.5s 网格线性插值）；fit 集已剔除全部已判行。"""
    try:
        d = json.loads(DIST.read_text(encoding="utf-8"))
        gx, gf = d["grid_x"], d["grid_f"]
    except Exception as e:  # noqa: BLE001
        print("[warn] dist_v1 不可用，pred 退化为 cos：", e)
        return None, None

    def f(dt):
        x = min(max(dt, gx[0]), gx[-1])
        i = min(int((x - gx[0]) / (gx[1] - gx[0])), len(gx) - 2)
        t = (x - gx[i]) / (gx[i + 1] - gx[i])
        return max(gf[i] + (gf[i + 1] - gf[i]) * t, 1e-6)
    return f, d


wf, dist_meta = _w_dt_fn()

# 预先已判定的 pair（用于把审计抽样限定在「未预先判定」的子集）
pre = {r["pair_id"] for r in rd(SRC / "pair_candidates_labeled.csv")
       if (r.get("human_same_person") or "").strip() in ("1", "0", "?")}

for p in pairs:
    w = wf(p["dt"]) if wf else 1.0
    p["w_dt"] = round(w, 6)

# cos 缺失（实体嵌入不在全局库）→ 用全库 cos 的 25 分位作**回退分**，
# 而不是赋 -9 让它垫底（否则 relay 支持的真配对会被埋没，见 TRUTH-FINDINGS §5.1）
avail = sorted(p["cos"] for p in pairs if p["cos"] > -8)
FALLBACK_COS = avail[len(avail) // 4] if avail else 0.0
n_missing = 0
for p in pairs:
    if p["cos"] > -8:
        p["cos_missing"] = False
        p["pred"] = round(p["cos"] + LAM * math.log(p["w_dt"]), 6)
    else:
        p["cos_missing"] = True
        n_missing += 1
        p["pred"] = round(FALLBACK_COS + LAM * math.log(p["w_dt"]), 6)

# 排名（pred 降序；同分按 dt 升序再按 pair_id 稳定）
for i, p in enumerate(sorted(pairs, key=lambda q: (-q["pred"], q["dt"], q["pair_id"])), 1):
    p["rank"] = i

# Top-K：按名次贪心，逐事件度数 ≤2（既保覆盖，又暴露互斥冲突）
deg, topk = {}, []
for p in sorted(pairs, key=lambda q: q["rank"]):
    if deg.get(p["a"], 0) >= 2 or deg.get(p["b"], 0) >= 2:
        continue
    topk.append(p["pair_id"])
    deg[p["a"]] = deg.get(p["a"], 0) + 1
    deg[p["b"]] = deg.get(p["b"], 0) + 1
    if len(topk) >= TOPK_K:
        break

# 审计：从「剩余且未预先判定」随机抽（seed 固定，无偏）
topk_set = set(topk)
rest_pool = [p["pair_id"] for p in pairs if p["pair_id"] not in topk_set
             and p["pair_id"] not in pre]
rng = random.Random(SEED)
audit = sorted(rng.sample(rest_pool, min(AUDIT_K, len(rest_pool))))
audit_set = set(audit)
for p in pairs:
    p["role"] = ("topk" if p["pair_id"] in topk_set
                 else "audit" if p["pair_id"] in audit_set else "rest")

state = {
    "meta": {
        "title": "短窗出入口事件真值（343 × 316 · synced-10min 08:55–09:05）",
        "cam_a": "343", "cam_b": "316",
        "n_pairs": len(pairs), "n_events": len(events), "n_thumb": n_thumb,
        "note": "343 侧缩略图偏暗（电梯间）；pred/cos 仅为排序建议，不是真值。"
                "判为「同人」时须满足互斥：一个 343 事件至多配一个 316 事件（反向亦然）。",
        "rules": "1=同人 0=不是 2=看不清；置信 high/mid/low；备注自由。",
        "select": {
            "mode": "topk+audit", "k_topk": len(topk), "k_audit": len(audit),
            "budget": len(topk) + len(audit), "seed": SEED, "lam": LAM,
            "pred": "cos_appear + λ·ln w_dt(dt)",
            "w_dt_source": "relay_v2/dist_v1.json（Parzen；fit 集剔除全部已判行 → 选择器与真值独立）",
            "cos_fallback_p25": round(FALLBACK_COS, 6), "n_cos_missing": n_missing,
            "audit_pool_excludes_prejudged": len(pre),
            "note": "同预算(50)对照：Top-K+审计 vs 纯随机50；审计分层用于无偏估计其余层同人率。",
        },
    },
    "pairs": pairs,
    "events": events,
}
(OUT / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

# 已有判定（供页面初值；后端另有 labels.json 为准）
seed = {}
for r in rd(SRC / "pair_candidates_labeled.csv"):
    v = (r.get("human_same_person") or "").strip()
    if v in ("1", "0", "?"):
        seed[r["pair_id"]] = {"v": 1 if v == "1" else (0 if v == "0" else 2),
                              "c": r.get("human_confidence", "") or "",
                              "n": r.get("human_note", "") or ""}
(OUT / "seed_labels.json").write_text(json.dumps(seed, ensure_ascii=False), encoding="utf-8")

print("pairs %d  events %d  thumbs %d  seed %d" % (len(pairs), len(events), n_thumb, len(seed)))
print("select: topk %d / audit %d / rest %d (budget %d, seed %d)"
      % (len(topk), len(audit), len(pairs) - len(topk) - len(audit), len(topk) + len(audit), SEED))
print("cos 缺失 %d 条 → 回退分 %.4f (全库 cos 25 分位)" % (n_missing, FALLBACK_COS))
print("relay v2 非空 %d 条（active %d / audit %d）"
      % (n_v2, sum(1 for x in v2.values() if x == "active"),
         sum(1 for x in v2.values() if x == "audit")))
print("->", OUT / "state.json", round((OUT / "state.json").stat().st_size / 1024), "KB")
