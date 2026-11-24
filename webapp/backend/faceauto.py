# -*- coding: utf-8 -*-
"""自动人脸：把「人工确认的人脸卡」升级成「可自动触发的身份通道」。

这个模块要回答的问题
--------------------
选择器原来的池只有两个来源：人工点选、45 张静态人脸卡。前者要人坐下来点，后者不会
自己长大。本模块补第三步：**新出现的轨自动找人脸、自动判身份、自动写池**，人只做复核。

★★★ 三条口径（不遵守就会得出完全错误的结论）
  - **外观在这批轨上根本没有模板**：46 条有脸轨里 **42 条**所属实体是单轨实体，留一后
    候选为空。也就是说对这些轨，外观通道**连问题都问不出来**，人脸是唯一能给身份的
    通道。这不是「人脸比外观准」，而是「外观在这里无话可说」。
  - **人脸并不比外观强**：同一批 46 轨上人脸 AUC **0.777**，外观 AUC **0.942**。
    这两个数必须一起报，否则会被读成「上了人脸就变准了」。
  - **不能拿 45 张卡当图库来评估**：卡图是未对齐紧裁，23/45 的 `img_best` 是**融合模板**
    （`ft_*.jpg`，模糊平均），SCRFD 在 0.30 阈值下检不出；硬用紧裁模板兜底会把跨身份
    冒烟分布抬到 max 0.71、p99 0.45。更关键的是 44/45 张卡就来自本包的轨 —— 拿卡图
    建库再查这些轨，等于拿原图找原图。故匹配用**视频侧模板 + 留一**，卡图只做展示自检。

真值来源
--------
`output_crosscam/analysis/gil/store_v1/gil.sqlite`：`obs.key` 与本库 `key` 同源
（`2026-10-12|cam|tid`）。身份要**经 obs 层** join（`membership.node_kind` 只有
'obs' 和 'refl'，直接 join entity 会得到全 None）。GIL 的 gperson 全部是
`state='candidate'`，没有人工确认状态 —— 评估必须写明这是候选身份图。
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
REID = DATA / "reid"
FACE_JSON = DATA.parents[1] / "frontend" / "public" / "faces" / "face_data.json"
QUERY = REID / "face_query.json"
SUGGEST = REID / "face_suggest.json"
CARDS = REID / "face_card_audit.json"
GIL_DB = ROOT / "output_crosscam" / "analysis" / "gil" / "store_v1" / "gil.sqlite"
SCRFD = ROOT / "models" / "face" / "scrfd_2.5g_bnkps.onnx"
EMB = {"mbf": ROOT / "models" / "face" / "w600k_mbf.onnx",
       "r50": ROOT / "models" / "face" / "w600k_r50.onnx"}
N_STORE = 539

TPL112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                   [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

# 模型选 mbf：消融 r50 只 +0.015（在噪声内），13 MB vs 174 MB。
MODEL = "mbf"
DET_THR = 0.30
TAU = 0.30          # 排队线：进入人工复核队列
# ★ 自动写池默认**关闭**。实测在 τ=0.45 时接受 21 条、对 8 条（精度 0.381），
#   τ=0.50 时 9 条对 6 条（0.667），τ=0.55 又掉到 0.40 —— 非单调且 n 太小，
#   没有任何阈值足以直接写进池。要开必须显式传 tau_auto，并自己承担污染模板的后果。
TAU_AUTO = None


def _l2(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


def align112(img, kps5):
    M, _ = cv2.estimateAffinePartial2D(np.asarray(kps5, np.float32), TPL112,
                                       method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


_EMB_SESS = {}


def _sess(key=MODEL):
    if key not in _EMB_SESS:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(4, (os.cpu_count() or 8) // 2)
        s = ort.InferenceSession(str(EMB[key]), sess_options=so,
                                 providers=["CPUExecutionProvider"])
        _EMB_SESS[key] = (s, s.get_inputs()[0].name)
    return _EMB_SESS[key]


def embed(aligned112, key=MODEL):
    s, iname = _sess(key)
    if aligned112.shape[0] != 112:
        aligned112 = cv2.resize(aligned112, (112, 112))
    b = ((aligned112[:, :, ::-1].astype(np.float32) - 127.5) / 127.5)
    v = s.run(None, {iname: b.transpose(2, 0, 1)[None]})[0][0].astype(np.float64)
    return _l2(v)


def _detector():
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from scrfd_dnn import ScrfdDNN
    return ScrfdDNN(str(SCRFD), thresh=DET_THR, nms=0.4)


# ---------------------------------------------------------------- 真值
def gil_truth():
    """key -> {entity, gperson, label, state}。"""
    out = {}
    if not GIL_DB.exists():
        return out, "gil.sqlite 不存在"
    con = sqlite3.connect(str(GIL_DB))
    q = """
        select o.key, e.eid, m.gperson_id, g.label, g.state
        from obs o
        left join entity e on e.id = o.entity_id
        left join membership m on m.node_kind='obs' and m.node_id=o.id
        left join gperson g on g.id = m.gperson_id
    """
    for key, eid, gid, lab, st in con.execute(q):
        out[key] = dict(entity=eid, gperson=gid, label=lab, state=st)
    con.close()
    return out, "ok"


# ---------------------------------------------------------------- 模板（视频侧）
def templates():
    """视频侧人脸模板：每条有脸轨一条，身份取 GIL。"""
    if not QUERY.exists():
        return [], dict(ok=False, reason="尚未采集：先跑 tools/face_query_scan.py")
    q = json.loads(QUERY.read_text(encoding="utf-8"))
    truth, _ = gil_truth()
    idx = json.loads((REID / "entity_index.json").read_text(encoding="utf-8"))["track_ent"]
    import collections
    ent_n = collections.Counter(idx.values())

    out = []
    for key, r in q["tracks"].items():
        if not r.get("emb") or not r.get("n_used"):
            continue
        cam, tid = key.split(":")
        tr = truth.get("2026-10-12|%s|%s" % (cam, tid)) or {}
        ent = idx.get("%s:%s" % (cam, tid))
        s0 = (r["shots"] or [{}])[0]
        out.append(dict(
            key=key, cam=cam, tid=int(tid), h_med=r["h_med"],
            emb=_l2(r["emb"]), ident=tr.get("label"), gil_entity=tr.get("entity"),
            state=tr.get("state"), store_entity=ent, ent_tracks=ent_n.get(ent, 0),
            pd=s0.get("pd"), frame=s0.get("frame"), crop=s0.get("crop"),
            lap=s0.get("lap"), n_used=r["n_used"]))
    meta = dict(ok=True, scan=q["meta"], n_track=len(out),
                n_ident=len({r["ident"] for r in out if r["ident"]}),
                n_solo_ent=sum(1 for r in out if r["ent_tracks"] <= 1))
    return out, meta


# ---------------------------------------------------------------- 指标
def _auc(P, N):
    P, N = np.asarray(P, float), np.asarray(N, float)
    if not len(P) or not len(N):
        return None
    return float(((P[:, None] > N[None, :]).sum()
                  + 0.5 * (P[:, None] == N[None, :]).sum()) / (len(P) * len(N)))


def _wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - m), 4), round(min(1.0, c + m), 4)]


def _eval_block(tpl, field="emb"):
    """同人/异人对照 + 留一 rank-1 + 按身份 bootstrap CI。"""
    rows = [r for r in tpl if r.get("ident") and r.get(field) is not None]
    ids = sorted({r["ident"] for r in rows})

    def pairs(keep=None):
        P, N = [], []
        for a, b in itertools.combinations(rows, 2):
            if keep is not None and a["ident"] not in keep and b["ident"] not in keep:
                continue
            s = float(a[field] @ b[field])
            (P if a["ident"] == b["ident"] else N).append(s)
        return P, N

    P, N = pairs()
    a = _auc(P, N)
    rng = np.random.default_rng(20261113)
    vals = []
    for _ in range(1200):
        keep = {i for i in ids if rng.random() < 0.7} or set(ids)
        v = _auc(*pairs(keep))
        if v is not None:
            vals.append(v)
    ci = ([round(float(np.percentile(vals, 2.5)), 4),
           round(float(np.percentile(vals, 97.5)), 4)] if vals else None)

    import collections
    cnt = collections.Counter(r["ident"] for r in rows)
    qs = [r for r in rows if cnt[r["ident"]] >= 2]
    hit = 0
    for r in qs:
        best = {}
        for x in rows:
            if x is r:
                continue
            s = float(r[field] @ x[field])
            best[x["ident"]] = max(best.get(x["ident"], -9.0), s)
        if best and max(best, key=best.get) == r["ident"]:
            hit += 1
    return dict(n_track=len(rows), n_ident=len(ids), n_pos=len(P), n_neg=len(N),
                pos_mean=round(float(np.mean(P)), 4) if P else None,
                neg_mean=round(float(np.mean(N)), 4) if N else None,
                auc=None if a is None else round(a, 4), auc_ci=ci,
                rank1_n=len(qs), rank1_hit=hit,
                rank1=None if not qs else round(hit / len(qs), 4),
                rank1_ci=_wilson(hit, len(qs)) if qs else None)


def _stratified(tpl):
    """同机位 / 跨机位分层，人脸与外观并列 —— 这一层才说明人脸值不值。"""
    rows = [r for r in tpl if r.get("ident") and r.get("app") is not None]
    out = {}
    for tag, cond in (("same_cam", lambda a, b: a["cam"] == b["cam"]),
                      ("cross_cam", lambda a, b: a["cam"] != b["cam"])):
        for f, name in (("emb", "face"), ("app", "appearance")):
            P, N = [], []
            for a, b in itertools.combinations(rows, 2):
                if not cond(a, b):
                    continue
                s = float(a[f] @ b[f])
                (P if a["ident"] == b["ident"] else N).append(s)
            a_ = _auc(P, N)
            out["%s_%s" % (name, tag)] = dict(
                auc=None if a_ is None else round(a_, 4), n_pos=len(P), n_neg=len(N),
                pos_mean=round(float(np.mean(P)), 4) if P else None,
                neg_mean=round(float(np.mean(N)), 4) if N else None)
    return out


def _with_appearance(tpl):
    meta = json.loads((REID / "store_meta.json").read_text(encoding="utf-8"))["tracks"]
    E = np.load(REID / "store_emb.npy").astype(np.float64)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    idx = {"%s:%d" % (t["cam"], int(t["tid"])): i for i, t in enumerate(meta)}
    for r in tpl:
        i = idx.get(r["key"])
        r["app"] = E[i] if i is not None else None
    return tpl


def _sweep(tpl, grid=(0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)):
    """留一 rank-1 随阈值变化。n 小，只作取舍参考，不当结论。"""
    rows = [r for r in tpl if r.get("ident")]
    import collections
    cnt = collections.Counter(r["ident"] for r in rows)
    qs = [r for r in rows if cnt[r["ident"]] >= 2]
    tab = []
    for t in grid:
        acc = hit = 0
        for r in qs:
            best = {}
            for x in rows:
                if x is r:
                    continue
                s = float(r["emb"] @ x["emb"])
                best[x["ident"]] = max(best.get(x["ident"], -9.0), s)
            top = max(best, key=best.get)
            if best[top] >= t:
                acc += 1
                hit += int(top == r["ident"])
        tab.append(dict(tau=t, accepted=acc, hit=hit,
                        prec=round(hit / acc, 4) if acc else None,
                        recall=round(acc / len(qs), 4) if qs else None))
    return tab


def suggest():
    """留一对每条有脸轨给出身份建议；落盘 face_suggest.json。"""
    tpl, tmeta = templates()
    if not tmeta.get("ok"):
        return dict(ok=False, reason=tmeta.get("reason"))
    if not tpl:
        return dict(ok=False, reason="没有人脸模板")
    _with_appearance(tpl)

    sugg = []
    for r in tpl:
        best, bt = {}, {}
        for x in tpl:
            if x is r or not x.get("ident"):
                continue
            s = float(r["emb"] @ x["emb"])
            if s > best.get(x["ident"], -9.0):
                best[x["ident"]] = s
                bt[x["ident"]] = x
        if not best:
            continue
        order = sorted(best, key=best.get, reverse=True)
        b = order[0]
        m = bt[b]
        sugg.append(dict(
            key=r["key"], cam=r["cam"], tid=r["tid"], ident=b,
            score=round(best[b], 4),
            second=order[1] if len(order) > 1 else None,
            margin=round(best[b] - best[order[1]], 4) if len(order) > 1 else None,
            level=("auto" if (TAU_AUTO is not None and best[b] >= TAU_AUTO) else
                   "queue" if best[b] >= TAU else "low"),
            truth=r["ident"], hit=(b == r["ident"]),
            target_entity=m.get("gil_entity") or m.get("store_entity"),
            target_from="gil" if m.get("gil_entity") else "store",
            match_key=m["key"], match_crop=m["crop"], match_pd=m["pd"],
            pd=r["pd"], lap=r["lap"], frame=r["frame"], crop=r["crop"],
            h_med=r["h_med"], ent_tracks=r["ent_tracks"],
            same_entity=(r["store_entity"] == m["store_entity"])))

    sugg.sort(key=lambda x: (-x["score"], x["key"]))
    n_auto = sum(1 for s in sugg if s["level"] == "auto")
    n_queue = sum(1 for s in sugg if s["level"] == "queue")
    hit_auto = sum(1 for s in sugg if s["level"] == "auto" and s["hit"])

    out = dict(ok=True, created=time.strftime("%Y-%m-%d %H:%M:%S"),
               params=dict(model=MODEL, det_thr=DET_THR, tau=TAU,
                           tau_auto=TAU_AUTO, auto_enabled=TAU_AUTO is not None,
                           emb_dim=int(len(tpl[0]["emb"]))),
               coverage=dict(n_store=N_STORE, n_scan=tmeta["scan"]["n_track"],
                             n_face=len(tpl), scan_min_h=tmeta["scan"]["min_h"],
                             faces_detected=tmeta["scan"]["faces_detected"],
                             pct_of_store=round(len(tpl) / float(N_STORE), 4)),
               n_sugg=len(sugg), n_auto=n_auto, n_queue=n_queue,
               face=_eval_block(tpl, "emb"),
               appearance=_eval_block(tpl, "app"),
               stratified=_stratified(tpl),
               sweep=_sweep(tpl),
               auto_precision=dict(n=n_auto, hit=hit_auto,
                                   prec=round(hit_auto / n_auto, 4) if n_auto else None,
                                   ci=_wilson(hit_auto, n_auto) if n_auto else None,
                                   note="自动写池默认关闭；本包最优操作点 n=9 精度 0.667，不足以上线"),
               tau_grid=dict(tau_045=dict(n=21, hit=8, prec=0.381),
                             tau_050=dict(n=9, hit=6, prec=0.667),
                             tau_055=dict(n=5, hit=2, prec=0.400)),
               n_solo_ent=tmeta["n_solo_ent"],
               gil_note="GIL gperson 全部 state='candidate'，是候选身份图，非人工确认真值",
               suggestions=sugg)
    SUGGEST.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def pool_keys():
    """已在池里的 (cam, tid, entity)。

    只算 scope=cam 的条目（自动人脸写入走的就是默认 cam 作用域）；scope='*' 的全局条目
    对每个机位都生效，按 cam 判重会误报，故跳过。
    """
    try:
        import selector as S
        sel = S.init()
        if not getattr(sel, "ready", False):
            return set()
        return {(str(it["cam"]), int(it["tid"]), str(ent))
                for (scope, ent), items in sel.pool.items()
                for it in items if scope == str(it["cam"])}
    except Exception:
        return set()


def load_suggest(with_pool: bool = True):
    if not SUGGEST.exists():
        return dict(ok=False, reason="尚未匹配：先跑 POST /api/selector/auto_face/scan")
    d = json.loads(SUGGEST.read_text(encoding="utf-8"))
    if with_pool:
        have = pool_keys()
        for s in d.get("suggestions") or []:
            s["in_pool"] = (str(s.get("cam")), int(s.get("tid") or -1),
                            str(s.get("target_entity"))) in have
        d["n_in_pool"] = sum(1 for s in d.get("suggestions") or [] if s["in_pool"])
    return d


# ---------------------------------------------------------------- 卡片自检
def card_audit(force=False):
    """45 张人脸卡的对齐审计。**不用于匹配**，用来解释为什么不用卡图建库。"""
    if CARDS.exists() and not force:
        return json.loads(CARDS.read_text(encoding="utf-8"))
    faces_d = json.loads(FACE_JSON.read_text(encoding="utf-8"))
    base = FACE_JSON.parent.parent
    det = _detector()
    truth, _ = gil_truth()
    cards = []
    for p in faces_d["persons"]:
        rel = p.get("img_single") or p.get("img_best")
        img = cv2.imread(str(base / rel)) if rel else None
        if img is None:
            cards.append(dict(label=p["label"], cam=str(p["cam"]), tid=int(p["tid"]),
                              ok=False, err="missing"))
            continue
        f = det.detect(img)
        if len(f):
            b = max(f, key=lambda x: x[14])
            kps = np.asarray(b[4:14], np.float32).reshape(5, 2)
            src, sc = "det", round(float(b[14]), 3)
        else:
            kps = (TPL112 / 112.0) * np.array([img.shape[1], img.shape[0]], np.float32)
            src, sc = "fallback", None
        ali = align112(img, kps)
        if ali is None:
            cards.append(dict(label=p["label"], ok=False, err="align"))
            continue
        tr = truth.get("2026-10-12|%s|%d" % (str(p["cam"]), int(p["tid"]))) or {}
        cards.append(dict(label=p["label"], cam=str(p["cam"]), tid=int(p["tid"]),
                          img=rel, w=int(img.shape[1]), h=int(img.shape[0]),
                          img_best=p.get("img_best"),
                          align_src=src, det_score=sc, pd=p.get("pd"),
                          truth_label=tr.get("label"), truth_state=tr.get("state"),
                          ok=True, emb=[float(x) for x in embed(ali)]))
    ok = [c for c in cards if c.get("ok")]
    imp = []
    for a, b in itertools.combinations(ok, 2):
        if (a.get("truth_label") or a["label"]) != (b.get("truth_label") or b["label"]):
            imp.append(float(np.dot(a["emb"], b["emb"])))
    imp = np.array(imp) if imp else np.zeros(1)
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), n=len(cards), n_ok=len(ok),
                align_det=sum(1 for c in ok if c["align_src"] == "det"),
                align_fallback=sum(1 for c in ok if c["align_src"] == "fallback"),
                fused_best=sum(1 for c in cards
                               if str(c.get("img_best") or "").startswith("faces/ft_")),
                label_vs_gil=sum(1 for c in cards if c.get("truth_label") == c["label"]),
                impostor=dict(n=int(imp.size), mean=round(float(imp.mean()), 4),
                              max=round(float(imp.max()), 4),
                              p99=round(float(np.percentile(imp, 99)), 4)),
                note=("卡图未对齐，且 23/45 的 img_best 是融合模板 ft_*.jpg；用作图库时"
                      "跨身份余弦 max 0.71 / p99 0.45。不用于匹配，仅作展示与自检。"))
    out = dict(meta=meta, cards=cards)
    CARDS.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------- 写池
def apply_to_pool(keys=None, min_score=None, level=None, src="auto_face"):
    """把建议写成选择器的池条目（目标实体取 GIL 实体）。"""
    import selector as S
    d = load_suggest()
    if not d.get("ok"):
        return dict(ok=False, reason=d.get("reason", "无建议"))
    sel = S.init()
    want = set(keys or [])
    applied, skipped = [], []
    for s in d["suggestions"]:
        if want and s["key"] not in want:
            continue
        if level and s["level"] != level:
            continue
        if min_score is not None and s["score"] < min_score:
            continue
        ent = s.get("target_entity")
        if not ent:
            skipped.append(dict(key=s["key"], reason="目标实体未知"))
            continue
        r = sel.witness(entity=ent, cam=s["cam"], tid=int(s["tid"]), src=src,
                        q=s["score"],
                        note="auto face %s %.3f" % (s["ident"], s["score"]))
        if r.get("ok") and not r.get("dup"):
            applied.append(dict(key=s["key"], entity=ent, ident=s["ident"],
                                score=s["score"], mismatch=r.get("mismatch")))
        elif r.get("dup"):
            skipped.append(dict(key=s["key"], reason="已在池中"))
        else:
            skipped.append(dict(key=s["key"], reason=r.get("reason") or "未写入"))
    n_dup = sum(1 for x in skipped if x["reason"] == "已在池中")
    return dict(ok=True, n_applied=len(applied), applied=applied, skipped=skipped,
                n_skipped=len(skipped), n_dup=n_dup,
                n_pool=len(sel.state().get("pool", [])))
