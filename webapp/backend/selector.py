# -*- coding: utf-8 -*-
"""运行时身份选择器：形态条件化模板 + 人脸触发的实时池。

为什么单独一个模块
------------------
`algo.ReidStore` 只回答「哪条轨最像这条轨」（单模板、对称余弦）。真正上线要回答的是
「这条轨是哪个身份」，而这件事在一天之内为什么难，有两个已量化的原因：

  1. **形态跨度**：同一个人框高跨度中位 3.76 倍（站立整身 / 行走 / 坐姿头肩），
     每身份一个均值模板在这种跨度下必然失配。按 exp(-|log(h_i/h_q)|/σ) 加权后，
     8 段 179 条可评查询上 τ=0.80 的召回从 56.4% → 68.2%（零训练、零新数据）。
  2. **换衣**：服装变了，历史库里的模板就全废。人脸若在同一时刻找到该人，
     就把**那一刻的全身特征**写进该身份的实时池；池不进聚类、不改簇结构，
     只改选择器能看到什么模板。8 段上 68.2% → 79.3%（+11.1），上限 92.7%。

本模块把这两件事做成可在控制台操作的服务，参数全部沿用已验证值（σ=0.6、
池 top-3 / 容量 8 / 按机位分层），不引入新超参。

★ 三条必须随结果一起引用的口径
  - **池的触发源**：控制台里是「人工确认过的人脸卡」(`source='face'`)，
    不是自动人脸识别。人脸识别全对时的收益是 79.3% 那组数字；现实折损来自
    覆盖（本包 45 张人脸卡 / 539 轨 = 8.4%）与误识别，二者本模块都不估。
  - **留一**：查询轨自身一定从模板与池中剔除。不剔的话查询轨就在自己身份的
    模板里，top-1 恒等于自己 → 分数必然虚高到 1.0。
  - **候选集只含可评身份**：本包 411 个实体里 325 个是单轨实体，留一后模板为空，
    它们不进候选（否则「没有模板」会被误读成「不是这个人」）。
  - **跨机位默认关闭**：外观跨机位在实测里是随机水平（6 种编码/对齐配置 AUC
    0.450–0.622，区间全跨 0.5），所以库与池都按机位分层。`scope='all'` 只作对照。
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
REID = DATA / "reid"
POOL_PATH = REID / "selector_pool.json"

# 已验证参数（改这些等于改实验结论，不要随手调）
D_SIGMA = 0.6        # 形态加权带宽：0.2–0.6 是平台，≥0.9 退化
D_POOL_K = 3         # 池内取最像的 k 条，不是池均值（池混着换衣前的旧样本）
D_POOL_MAX = 8       # 每身份池容量（FIFO）
D_MIN_H = 60.0       # 写入质量门：框高中位 px（<60 px 的外观特征不可靠）
D_TAU = 0.80         # 判定阈值（与 8 段实验同一操作点）
D_MARGIN = 0.10      # 相对间隔判定：顶一减顶二（D2 179 查询：召回 116 / 误受 1）
D_FLOOR = 0.60       # 间隔模式下仍要求的最低余弦（开集场景挡陌生人）


def _l2(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _cam_of(key: str) -> str:
    return key.rsplit("#", 1)[0].split("_")[0]


class Selector:
    def __init__(self):
        self.lock = threading.Lock()
        self.ready = False
        self.err = ""
        self.emb = None          # (N, D) 已 L2 归一
        self.meta = []           # 每行 dict：cam/tid/key/t0/t1/h_med/f_rep
        self.by_key = {}         # "cam:tid" -> row idx
        self.ent_of = {}         # row idx -> entity
        self.groups = {}         # (cam, ent) -> [(idx, h)]
        self.groups_all = {}     # ent -> [(idx, h)]   （scope='all' 对照用）
        self.pool = {}           # (scope_key, ent) -> [entry, ...]
        self.sigma = D_SIGMA
        self.pool_k = D_POOL_K
        self.pool_max = D_POOL_MAX
        self.min_h = D_MIN_H
        self._load()

    # ---------------- 加载 ----------------
    def _load(self):
        try:
            self.emb = _l2(np.load(REID / "store_emb.npy").astype(np.float64))
            m = json.loads((REID / "store_meta.json").read_text(encoding="utf-8"))
            self.version = m.get("version", {})
            self.meta = m["tracks"]
            idx = json.loads((REID / "entity_index.json").read_text(encoding="utf-8"))
            te = idx.get("track_ent", {})

            for i, t in enumerate(self.meta):
                self.by_key[f"{t['cam']}:{int(t['tid'])}"] = i
                ent = te.get(f"{t['cam']}:{int(t['tid'])}") or f"T{t['cam']}-{int(t['tid'])}"
                self.ent_of[i] = ent

            for i, t in enumerate(self.meta):
                h = float(t.get("h_med") or 0.0)
                if h <= 0:
                    continue
                ent = self.ent_of[i]
                self.groups.setdefault((t["cam"], ent), []).append((i, h))
                self.groups_all.setdefault(ent, []).append((i, h))

            self._load_pool()
            self.ready = True
        except Exception as e:  # pragma: no cover
            self.err = f"{type(e).__name__}: {e}"

    def _load_pool(self):
        self.pool = {}
        if not POOL_PATH.exists():
            return
        try:
            d = json.loads(POOL_PATH.read_text(encoding="utf-8"))
            for e in d.get("entries", []):
                key = f"{e['cam']}:{int(e['tid'])}"
                i = self.by_key.get(key)
                if i is None:
                    continue           # 换包后该轨不存在 → 丢弃，不静默串号
                scope = e.get("scope") or e["cam"]
                slot = (scope, e["entity"])
                self.pool.setdefault(slot, []).append({
                    "i": i, "cam": e["cam"], "tid": int(e["tid"]),
                    "src": e.get("src", "manual"), "q": e.get("q"),
                    "at": e.get("at"), "note": e.get("note", ""),
                })
        except Exception:
            self.pool = {}

    def _save_pool(self):
        out = {"version": 1, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "params": {"sigma": self.sigma, "pool_k": self.pool_k,
                          "pool_max": self.pool_max, "min_h": self.min_h},
               "entries": []}
        for (scope, ent), items in self.pool.items():
            for it in items:
                out["entries"].append({
                    "entity": ent, "scope": scope, "cam": it["cam"], "tid": it["tid"],
                    "src": it["src"], "q": it["q"], "at": it["at"], "note": it["note"],
                })
        POOL_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 基础 ----------------
    def row(self, i: int) -> dict:
        t = self.meta[int(i)]
        return {"cam": t["cam"], "tid": int(t["tid"]), "key": t["key"],
                "t0": t.get("t0"), "t1": t.get("t1"), "h_med": t.get("h_med"),
                "f_rep": t.get("f_rep"), "entity": self.ent_of[int(i)]}

    def idx_of(self, cam: str, tid: int):
        return self.by_key.get(f"{cam}:{int(tid)}")

    def _pool_scope(self, cam: str, scope: str) -> str:
        return cam if scope == "cam" else "*"

    # ---------------- 打分 ----------------
    def _lib_scores(self, qi: int, hq: float, scope: str, cam: str, exclude: set,
                    allow: set | None = None):
        """形态加权模板的余弦；返回 {ent: (score, n, detail)}。

        allow 限定「哪些轨可以作为模板」—— 跨时段评估时只允许用查询时刻之前的轨。
        """
        q = self.emb[qi]
        src = self.groups if scope == "cam" else self.groups_all
        out = {}
        if scope == "cam":
            items_iter = [((cam, ent), v) for (c, ent), v in src.items() if c == cam]
        else:
            items_iter = list(src.items())
        for key, items in items_iter:
            ent = key[1] if scope == "cam" else key
            acc = None
            parts = []
            for i, h in items:
                if i in exclude:
                    continue
                if allow is not None and i not in allow:
                    continue
                w = math.exp(-abs(math.log(h / hq)) / self.sigma)
                v = self.emb[i] * w
                acc = v if acc is None else acc + v
                parts.append({"cam": self.meta[i]["cam"], "tid": int(self.meta[i]["tid"]),
                              "h": round(h, 1), "w": round(w, 3),
                              "cos": round(float(q @ self.emb[i]), 4)})
            if acc is None:
                continue
            acc = _l2(acc)
            parts.sort(key=lambda p: -p["w"])
            out[ent] = (round(float(q @ acc), 4), len(parts), parts[:5])
        return out

    def _pool_scores(self, qi: int, scope: str, cam: str, exclude: set):
        q = self.emb[qi]
        ps = self._pool_scope(cam, scope)
        out = {}
        for (sc, ent), items in self.pool.items():
            if sc != ps:
                continue
            rows = [it for it in items if it["i"] not in exclude]
            if not rows:
                continue
            v = np.asarray([self.emb[it["i"]] for it in rows]) @ q
            if len(v) > self.pool_k:
                keep = np.argsort(v)[-self.pool_k:]
            else:
                keep = np.arange(len(v))
            best = rows[int(keep[int(np.argmax(v[keep]))])]
            out[ent] = (round(float(v[keep].mean()), 4), len(rows),
                        {"cam": best["cam"], "tid": int(best["tid"])})
        return out

    def query(self, cam: str, tid: int, scope: str = "cam", tau: float = D_TAU,
              mode: str = "abs", margin: float = D_MARGIN, floor: float = D_FLOOR,
              exclude_self: bool = True, lib_allow: set | None = None) -> dict:
        if not self.ready:
            raise RuntimeError(f"选择器未就绪：{self.err}")
        qi = self.idx_of(cam, tid)
        if qi is None:
            raise KeyError(f"{cam}#{tid} 不在 ReID 库")
        q = self.row(qi)
        hq = float(q["h_med"] or 0.0)
        if hq <= 0:
            raise ValueError(f"{cam}#{tid} 缺 h_med，形态加权无从谈起")

        exclude = {qi} if exclude_self else set()
        lib = self._lib_scores(qi, hq, scope, cam, exclude, allow=lib_allow)
        pool = self._pool_scores(qi, scope, cam, exclude)

        cands = []
        for ent in set(lib) | set(pool):
            s0 = lib.get(ent, (None, 0, []))[0]
            s1v = lib.get(ent, (None, 0, []))
            s2v = pool.get(ent, (None, 0, None))
            s1 = max([x for x in (s0, s2v[0]) if x is not None], default=None)
            # 缩略图代表：库模板里权重最高的那条；没有库模板就用池里最像的那条
            part0 = (s1v[2] or [None])[0]
            rep = ({"cam": part0["cam"], "tid": int(part0["tid"])} if part0
                   else s2v[2])
            cands.append({
                "entity": ent, "is_own": ent == q["entity"],
                "s_lib": s0, "s_pool": s2v[0], "s_both": s1,
                "n_lib": s1v[1], "n_pool": s2v[1],
                "lib_parts": s1v[2],
                "rep": rep,
            })

        def pick(key):
            ok = [c for c in cands if c.get(key) is not None]
            if not ok:
                return None
            # ★ 确定性 tie-break：同分按实体名。否则候选集由 set 迭代顺序决定，
            #   池一多候选集就变大，S0（纯库，与池无关）也会跟着抖 —— 实测差 0.4pt。
            ok.sort(key=lambda c: (-c[key], c["entity"]))
            b = ok[0]
            s2 = ok[1][key] if len(ok) > 1 else None
            gap = (b[key] - s2) if s2 is not None else None
            if mode == "margin":
                # 间隔判定：只看与次名的分差，另加一个绝对下限防开集陌生人
                accepted = gap is not None and gap >= margin and b[key] >= floor
            else:
                accepted = b[key] >= tau
            return {"entity": b["entity"], "score": b[key], "score2": s2,
                    "margin": gap, "is_own": b["is_own"],
                    "accepted": accepted, "n": len(ok)}

        for c in cands:
            c["rank_lib"] = c["rank_both"] = None
        order0 = sorted([c for c in cands if c["s_lib"] is not None],
                        key=lambda c: (-c["s_lib"], c["entity"]))
        order1 = sorted([c for c in cands if c["s_both"] is not None],
                        key=lambda c: (-c["s_both"], c["entity"]))
        for r, c in enumerate(order0, 1):
            c["rank_lib"] = r
        for r, c in enumerate(order1, 1):
            c["rank_both"] = r
        cands.sort(key=lambda c: (-(c["s_both"] if c["s_both"] is not None else -9), c["entity"]))

        return {
            "query": q,
            "scope": scope,
            "tau": tau,
            "mode": mode,
            "margin_thr": margin,
            "floor": floor,
            "top_lib": pick("s_lib"),
            "top_pool": pick("s_pool"),
            "top_both": pick("s_both"),
            "n_cand": len(cands),
            "candidates": cands[:40],
            "query_in_pool": any(it["i"] == qi for items in self.pool.values() for it in items),
            "note": ("查询轨已从模板中剔除（留一）；候选只含还有其它模板的身份" if exclude_self
                     else "未剔除自身，分数不可用于验收"),
        }

    # ---------------- 池 ----------------
    def witness(self, entity: str, cam: str, tid: int, src: str = "manual",
                q: float | None = None, note: str = "", scope: str = "cam",
                force: bool = False) -> dict:
        if not self.ready:
            raise RuntimeError(f"选择器未就绪：{self.err}")
        i = self.idx_of(cam, tid)
        if i is None:
            raise KeyError(f"{cam}#{tid} 不在 ReID 库")
        row = self.row(i)
        h = float(row["h_med"] or 0.0)
        if h < self.min_h and not force:
            return {"ok": False, "skipped": "quality",
                    "reason": f"框高中位 {h:.0f} px < {self.min_h:.0f} px 质量门",
                    "need_force": True}
        with self.lock:
            slot = (self._pool_scope(cam, scope), entity)
            items = self.pool.setdefault(slot, [])
            if any(it["i"] == i for it in items):
                return {"ok": True, "dup": True, "entity": entity,
                        "track_entity": row["entity"], "n_pool": len(items),
                        "reason": "该轨已在池中"}
            items.append({"i": i, "cam": cam, "tid": int(tid), "src": src,
                          "q": q, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "note": note})
            dropped = None
            if len(items) > self.pool_max:
                old = items.pop(0)
                dropped = {"cam": old["cam"], "tid": old["tid"], "at": old["at"]}
            self._save_pool()
            return {"ok": True, "entity": entity, "cam": cam, "tid": int(tid),
                    "n_pool": len(items), "dropped": dropped,
                    "track_entity": row["entity"],
                    "mismatch": row["entity"] != entity,
                    "slot": {"scope": slot[0], "entity": entity}}

    def revoke(self, entity: str, cam: str, tid: int, scope: str = "cam") -> dict:
        i = self.idx_of(cam, tid)
        with self.lock:
            slot = (self._pool_scope(cam, scope), entity)
            items = self.pool.get(slot) or []
            before = len(items)
            self.pool[slot] = [it for it in items if it["i"] != i]
            if not self.pool[slot]:
                self.pool.pop(slot, None)
            self._save_pool()
            return {"ok": True, "removed": before - len(self.pool.get(slot) or []),
                    "n_pool": len(self.pool.get(slot) or [])}

    def reset_pool(self) -> dict:
        with self.lock:
            n = sum(len(v) for v in self.pool.values())
            self.pool = {}
            self._save_pool()
            return {"ok": True, "cleared": n}

    def set_params(self, sigma=None, pool_k=None, pool_max=None, min_h=None) -> dict:
        with self.lock:
            if sigma is not None:
                self.sigma = float(min(1.5, max(0.05, sigma)))
            if pool_k is not None:
                self.pool_k = int(min(8, max(1, pool_k)))
            if pool_max is not None:
                self.pool_max = int(min(32, max(1, pool_max)))
            if min_h is not None:
                self.min_h = float(max(0.0, min_h))
            self._save_pool()   # 参数与池同文件，便于复现
            return self.params()

    def params(self) -> dict:
        return {"sigma": self.sigma, "pool_k": self.pool_k,
                "pool_max": self.pool_max, "min_h": self.min_h, "tau": D_TAU}

    # ---------------- 快照 ----------------
    def state(self) -> dict:
        if not self.ready:
            return {"ok": False, "error": self.err}
        multi = sum(1 for v in self.groups.values() if len(v) >= 2)
        eval_tracks = sum(len(v) for v in self.groups.values() if len(v) >= 2)
        ents = {}
        for (cam, ent), items in self.groups.items():
            e = ents.setdefault(ent, {"entity": ent, "cams": set(), "n_track": 0,
                                      "n_pool": 0, "tracks": [], "_best": None})
            e["cams"].add(cam)
            e["n_track"] += len(items)
            for i, h in items:
                # 缩略图代表 = 该身份里框最高的一条（最清楚，远景小框看着只有色块）
                if e["_best"] is None or h > e["_best"][2]:
                    e["_best"] = (self.meta[i]["cam"], int(self.meta[i]["tid"]), float(h))
        for (scope, ent), items in self.pool.items():
            if ent not in ents:
                continue
            ents[ent]["n_pool"] += len(items)
        out_ents = []
        for e in ents.values():
            best = e.pop("_best", None)
            out_ents.append({"entity": e["entity"], "cams": sorted(e["cams"]),
                             "n_track": e["n_track"], "n_pool": e["n_pool"],
                             "rep": ({"cam": best[0], "tid": best[1]} if best else None)})
        out_ents.sort(key=lambda x: (-x["n_pool"], -x["n_track"], x["entity"]))
        pool_rows = []
        for (scope, ent), items in self.pool.items():
            for it in items:
                r = self.row(it["i"])
                # ★ entity = 被锚定的槽位；track_entity = 该轨自己所属的实体。
                #   两者不同说明「人工把这条轨归到了另一个身份」，必须显示出来，
                #   否则锚错人会被静默掩盖。
                pool_rows.append({**r, "entity": ent, "track_entity": r["entity"],
                                  "scope": scope, "src": it["src"],
                                  "q": it["q"], "at": it["at"], "note": it["note"]})
        pool_rows.sort(key=lambda r: (r["entity"], r["t0"] or 0))
        return {
            "ok": True,
            "version": self.version,
            "n_track": len(self.meta),
            "n_entity": len(ents),
            "n_group_multi": multi,
            "n_eval_track": eval_tracks,
            "params": self.params(),
            "pool": pool_rows,
            "n_pool": len(pool_rows),
            "entities": out_ents,
            "note": ("可评轨 = 所属「机位×实体」组至少 2 条轨的那部分；"
                     "单轨实体留一后没有模板，不进候选"),
        }

    def tracks(self) -> list[dict]:
        out = []
        for i in range(len(self.meta)):
            r = self.row(i)
            grp = self.groups.get((r["cam"], r["entity"]), [])
            r["n_group"] = len(grp)
            out.append(r)
        out.sort(key=lambda r: (r["cam"], r["t0"] or 0))
        return out

    # ---------------- 实时增量：三方代价对账 ----------------
    def bind_cost(self, tau: float = 0.80, online_tau: float = 0.85,
                  online_margin: float = 0.05, face_gate: float = 0.50,
                  face_margin: float = 0.05) -> dict:
        """量测「实时增量写池」到底赚还是亏。

        为什么必须单独量：`evaluate(mode='oracle')` 假设每次写池的身份都对，等于
        「人脸识别全对」。真实部署里写池的身份会错，错了就污染别人的池。
        这里同时跑四种，把上限和现实摆在一起：

          history     只用历史库（t0 早于查询），不写池 —— 基线
          online      每条轨判定后，若置信度够，按**所判身份**写池（会写错）
          online_or   同上但按**真值身份**写池 —— 自学习的上限
          face        池只由**真实人脸绑定**写入（会写错）

        ★ 结构性提醒：如果池里的轨本来就都在历史库里，两者信息重叠，
          增益上限会接近 0 —— 这说明「本数据集测不出」，不是「机制无效」。
          跨时段的增益证据在 8 段：库均值 56.4% → 形态加权 68.2% → 加池 79.3%。
        """
        if not self.ready:
            raise RuntimeError(f"选择器未就绪：{self.err}")
        order = [i for i in sorted(range(len(self.meta)),
                                   key=lambda i: (float(self.meta[i].get("t0") or 0),
                                                  self.meta[i]["cam"]))
                 if float(self.meta[i].get("h_med") or 0) > 0]

        fb = []
        fp = REID / "face_suggest.json"
        if fp.exists():
            for s in json.loads(fp.read_text(encoding="utf-8")).get("suggestions", []):
                i = self.idx_of(s["cam"], int(s["tid"]))
                if i is None:
                    continue
                t = self.meta[i]
                a = float(t.get("t1") or 0)
                f, fr = s.get("frame"), t.get("f_rep")
                if f is not None and fr is not None:
                    a = min(max(float(t.get("t0") or 0) + (float(f) - float(fr)) / 25.0,
                                float(t.get("t0") or 0)), a)
                fb.append({"i": i, "ent": s["target_entity"],
                           "hit": bool(s.get("hit")), "score": float(s["score"]),
                           "margin": float(s.get("margin") or 0.0), "t": a})

        saved_groups, saved_pool = self.groups, self.pool
        out = {}
        try:
            for mode in ("history", "online", "online_or", "face"):
                pool = {}
                seen = {}
                n = hit = drift = 0
                for qi in order:
                    cam = self.meta[qi]["cam"]
                    ent = self.ent_of[qi]
                    tn = float(self.meta[qi].get("t0") or 0)
                    self.groups = {k: list(v) for k, v in seen.items() if k[0] == cam}
                    self.pool = {}
                    for (c, e), items in pool.items():
                        if c != cam:
                            continue
                        live = [{"i": i, "cam": c, "tid": int(self.meta[i]["tid"]),
                                 "src": mode, "q": None, "at": t, "note": ""}
                                for (t, i) in items if t < tn and i != qi]
                        if live:
                            self.pool[(c, e)] = live[-self.pool_max:]
                    try:
                        r = self.query(cam, int(self.meta[qi]["tid"]), scope="cam",
                                       tau=tau)
                    except Exception:
                        r = None
                    if r and r["candidates"]:
                        n += 1
                        tp = r["top_both"]
                        if tp and tp["accepted"]:
                            hit += int(tp["entity"] == ent)
                        tl = r["top_lib"]
                        if mode in ("online", "online_or") and tl and tl["accepted"] \
                                and tl["score"] >= online_tau:
                            cs = sorted([c for c in r["candidates"]
                                         if c["s_both"] is not None],
                                        key=lambda c: (-c["s_both"], c["entity"]))
                            mg = (cs[0]["s_both"] - cs[1]["s_both"]) if len(cs) > 1 else 1.0
                            if mg >= online_margin:
                                if cs[0]["entity"] != ent:
                                    drift += 1
                                we = ent if mode == "online_or" else cs[0]["entity"]
                                pool.setdefault((cam, we), []).append((tn, qi))
                    if mode == "face":
                        for b in fb:
                            if b["i"] == qi and b["score"] >= face_gate \
                                    and b["margin"] >= face_margin:
                                pool.setdefault((cam, b["ent"]), []).append((b["t"], qi))
                                if not b["hit"]:
                                    drift += 1
                    h = float(self.meta[qi].get("h_med") or 0)
                    seen.setdefault((cam, ent), []).append((qi, h))
                out[mode] = {
                    "n": n, "hit": hit,
                    "pct": round(100.0 * hit / n, 1) if n else 0.0,
                    "n_pool": sum(len(v) for v in pool.values()),
                    "n_wrong_write": drift,
                }
        finally:
            self.groups, self.pool = saved_groups, saved_pool

        base = out["history"]["pct"]
        for k in out:
            out[k]["delta"] = round(out[k]["pct"] - base, 1)
        n_face = sum(1 for b in fb if b["score"] >= face_gate
                     and b["margin"] >= face_margin)
        n_face_ok = sum(1 for b in fb if b["score"] >= face_gate
                        and b["margin"] >= face_margin and b["hit"])
        return {
            "ok": True, "tau": tau, "n_track": len(order),
            "params": {"online_tau": online_tau, "online_margin": online_margin,
                       "face_gate": face_gate, "face_margin": face_margin},
            "table": out,
            "face_bind": {"n": n_face, "hit": n_face_ok,
                          "prec": round(n_face_ok / n_face, 4) if n_face else None},
            "note": ("评估用簇号，只做相对比较，不能当对外精度。"
                     "history=不写池；online=按所判身份写（会写错）；"
                     "online_or=按真值写（上限）；face=真实人脸绑定（会写错）。"
                     "★ 若池内轨本就在历史库里，增益上限≈0 —— 那是数据集没有跨时段，"
                     "不是机制无效；跨时段证据在 8 段（56.4→68.2→79.3）。"),
        }

    # ---------------- 自评（包内相对口径） ----------------
    def evaluate(self, scope: str = "cam", tau: float = D_TAU,
                 mode: str = "real", cam: str = "", split: str = "loo") -> dict:
        """全量评估。

        split='loo'   包内留一：查询轨剔出模板，其余同身份轨都可作模板。
                      衡量「同一时段内能不能把人分开」，池在这里几乎不带来信息。
        split='half'  按机位把 10 分钟卷切成前后两半：前半段建库、后半段查询，
                      池只在查询段内按时间因果写入。这是 8 段「上午库 → 下午查询」
                      的包内类比，池的增益必须在这个口径下看。

        mode='real'   只用真实池（人工/人脸卡写入过的）。
        mode='oracle' 模拟「每次触发都识别正确」：每处理完一条轨，就把它写入自己
                      身份的池 → 这是池的**上限**，对应 8 段里「人脸识别全对」的口径。
        """
        if not self.ready:
            raise RuntimeError(f"选择器未就绪：{self.err}")
        if split not in ("loo", "half"):
            raise ValueError(f"split 非法: {split}")

        lib_allow: dict[str, set] = {}
        if split == "loo":
            ok = set()
            for (cam_k, ent), items in self.groups.items():
                if len(items) >= 2 and (not cam or cam_k == cam):
                    ok.update(i for i, _ in items)
        else:
            ok = set()
            for cam_k in {t["cam"] for t in self.meta}:
                if cam and cam_k != cam:
                    continue
                rows_cam = [i for i in range(len(self.meta))
                            if self.meta[i]["cam"] == cam_k
                            and float(self.meta[i].get("h_med") or 0) > 0]
                if len(rows_cam) < 4:
                    continue
                ts = sorted(float(self.meta[i].get("t0") or 0) for i in rows_cam)
                mid = ts[len(ts) // 2]
                allow = {i for i in rows_cam if float(self.meta[i].get("t0") or 0) < mid}
                lib_allow[cam_k] = allow
                seen = {(cam_k, self.ent_of[i]) for i in allow}
                for i in rows_cam:
                    if float(self.meta[i].get("t0") or 0) >= mid and \
                            (cam_k, self.ent_of[i]) in seen:
                        ok.add(i)

        # ★ 严格因果：按 t0 升序流式处理；打分之后再写池（这条查询看不到自己）
        rows = sorted(ok, key=lambda i: (self.meta[i]["cam"],
                                         float(self.meta[i].get("t0") or 0)))

        saved = self.pool
        self.pool = ({k: [dict(it) for it in v] for k, v in saved.items()}
                     if mode == "real" else {})
        n = 0
        hit = {"s_lib": 0, "s_pool": 0, "s_both": 0}
        cov = {"s_lib": 0, "s_pool": 0, "s_both": 0}
        skipped_no_lib = 0
        by_cam = {}
        try:
            for qi in rows:
                cam_k = self.meta[qi]["cam"]
                ent = self.ent_of[qi]
                try:
                    r = self.query(cam_k, int(self.meta[qi]["tid"]), scope=scope, tau=tau,
                                   lib_allow=lib_allow.get(cam_k))
                except Exception:
                    continue
                if r["candidates"]:
                    n += 1
                    st = by_cam.setdefault(cam_k, {"n": 0, "s_lib": 0, "s_both": 0})
                    st["n"] += 1
                    for k, tk in (("s_lib", "top_lib"), ("s_pool", "top_pool"),
                                  ("s_both", "top_both")):
                        t = r[tk]
                        if t and t["accepted"]:
                            cov[k] += 1
                            if t["entity"] == ent:
                                hit[k] += 1
                                if k != "s_pool":
                                    st[k] += 1
                else:
                    skipped_no_lib += 1
                if mode == "oracle":
                    slot = (self._pool_scope(cam_k, scope), ent)
                    items = self.pool.setdefault(slot, [])
                    items.append({"i": qi, "cam": cam_k, "tid": int(self.meta[qi]["tid"]),
                                  "src": "oracle", "q": None, "at": None, "note": ""})
                    if len(items) > self.pool_max:
                        del items[0]
        finally:
            self.pool = saved

        def pack(k):
            return {"cov": cov[k], "hit": hit[k], "n": n,
                    "cov_pct": round(100 * cov[k] / n, 1) if n else 0.0,
                    "same_pct": round(100 * hit[k] / cov[k], 1) if cov[k] else None}

        return {
            "ok": True, "scope": scope, "tau": tau, "mode": mode, "split": split, "n": n,
            "skipped_no_lib": skipped_no_lib,
            "n_lib_pool": sum(len(v) for v in lib_allow.values()),
            "s_lib": pack("s_lib"), "s_pool": pack("s_pool"), "s_both": pack("s_both"),
            "by_cam": by_cam,
            "note": ("实体来自聚类输出，不是人工真值 → 本表是**包内自证**口径，"
                     "只做相对比较（有池 vs 无池），不能当作对外精度。"
                     + (" oracle 池＝触发全对的上限，现实折损在覆盖与误识别。"
                        if mode == "oracle" else " real 池＝控制台上已写入的条目。")
                     + (" split=half 用前半段建库、后半段查询。"
                        if split == "half" else
                        " split=loo 是同时段留一，池与历史库信息高度冗余，"
                        "这里池的增益本就接近 0。")
                     + " 池增益的实测证据来自 8 段（跨时段）：库均值 56.4% → 形态加权 68.2%"
                       " → 加池 79.3%（n=179 可评），本包没有那种跨时段条件。"),
        }


SEL: Selector | None = None


def init(reset: bool = False) -> Selector:
    """返回进程内单例。

    ★ 必须单例：池条目存在实例内存里（`self.pool`），只有 `_save_pool()` 落盘时才写文件。
    若像以前那样每次 init 都 `Selector()` 新建实例，写入方与读取方（main 的 SEL）就会分叉：
    采纳写进了磁盘，控制台的池视图/打分仍读旧内存 —— 实测表现就是「点了采纳没任何反应」。
    独立工具脚本走独立进程，首次调用照样新建。
    """
    global SEL
    if reset or SEL is None or not getattr(SEL, "ready", False):
        SEL = Selector()
    return SEL
