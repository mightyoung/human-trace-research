# -*- coding: utf-8 -*-
"""算法工作台后端：跨镜头统一 ReID / 语义查人 / 聚类标注。

- ReidStore：539 条全局轨迹的 768d 嵌入（C-RADIOxGemma2 桥接），跨 6 机位检索
- 语义查人：中文/英文 query → 属性概念 → 词表向量组合 → 余弦检索（本地离线编码）
- 聚类工作台：可在界面上重命名/删除簇、挪动/移除成员、新建簇，状态落盘
"""
from __future__ import annotations
import json
import threading
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
REID = DATA / "reid"
CLUSTER_STATE = REID / "clusters_state.json"
CLUSTER_SEED = REID / "clusters.json"

# ---- 中文/英文关键词 → 概念（与 dataprep/build_text_vocab.py 保持一致）----
COLOR_ALIAS = {
    "white": ["白", "白色", "white"],
    "black": ["黑", "黑色", "black", "dark"],
    "gray": ["灰", "灰色", "gray", "grey"],
    "red": ["红", "红色", "red"],
    "blue": ["蓝", "蓝色", "blue"],
    "navy": ["藏青", "深蓝", "navy"],
    "green": ["绿", "绿色", "green"],
    "yellow": ["黄", "黄色", "yellow"],
    "brown": ["棕", "棕色", "咖啡色", "brown"],
    "beige": ["米白", "米色", "杏色", "beige"],
    "pink": ["粉", "粉色", "pink"],
    "orange": ["橙", "橙色", "orange"],
    "purple": ["紫", "紫色", "purple"],
}
GARMENT_ALIAS = {
    "top": ["上衣", "衬衫", "衬衣", "衣服", "T恤", "短袖", "shirt", "top", "t-shirt"],
    "jacket": ["外套", "夹克", "西装", "jacket", "coat"],
    "hoodie": ["卫衣", "连帽", "hoodie"],
    "pants": ["长裤", "裤子", "裤", "pants", "trousers"],
    "dress": ["连衣裙", "裙子", "裙", "dress", "skirt"],
    "hat": ["帽子", "帽", "hat", "cap"],
}
OTHER_ALIAS = {
    "backpack": ["背包", "书包", "双肩包", "backpack"],
    "glasses": ["眼镜", "戴眼镜", "glasses"],
    "mask": ["口罩", "mask"],
    "scarf": ["围巾", "scarf"],
    "umbrella": ["打伞", "雨伞", "伞", "umbrella"],
    "hair_long": ["长发", "long hair"],
    "man": ["男士", "男人", "男的", "man"],
    "woman": ["女士", "女人", "女的", "woman"],
    "clean": ["保洁", "拖地", "打扫", "清洁", "cleaning"],
    "fur_coat": ["毛皮领", "皮领", "毛领", "fur collar"],
}


def _l2(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


class ReidStore:
    """全局 ReID 库：加载一次，内存检索。"""

    def __init__(self):
        self.ready = False
        self.emb: np.ndarray | None = None
        self.meta: list[dict] = []
        self.by_camtid: dict[tuple, dict] = {}
        self.by_key: dict[str, dict] = {}
        self.vocab_keys: list[str] = []
        self.vocab_emb: np.ndarray | None = None
        self.vocab_phrase: dict[str, str] = {}
        self._load()

    def _load(self):
        try:
            self.emb = _l2(np.load(REID / "store_emb.npy").astype(np.float32))
            m = json.loads((REID / "store_meta.json").read_text(encoding="utf-8"))
            self.meta = m["tracks"]
            self.version = m.get("version", {})
            self.by_camtid = {(t["cam"], t["tid"]): t for t in self.meta}
            self.by_key = {t["key"]: t for t in self.meta}
            v = np.load(REID / "text_vocab.npz", allow_pickle=True)
            self.vocab_keys = [str(x) for x in v["keys"]]
            self.vocab_emb = _l2(v["emb"].astype(np.float32))
            self.vocab_phrase = {str(k): str(p) for k, p in zip(v["keys"], v["phrases"])}
            self.ready = True
        except Exception as e:  # pragma: no cover
            self.err = str(e)

    # ---------- 检索 ----------
    def _pack(self, i: int, score: float, **extra) -> dict:
        m = self.meta[int(i)]
        return {"cam": m["cam"], "tid": m["tid"], "key": m["key"], "i": int(i),
                "t0": m["t0"], "t1": m["t1"], "f_rep": m.get("f_rep"),
                "h_med": m["h_med"], "score": round(float(score), 4), **extra}

    def similar(self, cam: str, tid: int, k: int = 12) -> dict | None:
        t = self.by_camtid.get((cam, int(tid)))
        if not t:
            return None
        q = self.emb[t["i"]]
        sim = self.emb @ q
        order = np.argsort(-sim)
        out = []
        for i in order:
            if int(i) == t["i"]:
                continue
            out.append(self._pack(int(i), sim[i]))
            if len(out) >= k:
                break
        return {"query": {**t, "score": 1.0}, "results": out,
                "same_cam_rank": None}

    def search_vector(self, idx: int, k: int = 24) -> list[dict]:
        if idx < 0 or idx >= len(self.meta):
            return []
        q = self.emb[idx]
        sim = self.emb @ q
        order = np.argsort(-sim)
        out = []
        for i in order:
            if int(i) == idx:
                continue
            out.append(self._pack(int(i), sim[i]))
            if len(out) >= k:
                break
        return out

    def parse_text(self, query: str) -> dict:
        """把 query 解析成概念键列表（位置配对着色 + 服饰）。"""
        q = (query or "").lower()
        raw = []
        for ckey, aliases in COLOR_ALIAS.items():
            for a in aliases:
                s = 0
                while True:
                    p = q.find(a, s)
                    if p < 0:
                        break
                    raw.append((p, len(a), "color", ckey))
                    s = p + len(a)
        for gkey, aliases in GARMENT_ALIAS.items():
            for a in aliases:
                s = 0
                while True:
                    p = q.find(a, s)
                    if p < 0:
                        break
                    raw.append((p, len(a), "garment", gkey))
                    s = p + len(a)
        # 贪心去重叠（长匹配优先），避免"帽子/帽"、"深蓝/蓝"重复命中
        raw.sort(key=lambda e: (e[0], -e[1]))
        events, taken = [], []
        for p, ln, typ, key in raw:
            if any(not (p + ln <= a or p >= b) for a, b in taken):
                continue
            taken.append((p, p + ln))
            events.append((p, typ, key))
        parts, pending, used = [], None, set()
        for pos, typ, key in events:
            if typ == "color":
                pending = key
            else:
                conc = f"{pending}|{key}" if pending else f"any|{key}"
                if conc not in used:
                    parts.append(conc); used.add(conc)
                pending = None
        if pending:
            conc = f"{pending}|color"
            if conc not in used:
                parts.append(conc); used.add(conc)
        for okey, aliases in OTHER_ALIAS.items():
            if any(a in q for a in aliases) and f"-|{okey}" not in used:
                parts.append(f"-|{okey}"); used.add(f"-|{okey}")
        # 过滤词表中不存在的键
        parts = [p for p in parts if p in set(self.vocab_keys)]
        return {"parts": parts, "matched": bool(parts)}

    def search_text(self, query: str, k: int = 24) -> dict:
        pr = self.parse_text(query)
        parts = pr["parts"] or ["-|generic"]
        ki = {kk: i for i, kk in enumerate(self.vocab_keys)}
        vecs = [self.vocab_emb[ki[p]] for p in parts if p in ki]
        qv = _l2(np.mean(vecs, axis=0))
        sim = self.emb @ qv
        order = np.argsort(-sim)
        out = [self._pack(int(i), sim[i]) for i in order[:k]]
        return {"query": query, "parts": pr["parts"], "matched": pr["matched"],
                "results": out,
                "phrases": [self.vocab_phrase.get(p, p) for p in pr["parts"]]}

    def entity_of(self, cam: str, tid: int) -> dict:
        idx = json.loads((REID / "entity_index.json").read_text(encoding="utf-8"))
        return {"entity": idx["track_ent"].get(f"{cam}:{tid}"),
                "chain": idx["track_chain"].get(f"{cam}:{tid}")}


# ---- 聚类工作台状态 ----
class ClusterDesk:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = None
        self._load()

    def _seed(self):
        d = json.loads(CLUSTER_SEED.read_text(encoding="utf-8"))
        cl = [{"cid": c["cid"], "label": c.get("label", ""), "members": c["members"],
               "rep": c["rep"], "origin": "auto"} for c in d["clusters"]]
        return {"method": d["method"], "k": d["k"], "thr": d["thr"], "n": d["n"],
                "n_total": len(cl), "clusters": cl, "unassigned": [],
                "revision": 0, "log": []}

    def _load(self):
        if CLUSTER_STATE.exists():
            try:
                self.state = json.loads(CLUSTER_STATE.read_text(encoding="utf-8"))
                return
            except Exception:
                pass
        self.state = self._seed()
        self._save()

    def _save(self):
        CLUSTER_STATE.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")

    def _next_cid(self):
        used = {c["cid"] for c in self.state["clusters"]}
        i = 1
        while f"U{i:03d}" in used:
            i += 1
        return f"U{i:03d}"

    def _log(self, action, detail):
        self.state["log"].insert(0, {"action": action, "detail": detail})
        self.state["log"] = self.state["log"][:50]
        self.state["revision"] += 1

    def snapshot(self, store: ReidStore):
        out = []
        for c in self.state["clusters"]:
            mem = [store._pack(i, 0.0) for i in c["members"] if 0 <= i < len(store.meta)]
            rep = store._pack(c["rep"], 0.0) if 0 <= c.get("rep", -1) < len(store.meta) else None
            out.append({"cid": c["cid"], "label": c["label"], "size": len(c["members"]),
                        "origin": c.get("origin", "auto"), "rep": rep, "members": mem})
        un = [store._pack(i, 0.0) for i in self.state.get("unassigned", [])]
        return {"method": self.state["method"], "k": self.state["k"], "thr": self.state["thr"],
                "n": self.state["n"], "n_total": len(out), "revision": self.state["revision"],
                "clusters": out, "unassigned": un, "log": self.state["log"][:20]}

    def rename(self, cid, label):
        with self.lock:
            for c in self.state["clusters"]:
                if c["cid"] == cid:
                    c["label"] = label or ""
                    self._log("rename", f"{cid} -> {label}")
                    self._save()
                    return True
        return False

    def delete(self, cid):
        with self.lock:
            for c in self.state["clusters"]:
                if c["cid"] == cid:
                    self.state["unassigned"] = list(dict.fromkeys(
                        self.state.get("unassigned", []) + c["members"]))
                    self.state["clusters"].remove(c)
                    self._log("delete", f"删除簇 {cid}（{len(c['members'])} 项转为未分配）")
                    self._save()
                    return True
        return False

    def move(self, items, to_cid, create_new=False, label=""):
        with self.lock:
            items = [int(x) for x in (items or [])]
            srcs = self.state["clusters"]
            moved = []
            for it in items:
                for c in srcs:
                    if it in c["members"]:
                        c["members"].remove(it); moved.append(it); break
                if it in self.state.get("unassigned", []):
                    self.state["unassigned"].remove(it); moved.append(it)
            if create_new or not to_cid:
                to_cid = self._next_cid()
                srcs.append({"cid": to_cid, "label": label or to_cid, "members": [],
                             "rep": moved[0] if moved else -1, "origin": "manual"})
            target = next((c for c in srcs if c["cid"] == to_cid), None)
            if target is None:
                return False
            target["members"] = list(dict.fromkeys(target["members"] + moved))
            if moved:
                target["rep"] = moved[0]
            # 清理空簇
            self.state["clusters"] = [c for c in srcs if c["members"]]
            self._log("move", f"{len(moved)} 项 -> {to_cid}")
            self._save()
            return True

    def remove_items(self, cid, items):
        with self.lock:
            c = next((c for c in self.state["clusters"] if c["cid"] == cid), None)
            if c is None:
                return False
            items = [int(x) for x in (items or [])]
            for it in items:
                if it in c["members"]:
                    c["members"].remove(it)
                    self.state.setdefault("unassigned", []).append(it)
            if not c["members"]:
                self.state["clusters"].remove(c)
            self.state["unassigned"] = list(dict.fromkeys(self.state["unassigned"]))
            self._log("remove", f"从 {cid} 移除 {len(items)} 项 -> 未分配")
            self._save()
            return True

    def reset(self):
        with self.lock:
            self.state = self._seed()
            self._save()
            return True


STORE: ReidStore | None = None
DESK: ClusterDesk | None = None


def init():
    global STORE, DESK
    STORE = ReidStore()
    DESK = ClusterDesk()
    return STORE, DESK
