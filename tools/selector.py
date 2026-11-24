#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""身份选择器（Identity Selector）：输入 ReID 向量 → 输出身份分类（簇 ID 或 unknown）。

v2（2026-10-14）：多原型 + per-camera 阈值（依 selector_upgrade_study 实验结论）
  - 多原型：每簇可有多个子原型（PCLHD 式多原型借鉴），簇级 score = 子原型 max 聚合；
    兼容单原型库（proto_cluster=None 时行为与 v1 完全一致）。
  - per-camera 阈值：thr_unknown_per_cam 按样本所属摄像头取阈值（折中方案 F）；
    未提供 cams 或该摄像头无校准时回退全局 thr。
  - 两档阈值：selfcal（保守档，默认）/ loose（覆盖档，thr_loose*）。

训练（无监督）由 train_selector_cluster.py（v1）/ build_selector_v2.py（v2）完成，
本模块提供运行时接口与持久化。

用法：
    sel = IdentitySelector.load(".../selector_v2.npz")
    ids, sims = sel.predict(vec, cams=["305"])        # vec: [D] 或 [N, D]
    ids_loose, sims = sel.predict(vec, cams, loose=True)   # 覆盖档
    sc = sel.sims_all(vec)   # [N, K] 簇级 score 矩阵
"""
import json
import warnings

import numpy as np


def _norm(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


class IdentitySelector:
    def __init__(self, protos, thr_unknown=0.35, proto_cluster=None,
                 thr_unknown_per_cam=None, thr_loose=None,
                 thr_loose_per_cam=None, margin_prec=None,
                 margin_prec_per_cam=None, margin_ret=None, meta=None):
        """protos: [S, D] 已归一化原型（v2: 子原型；v1: 每簇一个）。
        proto_cluster: [S] 每个子原型所属簇 ID（None = v1 行为，每原型一簇）。
        thr_unknown_per_cam: {cam: thr}（None = 仅全局 thr）。
        margin_prec: 推荐精度档 margin（全局参考值，top1-top2 簇分差下限）。
        margin_prec_per_cam: {cam: margin}（分位校准档；predict(margin="prec") 优先用）。
        margin_ret: 分位校准的目标保留率（语义记录，如 0.7）。
        """
        self.protos = _norm(np.asarray(protos, dtype=np.float32))
        self.thr_unknown = float(thr_unknown)
        self.thr_unknown_per_cam = (dict(thr_unknown_per_cam)
                                    if thr_unknown_per_cam else None)
        self.thr_loose = float(thr_loose) if thr_loose is not None else None
        self.thr_loose_per_cam = (dict(thr_loose_per_cam)
                                  if thr_loose_per_cam else None)
        self.margin_prec = float(margin_prec) if margin_prec is not None else None
        self.margin_prec_per_cam = (dict(margin_prec_per_cam)
                                    if margin_prec_per_cam else None)
        self.margin_ret = float(margin_ret) if margin_ret is not None else None
        self.meta = meta or {}
        if proto_cluster is None:
            self.proto_cluster = None
            self.n_clusters = int(self.protos.shape[0])
        else:
            self.proto_cluster = np.asarray(proto_cluster, dtype=np.int64)
            self.n_clusters = int(self.proto_cluster.max()) + 1

    # ---------------- 内部 ----------------
    def _cluster_scores(self, S):
        """子原型 sim 矩阵 [N, S] → 簇级 score [N, K]（簇内 max 聚合）。

        ★ 防御：空簇（proto_cluster 中无该簇 ID 的稀疏场景）填 -inf——
        该簇永不被 argmax 命中，也不再抛 ValueError（S[:, cols].max(1) 空列聚合崩溃修复）。
        正常配置（每簇至少一个原型）下输出与旧版逐位相同。
        """
        if self.proto_cluster is None:
            return S
        out = np.full((S.shape[0], self.n_clusters), -np.inf, dtype=np.float32)
        for c in range(self.n_clusters):
            cols = np.where(self.proto_cluster == c)[0]
            if cols.size == 0:
                continue
            out[:, c] = S[:, cols].max(1)
        return out

    def _thr_for(self, cams, loose=False):
        """返回逐样本阈值数组（cams 提供且有空 per-cam 校准时）或全局标量。"""
        t_glob = self.thr_unknown
        if loose and self.thr_loose is not None:
            t_glob = self.thr_loose
        if cams is None:
            return float(t_glob)
        pm = self.thr_loose_per_cam if loose else self.thr_unknown_per_cam
        if not pm:
            return float(t_glob)
        return np.asarray([pm.get(str(c), t_glob) for c in cams], dtype=np.float32)

    # ---------------- 公共接口 ----------------
    def _margin_for(self, cams, margin):
        """解析 margin 参数 → 逐样本数组或标量。
        margin: None→0；float→全局；"prec"→库带推荐值（per-cam 优先，回退全局）。"""
        if margin == "prec":
            if self.margin_prec_per_cam and cams is not None:
                g = self.margin_prec if self.margin_prec is not None else 0.0
                return np.asarray([self.margin_prec_per_cam.get(str(c), g) for c in cams],
                                  dtype=np.float32)
            return float(self.margin_prec if self.margin_prec is not None else 0.0)
        if margin is None:
            return 0.0
        return float(margin)

    def predict(self, v, cams=None, loose=False, margin=None,
                refl=None, refl_tau=None, refl_mode="soft",
                refl_base=None, refl_alpha=None):
        """v: [D] 或 [N, D]；cams: None 或长度 N 的摄像头标签列表。
        loose: 用覆盖档阈值；margin: 精度档一致性拒识（top1-top2 簇分差下限），
        None=不过滤；"prec"=用库带推荐值（margin_prec_per_cam 优先）。

        ★ 倒影门（S7 接入，默认 refl=None → 行为与旧版**完全一致**）：
        refl: 长度 N 的倒影分（0..1，越大越像倒影，来自 obs.flags.reflection_score）；
        refl_mode='soft'（默认）：**只作用于接受判定、不改簇归属**——
            eff = best − alpha·max(0, refl − base)，用 eff 与阈值比较；低于 base 的样本不受影响；
        refl_mode='hard'：refl ≥ refl_tau 直接判 unknown。
        refl_tau/base/alpha: None → 取 meta['reflection_gate'] 同名键，再回退 tau=0.5 / base=0.5 / alpha=1.0。
        返回 (ids, sims)：ids=-1 表示 unknown（低于阈值或被子门拦下）。"""
        v = np.asarray(v, dtype=np.float32)
        if not np.isfinite(v).all():
            raise ValueError("selector.predict: 输入向量含 NaN/Inf（上游缺陷，拒绝静默输出）")
        v = np.atleast_2d(v)
        v = _norm(v, axis=1)
        S = v @ self.protos.T
        sc = self._cluster_scores(S)
        idx = sc.argmax(1)
        best = sc[np.arange(len(idx)), idx]
        thr = self._thr_for(cams, loose)
        gate = self.meta.get("reflection_gate", {}) or {}
        eff = best
        hard = None
        if refl is not None:
            r = np.asarray(refl, dtype=np.float32).reshape(-1)
            if not np.isfinite(r).all():
                raise ValueError("selector.predict: refl 含 NaN/Inf")
            r = np.clip(r, 0.0, 1.0)
            if r.shape[0] != sc.shape[0]:
                raise ValueError("refl 长度需与样本数一致")
            if refl_mode == "hard":
                tau = refl_tau if refl_tau is not None else gate.get("tau", 0.5)
                hard = r >= float(tau)
            else:
                base = refl_base if refl_base is not None else gate.get("base", 0.5)
                alpha = refl_alpha if refl_alpha is not None else gate.get("alpha", 1.0)
                eff = best - float(alpha) * np.clip(r - float(base), 0.0, None)
        m = self._margin_for(cams, margin)
        if np.isscalar(m) and m <= 0:
            out = np.where(eff >= thr, idx, -1)
        elif sc.shape[1] < 2:
            # ★ 防御：K=1 时 top1-top2 分差无定义（旧版 srt[:, -2] 越界崩溃）。
            # 语义选择：跳过 margin 条件（降级为纯阈值判定）——而非全判 unknown，
            # 后者会砍掉单簇库的全部检索结果。
            warnings.warn("selector: 簇数<2，margin 条件已跳过（退化配置，无第二候选可比）")
            out = np.where(eff >= thr, idx, -1)
        else:
            srt = np.sort(sc, axis=1)
            mg = srt[:, -1] - srt[:, -2]
            out = np.where((eff >= thr) & (mg >= m), idx, -1)
        if hard is not None:
            out = np.where(hard, -1, out)
        if len(out) == 1:
            return int(out[0]), float(best[0])
        return out, best

    def sims_all(self, v):
        """簇级 score 矩阵 [N, K]（v1 单原型时即原型 sim 矩阵，行为不变）。"""
        v = np.asarray(v, dtype=np.float32)
        if not np.isfinite(v).all():
            raise ValueError("selector.sims_all: 输入向量含 NaN/Inf")
        v = _norm(np.atleast_2d(v), axis=1)
        return self._cluster_scores(v @ self.protos.T)

    def save(self, path):
        np.savez(path, protos=self.protos,
                 thr_unknown=self.thr_unknown,
                 proto_cluster=(self.proto_cluster if self.proto_cluster is not None
                                else np.array([], dtype=np.int64)),
                 thr_unknown_per_cam=np.array(json.dumps(self.thr_unknown_per_cam or {})),
                 thr_loose=np.array(self.thr_loose if self.thr_loose is not None else -1.0),
                 thr_loose_per_cam=np.array(json.dumps(self.thr_loose_per_cam or {})),
                 margin_prec=np.array(self.margin_prec if self.margin_prec is not None else -1.0),
                 margin_prec_per_cam=np.array(json.dumps(self.margin_prec_per_cam or {})),
                 margin_ret=np.array(self.margin_ret if self.margin_ret is not None else -1.0),
                 meta=np.asarray([self.meta], dtype=object))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        meta = z["meta"][0] if "meta" in z.files and len(z["meta"]) else {}
        pc = (z["proto_cluster"] if "proto_cluster" in z.files
              and z["proto_cluster"].size else None)
        if pc is not None:
            ids_pc = np.asarray(pc)
            if ids_pc.size and (ids_pc.min() < 0
                                or len(set(ids_pc.tolist())) < int(ids_pc.max()) + 1):
                warnings.warn(
                    f"selector.load: proto_cluster 含空簇或非法 ID"
                    f"（ID 集 {sorted(set(ids_pc.tolist()))}，n_clusters={int(ids_pc.max()) + 1}）；"
                    f"空簇打分时填 -inf，永不命中")
        pm = json.loads(str(z["thr_unknown_per_cam"])) if "thr_unknown_per_cam" in z.files else None
        tl = float(z["thr_loose"]) if "thr_loose" in z.files else None
        tlpm = json.loads(str(z["thr_loose_per_cam"])) if "thr_loose_per_cam" in z.files else None
        mg = float(z["margin_prec"]) if "margin_prec" in z.files else None
        mpm = json.loads(str(z["margin_prec_per_cam"])) if "margin_prec_per_cam" in z.files else None
        mret = float(z["margin_ret"]) if "margin_ret" in z.files else None
        return IdentitySelector(z["protos"], float(z["thr_unknown"]),
                                proto_cluster=pc,
                                thr_unknown_per_cam=(pm if pm else None),
                                thr_loose=(tl if (tl is not None and tl >= 0) else None),
                                thr_loose_per_cam=(tlpm if tlpm else None),
                                margin_prec=(mg if (mg is not None and mg >= 0) else None),
                                margin_prec_per_cam=(mpm if mpm else None),
                                margin_ret=(mret if (mret is not None and mret >= 0) else None),
                                meta=meta if isinstance(meta, dict) else {})
