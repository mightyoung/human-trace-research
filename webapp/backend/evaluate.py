# -*- coding: utf-8 -*-
"""算法配置的在线重算：在**已存分数**上做阈值扫描与分层复核（不重跑 GPU 推理）。

诚实边界：本模块只重算「跨机候选对的判定分层」——即 crosscam_pipeline 的
时间/时空/外观三门与 AUTO/REVIEW/DROP 分层，输入是已有的 reid_cos / sp_median 等分数。
检测级参数（box_threshold / containment）需要 GPU 重跑，本模块**不伪造**，明确标注为需远程重跑。
"""
from __future__ import annotations
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

DEFAULT_PARAMS = {
    "sp_threshold": 0.125,      # 时空残差中位数门（越小越严）
    "auto_reid_min": 0.50,      # AUTO 层外观门
    "gray_mult": 2.0,           # REVIEW 灰区倍数
    "min_covis_sec": 0.0,       # 最小共视时长（秒）
    "min_sp_n": 0,              # 最小配对样本数
}

# 检测/跟踪级参数：本模块不改，仅供前端展示与"需远程重跑"提示
INFERENCE_PARAMS = {
    "box_threshold": 0.30,
    "text_threshold": 0.25,
    "containment_threshold": 0.85,
    "track_activation": 0.10,
    "stride": 5,
}


def _load_pairs() -> list:
    p = DATA / "pairs.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _tier(p: dict, params: dict) -> str:
    sp = p.get("sp_median")
    cos = p.get("reid_cos")
    sp_thr = float(params["sp_threshold"])
    cos_min = float(params["auto_reid_min"])
    gray = float(params["gray_mult"])
    if p.get("covis_sec") is not None and p["covis_sec"] < float(params.get("min_covis_sec", 0)):
        return "DROP"
    if p.get("sp_n") is not None and p["sp_n"] < int(params.get("min_sp_n", 0)):
        return "DROP"
    sp_ok = sp is not None and sp <= sp_thr
    cos_ok = cos is not None and cos >= cos_min
    if sp_ok and cos_ok:
        return "AUTO"
    # REVIEW 灰区 = 时空残差落在 [sp_thr, sp_thr*gray] 内（对齐 crosscam_pipeline 口径）
    if sp is not None and sp <= sp_thr * gray:
        return "REVIEW"
    return "DROP"


def evaluate(params: dict) -> dict:
    pairs = _load_pairs()
    tiers = {"AUTO": [], "REVIEW": [], "DROP": []}
    for p in pairs:
        tiers[_tier(p, params)].append(p)

    def label_counts(items):
        c = {"CONF": 0, "MODERATE": 0, "REJ": 0, "UNLABELED": 0}
        for p in items:
            lab = p.get("label", "UNLABELED")
            if lab in c:
                c[lab] += 1
            else:
                c["UNLABELED"] += 1
        return c

    # AUTO 层正例率（与人工标签比对）
    auto = tiers["AUTO"]
    auto_pos = sum(1 for p in auto if p.get("label") in ("CONF", "MODERATE"))
    auto_lab = sum(1 for p in auto if p.get("label") in ("CONF", "MODERATE", "REJ"))
    auto_prec = (auto_pos / auto_lab) if auto_lab else None

    # 分数层面的 AUC（reid_cos：CONF vs REJ）
    pos = [p["reid_cos"] for p in pairs if p.get("label") == "CONF" and p.get("reid_cos") is not None]
    neg = [p["reid_cos"] for p in pairs if p.get("label") == "REJ" and p.get("reid_cos") is not None]
    auc = _auc(pos, neg)

    return {
        "params": params,
        "inference_params": INFERENCE_PARAMS,
        "summary": {k: len(v) for k, v in tiers.items()},
        "validation": {k: label_counts(v) for k, v in tiers.items()},
        "auto_precision": auto_prec,
        "auc_reid_cos": auc,
        "n_pairs": len(pairs),
        "auto_list": [f'{p["cam_a"]}:{p["tid_a"]}|{p["cam_b"]}:{p["tid_b"]}' for p in auto],
        "note": "基于已存分数重算分层；检测级参数需远程 GPU 重跑，本页不伪造。",
    }


def sweep(base: dict, key: str, values: list) -> dict:
    out = []
    for v in values:
        p = dict(DEFAULT_PARAMS)
        p.update(base or {})
        p[key] = v
        r = evaluate(p)
        out.append({
            "value": v,
            "AUTO": r["summary"]["AUTO"],
            "REVIEW": r["summary"]["REVIEW"],
            "DROP": r["summary"]["DROP"],
            "auto_precision": r["auto_precision"],
        })
    return {"key": key, "points": out}


def _auc(pos: list, neg: list):
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))
