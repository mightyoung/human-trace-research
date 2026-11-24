# -*- coding: utf-8 -*-
"""科研工作台 · 后端接口（/api/workbench/*）

权威库仍在本机 `ops/ops.db`（与 ops_api 同一连接方式）。本模块**可选**：
找不到台账时返回 503 + 原因，不影响既有页面。

职责边界（对齐 2026-11-23 研究报告 §08/§09/§10）：
* 实验卡 = 一个主要问题 + 基线 + 候选改动 + 主指标 + 失败条件 + 预算；
* 四关 A 立项 / B 数据与真值 / C 边界变化 / D 成果验收 只记**人的决定**，append-only；
* 状态机由人推进，A 通过→已批准、D 通过→已采纳 自动联动，避免「批了但没走」；
* 七件套交付物勾选与运行挂载只是**证据索引**，不代替原始预测与指标。

刻意不做的：不自动生成结论摘要、不把「已采纳」写成实验成功、不按缺失指标补零。
"""
from __future__ import annotations

import os
import sys
from typing import Any

from fastapi import APIRouter, Body, HTTPException

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

router = APIRouter(prefix="/api/workbench", tags=["workbench"])

_OPS_ERR: str | None = None
try:
    import ops_db as opdb          # noqa: E402
except Exception as e:            # pragma: no cover - 远程无台账时走这里
    opdb = None
    _OPS_ERR = f"{type(e).__name__}: {e}"

ACTOR = "console"
SEED_KEY = "exp_card_seed_20261123"

# 四关：人只在这些点做决定；通过后由执行器自动推进（报告 §09）
GATES = {
    "A": {"name": "立项", "auto": "按已定问题、对照、主指标、预算与改动范围，实现单项增量",
          "output": "实验卡、基线提交、成功与停止标准"},
    "B": {"name": "数据与真值", "auto": "复用已授权且版本不变的数据；执行器只读获准数据，真值留在独立评测器",
          "output": "来源、时间划分、哈希、评测器版本、例外说明"},
    "C": {"name": "边界变化", "auto": "修复与重试在既有预算内；改人脸用途 / 主指标 / 时间窗口 / 预算须单开分支",
          "output": "变更理由、影响范围、新旧实验编号"},
    "D": {"name": "成果验收", "auto": "由固定程序评分，分析 Agent 汇总正反证据；负责人决定采纳 / 继续 / 停止",
          "output": "原始预测、配对指标、失败样例、模型版本"},
}

DECISIONS = {"approve": "同意", "return": "退回修改", "hold": "暂不实施"}

EVIDENCE_KEYS = [
    ("card", "实验卡与审批事件"),
    ("diff", "源码提交与差分说明"),
    ("data", "数据切分、真值版本与权限"),
    ("model", "权重、配置、依赖与随机种子"),
    ("logs", "原始预测、训练与评测日志"),
    ("metrics", "定量指标与分组置信区间"),
    ("samples", "成功 / 失败样例与结论边界"),
]

STATE_FLOW = {
    "draft": {"proposed", "archived"},
    "proposed": {"approved", "revise", "archived"},
    "approved": {"building", "paused", "archived"},
    "building": {"testing", "paused", "failed"},
    "testing": {"running", "paused", "failed"},
    "running": {"scored", "paused", "failed"},
    "scored": {"reviewed", "revise", "failed"},
    "reviewed": {"accept_pending", "revise"},
    "accept_pending": {"adopted", "revise", "archived"},
    "revise": {"proposed", "archived"},
    "paused": {"building", "testing", "running", "archived"},
    "failed": {"revise", "archived"},
    "adopted": {"archived"},
    "archived": set(),
}
STATE_LABEL = {
    "draft": "草稿", "proposed": "待立项审批", "approved": "已批准",
    "building": "方案/编码", "testing": "自动测试", "running": "运行",
    "scored": "独立评测", "reviewed": "自动复核", "accept_pending": "待成果验收",
    "adopted": "已采纳", "revise": "需修订", "paused": "暂停",
    "failed": "失败", "archived": "已归档",
}


def _conn():
    if opdb is None:
        raise HTTPException(503, f"台账未部署（本机为主库，远程仅预览）：{_OPS_ERR}")
    if not os.path.isfile(opdb.DB_PATH):
        raise HTTPException(503, f"台账库不存在：{opdb.rel(opdb.DB_PATH)}，先跑 "
                                f"`python tools/ops_db.py init`")
    c = opdb.connect()
    opdb.init_db(c)
    _seed(c)
    return c


# ------------------------------------------------------------------ 种子卡
# 五张卡的正文来自 `docs/multicamera_reid_research_report_2026-11-23.html` §10。
# ★ 只种「计划」，不种任何成绩：截至 11-23 五张卡均未产生项目实测结果，state 停在
#   draft / proposed，metric_json 里没有任何数值。
SEED_CARDS: list[dict[str, Any]] = [
    {
        "id": "EXP-001",
        "title": "人员交汇后能否接回原身份",
        "question": "一次交汇后身份编号变错，先错在局部跟踪还是全局关联？把任一层换成单变量增量，能否接回原身份且不牺牲覆盖？",
        "hypothesis": "若全局编号先错，交汇期暂停原型写入可减少后续污染；若局部编号先错，延迟确认能提高恢复率但必须单报延迟。",
        "baseline": {"text": "固定检测器 / 权重 / 低置信框阈值；ByteTrack 运动+框重叠两阶段关联；同时记录局部 tid 与全局身份编号", "runs": []},
        "candidate": {"text": "只在「已定位到错号的那一层」改一处：全局层=交汇期冻结原型写入；局部层=延迟确认", "runs": []},
        "metric": {"primary": "交汇后恢复原 ID 的比例",
                   "secondary": ["IDF1", "AssA", "IDSW", "碎片数", "未关联率", "额外延迟"],
                   "failure": "仅靠「不再分配 ID」降低错号（覆盖同步下降）即判不通过；不报覆盖的单指标提升无效"},
        "budget": {"data": "DanceTrack 压测 + CHIRLA 室内遮挡片段复核", "human_min": None, "gpu_h": None},
        "roles": {
            "human": ["确定交汇事件起止、可接受滞后与碎片数", "抽看一组局部及全局错号", "冻结主指标与失败条件后才开最终测试"],
            "agent": ["导出双层编号并重放相同检测", "交付发生换号的帧、前后轨迹与差分评分", "按统一模板报覆盖与延迟"],
        },
        "risk": "标准 ByteTrack 默认不读 OSNet：若 OSNet 只在后端聚类使用，冻结原型改不动该环节输出 —— 先审计调用路径。",
        "state": "proposed",
    },
    {
        "id": "EXP-002",
        "title": "跨镜连续轨迹与聚集去重",
        "question": "重叠区与非重叠区分别需要哪类证据，才能既接对人又不把两人合成一个？",
        "hypothesis": "重叠区加标定几何优于纯外观；非重叠区加出入口与转移时间约束优于同一外观库无约束。",
        "baseline": {"text": "重叠区=纯外观匹配；非重叠区=同一外观库、无转移约束", "runs": []},
        "candidate": {"text": "重叠区=外观+标定几何；非重叠区=外观+入口/出口/转移时间约束（只用历史数据估计）", "runs": []},
        "metric": {"primary": "跨镜同人关联（接对/错接分开计）",
                   "secondary": ["漏接", "重叠区人数绝对误差（WILDTRACK）", "每人局部轨迹准确率"],
                   "failure": "三件事分别计算：每人局部轨迹准、双镜合计恰好数对、盲区交接正确 —— 合并成一个数即无效"},
        "budget": {"data": "CHIRLA（跨镜）+ WILDTRACK（重叠区计数）", "human_min": None, "gpu_h": None},
        "roles": {
            "human": ["确认摄像头拓扑、重叠区、可通行出入口、时间同步误差与真值定义"],
            "agent": ["按同一批跨镜交接计算接对与错接", "单独计算同一时刻重叠区计数误差"],
        },
        "risk": "时间同步误差与拓扑错误会先于算法失效；标定与通行时间不得用评测日数据估计。",
        "state": "draft",
    },
    {
        "id": "EXP-003",
        "title": "可信身份锚定能否帮助后续无脸片段",
        "question": "人脸锚定之后，来自独立轨迹、无人脸可用的片段能否被接回？",
        "hypothesis": "只有「确认当前身份 + 把当前服装/姿态写成该身份原型」才可能外溢到无脸片段；单纯当下融合不产生后续收益。",
        "baseline": {"text": "A 固定人体特征库", "runs": []},
        "candidate": {"text": "B 多帧人脸+人体轨迹配对，只确认当前身份；C 在 B 上写入当前服装/姿态原型（B、C 接收完全相同的锚定事件）", "runs": []},
        "metric": {"primary": "锚定之后、独立轨迹且无可用人脸片段的成功接回率（同等误合并风险下）",
                   "secondary": ["弃判率", "锚点准确率", "锚点覆盖率", "已见/新来分组", "是否换装分组"],
                   "failure": "若片段仍在同一条局部轨迹上，正确编号可能只是连续跟踪继承 → 必须移出主结论"},
        "budget": {"data": "CHIRLA 锚定事件 + 后续无脸独立轨迹", "human_min": None, "gpu_h": None},
        "roles": {
            "human": ["确认可用人脸与全身轨迹的配对条件", "冻结误合并容忍度", "抽检可用锚点与后续无脸事件"],
            "agent": ["记录锚定与写入时刻", "对同一组后续无脸独立轨迹比较固定库 / 仅确认 / 确认后写入"],
        },
        "risk": "用真值身份模拟锚点只能估上界；本项目实测人脸 AUC .777 < 外观 .942，覆盖才是瓶颈（可用脸 12.1%）。",
        "state": "draft",
    },
    {
        "id": "EXP-004",
        "title": "给相同人工时间，哪种纠错更有效",
        "question": "在固定人工分钟数下，逐轨迹核验、随机簇纠错、按风险排序的簇纠错，哪种传播得更准？",
        "hypothesis": "按风险排序的簇纠错在同等人工时间下带来更高的身份关系正确率，且误传播更少。",
        "baseline": {"text": "逐轨迹身份核验", "runs": []},
        "candidate": {"text": "随机簇纠错 / 按风险排序的簇纠错（共享同一冻结轨迹与初始簇）", "runs": []},
        "metric": {"primary": "单位人工分钟纠正的正确身份关系数",
                   "secondary": ["核验耗时", "误传播到其他轨迹的条数", "后续保留数据的关联质量"],
                   "failure": "真值只给独立评测器；模拟用户按真值点击只能估潜在收益 —— 声称省人工必须有真人计时"},
        "budget": {"data": "固定冻结的轨迹与初始簇（8 段 + long35）", "human_min": None, "gpu_h": None},
        "roles": {
            "human": ["定人工时间预算与操作指引", "核验影响范围并实记操作时间"],
            "agent": ["导出各策略任务清单与耗时", "记录每次纠错的改善与误传播轨迹关系"],
        },
        "risk": "本项目实测：可疑优先队列只有 18 簇（0.5%）可用，唯一有效信号是时空互斥；不要用「准静止+小框」当假框判据。",
        "state": "draft",
    },
    {
        "id": "EXP-005",
        "title": "周期性更新 OSNet 是否有额外收益",
        "question": "在等训练预算下，更新权重与更新记忆各自贡献多少？",
        "hypothesis": "记忆与权重分离可单独计价；若权重更新无增益，记忆侧结果仍可独立成立。",
        "baseline": {"text": "固定模型 + 固定记忆", "runs": []},
        "candidate": {"text": "只更新记忆 / 只更新权重 / 两者都更新；先只选一种伪标签方法，再单独加可信筛选与历史回放", "runs": []},
        "metric": {"primary": "未来观测日的同日配对增益",
                   "secondary": ["历史回归（遗忘）", "陌生人误接", "耗时", "新旧嵌入兼容维护成本"],
                   "failure": "跨日独立样本不足时只报初步趋势；「伪标签自洽」不得当作准确率"},
        "budget": {"data": "前 N 天可信轨迹 + 历史回放；封存未来窗口", "human_min": None, "gpu_h": None},
        "roles": {
            "human": ["冻结历史截止日、正式测试窗口、训练预算与候选晋级条件"],
            "agent": ["保存权重、伪标签与记忆库版本", "做四组等条件对照，检查中断恢复、历史遗忘与嵌入兼容"],
        },
        "risk": "本项目实测：域内微调在留出身份上 .595/.554（闭集 .986 是假象）；改权重后历史向量需重编码或版本隔离。",
        "state": "draft",
    },
]
SOURCE = "multicamera_reid_research_report_2026-11-23.html §10"


def _seed(c) -> None:
    if c.execute("SELECT 1 FROM meta WHERE k=?", (SEED_KEY,)).fetchone():
        return
    have = {r[0] for r in c.execute("SELECT id FROM exp_card")}
    for card in SEED_CARDS:
        if card["id"] in have:
            continue
        c.execute(
            "INSERT INTO exp_card(id, title, question, hypothesis, baseline_json, candidate_json, "
            "metric_json, budget_json, roles_json, evidence_json, runs_json, state, risk, source, "
            "created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (card["id"], card["title"], card["question"], card["hypothesis"],
             opdb.jdump(card["baseline"]), opdb.jdump(card["candidate"]),
             opdb.jdump(card["metric"]), opdb.jdump(card["budget"]),
             opdb.jdump(card["roles"]),
             opdb.jdump({"items": [{"k": k, "label": lb, "done": False, "note": ""}
                                   for k, lb in EVIDENCE_KEYS]}),
             opdb.jdump([]), card["state"], card["risk"], SOURCE,
             opdb.now(), opdb.now()))
        opdb.audit(c, ACTOR, "exp_card.seed", "exp_card", card["id"], None, {"source": SOURCE})
    c.execute("INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
              (SEED_KEY, opdb.now()))


# ------------------------------------------------------------------ 读
def _card_row(c, row) -> dict:
    d = dict(row)
    for f in ("baseline_json", "candidate_json", "metric_json", "budget_json",
              "roles_json", "evidence_json", "runs_json"):
        d[f[:-5]] = opdb.jload(d.pop(f), None)
    d["baseline"] = d["baseline"] or {"text": "", "runs": []}
    d["candidate"] = d["candidate"] or {"text": "", "runs": []}
    d["metric"] = d["metric"] or {"primary": "", "secondary": [], "failure": ""}
    d["budget"] = d["budget"] or {}
    d["roles"] = d["roles"] or {"human": [], "agent": []}
    d["evidence"] = (d["evidence"] or {}).get("items") or []
    d["runs"] = d["runs"] or []
    d["state_label"] = STATE_LABEL.get(d["state"], d["state"])
    d["next_states"] = sorted(STATE_FLOW.get(d["state"], set()))
    return d


def _gates_of(c, cid: str) -> tuple[dict, list]:
    """最新一条决定 + 全量记录（append-only，倒序）。"""
    rows = [dict(r) for r in c.execute(
        "SELECT gate, decision, note, actor, at FROM exp_gate WHERE card_id=? ORDER BY id", (cid,))]
    latest: dict[str, dict] = {}
    for r in rows:
        latest[r["gate"]] = r
    return latest, list(reversed(rows))


@router.get("/status")
def status():
    ok = opdb is not None and os.path.isfile(opdb.DB_PATH)
    return {"ok": ok, "db": opdb.rel(opdb.DB_PATH) if opdb else None, "error": None if ok else _OPS_ERR}


@router.get("/overview")
def overview():
    c = _conn()
    try:
        cards = [_card_row(c, r) for r in c.execute("SELECT * FROM exp_card ORDER BY id")]
        gates = {d["id"]: _gates_of(c, d["id"])[0] for d in cards}
        pending = [d["id"] for d in cards
                   if d["state"] in ("proposed", "accept_pending", "reviewed") or not gates[d["id"]]]
        by_state: dict[str, int] = {}
        for d in cards:
            by_state[d["state"]] = by_state.get(d["state"], 0) + 1
        return {
            "ok": True, "db": opdb.rel(opdb.DB_PATH), "gates": GATES,
            "decisions": DECISIONS, "states": STATE_LABEL,
            "counts": {"cards": len(cards), "pending": len(pending),
                       "adopted": by_state.get("adopted", 0),
                       "running": sum(by_state.get(k, 0) for k in ("building", "testing", "running"))},
            "pending": pending,
            "evidence_done": sum(1 for d in cards for it in d["evidence"] if it.get("done")),
            "evidence_total": len(cards) * len(EVIDENCE_KEYS),
            "generated_at": opdb.now(),
        }
    finally:
        c.close()


@router.get("/cards")
def cards():
    c = _conn()
    try:
        out = []
        for r in c.execute("SELECT * FROM exp_card ORDER BY id"):
            d = _card_row(c, r)
            latest, log = _gates_of(c, d["id"])
            d["gate_state"] = {g: latest.get(g) for g in GATES}
            d["gate_log"] = log
            out.append(d)
        return {"ok": True, "gates": GATES, "decisions": DECISIONS,
                "states": STATE_LABEL, "cards": out, "db": opdb.rel(opdb.DB_PATH)}
    finally:
        c.close()


# ------------------------------------------------------------------ 写
def _one(c, cid: str) -> dict:
    r = c.execute("SELECT * FROM exp_card WHERE id=?", (cid,)).fetchone()
    if not r:
        raise HTTPException(404, f"实验卡不存在：{cid}")
    d = _card_row(c, r)
    latest, log = _gates_of(c, cid)
    d["gate_state"] = {g: latest.get(g) for g in GATES}
    d["gate_log"] = log
    return d


def _set_state(c, cid: str, to: str, note: str | None, why: str) -> tuple[bool, str]:
    cur = c.execute("SELECT state FROM exp_card WHERE id=?", (cid,)).fetchone()["state"]
    if to == cur:
        return False, cur
    if to not in STATE_FLOW.get(cur, set()):
        raise HTTPException(400, f"非法流转 {cur} → {to}；允许：{sorted(STATE_FLOW.get(cur, set())) or '无'}")
    c.execute("UPDATE exp_card SET state=?, updated_at=? WHERE id=?", (to, opdb.now(), cid))
    opdb.audit(c, ACTOR, "exp_card.state", "exp_card", cid,
               {"state": cur}, {"state": to, "by": why, "note": note})
    return True, cur


@router.post("/cards/{cid}/gate")
def gate(cid: str, payload: dict = Body(...)):
    """记录一次关口决定；A 通过→已批准、D 通过→已采纳 自动联动。"""
    g = str(payload.get("gate") or "").strip().upper()
    dec = str(payload.get("decision") or "").strip()
    note = (payload.get("note") or "").strip()
    if g not in GATES:
        raise HTTPException(400, f"gate 必须是 {sorted(GATES)} 之一")
    if dec not in DECISIONS:
        raise HTTPException(400, f"decision 必须是 {sorted(DECISIONS)} 之一")
    if dec != "approve" and not note:
        raise HTTPException(400, "退回/暂缓必须写明理由（会进审批记录）")
    c = _conn()
    try:
        _one(c, cid)
        c.execute("INSERT INTO exp_gate(card_id, gate, decision, note, actor, at) VALUES(?,?,?,?,?,?)",
                  (cid, g, dec, note, ACTOR, opdb.now()))
        opdb.audit(c, ACTOR, f"exp_gate.{g}.{dec}", "exp_card", cid,
                   None, {"note": note})
        moved = None
        if g == "A" and dec == "approve":
            ok, cur = _set_state(c, cid, "approved", note, "gateA")
            moved = "approved" if ok else None
        elif g == "A" and dec == "return":
            row = c.execute("SELECT state FROM exp_card WHERE id=?", (cid,)).fetchone()
            if row["state"] in ("proposed", "approved"):
                ok, _ = _set_state(c, cid, "revise", note, "gateA-return")
                moved = "revise" if ok else None
        elif g == "D" and dec in ("approve", "return"):
            row = c.execute("SELECT state FROM exp_card WHERE id=?", (cid,)).fetchone()
            if row["state"] == "accept_pending":
                to = "adopted" if dec == "approve" else "revise"
                ok, _ = _set_state(c, cid, to, note, "gateD")
                moved = to if ok else None
        return {"ok": True, "moved_to": moved, "card": _one(c, cid)}
    finally:
        c.close()


@router.post("/cards/{cid}/state")
def set_state(cid: str, payload: dict = Body(...)):
    to = str(payload.get("state") or "").strip()
    if to not in STATE_FLOW:
        raise HTTPException(400, f"未知状态：{to}")
    c = _conn()
    try:
        d = _one(c, cid)
        if to in ("adopted",) and (d["gate_state"].get("D") or {}).get("decision") != "approve":
            raise HTTPException(400, "D 成果验收未通过，不能标为已采纳")
        ok, cur = _set_state(c, cid, to, payload.get("note"), "manual")
        return {"ok": True, "moved": ok, "from": cur, "card": _one(c, cid)}
    finally:
        c.close()


@router.post("/cards/{cid}/evidence")
def evidence(cid: str, payload: dict = Body(...)):
    """七件套勾选：只记「交了/没交 + 路径」，不判断交付质量。"""
    key = str(payload.get("key") or "").strip()
    if key not in dict(EVIDENCE_KEYS):
        raise HTTPException(400, f"key 必须是 {[k for k, _ in EVIDENCE_KEYS]} 之一")
    done = bool(payload.get("done"))
    note = (payload.get("note") or "").strip()
    c = _conn()
    try:
        d = _one(c, cid)
        items = d["evidence"]
        for it in items:
            if it["k"] == key:
                it["done"] = done
                if note:
                    it["note"] = note
        c.execute("UPDATE exp_card SET evidence_json=?, updated_at=? WHERE id=?",
                  (opdb.jdump({"items": items}), opdb.now(), cid))
        opdb.audit(c, ACTOR, "exp_card.evidence", "exp_card", cid,
                   {"key": key}, {"done": done, "note": note})
        return {"ok": True, "card": _one(c, cid)}
    finally:
        c.close()


@router.post("/cards/{cid}/run")
def attach_run(cid: str, payload: dict = Body(...)):
    """把一次运行挂到卡上（基线/变体/消融）。只校验 run 存在，不评价它。"""
    rid = str(payload.get("run_id") or "").strip()
    role = str(payload.get("role") or "variant").strip()
    if role not in ("baseline", "variant", "ablation"):
        raise HTTPException(400, "role 必须是 baseline / variant / ablation")
    if not rid:
        raise HTTPException(400, "run_id 必填")
    c = _conn()
    try:
        if not c.execute("SELECT 1 FROM run WHERE id=?", (rid,)).fetchone():
            raise HTTPException(404, f"run 不存在：{rid}")
        d = _one(c, cid)
        runs = [r for r in d["runs"] if r.get("run_id") != rid]
        runs.append({"run_id": rid, "role": role, "at": opdb.now()})
        c.execute("UPDATE exp_card SET runs_json=?, updated_at=? WHERE id=?",
                  (opdb.jdump(runs), opdb.now(), cid))
        opdb.audit(c, ACTOR, "exp_card.run", "exp_card", cid, None,
                   {"run_id": rid, "role": role})
        return {"ok": True, "card": _one(c, cid)}
    finally:
        c.close()
