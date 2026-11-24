# -*- coding: utf-8 -*-
"""影子运行 harness（EVAL-5）：候选变更在沙盒执行 → 影响面表 + 生产文件字节级护栏。

设计（对应评测协议 §6 的"冻结窗口→候选→回放→离线影子→发布/回滚"）：
  - 所有 job 只在 `analysis/shadow/<时间戳>/` 下产出，绝不写生产路径；
  - 运行前后对 PROD_FILES 做 md5 对账：任何差异 → report.guard = FAIL（供回滚告警）；
  - 产出 shadow_report.json：每个 job 的命令、状态、**影响面摘要**（关键指标）、护栏结果。

支持 job：
  registry                  重跑 EVAL-1 规则库（沙盒副本）
  zone                      重跑 TOP-3 zone-pair（沙盒副本）
  topo                      重跑 TOP-1 窄窗显著性（long35，少量置换）
  replay:<panel>.<key>=<v1,v2>   参数扫描（如 replay:registry.sim_hi=0.945,0.955）

用法：
  python tools/shadow_run.py --job registry --job zone
  python tools/shadow_run.py --job "replay:registry.sim_hi=0.945,0.95,0.955,0.96"
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
PY = sys.executable

PROD_FILES = [
    os.path.join(ANA, "relay_v2", "edges_v2_gated.json"),
    os.path.join(ANA, "relay_v2", "edge_states.json"),
    os.path.join(ANA, "relay_v2", "scored_v2.json"),
    os.path.join(ANA, "c0", "c0_baseline_v7.json"),
    os.path.join(ANA, "gil", "store_v1", "gil.sqlite"),
    os.path.join(ANA, "topology", "portal_graph_v2.json"),
    os.path.join(ANA, "eval_registry", "registry_v0.json"),
    os.path.join(ANA, "selector_1012", "selector_v2.npz"),
]


def md5(p, bs=1 << 20):
    if not os.path.exists(p):
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def snapshot():
    return {p: md5(p) for p in PROD_FILES}


def run_cmd(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def job_registry(sb):
    out = os.path.join(sb, "registry")
    rc, log = run_cmd([PY, "tools/cluster_eval_registry.py", "--out", out])
    summary = None
    p = os.path.join(out, "registry_v0.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        summary = dict(action_counts=d["action_counts"], claimable=d["claimable_pool"],
                       agreement=d.get("agreement_summary"),
                       reconcile=f"conflict {d['reconcile_states']['conflict']}"
                                 f" (hard {d['reconcile_states'].get('hard_conflict')})")
    return dict(job="registry", rc=rc, output=os.path.relpath(p, ROOT) if os.path.exists(p) else None,
                summary=summary, log_tail=log[-400:])


def job_zone(sb):
    out = os.path.join(sb, "portal_graph_v2.json")
    rc, log = run_cmd([PY, "tools/portal_edge_confidence.py", "--out", out])
    summary = None
    if os.path.exists(out):
        d = json.load(open(out, encoding="utf-8"))
        summary = dict(n_zones=d["n_zones"], main=d["eval"]["main"])
    return dict(job="zone", rc=rc, output=os.path.relpath(out, ROOT) if os.path.exists(out) else None,
                summary=summary, log_tail=log[-400:])


def job_topo(sb):
    out = os.path.join(sb, "topo_frag_narrow_long35.json")
    data = os.path.join(ANA, "topology", "long35_data")
    rc, log = run_cmd([PY, "tools/topo_frag_narrow.py", data, out,
                       "--tag", "long35-shadow", "--shifts", "500"])
    summary = None
    if os.path.exists(out):
        d = json.load(open(out, encoding="utf-8"))
        k = [k for k in d if k.startswith("frag[")]
        main = d.get("frag[-1.0,10.0]") or d.get(k[0])
        summary = dict(zmax=main["zmax_obs"], p_family=main["p_family"],
                       n_z_ge3=main["n_z_ge3"])
    return dict(job="topo", rc=rc, output=os.path.relpath(out, ROOT) if os.path.exists(out) else None,
                summary=summary, log_tail=log[-400:])


def job_replay(sb, spec):
    """spec: panel.key=v1,v2  （如 registry.sim_hi=0.94,0.95）"""
    param, vals = spec.split("=", 1)
    out = os.path.join(sb, f"sweep_{param.replace('.', '_')}.json")
    rc, log = run_cmd([PY, "tools/replay_eval.py", "--sweep", param, "--values", vals, "--out", out])
    summary = None
    if os.path.exists(out):
        d = json.load(open(out, encoding="utf-8"))
        rows = d["rows"]
        summary = dict(param=param, n_rows=len(rows),
                       rows=[{k: r[k] for k in list(r)[:8]} for r in rows[:6]])
    return dict(job=f"replay:{param}", rc=rc,
                output=os.path.relpath(out, ROOT) if os.path.exists(out) else None,
                summary=summary, log_tail=log[-400:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", default=[],
                    help="registry | zone | topo | replay:panel.key=v1,v2")
    ap.add_argument("--sandbox", default=None)
    args = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sb = args.sandbox or os.path.join(ANA, "shadow", ts)
    os.makedirs(sb, exist_ok=True)

    before = snapshot()
    jobs = []
    for j in (args.job or ["registry", "zone"]):
        if j == "registry":
            r = job_registry(sb)
        elif j == "zone":
            r = job_zone(sb)
        elif j == "topo":
            r = job_topo(sb)
        elif j.startswith("replay:"):
            r = job_replay(sb, j.split(":", 1)[1])
        else:
            r = dict(job=j, rc=-1, summary=None, log_tail="unknown job")
        jobs.append(r)
        print(f"[{r['job']}] rc={r['rc']} out={r.get('output')}", flush=True)
        if r.get("summary"):
            print("   summary:", json.dumps(r["summary"], ensure_ascii=False)[:300], flush=True)

    after = snapshot()
    guard = {p: dict(before=before[p], after=after[p], unchanged=(before[p] == after[p]))
             for p in PROD_FILES}
    ok = all(v["unchanged"] for v in guard.values())

    report = dict(version="shadow/1", ts=ts, sandbox=os.path.relpath(sb, ROOT),
                  jobs=jobs, guard=dict(pass_=ok, detail=guard),
                  note="影子运行：实现不触生产；guard= 生产文件 md5 对账（全过才可进入发布评审）")
    rp = os.path.join(sb, "shadow_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"[guard] {'PASS' if ok else 'FAIL'}")
    print("saved", rp)


if __name__ == "__main__":
    main()
