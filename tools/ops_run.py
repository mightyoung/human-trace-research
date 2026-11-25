#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ops run · 算法运行包装器

把任意一条已有命令变成一条**可追溯的 run**：自动采集参数/环境/git/host/gpu、
落日志、跑完登记产物与退出码。现有脚本不改，只在入口与出口加钩子。

用法
----
    # 跑一条命令（推荐：-- 之后的参数原样透传）
    python tools/ops_run.py --name wave4 --kind infer \
        --dataset ds_6cam_synced10 --out output_crosscam/analysis/wave4 \
        -- bash tools/run_wave4.sh

    # 本机小任务
    python tools/ops_run.py --name build-webdata --kind build \
        -- python webapp/dataprep/build_web_data.py

    # 只登记一条已经跑完的任务（补历史，不执行）
    python tools/ops_run.py --record-only --name legacy-wave4 --kind infer \
        --dataset ds_6cam_synced10 --out output_crosscam/analysis --status succeeded

    # 演练：只打印将执行的命令与采集结果，不落库
    python tools/ops_run.py --dry-run --name t -- python -c "print(1)"

约定
----
* **研究类 kind（train/eval/infer/encode）必须先挂实验卡**：`--card EXP-00x` 且该卡已过
  A 关（`exp_gate` 里 A 最新为 approve）；探索性运行用 `--no-card --note "理由"` 显式声明。
  两者都没给 → exit 4。`build/export/other` 不受此限。
* run_id 形如 run_20261102_131500_a1b2c3
* 日志落 ops/logs/<run_id>.log（相对路径入库）
* 产物扫 out_dir，只登记 uri + bytes + sha1（大文件超 --max-hash-mb 只记 bytes）
* 失败不掩盖：非零退出码 → status=failed，并原样返回该退出码
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops_db as db  # noqa: E402

ROOT = db.ROOT
LOG_DIR = os.path.join(ROOT, "ops", "logs")

# 不纳入产物登记的后缀/目录
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv"}
SKIP_EXT = {".pyc", ".log", ".tmp", ".part"}


# ------------------------------------------------------------------ 采集
def git_rev() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def code_fingerprint(cmd: list[str]) -> dict | None:
    """给本次真正执行的脚本算 sha1。本仓无 git 时，这是唯一的代码身份。"""
    for tok in cmd:
        if not tok.lower().endswith(".py"):
            continue
        p = tok if os.path.isabs(tok) else os.path.join(ROOT, tok)
        if not os.path.isfile(p):
            continue
        try:
            return {"path": db.rel(p), "sha1": db.sha1_file(p),
                    "bytes": os.path.getsize(p)}
        except OSError:
            return None
    return None


def gpu_info() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "cpu"
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=6)
        lines = [x.strip() for x in out.stdout.strip().splitlines() if x.strip()]
        return " | ".join(lines) if lines else "cpu"
    except Exception:
        return "cpu"


def env_fingerprint() -> dict:
    info = {
        "python": sys.version.split()[0],
        "executable": db.rel(sys.executable),
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "cwd": db.rel(os.getcwd()),
    }
    try:
        import torch  # noqa: F401
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
    except Exception:
        info["torch"] = None
    return info


def collect_params(args: argparse.Namespace, cmd: list[str]) -> dict:
    """参数快照：显式 tag + 透传命令。不解析子命令语义（避免猜测）。"""
    p = {"cmd": cmd}
    if args.tag:
        for kv in args.tag:
            if "=" in kv:
                k, v = kv.split("=", 1)
                p[k.strip()] = v.strip()
            else:
                p[kv.strip()] = True
    return p


# ------------------------------------------------------------------ 执行准入
# 研究类 run 会产出被引用的数字，必须先挂在一张**已过 A 关（立项）**的实验卡下；
# 没有卡就必须显式 --no-card，理由写进 note 存档，不能静默跑。
RESEARCH_KINDS = {"train", "eval", "infer", "encode"}
RUNNABLE_STATES = {"approved", "building", "testing", "running"}
STATE_BLOCK = {
    "draft": "草稿，尚未提交立项（A 关）",
    "proposed": "待立项审批，A 关未通过",
    "revise": "需修订，未重新提交",
    "paused": "已暂停；要续跑请先转回 building/testing/running",
    "failed": "已标记失败；请修订后重新过 A 关",
    "scored": "已进入独立评测，重跑会作废已登记的数字；先新建卡或回退到 running",
    "reviewed": "已进入自动复核，同上",
    "accept_pending": "待成果验收，同上",
    "adopted": "已采纳，不能原地重跑；新问题请开新卡",
    "archived": "已归档",
}


def gate_check(conn, card_id: str) -> tuple[bool, str]:
    """校验实验卡是否准入。返回 (ok, 说明)。"""
    row = conn.execute("SELECT id, state, title FROM exp_card WHERE id=?", (card_id,)).fetchone()
    if not row:
        have = [r["id"] for r in conn.execute("SELECT id FROM exp_card ORDER BY id LIMIT 12")]
        return False, f"实验卡 {card_id} 不存在。现有：{', '.join(have) or '（无）'}"
    state = (row["state"] or "draft").strip()
    if state not in RUNNABLE_STATES:
        why = STATE_BLOCK.get(state, "状态未知")
        return False, (f"实验卡 {card_id} 当前 state={state}，不可执行：{why}。\n"
                       f"      先在 #/workbench 过 A 关（立项审批），或确认这是探索性运行并改用 --no-card。")
    gates = conn.execute(
        "SELECT gate, decision FROM exp_gate WHERE card_id=? ORDER BY id DESC", (card_id,)).fetchall()
    latest = {}
    for g in gates:                        # append-only，最新一条即当前
        latest.setdefault(g["gate"], g["decision"])
    if latest.get("A") != "approve":
        return False, (f"实验卡 {card_id} 没有 A 关（立项）的 approve 记录（最新：{latest.get('A') or '无'}）。\n"
                       f"      state 可能是手改的；请在 #/workbench 补记 A 关决定。")
    return True, f"{card_id} state={state}，A 关已通过"


def link_card(conn, card_id: str, run_id: str, role: str | None) -> str:
    """把 run 追加到 exp_card.runs_json（幂等），返回说明。"""
    row = conn.execute("SELECT runs_json FROM exp_card WHERE id=?", (card_id,)).fetchone()
    runs = db.jload(row["runs_json"], []) if row else []
    if not isinstance(runs, list):
        runs = []
    if any((r or {}).get("run_id") == run_id for r in runs):
        return "已挂卡（重复登记已跳过）"
    runs.append({"run_id": run_id, "role": role or "explore"})
    conn.execute("UPDATE exp_card SET runs_json=?, updated_at=? WHERE id=?",
                 (db.jdump(runs), db.now(), card_id))
    db.audit(conn, "ops_run", "exp_card.attach_run", "exp_card", card_id, None,
             {"run_id": run_id, "role": role or "explore"})
    return f"已挂到 {card_id}（role={role or 'explore'}，第 {len(runs)} 条）"


# ------------------------------------------------------------------ 产物
def scan_artifacts(conn, run_id: str, out_dir: str, max_files: int, max_hash_mb: int,
                   since: float | None = None) -> dict:
    """扫产物目录。

    since（epoch 秒）非空时只收本次运行期间修改过的文件——复用同一 out_dir 时
    避免把上一次的产物重复登记到本次 run。
    """
    if not out_dir or not os.path.isdir(out_dir):
        return {"n": 0, "bytes": 0, "note": "out_dir 不存在，跳过"}
    picked: list[tuple[int, str]] = []
    for root, dirs, files in os.walk(out_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SKIP_EXT:
                continue
            p = os.path.join(root, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if since is not None and st.st_mtime < since - 1.0:
                continue
            picked.append((st.st_size, p))
    picked.sort(reverse=True)
    truncated = len(picked) > max_files
    picked = picked[:max_files]

    n, total = 0, 0
    for size, p in picked:
        if size > max_hash_mb * 1024 * 1024:
            # 大文件只登记大小，不计算 sha1（避免拖慢收尾）
            uri = db.rel(p)
            conn.execute(
                "INSERT INTO artifact(run_id, kind, uri, bytes, sha1, summary_json, at) "
                "VALUES(?,?,?,?,?,?,?)",
                (run_id, os.path.splitext(p)[1].lstrip(".").lower() or "file", uri, size,
                 None, db.jdump({"sha1": "skipped:large"}), db.now()))
            n += 1
            total += size
        else:
            if db.add_artifact(conn, run_id, p):
                n += 1
                total += size
    return {"n": n, "bytes": total, "truncated": truncated, "seen": len(picked)}


def find_metrics(out_dir: str) -> dict | None:
    """约定：out_dir 下的 metrics.json / metrics_full.json / summary.json 视为本次 run 的指标。

    写入前归一（db.normalize_metrics），这样训练页不必认识每个脚本的私有格式；
    原始内容完整保留在 metrics["_raw"]。

    顺序有讲究：`metrics_full.json`（intent_model_cc / social）比 `metrics.json`
    信息更全（含 per_cam / per_known / hist），所以优先取它。
    """
    for name in ("metrics_full.json", "metrics.json", "summary.json"):
        p = os.path.join(out_dir or "", name)
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return db.normalize_metrics(json.load(f))
            except Exception:
                return None
    return None


# ------------------------------------------------------------------ 执行
def attach_fingerprint(conn, rid: str, dataset_id: str | None, model_id: str | None,
                       env: dict) -> None:
    """记录输入指纹：数据快照 + 模型权重 + 执行脚本。

    有这组分量，日后 `tools/ops_stale.py check` 才能回答"这份结果是不是过期了"。
    任一分量缺失就不写总 hash（不假装算得准），但仍保留已有的分量。
    """
    code = (env or {}).get("code") or {}
    h, parts = db.input_fingerprint(conn, dataset_id, model_id, code.get("sha1"))
    if parts:
        conn.execute("UPDATE run SET input_hash=?, input_breakdown=? WHERE id=?",
                     (h, db.jdump(parts), rid))


def run_command(cmd: list[str], log_path: str, extra_env: dict | None = None) -> int:
    """执行命令，stdout/stderr 同时透传终端与日志。返回退出码。

    extra_env 注入子进程（OPS_RUN_ID 等），使训练脚本能把 epoch 指标写回台账。
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        head = f"$ {' '.join(cmd)}\n[{db.now()}] start\n"
        log.write(head)
        sys.stdout.write(head)
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=ROOT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
        tail = f"[{db.now()}] exit={proc.returncode}\n"
        log.write(tail)
        sys.stdout.write(tail)
        return proc.returncode


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(add_help=True, description="把命令包成可追溯的 run")
    ap.add_argument("--name", required=True, help="运行名，如 wave4")
    ap.add_argument("--kind", default="other",
                    choices=["infer", "encode", "train", "eval", "build", "export", "other"])
    ap.add_argument("--dataset", default=None, help="dataset id（不存在则轻量创建）")
    ap.add_argument("--dataset-root", default=None, help="dataset 根目录（配合 --dataset）")
    ap.add_argument("--dataset-dialect", default=None, help="口径，如 clean2")
    ap.add_argument("--model", default=None, help="model id（可空）")
    ap.add_argument("--out", default=None, help="产物目录（跑完扫这个目录）")
    ap.add_argument("--tag", action="append", help="参数快照 key=value，可多次")
    ap.add_argument("--note", default=None)
    ap.add_argument("--card", default=None, help="实验卡 id（EXP-00x）；研究类 run 必需")
    ap.add_argument("--role", default=None, choices=["baseline", "variant", "ablation", "explore"],
                    help="该 run 在卡内的角色（配合 --card）")
    ap.add_argument("--no-card", action="store_true",
                    help="探索性运行：不挂卡，但必须显式声明（理由写进 --note）")
    ap.add_argument("--status", default=None,
                    help="配合 --record-only：succeeded|failed|running")
    ap.add_argument("--record-only", action="store_true", help="只登记，不执行命令")
    ap.add_argument("--dry-run", action="store_true", help="演练：不落库、不执行")
    ap.add_argument("--max-artifacts", type=int, default=500)
    ap.add_argument("--max-hash-mb", type=int, default=512)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- 之后为要执行的命令")
    args = ap.parse_args(argv)

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    env = env_fingerprint()
    fp = code_fingerprint(cmd)
    if fp:
        env["code"] = fp
    params = collect_params(args, cmd)
    meta = {
        "name": args.name, "kind": args.kind, "params": params, "env": env,
        "host": env.get("host"), "gpu": gpu_info(), "git_rev": git_rev(),
        "dataset": args.dataset, "out": args.out,
    }

    if args.dry_run:
        print(json.dumps(meta, ensure_ascii=False, indent=1))
        print("\n将执行:", " ".join(cmd) if cmd else "(record-only)")
        # 演练也要能看到会不会被门禁拒（不落库）
        c2 = db.connect()
        try:
            db.init_db(c2)
            if args.card:
                ok, why = gate_check(c2, args.card)
                print("[ops] 准入预检：%s %s" % ("通过" if ok else "会被拒绝", why))
            elif args.kind in RESEARCH_KINDS and not args.no_card:
                print("[ops] 准入预检：会被拒绝（研究类需要 --card 或 --no-card）")
            else:
                print("[ops] 准入预检：不挂卡（--no-card 或非研究类）")
        finally:
            c2.close()
        return 0

    conn = db.connect()
    try:
        db.init_db(conn)

        # 执行准入门禁：研究类 run 必须先挂到一张已过 A 关的实验卡下
        if args.card:
            ok, why = gate_check(conn, args.card)
            if not ok:
                print(f"[ops] 拒绝启动：{why}", file=sys.stderr)
                return 4
            print(f"[ops] 准入：{why}")
        elif args.no_card or args.record_only:
            if not args.record_only:
                print("[ops] 准入：--no-card，本次不挂实验卡（已记入 run.note）")
                args.note = "[no-card] " + (args.note or "")
        elif args.kind in RESEARCH_KINDS:
            cards = [f"{r['id']} {r['state']}" for r in
                     conn.execute("SELECT id, state FROM exp_card ORDER BY id")]
            print(f"[ops] 拒绝启动：kind={args.kind} 属研究类，必须二选一：\n"
                  f"      --card EXP-00x（挂到已过 A 关的实验卡，推荐）\n"
                  f"      --no-card --note \"理由\"（探索性运行，显式声明）\n"
                  f"      现有卡：{', '.join(cards) or '（无，先在 #/workbench 建卡）'}", file=sys.stderr)
            return 4

        if args.dataset:
            db.upsert_dataset(conn, args.dataset, args.dataset,
                              root_uri=db.rel(args.dataset_root) if args.dataset_root else None,
                              dialect=args.dataset_dialect, note="ops_run 自动登记")

        # 并行防冲突：同一 out 目录同时被两个 run 写，产物会互相覆盖且无法追溯归谁
        if args.out:
            busy = conn.execute(
                "SELECT id, name, started_at FROM run WHERE out_dir=? AND status='running' "
                "ORDER BY started_at DESC LIMIT 1", (args.out,)).fetchone()
            if busy:
                print(f"[ops] 拒绝启动：out 目录被进行中的 run 占用\n"
                      f"      run_id={busy['id']} name={busy['name']} since={busy['started_at']}\n"
                      f"      {args.out}\n"
                      f"      换一个 --out，或确认对方已结束再重试。", file=sys.stderr)
                return 3
            done = conn.execute(
                "SELECT id, name, ended_at FROM run WHERE out_dir=? ORDER BY started_at DESC LIMIT 1",
                (args.out,)).fetchone()
            if done:
                print(f"[ops] 注意：out 目录被早先的 run 用过（{done['id']} {done['name']} "
                      f"@{done['ended_at']}），本次只登记新改动的文件。")

        if args.record_only:
            rid = db.create_run(conn, args.name, args.kind, entrypoint=" ".join(cmd) or None,
                                params=params, env=env, dataset_id=args.dataset,
                                model_id=args.model, host=env.get("host"), gpu=meta["gpu"],
                                git_rev=meta["git_rev"], out_dir=args.out,
                                status=args.status or "succeeded", note=args.note,
                                card_id=args.card, card_role=args.role)
            if args.card:
                print("[ops] " + link_card(conn, args.card, rid, args.role))
            attach_fingerprint(conn, rid, args.dataset, args.model, env)
            art = scan_artifacts(conn, rid, args.out, args.max_artifacts, args.max_hash_mb)
            db.finish_run(conn, rid, args.status or "succeeded", None,
                          metrics=find_metrics(args.out or ""))
            print(json.dumps({"run_id": rid, "artifacts": art}, ensure_ascii=False, indent=1))
            return 0

        if not cmd:
            print("缺少要执行的命令（-- 之后）", file=sys.stderr)
            return 2

        rid = db.create_run(conn, args.name, args.kind, entrypoint=" ".join(cmd),
                            params=params, env=env, dataset_id=args.dataset,
                            model_id=args.model, host=env.get("host"), gpu=meta["gpu"],
                            git_rev=meta["git_rev"], out_dir=args.out, note=args.note,
                            status="running", card_id=args.card, card_role=args.role)
        if args.card:
            print("[ops] " + link_card(conn, args.card, rid, args.role))
        attach_fingerprint(conn, rid, args.dataset, args.model, env)
        log_path = os.path.join(LOG_DIR, f"{rid}.log")
        conn.execute("UPDATE run SET log_uri=? WHERE id=?", (db.rel(log_path), rid))
        print(f"[ops] run_id={rid}\n[ops] log={db.rel(log_path)}\n[ops] gpu={meta['gpu']}")

        t0 = time.time()
        t_start = t0
        status, exit_code = "succeeded", None
        try:
            # PYTHONPATH 带上 tools/：训练脚本不在 tools/ 下（如 intent_cc/xxx.py）
            # 时也能 from ops_metric import Reporter
            _pp = [os.path.join(ROOT, "tools")]
            if os.environ.get("PYTHONPATH"):
                _pp.append(os.environ["PYTHONPATH"])
            exit_code = run_command(cmd, log_path,
                                    extra_env={"OPS_RUN_ID": rid, "OPS_DB": db.DB_PATH,
                                               "OPS_ROOT": ROOT,
                                               "PYTHONPATH": os.pathsep.join(_pp)})
            if exit_code != 0:
                status = "failed"
        except KeyboardInterrupt:
            status, exit_code = "killed", -2
            print("\n[ops] 收到中断，记为 killed", file=sys.stderr)
        except Exception:
            status, exit_code = "failed", -1
            traceback.print_exc()
        finally:
            art = scan_artifacts(conn, rid, args.out, args.max_artifacts, args.max_hash_mb,
                                 since=t_start)
            db.finish_run(conn, rid, status, exit_code,
                          metrics=find_metrics(args.out or ""), log_uri=db.rel(log_path))
            dur = round(time.time() - t0, 1)
            print(json.dumps({"run_id": rid, "status": status, "exit_code": exit_code,
                              "seconds": dur, "artifacts": art},
                             ensure_ascii=False, indent=1))
        return 0 if status == "succeeded" else (exit_code or 1)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
