#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练脚本 → 台账的指标上报口（可选依赖，永不阻断训练）。

训练脚本只需要三行：

    from ops_metric import Reporter
    rep = Reporter()                       # 读环境变量 OPS_RUN_ID / OPS_DB
    rep.metric(ep, loss=..., val_acc=..., val_f1=...)
    rep.close()

行为：
- 有 OPS_RUN_ID 且 ops.db 可写 → 每个 epoch 写一条 run_metric（幂等，同 step 覆盖）
- 没有 OPS_RUN_ID（直接跑脚本）→ 完全 no-op，只在 enabled=False
- ops.db 不存在 / 表缺失 / 写失败 → 打印一次告警后转为 no-op，**绝不抛异常、绝不中断训练**

用法（推荐走包装器，自动注入 OPS_RUN_ID）：

    python tools/ops_run.py --name intent-v3 --kind train --out <dir> -- \
        python tools/intent_model.py --train ... --val ... --test ... --out-dir <dir>
"""
from __future__ import annotations

import os
import sys


class Reporter:
    """把 epoch 级指标写进台账；不可用时静默降级。"""

    def __init__(self, run_id: str | None = None, db_path: str | None = None,
                 quiet: bool = False):
        self.run_id = run_id or os.environ.get("OPS_RUN_ID") or ""
        self.db_path = db_path or os.environ.get("OPS_DB") or ""
        self.quiet = quiet
        self.enabled = False
        self.written = 0
        self._conn = None
        self._db = None
        self._warned = False
        if self.run_id:
            self._open()

    # ------------------------------------------------------------ 内部
    def _warn(self, msg: str) -> None:
        if self._warned:
            return
        self._warned = True
        if not self.quiet:
            print(f"[ops_metric] {msg}，指标不回写台账（训练继续）", file=sys.stderr)

    def _open(self) -> None:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import ops_db as db  # noqa: WPS433  (延迟导入，避免训练脚本强依赖)
            path = self.db_path or db.DB_PATH
            if not os.path.isfile(path):
                self._warn(f"台账不存在: {path}")
                return
            self._db = db
            self._conn = db.connect(path)
            self.enabled = True
        except Exception as e:                       # noqa: BLE001
            self._warn(f"台账不可用（{type(e).__name__}: {e}）")

    # ------------------------------------------------------------ 公开
    def metric(self, step: int, **values: float) -> None:
        """写一个 step 的多个指标。values 形如 loss=0.4, val_acc=0.83。"""
        if not self.enabled or not self._conn:
            return
        try:
            for name, val in values.items():
                if val is None:
                    continue
                self._db.set_metric(self._conn, self.run_id, int(step), name, float(val))
            self._conn.commit()
            self.written += 1
        except Exception as e:                       # noqa: BLE001
            self._warn(f"写入失败（{type(e).__name__}: {e}）")
            self.enabled = False

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:                        # noqa: BLE001
                pass
            self._conn = None
        self.enabled = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
