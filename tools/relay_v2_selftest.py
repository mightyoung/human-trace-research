#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L0 单元自检：Relay v2 / F1 预算多对多分配（合成案例，无真实数据依赖）

案例（对齐 docs/relay-v2-plan.md §7.1）：
  T1 一人多次通行   —— 1 exit + 3 同人 entry（预算内）全部恢复
  T2 预算溢出+冲突  —— K_a 满后，高 sim(≥0.95) 挤出 → blocked；低分挤出 → 静默
  T3 K_b 回穿拦截   —— 同 entry 被 3 个 exit 指向，K_b=2 → 第 3 条 blocked
  T4 确定性排序     —— sim 相同按 dt 升序（小 dt 优先）
  T5 空池边界       —— 无高 sim 候选时不崩、输出为空

用法: python tools/relay_v2_selftest.py  （退出码 0=全过, 1=有失败）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relay_expand import expand_assign  # noqa: E402


def C(a, b, sim, dt=10.0):
    return {"a": a, "b": b, "sim": sim, "dt": dt,
            "dir": f"{a.split(':')[0]}->{b.split(':')[0]}"}


def keys(rows):
    return {(r["a"], r["b"]) for r in rows}


FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def t1():
    print("[T1] 一人多次通行：1 exit + 3 同人 entry（K_a=3 预算内全部恢复）")
    cands = [
        C("A:a0", "B:b0", 0.970),                       # primary
        C("A:a0", "B:b1", 0.960, dt=1.0),               # 同人 1
        C("A:a0", "B:b2", 0.955, dt=30.0),              # 同人 2
        C("A:a0", "B:b3", 0.946, dt=50.0),              # 同人 3
        C("A:a9", "B:b9", 0.968),                       # 无关 primary
    ]
    r = expand_assign(cands, 0.943, 3, 2, 0.95)
    check("primary 含 (a0,b0),(a9,b9)", keys(r["primary"]) == {("A:a0", "B:b0"), ("A:a9", "B:b9")})
    check("3 条同人全部恢复为 secondary",
          keys(r["secondary"]) == {("A:a0", "B:b1"), ("A:a0", "B:b2"), ("A:a0", "B:b3")})


def t2():
    print("[T2] 预算溢出：K_a=3 满后 sim≥0.95 → blocked；<0.95 → 静默")
    cands = [
        C("A:a0", "B:b0", 0.970),                       # primary
        C("A:a0", "B:b1", 0.965), C("A:a0", "B:b2", 0.960), C("A:a0", "B:b3", 0.955),
        C("A:a0", "B:b4", 0.951),                       # 挤出但 ≥0.95 → blocked
        C("A:a0", "B:b5", 0.948),                       # 挤出且 <0.95 → 静默淘汰
    ]
    r = expand_assign(cands, 0.943, 3, 2, 0.95)
    check("secondary = b1..b3", keys(r["secondary"]) == {("A:a0", "B:b1"), ("A:a0", "B:b2"), ("A:a0", "B:b3")})
    check("blocked 恰含 b4", keys(r["blocked"]) == {("A:a0", "B:b4")})
    check("b5 不出现（静默）", ("A:a0", "B:b5") not in keys(r["blocked"]) | keys(r["secondary"]))


def t3():
    print("[T3] K_b 回穿拦截：3 exit → 同 entry，K_b=2 → 第 3 条 blocked")
    cands = [
        # 干扰 primary（让 a1/a2/a3 各自有更强候选 → 其 b0 候选非 mutual-best）
        C("A:a1", "B:bX", 0.970), C("A:a2", "B:bY", 0.970), C("A:a3", "B:bZ", 0.970),
        C("A:a1", "B:b0", 0.960, dt=1.0),
        C("A:a2", "B:b0", 0.958, dt=2.0),
        C("A:a3", "B:b0", 0.956, dt=3.0),               # 第 3 条 → K_b 拦截
    ]
    r = expand_assign(cands, 0.943, 3, 2, 0.95)
    check("secondary 收 a1/a2→b0", keys(r["secondary"]) == {("A:a1", "B:b0"), ("A:a2", "B:b0")})
    check("blocked 含 a3→b0", keys(r["blocked"]) == {("A:a3", "B:b0")})


def t4():
    print("[T4] 确定性排序：同 sim 按 dt 升序（小 dt 优先），落选者 blocked")
    cands = [
        C("A:a0", "B:b0", 0.970),                       # primary
        C("A:a0", "B:b1", 0.960, dt=20.0),              # 大 dt → 落选
        C("A:a0", "B:b2", 0.960, dt=5.0),               # 小 dt → 入选
    ]
    r = expand_assign(cands, 0.943, 1, 2, 0.95)         # K_a=1
    check("secondary = b2（dt 小者）", keys(r["secondary"]) == {("A:a0", "B:b2")})
    check("b1 进 blocked", keys(r["blocked"]) == {("A:a0", "B:b1")})


def t5():
    print("[T5] 空池边界：无高分候选不崩、输出为空")
    cands = [C("A:a0", "B:b0", 0.90), C("A:a1", "B:b1", 0.80)]
    r = expand_assign(cands, 0.943, 3, 2, 0.95)
    check("全空", not r["primary"] and not r["secondary"] and not r["blocked"])


def t6():
    print("[T6] score 排序：低 sim 短 dt（主峰）胜出，高 sim 长 dt（长尾）让位")
    # 对应真实场景：E042→E046(0.9445, dt=1) vs E042→E050(0.9499, dt=47.6)
    c_short = C("A:a0", "B:b1", 0.9445, dt=1.0)
    c_long = C("A:a0", "B:b2", 0.9499, dt=47.6)
    c_short["score"] = 0.9445 + 0.01            # 主峰 w≈1 → +0.01
    c_long["score"] = 0.9499 + 0.01 * (-1.5)    # 长尾 w≈0.22 → -0.015
    cands = [C("A:a0", "B:b0", 0.970), c_short, c_long]
    cands[0]["score"] = 0.970
    r = expand_assign(cands, 0.943, 1, 2, 0.95, sort_key="score")   # K_a=1
    check("短 dt 者入选", keys(r["secondary"]) == {("A:a0", "B:b1")})
    check("长 dt 者让位", keys(r["secondary"]) == {("A:a0", "B:b1")} and
          ("A:a0", "B:b2") not in keys(r["secondary"]))


def main():
    print("=== Relay v2 / L0 自检（F1 分配） ===")
    t1(); t2(); t3(); t4(); t5(); t6()
    print(f"\n结果: {'全部通过 ✓' if not FAILS else f'{len(FAILS)} 项失败: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
