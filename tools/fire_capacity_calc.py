#!/usr/bin/env python3
"""火焰/烟雾识别容量与误报预算计算器。

两种形态：
  campus  园区摄像头轮询（snapshot 为主）
  robot   机器狗车载（连续推理 + 停步复检）

所有模型延迟、单帧误报率均为「规划假设」，不是实测值；用 --latency-ms /
--fp-frame 替换为你自己的实测数。脚本只做算术，不做任何推断。

用法：
  python tools/fire_capacity_calc.py campus --cameras 200 --interval 5
  python tools/fire_capacity_calc.py campus --sweep
  python tools/fire_capacity_calc.py robot --fps 15 --hours 4
  python tools/fire_capacity_calc.py campus --cameras 200 --json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

SECONDS_PER_DAY = 86_400


@dataclass
class CampusResult:
    cameras: int
    interval_s: float
    aggregate_fps: float
    latency_ms: float
    gpu_load: float            # 单卡占用（0-1+），按串行等效
    cpu_cores: float           # 纯 CPU 路线所需等效核数（按 cpu_latency_ms 计）
    frames_per_cam_day: int
    candidate_fp_per_cam_day: float
    alarm_fp_per_cam_day: float
    alarms_per_day: float
    operator_minutes_per_day: float
    bandwidth_mbps: float
    storage_gb_per_day: float


def campus(
    cameras: int,
    interval_s: float,
    latency_ms: float,
    fp_frame: float,
    confirm_factor: float,
    snapshot_kb: float,
    clip_mb: float,
    review_min: float,
    gpu_fps_capacity: float,
    cpu_latency_ms: float = 110.0,
) -> CampusResult:
    aggregate_fps = cameras / interval_s
    frames_per_cam_day = int(round(SECONDS_PER_DAY / interval_s))
    candidate_fp = frames_per_cam_day * fp_frame
    alarm_fp = candidate_fp / confirm_factor
    alarms = cameras * alarm_fp
    return CampusResult(
        cameras=cameras,
        interval_s=interval_s,
        aggregate_fps=round(aggregate_fps, 2),
        latency_ms=latency_ms,
        gpu_load=round(aggregate_fps / gpu_fps_capacity, 3),
        cpu_cores=round(aggregate_fps * cpu_latency_ms / 1000 / 0.6, 2),  # 0.6 = 保守有效利用率
        frames_per_cam_day=frames_per_cam_day,
        candidate_fp_per_cam_day=round(candidate_fp, 3),
        alarm_fp_per_cam_day=round(alarm_fp, 4),
        alarms_per_day=round(alarms, 1),
        operator_minutes_per_day=round(alarms * review_min, 0),
        bandwidth_mbps=round(cameras * snapshot_kb * 8 / interval_s / 1000, 1),
        storage_gb_per_day=round(alarms * clip_mb / 1024, 2),
    )


@dataclass
class RobotResult:
    fps: float
    hours: float
    latency_ms: float
    frames_per_day: int
    candidate_fp_per_shift: float
    alarm_fp_per_shift: float
    compute_load: float
    power_w: float
    energy_wh_per_shift: float
    bandwidth_mbps: float


def robot(
    fps: float,
    hours: float,
    latency_ms: float,
    fp_frame: float,
    confirm_factor: float,
    edge_fps_capacity: float,
    power_w: float,
    bitrate_mbps: float,
) -> RobotResult:
    frames_per_day = int(round(fps * SECONDS_PER_DAY))
    candidate = fps * hours * 3600 * fp_frame
    return RobotResult(
        fps=fps,
        hours=hours,
        latency_ms=latency_ms,
        frames_per_day=frames_per_day,
        candidate_fp_per_shift=round(candidate, 3),
        alarm_fp_per_shift=round(candidate / confirm_factor, 4),
        compute_load=round(fps / edge_fps_capacity, 3),
        power_w=power_w,
        energy_wh_per_shift=round(power_w * hours, 1),
        bandwidth_mbps=bitrate_mbps,
    )


def fmt_campus(r: CampusResult) -> str:
    verdict = "OK" if r.gpu_load <= 0.7 else ("需加卡" if r.gpu_load > 1 else "接近上限")
    return "\n".join([
        f"园区轮询：{r.cameras} 路 @ 1 帧/{r.interval_s:g}s",
        f"  聚合帧率        {r.aggregate_fps:>8.1f} fps",
        f"  单卡占用(规划)  {r.gpu_load*100:>8.1f} %   [{verdict}]",
        f"  纯 CPU 等效核   {r.cpu_cores:>8.1f}",
        f"  每路每天帧数    {r.frames_per_cam_day:>8d}",
        f"  候选误报/路/天  {r.candidate_fp_per_cam_day:>8.3f}",
        f"  告警误报/路/天  {r.alarm_fp_per_cam_day:>8.4f}",
        f"  全园告警/天     {r.alarms_per_day:>8.1f}",
        f"  值班复核/天     {r.operator_minutes_per_day:>8.0f} 分钟",
        f"  抓图带宽        {r.bandwidth_mbps:>8.1f} Mbps",
        f"  事件存储        {r.storage_gb_per_day:>8.2f} GB/天",
        "",
        "注：告警数按「等效削减」估算，默认误报在时间/空间上可被规则削减；",
        "    真实误报相关时该系数会偏离，必须用连续录像回放实测，不可当独立性假设。",
    ])


def fmt_robot(r: RobotResult) -> str:
    verdict = "OK" if r.compute_load <= 0.7 else "需降帧/降分辨率"
    return "\n".join([
        f"机器狗：{r.fps:g} fps × {r.hours:g} h/班",
        f"  每天处理帧数    {r.frames_per_day:>8d}",
        f"  候选误报/班     {r.candidate_fp_per_shift:>8.3f}",
        f"  告警误报/班     {r.alarm_fp_per_shift:>8.4f}",
        f"  边缘算力占用    {r.compute_load*100:>8.1f} %   [{verdict}]",
        f"  计算功耗        {r.power_w:>8.1f} W",
        f"  每班计算耗电    {r.energy_wh_per_shift:>8.1f} Wh",
        f"  图传带宽        {r.bandwidth_mbps:>8.1f} Mbps",
    ])


def main() -> None:
    p = argparse.ArgumentParser(description="火焰/烟雾识别容量与误报预算")
    sub = p.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("campus", help="园区摄像头轮询")
    c.add_argument("--cameras", type=int, default=100)
    c.add_argument("--interval", type=float, default=5.0, help="每路抽帧间隔（秒）")
    c.add_argument("--latency-ms", type=float, default=8.0, help="单帧端到端延迟（GPU 规划值）")
    c.add_argument("--fp-frame", type=float, default=1e-4, help="负样本单帧误报率")
    c.add_argument("--confirm-factor", type=float, default=30.0,
                   help="时序/几何/多机规则对误报的等效削减倍数（非独立假设）")
    c.add_argument("--snapshot-kb", type=float, default=250.0)
    c.add_argument("--clip-mb", type=float, default=8.0)
    c.add_argument("--review-min", type=float, default=3.0, help="单条告警人工复核分钟")
    c.add_argument("--gpu-fps-capacity", type=float, default=400.0,
                   help="单卡在目标分辨率下的吞吐（规划值）")
    c.add_argument("--sweep", action="store_true", help="输出 25/50/100/200/500 路对比")
    c.add_argument("--json", action="store_true")
    c.add_argument("--cpu-latency-ms", type=float, default=110.0)

    r = sub.add_parser("robot", help="机器狗车载")
    r.add_argument("--fps", type=float, default=15.0)
    r.add_argument("--hours", type=float, default=4.0, help="每班巡逻小时")
    r.add_argument("--latency-ms", type=float, default=12.0)
    r.add_argument("--fp-frame", type=float, default=1e-4)
    r.add_argument("--confirm-factor", type=float, default=200.0,
                   help="停步+稳像+多帧复检的等效削减倍数（非独立假设）")
    r.add_argument("--edge-fps-capacity", type=float, default=60.0,
                   help="边缘模组吞吐（规划值，Orin NX 级）")
    r.add_argument("--power-w", type=float, default=18.0)
    r.add_argument("--bitrate-mbps", type=float, default=4.0)
    r.add_argument("--json", action="store_true")

    cap = sub.add_parser("capture", help="轮询采样覆盖概率 P=min(1, D/T)")
    cap.add_argument("--duration", type=float, default=1.0, help="事件可见时长（秒）")
    cap.add_argument("--interval", type=float, default=5.0, help="轮询间隔（秒）")
    cap.add_argument("--table", action="store_true", help="输出时长×间隔矩阵")
    cap.add_argument("--json", action="store_true")

    a = p.parse_args()

    if a.mode == "capture":
        if a.table:
            ds = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
            ts = (2.0, 5.0, 10.0, 30.0)
            head = "可见时长\\间隔 " + "".join(f"{t:>8g}s" for t in ts)
            print(head)
            print("-" * len(head))
            for d in ds:
                print(f"{d:>10g}s " + "".join(f"{min(1.0, d/t)*100:>8.0f}%" for t in ts))
            print("\n注：均匀随机相位、瞬时采样、检测器完美；周期性与轮询同频时假设失效。")
            return
        prob = min(1.0, a.duration / a.interval)
        out = {"duration_s": a.duration, "interval_s": a.interval,
               "capture_prob": round(prob, 4), "expected_wait_s": round(a.interval / 2, 2)}
        if a.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"可见时长 {a.duration:g}s / 轮询间隔 {a.interval:g}s")
            print(f"  采到概率      {prob*100:>6.1f}%")
            print(f"  平均等待      {a.interval/2:>6.1f}s（均匀相位假设）")
            print("  注：不含检测器漏检；事件与轮询同周期时可能恒漏或恒中。")
        return

    if a.mode == "campus":
        if a.sweep:
            rows = []
            for n in (25, 50, 100, 200, 500):
                res = campus(n, a.interval, a.latency_ms, a.fp_frame, a.confirm_factor,
                             a.snapshot_kb, a.clip_mb, a.review_min, a.gpu_fps_capacity,
                             a.cpu_latency_ms)
                rows.append(res)
            if a.json:
                print(json.dumps([asdict(x) for x in rows], ensure_ascii=False, indent=2))
                return
            header = f"{'路数':>5} {'聚合fps':>8} {'单卡占用':>9} {'告警/天':>9} {'复核min':>8} {'带宽Mbps':>9}"
            print(header)
            print("-" * len(header))
            for x in rows:
                print(f"{x.cameras:>5} {x.aggregate_fps:>8.1f} {x.gpu_load*100:>8.1f}% "
                      f"{x.alarms_per_day:>9.1f} {x.operator_minutes_per_day:>8.0f} {x.bandwidth_mbps:>9.1f}")
            print("\nCPU 路线（110 ms/帧，0.6 有效利用率）所需等效核：")
            for n in (25, 50, 100, 200, 500):
                print(f"  {n:>4} 路 → {n / a.interval * 0.11 / 0.6:>5.1f} 核")
            return
        res = campus(a.cameras, a.interval, a.latency_ms, a.fp_frame, a.confirm_factor,
                     a.snapshot_kb, a.clip_mb, a.review_min, a.gpu_fps_capacity,
                     a.cpu_latency_ms)
        print(json.dumps(asdict(res), ensure_ascii=False, indent=2) if a.json else fmt_campus(res))
        return

    res = robot(a.fps, a.hours, a.latency_ms, a.fp_frame, a.confirm_factor,
                a.edge_fps_capacity, a.power_w, a.bitrate_mbps)
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2) if a.json else fmt_robot(res))


if __name__ == "__main__":
    main()
