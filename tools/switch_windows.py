#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-A 共用：46 条串人轨迹的「复核窗口」计算（渲染与页面必须用同一份）。

窗口规则（先定窗口，再画框，最后页面按同一份 JSON 播放）：

1. 原记录点可用（落在轨迹内部）→ focus = 记录点 ±8s
2. 原记录点不可用（末帧/越界/无法解析）→ focus = 轨迹末 20s
3. **覆盖兜底**：若窗口内目标检测覆盖率 < 30%，在该轨迹范围内滑动窗口，
   取检测点最多的一段（步长 1s）。原因：部分 tid 是断裂的（例如
   `318_train_525#1041` 只有 25 个检测点、中间断 580 帧），
   照 1/2 取窗口会得到一段几乎全是空画面的视频，人工没法看。

scan = 整条轨迹 `[t_start−1s, t_end+1s]`，5fps 抽样（时长与真实一致）。

用法：
  PY tools/switch_windows.py            # 重算并写 gt/switch_windows.json
  PY tools/switch_windows.py --show     # 只打印
"""
import bisect
import csv
import json
import os

FPS = 25.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
LEDGER = os.path.join(DELIV, 'ledger/tracks.csv')
GAL = os.path.join(DELIV, 'gt/gt_gallery_members_FILLED.csv')
DET_DIR = os.path.join(ROOT, 'output_crosscam_local/1012_yolo8_dd')
CACHE = os.path.join(DELIV, 'gt/switch_windows.json')

FOCUS_PAD_S = 8.0
FOCUS_TAIL_S = 20.0
MIN_COVER = 0.30


class DetStore:
    """按段缓存逐帧检测（生产检测 stride=10）。"""

    def __init__(self, seg):
        p = os.path.join(DET_DIR, seg, '%s_detections.jsonl' % seg)
        self.frames = []
        self.by_frame = {}
        with open(p, encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                self.frames.append(d['frame'])
                self.by_frame[d['frame']] = {int(t): b for t, b in zip(d['track_ids'], d['boxes'])}
        self.span = (self.frames[1] - self.frames[0]) if len(self.frames) > 1 else 10

    def nearest(self, frame):
        i = bisect.bisect_left(self.frames, frame)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self.frames)]
        j = min(cands, key=lambda k: abs(self.frames[k] - frame))
        return self.frames[j], self.by_frame[self.frames[j]]

    def series(self, tid):
        return [(f, self.by_frame[f][tid]) for f in self.frames if tid in self.by_frame[f]]


def target_box(series, frame, max_gap=40, max_hold=30):
    """线性插值出目标轨迹在该帧的框；缺口过大返回 None。"""
    if not series:
        return None
    fs = [x[0] for x in series]
    i = bisect.bisect_left(fs, frame)
    if i < len(series) and fs[i] == frame:
        return series[i][1]
    prev = series[i - 1] if i > 0 else None
    nxt = series[i] if i < len(series) else None
    if prev and nxt and (nxt[0] - prev[0]) <= max_gap:
        a = (frame - prev[0]) / float(nxt[0] - prev[0])
        return [prev[1][k] + a * (nxt[1][k] - prev[1][k]) for k in range(4)]
    if prev and (frame - prev[0]) <= max_hold:
        return prev[1]
    if nxt and (nxt[0] - frame) <= max_hold:
        return nxt[1]
    return None


def coverage(series, fs, fe, span):
    if fe <= fs:
        return 0.0
    n = sum(1 for f, _ in series if fs <= f < fe)
    return n / float(max(1, (fe - fs) / float(span)))


def best_window(series, lo, hi, length, span, step=25):
    """在 [lo, hi) 内滑动长度 length 的窗口，返回检测点最多的起点。"""
    if hi - lo <= length:
        return lo
    fs_list = [f for f, _ in series]
    best_s, best_n = lo, -1
    s = lo
    while s + length <= hi:
        n = sum(1 for f in fs_list if s <= f < s + length)
        if n > best_n:
            best_s, best_n = s, n
        s += step
    return best_s


def load_items():
    led = {r['track_id']: r for r in csv.DictReader(open(LEDGER, encoding='utf-8-sig'))}
    gt = [r for r in csv.DictReader(open(GAL, encoding='utf-8-sig'))
          if r['id_switch'].strip() in ('1', '1.0')]
    stores = {}
    items = []
    for r in gt:
        t = led[r['track_id']]
        seg = t['seg']
        det = stores.get(seg)
        if det is None:
            det = stores[seg] = DetStore(seg)
        series = det.series(int(t['tid']))

        t0f, t1f = int(t['t_start_frame']), int(t['t_end_frame'])
        raw = (r['id_switch_frame'] or '').strip()
        num = ''.join(ch if ch.isdigit() else ' ' for ch in raw).split()
        old = int(num[0]) if num else None
        usable = old is not None and (t0f + 50) <= old <= (t1f - 50)

        if usable:
            fs = max(t0f - 25, old - int(FOCUS_PAD_S * FPS))
            fe = min(t1f + 25, old + int(FOCUS_PAD_S * FPS))
            mode = 'record'
        else:
            fs = max(t0f - 25, t1f - int(FOCUS_TAIL_S * FPS))
            fe = t1f + 25
            mode = 'tail'
        if fe - fs < int(4 * FPS):
            fs = max(0, fe - int(4 * FPS))

        cov = coverage(series, fs, fe, det.span)
        slid = False
        if cov < MIN_COVER and (t1f - t0f) > (fe - fs):
            s2 = best_window(series, max(0, t0f - 25), t1f + 25, fe - fs, det.span)
            if s2 != fs:
                c2 = coverage(series, s2, s2 + (fe - fs), det.span)
                if c2 > cov + 0.05:
                    fs, fe, slid, cov, mode = s2, s2 + (fe - fs), True, c2, 'slide'

        if mode == 'record':
            label = '以原记录点为中心 ±%ds' % int(FOCUS_PAD_S)
        elif mode == 'tail':
            label = '轨迹末 %.0fs' % ((fe - fs) / FPS)
        else:
            label = '检测最密处 %.0fs' % ((fe - fs) / FPS)

        items.append(dict(
            track_id=r['track_id'], file_base=r['track_id'].replace('#', '_'),
            seg=seg, cam=t['cam'], tid=int(t['tid']), n_frames=int(t['n_frames']),
            t0_f=t0f, t1_f=t1f, t0=t0f / FPS, t1=t1f / FPS,
            old_frame=old, old_usable=usable,
            focus_start=fs, focus_end=fe, scan_start=max(0, t0f - 25), scan_end=t1f + 25,
            win_mode=mode, coverage=round(cov, 3), win_label=label,
            n_det=len(series)))
    return items


def save(items, path=CACHE):
    json.dump(items, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return path


def load_or_make(path=CACHE):
    if os.path.exists(path):
        return json.load(open(path, encoding='utf-8'))
    items = load_items()
    save(items, path)
    return items


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--out', default=CACHE)
    a = ap.parse_args()
    items = load_items()
    if not a.show:
        save(items, a.out)
    n_slide = sum(1 for it in items if it['win_mode'] == 'slide')
    print('%d 条：记录点可用 %d ｜ 末段兜底 %d ｜ 滑动到最密处 %d'
          % (len(items), sum(1 for it in items if it['win_mode'] == 'record'),
             sum(1 for it in items if it['win_mode'] == 'tail'), n_slide))
    if a.show or n_slide:
        for it in items:
            if it['win_mode'] != 'record':
                print('  %-22s %-6s %s  cov=%.2f  %d–%d'
                      % (it['track_id'], it['win_mode'], it['win_label'],
                         it['coverage'], it['focus_start'], it['focus_end']))
    if not a.show:
        print('→ %s' % a.out)


if __name__ == '__main__':
    main()
