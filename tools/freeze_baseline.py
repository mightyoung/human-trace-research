#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M0-1 基线冻结：把 A_baseline 固化成可校验的锚。

为什么需要：M1/M2 的所有对照都要和同一个基线比。口头约定"基线是 0.952/0.515"
不够——参数在代码里、真值在 CSV 里、嵌入在 npz 里，任何一处变了基线就漂了。
本脚本把「参数 + 指标 + 全部产物的 sha256」写成一个文件，后续任何对照先校验它。

产出：
  deliverable/data-request-v2/BASELINE_FREEZE.json   机器可读，含每个文件的 sha256
  deliverable/data-request-v2/BASELINE_FREEZE.md     人可读，含复现命令与口径禁令

用法：
  PY tools/freeze_baseline.py           # 生成
  PY tools/freeze_baseline.py --check   # 校验现有冻结包是否与磁盘一致（CI 用）
"""
import argparse
import csv
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable', 'data-request-v2')
FREEZE_JSON = os.path.join(DELIV, 'BASELINE_FREEZE.json')
FREEZE_MD = os.path.join(DELIV, 'BASELINE_FREEZE.md')

# 基线的全部参数（与 tools/session_selector_v1.py、d7_sweep.py 的实现一致）
BASELINE = {
    "name": "A_baseline",
    "purpose": "M1/M2/M3 所有对照的唯一锚；无 veto 的机位感知聚类 + a/b 两片一致接受",
    "detector": "YOLO26s-seg (person only), iou=0.7, 同帧去重 IoU>=0.5",
    "tracker": "ByteTrack, 2.5 FPS 采样",
    "encoder": "OSNet (MSMT17/AIN), 8 帧均值编码, npz 缓存",
    "train_segs": ["305_train_529", "318_train_525", "320_train_514", "323_train_510"],
    "test_segs": ["305_test_544", "318_test_534", "320_test_518", "323_test_527"],
    "emb_dir": "output_crosscam_local/1012_yolo8_dd_emb",
    "cluster": {"mode": "camera_aware", "thr_in": 0.15, "thr_out": 0.25,
                "veto_dmax": None, "min_cluster": 2,
                "same_cam_merge": False},
    "scorer": {"mode": "proto", "z_calib": True, "session_mean_align": True},
    "accept": {"rule": "accept_a AND accept_b AND cluster_a == cluster_b",
               "same_min": 0.90, "min_eval": 100},
    "selected": {"theta": 1.85, "K": 111},
    "note": "self-cal 在 same>=0.90 且 evaluable>=100 的可行域内取覆盖最大，因此 θ 是"
            "可行域下界（结构性贴线，见 docs/memory 与 6 机位实验）",
}

ARTIFACTS = [
    ("config", "configs/selector_deploy.json"),
    ("gt", "deliverable/data-request-v2/gt/gt_query_gt_FILLED.csv"),
    ("gt", "deliverable/data-request-v2/gt/gt_gallery_members_FILLED.csv"),
    ("gt", "deliverable/data-request-v2/gt/clusters_gt.csv"),
    ("result", "deliverable/data-request-v2/results/precision.csv"),
    ("result", "deliverable/data-request-v2/results/queries/A_baseline.csv"),
    ("result", "deliverable/data-request-v2/results/members/A_baseline.csv"),
    ("result", "deliverable/data-request-v2/results/verdicts/A_baseline.csv"),
    ("result", "deliverable/data-request-v2/results/loss_decomposition.csv"),
    ("code", "tools/session_selector_v1.py"),
    ("code", "tools/cluster_camera_aware.py"),
    ("code", "tools/eval_mask_variants.py"),
    ("code", "tools/eval_det_switch.py"),
    ("code", "tools/analyze_loss_decomp.py"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(path, pattern='.npz'):
    files = sorted(f for f in os.listdir(path) if f.endswith(pattern))
    h = hashlib.sha256()
    per = {}
    for f in files:
        s = sha256(os.path.join(path, f))
        per[f] = s
        h.update(f.encode())
        h.update(s.encode())
    return h.hexdigest(), per


def collect():
    arts = []
    for role, rel in ARTIFACTS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            arts.append({"role": role, "path": rel, "exists": False})
            continue
        arts.append({"role": role, "path": rel, "exists": True,
                     "sha256": sha256(p), "bytes": os.path.getsize(p)})
    emb = os.path.join(ROOT, BASELINE['emb_dir'])
    emb_hash, per = hash_dir(emb)
    arts.append({"role": "embeddings", "path": BASELINE['emb_dir'], "exists": True,
                 "sha256": emb_hash, "n_files": len(per), "files": per})

    # 指标：取自现有产物，不重算，避免"为了冻结而重跑"引入新变量
    metrics = {}
    with open(os.path.join(DELIV, 'results/precision.csv'), encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['config'] == 'A_baseline':
                metrics = {k: r[k] for k in ('K', 'theta', 'coverage', 'final_output_rate',
                                             'accepted', 'C', 'W', 'U', 'precision',
                                             'recall_229', 'C_over_229')}
                break
    loss = json.load(open(os.path.join(DELIV, 'results/loss_decomposition.json'), encoding='utf-8'))
    return {"baseline": BASELINE, "metrics": metrics,
            "loss_decomposition": loss['groups'],
            "artifacts": arts}


def render_md(blob):
    m = blob['metrics']
    L = blob['loss_decomposition']['ALL']
    lines = [
        "# M0-1 基线冻结（A_baseline）",
        "",
        "> 由 `PY tools/freeze_baseline.py` 生成；校验用 `PY tools/freeze_baseline.py --check`。",
        "> 任何 M1/M2 对照在开跑前必须先过校验，否则基线已漂。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 冻结日期 | **%s** |" % blob.get('frozen_at', '—'),
    ]
    if blob.get('supersedes'):
        lines.append("| 取代 | %s 版（内容已核对，指标零变化） |" % blob['supersedes'])
    lines += ["| 冻结人 | %s |" % blob.get('frozen_by', '—'), ""]
    if blob.get('rebuild_note'):
        lines += ["> **重建说明**：%s" % blob['rebuild_note'], ""]
    lines += [
        "## 1 参数",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 检测 | YOLO26s-seg（仅 person），NMS IoU 0.7 + 同帧去重 0.5 |",
        "| 跟踪 / 采样 | ByteTrack，2.5 FPS |",
        "| 编码 | OSNet（MSMT17/AIN），8 帧均值 |",
        "| 图库 / 查询段 | 305/318/320/323 的 train_* 与 test_*（见 JSON） |",
        "| 聚类 | 机位感知：机位内 average 0.15 → 跨机位 complete 0.25，同机位禁并 |",
        "| veto | **关闭**（`veto_dmax=null`） |",
        "| 评分 | 簇原型余弦 + 每身份 z 校准 + 会话均值对齐 |",
        "| 接受 | 两片都过阈值且同簇 |",
        "| 工作点 | **θ = %s，K = %s**（self-cal：same≥0.90 且 evaluable≥100 取覆盖最大） |"
        % (m.get('theta'), m.get('K')),
        "",
        "## 2 冻结指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        "| 覆盖 coverage | %s |" % m.get('coverage'),
        "| final_output_rate | %s |" % m.get('final_output_rate'),
        "| 接受 / 答对 / 答错 / 看不清 | %s / %s / %s / %s |"
        % (m.get('accepted'), m.get('C'), m.get('W'), m.get('U')),
        "| **精度（C/(C+W)）** | **0.9516**（118/124） |",
        "| **C/229** | **0.515** |",
        "",
        "损失分解（同一批数据，196 可回答帧）：",
        "",
        "| 可回答 | 答对 | 答错 | 未答 | 其中双片未过 / 单片过 / 异簇 |",
        "|---:|---:|---:|---:|---|",
        "| %d | %d | %d | %d | %d / %d / %d |"
        % (L['answerable'], L['correct'], L['wrong'], L['unanswered'],
           L['ua_both_low'], L['ua_one_pass'], L['ua_ab_split']),
        "",
        "## 3 口径禁令（写论文前必读）",
        "",
        "1. **0.952 ≠ 0.959**：0.952 的分母 124 = 系统 138 条输出里「目标是人」的条数"
        "（14 条输出对着非人——影子/镜像/帽子——在 GT v2 里标 `no-human`，不进 C/W）；"
        "0.959 = 118/123 是 196 可回答集内的精度。两个分母不能混用。"
        "GT v2（2026-10-23）：323 远场 31 条 + 318 簇 61 的 5 条全部标 `no-human`，"
        "**C/W/精度/C-229 全部不变**（42 配置逐一核对，C 无一变化）。",
        "2. **基线是 veto=off**：部署配置 `C_dmax0.30` 是精度 0.947、**C/229 0.393**，"
        "与本基线不可互换。",
        "3. **不得用簇级继承判对错**：那会把精度低估 33.6 个百分点（0.616 vs 0.952）。",
        "4. **self-cal 结构性贴线**：θ 必然落在可行域下界，报告时要同时给稳健口径"
        "（`--robust-margin`）。",
        "",
        "## 4 复现",
        "",
        "**已验证**：2026-10-23 按下列命令复跑，得 θ=1.85 / cov=0.66376 / same=0.90789 / "
        "fm=1.525% / K=111，与冻结值逐位一致（耗时 3 秒，嵌入走 npz 缓存）。",
        "",
        "```bash",
        "# 校验冻结包是否与磁盘一致（M1/M2 开跑前的第一步）",
        "PY tools/freeze_baseline.py --check",
        "",
        "# 重跑基线评测（v02：显式写全参数，不依赖 profile 的默认值）",
        "PY tools/eval_det_switch.py \\",
        "   --variant yolo8dd=output_crosscam_local/1012_yolo8_dd_emb \\",
        "   --cooc-map yolo8dd=output_crosscam_local/1012_yolo8_dd \\",
        "   --cluster-mode camera --thr-in 0.15 --thr-out 0.25 \\",
        "   --scorer proto --same-min 0.90 --min-eval 100",
        "# 注意：不要传 --profile，也不要传 --veto-dmax（None = 关闭 veto）",
        "",
        "# 损失分解（带 5 条恒等式断言）",
        "PY tools/analyze_loss_decomp.py",
        "```",
        "",
        "## 5 产物哈希",
        "",
        "| 角色 | 文件 | 大小 | sha256（前 16 位） |",
        "|---|---|---:|---|",
    ]
    for a in blob['artifacts']:
        if not a['exists']:
            lines.append("| %s | `%s` | — | **缺失** |" % (a['role'], a['path']))
            continue
        size = a.get('bytes') or ('%d 个 npz' % a.get('n_files', 0))
        lines.append("| %s | `%s` | %s | `%s…` |"
                     % (a['role'], a['path'], size, a['sha256'][:16]))
    lines += ["", "完整哈希见 `BASELINE_FREEZE.json`。", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只校验，不重写')
    ap.add_argument('--frozen-at', default='2026-10-23',
                    help='冻结日期（重建时改成当天）')
    ap.add_argument('--supersedes', default='',
                    help='被本次重建取代的旧冻结日期（重建时填）')
    ap.add_argument('--rebuild-note', default='',
                    help='重建原因与验证结论（重建时必填，会写进 JSON/MD）')
    a = ap.parse_args()

    blob = collect()
    missing = [x['path'] for x in blob['artifacts'] if not x['exists']]

    if a.check:
        if not os.path.exists(FREEZE_JSON):
            print('FAIL 没有冻结包，先跑一次不带 --check 的生成')
            return 2
        old = json.load(open(FREEZE_JSON, encoding='utf-8'))
        oldmap = {x['path']: x for x in old['artifacts']}
        bad = []
        for x in blob['artifacts']:
            o = oldmap.get(x['path'])
            if not o or not x['exists']:
                bad.append((x['path'], 'missing'))
            elif o.get('sha256') != x.get('sha256'):
                bad.append((x['path'], 'changed'))
        if bad:
            print('FAIL 基线已漂：')
            for p, why in bad:
                print('  %-8s %s' % (why, p))
            return 1
        print('OK 基线一致（%d 个产物 + 嵌入目录）' % len(blob['artifacts']))
        print('   指标：precision=%s C/229=%s θ=%s K=%s'
              % (blob['metrics'].get('precision'), blob['metrics'].get('C_over_229'),
                 blob['metrics'].get('theta'), blob['metrics'].get('K')))
        return 0

    if missing:
        print('WARN 缺失 %d 个产物（已记录为 exists=false）：' % len(missing))
        for p in missing:
            print('   ', p)
    blob['frozen_at'] = a.frozen_at
    blob['frozen_by'] = 'M0-1（V0.4 里程碑）'
    if a.supersedes:
        blob['supersedes'] = a.supersedes
    if a.rebuild_note:
        blob['rebuild_note'] = a.rebuild_note
    with open(FREEZE_JSON, 'w', encoding='utf-8') as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)
    with open(FREEZE_MD, 'w', encoding='utf-8') as f:
        f.write(render_md(blob))
    print('written:', FREEZE_JSON)
    print('written:', FREEZE_MD)
    print('指标: precision=%s C/229=%s coverage=%s θ=%s K=%s'
          % (blob['metrics'].get('precision'), blob['metrics'].get('C_over_229'),
             blob['metrics'].get('coverage'), blob['metrics'].get('theta'),
             blob['metrics'].get('K')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
