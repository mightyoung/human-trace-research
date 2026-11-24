"""生成 MANIFEST.md：文件树 + 哈希 + 复跑说明。"""
import argparse
import datetime as dt
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "deliverable", "data-request-v2")

_ap = argparse.ArgumentParser()
_ap.add_argument("--dir", default=D, help="要生成清单的目录（默认源目录）")
_ap.add_argument("--noimage", action="store_true",
                 help="去图版：人像拼图未随包分发")
_a = _ap.parse_args()
D = _a.dir
NOIMAGE = _a.noimage

# 去图版：staging 里没有 crops，被移出的规模从源目录统计
REAL_CROP_N = REAL_CROP_SZ = 0
if NOIMAGE:
    _src = os.path.join(ROOT, "deliverable", "data-request-v2")
    for _sub in ("gt/crops", "gt_new/crops"):
        _d = os.path.join(_src, _sub)
        if not os.path.isdir(_d):
            continue
        for _dp, _, _fs in os.walk(_d):
            for _f in _fs:
                REAL_CROP_N += 1
                REAL_CROP_SZ += os.path.getsize(os.path.join(_dp, _f))


def sha(p, n=16):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


lines = []
rows = []
crop_n = crop_size = 0
crop_h = hashlib.sha256()
for dp, dn, fn in os.walk(D):
    dn.sort()
    for f in sorted(fn):
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, D).replace("\\", "/")
        sz = os.path.getsize(p)
        if rel.startswith("gt/crops/"):
            crop_n += 1
            crop_size += sz
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    crop_h.update(chunk)
            continue
        rows.append((rel, sz, sha(p)))

rows.sort()
total = sum(r[1] for r in rows) + crop_size

lines.append("# 文件清单与哈希\n")
if NOIMAGE:
    lines.append(f"> 生成于 {dt.datetime.now():%Y-%m-%d %H:%M}｜**去图版**：非图片文件 {len(rows)} 个｜"
                 f"人像拼图 {REAL_CROP_N} 张（{human(REAL_CROP_SZ)}）**未随包分发**\n")
    lines.append("本包不含任何视频帧、人像裁图或内嵌图片。原 `gt/crops/` 与各标注页里的内嵌图已全部移出，"
                 "理由见 `PRIVACY.md`。标注结论、簇级真值与逐条真值均完整保留。\n")
else:
    lines.append(f"> 生成于 {dt.datetime.now():%Y-%m-%d %H:%M}｜非图片文件 {len(rows)} 个 + 拼图 {crop_n} 张｜"
                 f"合计 {human(total)}\n")
lines.append("哈希为 `sha256` 前 16 位。项目无 git 仓库，**文件哈希是唯一的版本冻结手段**：\n"
             "引用本包任何数字时，请一并记录对应文件的哈希。\n")
lines.append("## 顶层\n")
lines.append("| 文件 | 字节 | sha256[:16] |")
lines.append("|---|---|---|")
for rel, sz, h in rows:
    if "/" not in rel:
        lines.append(f"| `{rel}` | {sz} | `{h}` |")
lines.append("")
for sub in ["ledger", "gt", "results"]:
    lines.append(f"## `{sub}/`\n")
    lines.append("| 文件 | 字节 | sha256[:16] |")
    lines.append("|---|---|---|")
    for rel, sz, h in rows:
        if rel.startswith(sub + "/"):
            lines.append(f"| `{rel}` | {sz} | `{h}` |")
    lines.append("")

lines.append("## `gt/crops/`\n")
if NOIMAGE:
    lines.append(f"**不随包分发。** 原为 {REAL_CROP_N} 张 8 帧拼图（{human(REAL_CROP_SZ)}，"
                 "`gt/crops` + `gt_new/crops`），内容为真实行人裁图，属个人影像。\n")
    lines.append("清单见包外 `crops_manifest_noimg.csv`（路径 + 字节 + sha256，无像素）。"
                 "复现或复核需要这些图时，请按 `PRIVACY.md` 走授权渠道单独索取。\n")
else:
    lines.append(f"{crop_n} 张 8 帧拼图，{human(crop_size)}，单独打包为 `_crops.zip`。\n")
    lines.append(f"聚合 sha256[:16]（按遍历顺序串流）：`{crop_h.hexdigest()[:16]}`\n")
    lines.append("文件名 `<seg>/<tid>.jpg` 与 `ledger/tracks.csv` 的 `track_id` 一一对应。\n")

lines.append("""## 数据流

```
media-resource/1012/segments/*.mp4
        │  YOLO26s-seg + dedup_detections --iou 0.5（检测，--iou 0.7）
        ▼
output_crosscam_local/1012_yolo8_dd/<seg>/<seg>_detections.jsonl
        │  make_ab_jsonl.py（轨迹前后半切分）+ encode_1012_osnet_seq.py（OSNet 冻结编码）
        ▼
output_crosscam_local/1012_yolo8_dd_emb/<seg>_emb_{a,b}.npz
        │  cluster_camera_aware.py（机位内 average 0.15 → 跨机位 complete 0.25）
        ▼
ledger/clusters_geom.csv ← export_paper_ledger.py
        │  d7_sweep.py（42 配置 × 63 θ 点）
        ▼
results/summary.csv · curve.csv · best.json · queries/ · members/
        │  eval_precision.py（接 gt/ 两套真值）
        ▼
results/precision.csv · verdicts/
        │  compare_cfgs.py（McNemar + bootstrap）
        ▼
results/compare.csv
```

## 复跑

```bash
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"
export PYTHONPATH=D:/human-count/pylibs
export TMPDIR=D:/human-count/tmp TEMP=D:/human-count/tmp

$PY tools/export_paper_ledger.py            # 重建 ledger/（含两配置逐查询输出）
$PY tools/merge_review_d3.py                # 合并复核（写前备份到 gt/_backup/）
$PY tools/derive_clusters_gt.py             # 重派生 gt/clusters_gt.csv
$PY tools/d7_sweep.py                       # 42 配置 × 63 θ 点，约 8 秒
$PY tools/d7_sweep.py --robust-margin 0.10 --out .../results/robust
$PY tools/eval_precision.py                 # → precision.csv + verdicts/
$PY tools/eval_precision.py --run results/robust
$PY tools/compare_cfgs.py                   # → compare.csv
$PY tools/diag_cluster_harm.py              # d_max 的两种 AUC
$PY tools/diag_score.py                     # 分数门
$PY tools/diag_errors.py                    # 6 条错判 + veto 逐簇账
$PY tools/plot_d4.py                        # → fig_d4_truth.png
```

## 环境

| 项 | 值 |
|---|---|
| 检测 | YOLO26s-seg（`--iou 0.7`）+ `dedup_detections --iou 0.5` |
| 追踪 | ByteTrack（随检测脚本内置） |
| 编码器 | OSNet，权重 `*_msmt17_combineall_*` —— **MSMT17 全量训练，公开实验不可用 MSMT17** |
| 运行 | 纯 CPU；YOLO 约 3.3 帧/秒，OSNet 单裁剪 112–229 ms |
| 时间隔离 | TRAIN = 305/318/320/323 的 `_train_*` 段；TEST = `_test_*` 段 |
| 四机位时钟 | 323 12:57 / 305 13:28 / 318 14:22 / 320 15:03，各 15 分钟，**互不重叠** |
""")

open(os.path.join(D, "MANIFEST.md"), "w", encoding="utf-8").write("\n".join(lines))
print(f"[out] MANIFEST.md  {len(rows)} 行 + {crop_n} 张拼图  {human(total)}")
