# long35 全链路重跑（2026-11-09）

> 用户决策：**方案 B** —— 把可视化/分析从 10min 扩到 35min，且"用现在的最新算法模型重跑"。
> 本文件是跑批记录：命令、口径、坑、结果。配套设计文档 §20.2。

## 0. 为什么是"延长"而不是"换片段"

305 前 3000 行检测与现网 10min 版**逐位一致（3000/3000，track_ids 也一致）**，
说明 `long-35min` 就是同一段素材的前 35 分钟，`synced-10min` 是它的前 10 分钟子集。
→ 扩到 35min 是**延长**，不是换素材，前 10 分钟的所有坐标/口径都继续成立。

## 1. 数据与模型

| 项 | 选择 | 依据 |
|---|---|---|
| 检测 | `output_crosscam/long35/{cam}_long35_detections.jsonl` | 已在盘，6 路各 10480~10517 行（35min @ stride 5） |
| 清洗轨迹 | `long35/cleaning/{cam}_cleaned.jsonl`（316 用 `_masked_cleaned_v2`） | 已在盘 5 路；**305 缺清洗版 → 用 raw** |
| 编码器 | **OSNet-x1.0 MSMT17**（`models/reid_osnet/...combineall...pth`） | exp8 冻结集：AUC **0.922** vs DINOv3-B CLS **0.628** |
| 编码口径 | 每轨 ≤8 张、per-crop L2 → 均值 → L2、512 维、FTZ 开 | 与 `local_encode_osnet.py` 既有口径一致 |
| 同机位拼接 | `track_stitching.py`（cos≥0.80、pe_adj≤0.08、mutual-best 无传递闭包） | 与 10min 生产口径一致 |
| 实体图 | `rebuild_clean2_entities.py --src long35 --stitch ...`（本轮新增两个参数） | 同一算法，只换数据根 |

## 2. 命令序列

```bash
PY=C:/Users/might/.workbuddy/binaries/python/envs/default/Scripts/python.exe

# 1) 编码（6 路，串行；seek 模式 151~167 ms/crop）
$PY tools/long35_pipeline.py --only encode

# 2) 同机位拼接
$PY tools/long35_pipeline.py --only stitch

# 3) 实体图（clean2 口径，输入重定向到 long35）
$PY tools/long35_pipeline.py --only entities

# 4) 闸口事件 + 跨机接力（环境变量切数据根）
DET_VIEW=clean2 DET_ROOT=output_crosscam/long35 \
NET_DIR=output_crosscam/long35/stitch_long35 \
CROSSCAM_RELAY_OUT=output_crosscam/long35/relay \
$PY tools/crosscam_relay.py events
DET_VIEW=clean2 DET_ROOT=output_crosscam/long35 \
NET_DIR=output_crosscam/long35/stitch_long35 \
CROSSCAM_RELAY_OUT=output_crosscam/long35/relay \
$PY tools/crosscam_relay.py links

# 5) Web 数据包（35min 窗口）
$PY webapp/dataprep/build_web_data.py \
  --root output_crosscam/long35 --det-suffix _long35_detections.jsonl \
  --entities output_crosscam/long35/stitch_long35 \
  --relay-dir output_crosscam/long35/relay \
  --window-sec 2100 --base-wall 08:55:00 \
  --out-dir webapp/backend/data_35min

# 6) 媒体转码（540p；imageio_ffmpeg 自带 ffmpeg，本机无系统 ffmpeg）
$PY tools/long35_media.py --dst-dir webapp/backend/data/media_long35
```

## 3. 本轮改的代码

| 文件 | 改动 | 为什么 |
|---|---|---|
| `tools/local_encode_osnet.py` | 加 `--seek sequential/random`、`--threads`；新增流式 `encode_sequential` | 长视频顺序解码可省 seek 开销；但**必须边解码边编码**，否则 6 路 × 每轨 8 张 2560×1440 裁片直接 OOM |
| `tools/rebuild_clean2_entities.py` | 加 `--src` / `--stitch` / `--cams` | 换数据根复用同一算法，不改口径 |
| `tools/det_view.py` | 加 `DET_ROOT` 环境变量 | 与 `DET_VIEW`（清洗口径）正交，支持按片段切根 |
| `webapp/dataprep/build_web_data.py` | 加 `--root/--det-suffix/--window-sec/--base-wall/--entities/--relay-dir`；11 处硬编码路径改为变量 | 让同一构建器能出 10min 与 35min 两套数据包 |
| `tools/long35_pipeline.py` | 新增（编码→拼接→实体的驱动） | 六路同参数，单步可重跑（`--only`） |
| `tools/long35_media.py` | 新增（转码 + 备份 + 时长校验） | 本机没有系统 ffmpeg；`--limit` 只出旁路样本，不覆盖正式文件 |

## 4. 踩过的坑（都已在代码里修复）

1. **顺序解码 OOM**：把每轨 8 张裁片攒在内存里，第 16267 帧 `Insufficient memory`（要 11 MB 都拿不出）。
   → 改成固定缓冲（64 张）满即前向，只留「向量和 + 计数」。
2. **`--limit` 测速覆盖了正式文件**：`305.mp4` 一度被替换成 30 秒版本（已从 `media_10min/` 恢复）。
   → `--limit` 现在输出到 `{cam}.limit{N}s.mp4`，绝不 `os.replace` 到正式名。
3. **sequential 对稀疏样本反而更慢**：35min 只取 ~3000 张裁片时，顺序解码要解完 5 万帧（8~10 分钟），
   而逐 crop seek 只要 6 分钟。→ `--seek` 默认改回 `random`（sequential 仅在 crop 密集时有优势）。
4. **符号链接在这个目录不可靠**（Syncthing 同步目录）：`os.symlink` 建出 0 字节普通文件。
   → 改 `shutil.copy2`。
5. **两个长任务并行会互相拖死**（编码 + 转码同时跑，10 分钟没进展）→ 串行执行。

## 5. 口径披露（必须随结果一起说）

- **305 未做 long35 清洗**：缺 long35 的 FP 审计清单（`track_cleaning.json` 只覆盖 10min 与 316 long35）。
  305 用 raw 检测（367 轨），会带入倒影/误检轨迹 → 其实体数偏高。其余 5 路用清洗版。
- **35min 无真值标注**：C0 类"精度/召回"指标**不可算**，只能报结构指标（实体数、链数、链接率、跨机边数）。
- **人工修正是 track-id 级的**：318/320 的黑名单/强制合并来自 10min 核定，因前 10min 的 track_id
  与 long35 逐位一致，这部分仍然有效；35min 新增轨迹不受影响（也不会被误伤）。

## 6. 结果

（跑完后补：6 路轨数/实体数、链路总时长、数据包体积、页面截图）

---

## 8. 结果（全部实测，2026-11-09 21:40 收尾）

### 8.1 各环节规模

| 环节 | 10min（现行生产） | 35min（本轮） | 倍率 |
|---|---|---|---|
| 检测帧（每路） | 2981~3005 行 | 10480~10517 行 | 3.5x |
| 轨迹总数 | 541 | **1779** | 3.29x |
| 实体总数 | 411 | **1271** | 3.09x |
| 跨机接力边 | 44（v6，人工否证后） | **135** | 3.07x |
| 接力链 | 29 | **88**（多跳 37） | 3.03x |

分机位：

| cam | 10min 轨/实体 | 35min 轨/实体 | 说明 |
|---|---|---|---|
| 305 | 107 / 81 | 367 / 250 | **raw 检测**（缺 35min FP 审计），含倒影误检 |
| 316 | 68 / 61 | 150 / 140 | 走廊，用 masked_cleaned_v2 |
| 318 | 151 / 105 | 478 / 338 | |
| 320 | 24 / 20 | 176 / 111 | 10min 清洗更狠（clean2 只剩 24 轨），不可直接比 |
| 323 | 138 / 98 | 508 / 350 | |
| 343 | 53 / 46 | 100 / 82 | 电梯间人流稀疏，实体增幅最小（1.78x） |

### 8.2 编码耗时（CPU，逐 crop seek）

| cam | 轨 | crop | 耗时 | 单 crop |
|---|---|---|---|---|
| 305 | 367 | 2317 | 371s | 160ms |
| 316 | 150 | ~1000 | 120s | 182ms |
| 318 | 478 | 2555 | 693s | 271ms |
| 320 | 176 | 1179 | 320s | 271ms |
| 323 | 508 | 3199 | 456s | 142ms |
| 343 | 100 | 621 | 113s | 182ms |
| **合计** | **1779** | **~10900** | **~35 min** | — |

转码（ffmpeg，960×540 / CRF 26 / `+faststart`）：6 路 35min 共 **200.5 MB**，单路 130~530s。

### 8.3 阈值选择与"未标定"的边界

`relay_rescore score` 用 OSNet 给 5835 条时间门候选打分：**p50=0.540 / p90=0.678 / p99=0.815 / max=0.904**。
扫阈值看 mutual 边数：0.90→3、0.80→56、0.75→85、0.70→112、0.65→**135**、0.60→142、0.55→145。

边数在 0.60~0.65 处饱和（mutual-best 约束本身在压制噪声），取 **0.65**：与 10min 的 44 边比例 3.07x，接近时长比 3.5x。

★ **必须说清楚**：这个阈值是"结构合理性"选的，**不是精度校准**——35min 没有真值，无法报 P/R/F1。

### 8.4 ★ 与 10min 边集的一致性：36%，但这不是"精度 36%"

10min v6 的 44 条边在 35min 的 135 条边里命中 **16 条（36%）**。逐条追查未命中的原因：

| 类型 | 例子 | 原因 |
|---|---|---|
| 被更高分边抢走 | `316:t26→318:t62`：35min 里该实体对 sim=0.7605 过门，但 316-E013 有更相似的 b | **mutual-best 是全局唯一匹配**，实体分段时间变长后，竞争关系变了 |
| 时间门失效 | `305:t51→320:t11`：35min 里 320-E009 起点提前 → dt 超 60s 窗 | 实体被拼成更长实体，首末时间变了 |
| 实体归属变了 | 同一条边两侧 frag 落进不同实体 | 35min 的拼接把 10min 里的两个实体合成一个 |

结论：**边集合不可直接对齐，因为"实体"的定义随观测时长变了**。这不代表 35min 链路错，也不代表 10min 错——两者是不同切分粒度下的产物。
前 10min 的 track id 逐位一致（6 路已核，仅 320 多 1 条边界轨），所以**候选级可比、实体级不可比**。

要真正验证 35min 的链路质量，需要 35min 段的人工标注（当前只有前 10min 的 241 对真值）——属未做项。

## 9. 页面切换与验收

### 9.1 数据包

```
webapp/dataprep/build_web_data.py \
  --root output_crosscam/long35 \
  --det-dir output_crosscam/long35/cleanset_v2 \
  --entities output_crosscam/long35/stitch_long35 \
  --relay-dir output_crosscam/long35/relay_long35 \
  --relay osnet --c0 none --window-sec 2100 \
  --out-dir webapp/backend/data_long35
```

本轮给 `build_web_data.py` 加了 4 个参数：`--det-dir`、`--relay osnet`、`--c0 none`、以及此前的 `--root/--entities/--relay-dir/--det-suffix/--window-sec/--base-wall/--out-dir`。
产物：`tracks/` 7.3 MB（10min 版 1.8 MB）、`chains.json` 48 K、`entities.json` 92 K、`cams.json` 28 K。

### 9.2 切换

```
python tools/long35_switch.py snapshot 10min     # 105 MB 快照（含 media）
python tools/long35_switch.py use long35         # 覆盖 data/，媒体从 data/media_long35 拷
python tools/long35_switch.py use snap_10min     # 回滚
```

★ 切换时**不动 `labels.json` / `mirror.json` / `pairs.json`**：判定属于"人"不属于"片段"。
已核 6 路前 10min 的 track id 与 10min 版逐位一致（305/316/318/323/343 完全一致，320 多 1 条边界轨），
所以 10min 期间人工判的 75 对 / 56 条倒影判定在 35min 里仍指向同一个人（切换后实测：pair 75 / mirror 56 / glass 3 全部保留）。

### 9.3 验收（本机 8025）

| 项 | 结果 |
|---|---|
| 时间轴 | `00:26 / 35:00`，`session.window_sec=2100`，墙钟基准 08:55:00 不变 |
| 首页指标 | 1271 实体 / 1779 轨 / 88 链 / 135 边 |
| 六路画面 | 全部出画（首帧修复后）；305 办公室 9 人、318 五人、323 四人 |
| ★ 时间对齐 | t=1200s 抽帧：305 OSD **09:14:50**（页面 09:14:55，offset −5）、318 OSD **09:15:18**（页面 09:15:20，offset +20） |
| ★ 框对齐 | t=1200s 叠加：305 9 框 / 318 5 框 / 323 4 框，框均落在人身上（305 有 2 处桌面误检，属 raw 检测已知问题） |
| 标注页 | 75/75 全判，画面正常，判定保留 |
| 真值页 | 241/107/75 全判，裁剪图正常 |
| 总览 | 数字正常；`1 个 run 失败`为包装器试跑被 SIGTERM（note 已标"由 run_long35_encode_6cam 完成"） |
| 台账 run | 回填 4 条：`run_long35_{encode_6cam,stitch_6cam,relay_osnet,webpack}`，含产物 sha1 与指标 |

### 9.4 顺带修的两个真实缺陷

1. **首帧黑屏**：`currentTime` 已是 0 时再赋 0 是 no-op，浏览器不解码 → 面板保持黑屏。
   35min 素材体积大（每路 23~42 MB）时必现。改为 `< 0.02` 时推一帧（0.04s）强制走一次 seek。
2. **`state.stats`** 与 **`ingest_truth` 分层**（见 §19.4）——上一轮已修，本轮切换后数字保持一致。

### 9.5 新增的深链接

`#/?t=1200` 直达第 1200 秒（验证与分享单帧用），优先于"首屏停在所有机位进入时刻"的默认逻辑。

## 10. 未做 / 不假装做了

| 项 | 状态 |
|---|---|
| 35min 真值 | **无**。全链路精度（P/R/F1）无法计算，只能报结构指标 |
| C0 口径 | **无**。`--c0 none`，配置页 C0 卡片空 |
| 305 的 FP 审计 | **无**。`track_cleaning.json` 只覆盖 10min 与 316 → 305 用 raw，实体数偏高 |
| 跨机空间映射（mapcover） | 未重跑。`relay_rescore` 只做时间门 + 外观 mutual，未加平面映射约束 |
| 倒影/源标注（floorref） | 未重跑，`mirror.json` 为空（10min 版同为空，判定在 `labels.json.mirror_decisions`） |
| tracks 分片 | 未做。7.3 MB 首屏本地 0.6s/路，远程会慢 |
| 远程发布 | 未做。远程 dist 仍 10-14 版，且无 `data_long35` |

## 11. 回滚

```
python tools/long35_switch.py use snap_10min
```

媒体、cams/tracks/entities/chains/metrics/masks/gil 全部回到 10min 版；`labels.json` 等判定文件全程未动。
