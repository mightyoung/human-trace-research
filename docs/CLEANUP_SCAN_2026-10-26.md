# 冗余文件扫描报告（只读，未做任何删除/移动）

扫描时间：2026-10-26 13:20　范围：`D:/dev/human-count`（未覆盖 `datasets/` 681k 张 jpg，见文末说明）

## 结论

目录内确实存在可清理的冗余，按"删除风险由低到高"分四档。**A 档（约 400MB）可安全清理**，B/C 档需你逐项确认，D 档不建议删。

| 档次 | 内容 | 可回收 | 风险 |
|---|---|---|---|
| A 安全 | 同步冲突残留、构建产物、重复快照 | ~400 MB | 极低 |
| B 待确认 | 旧版产物/备份/旧交付目录 | ~750 MB | 低（有历史价值） |
| C 需论证 | 分析产物中的中间件 | ~1.5 GB | 中（可能需复跑） |
| D 不动 | 原始素材、模型、源码、当前数据 | 39G+25G | — |

---

## A 档：可安全清理（~400 MB）

### A1. Syncthing 同步冲突文件（6 个，~90 KB）
`.stignore` 规则 `(?d).*` 只忽略点文件，未忽略 `*.sync-conflict-*`，所以这些文件会持续堆积。

全部 6 个都与现行文件有差异，但差异方向是**冲突件更旧**（10-10 时间戳，均为 DirectML/ffmpeg 改造前版本）：

| 冲突文件 | 对比基准 | 差异内容 |
|---|---|---|
| `Cargo.sync-conflict-20261026-012544-D4DTDW4.toml` | `Cargo.toml` | 缺 `directml` feature |
| `config.sync-conflict-20261026-012544-D4DTDW4.toml` | `config.toml` | 缺 `execution_provider`，ffmpeg 路径为旧版 |
| `src/config.sync-conflict-20261026-065947-D4DTDW4.rs` | `src/config.rs` | 缺 `execution_provider` 字段 |
| `src/detector.sync-conflict-20261026-070004-D4DTDW4.rs` | `src/detector.rs` | 缺 DirectML EP 分支 |
| `tools/rtsp_sim.sync-conflict-20261026-070107-D4DTDW4.sh` | `tools/rtsp_sim.sh` | ffmpeg 路径旧 |
| `docs/commit-015/verify_work/EXPERIMENTS_SUMMARY.sync-conflict-....md` | 文档 | 需人工看一眼（摘要类） |

**建议**：删除 5 个代码/配置冲突件（内容已被正式文件完全覆盖），文档那个先 diff 一眼。同时把 `*.sync-conflict-*` 加进 `.stignore` 阻止继续产生。

### A2. 构建产物（~2.0 GB，重建代价低）
- `target/` 1.8 GB — Rust 构建目录，`.gitignore` 已列。`cargo build` 可重建。
- `tests-pure/` 243 MB — 纯逻辑测试 crate 产物（`.o/.rlib/.pdb`），同样可重建。
- 注：`webapp/frontend/node_modules/` 120 MB **不建议删**（npm 重装耗时且可能受网络影响），但若磁盘紧张可删。

### A3. tmp 内完全重复的快照（~333 MB）
经 diff 校验确认逐字节一致或仅差元数据：

| 路径 | 大小 | 说明 |
|---|---|---|
| `tmp/verify_v13/human-count-package-2026-10-26` | 24 MB | 与 `tmp/v12/...` **完全相同**（diff 无输出） |
| `tmp/site_strict_v2bak` | 362 KB | 与 `tmp/site_strict` 完全相同 |
| `tmp/v8/human-count-package-2026-10-26` | 24 MB | 旧版本包（v12 是其迭代），保留 v12 即可 |
| `tmp/pkg_v7/human-count-package-2026-10-25` | 27 MB | 前一日旧包 |
| `webapp/backend/data.bak_1014_1755/` | 116 MB | 控制台数据整目录备份，与现行 `data/`（144 文件）几乎同构 |
| `webapp/backend/data/chains.json.bak_pre_v6_1014` | 16 KB | 单文件备份 |

**注意**：`webapp/backend/data.bak_1014_1755` 是全目录快照，`data/` 现行版本已迭代 12 天，删前建议先确认线上控制台的 `chains.json` 等无需回滚。

### A4. 已废弃的早期输出（~14 MB）
`output_preview/`、`output_dedup/`、`output_tracked/` — 10-10 的同一批 clip01/clip02 演示产物，三份互为迭代版本，`.gitignore` 已列。如论文/交付不再引用可删。

---

## B 档：待确认（~750 MB）

### B1. `.bak_pre_*` 备份（29 个，~4.9 MB，但都是小文件）
两个集中点：
- `output_crosscam/analysis/gil/store_v1/gil.sqlite.bak_pre_*`（**5 份 GIL 数据库，每份 897 KB**）——对应 entities/gperson/s3v2/relaysync/human_adj 各次迁移。既然已有 `store_v1/manifest.json` 与代际存储 v2，早期 3 份（entities/human_adj/human_gperson）可考虑只留最新一份。
- `tools/*.bak_pre_*`（17 个 .py）——batchA / scrfd_fix / v2 改造前的备份。**scrfd 那 3 个（`gil_face_capture`、`scrfd_dnn`、`_scrfd_cv2dnn_probe`）建议保留**，是坐标体系变更前的唯一参照。

### B2. 重复打包产物（~4.7 MB）
- `human-count-research-package-v1.tar.gz`（2.19 MB）与 `research_package.zip`（2.57 MB），对应同一份 `research_package/` 目录。交付已完成，可只留 tar.gz（有 SHA256 记录）或只留一个。
- `handoff_out/human-count-research-handoff.zip` 与同名目录并存。

### B3. 交付目录与 dist 的时间切片堆积
- `dist/export-2026-10-24/` 58 MB — 8 个分类子目录，与 `validation/`(53 MB)、`deliverable/` 内容有重叠但经过挑选重组，**不建议直接等同删除**。
- `deliverable/data-request-v2/` 342 MB — 当前交付物主体，**保留**。
- `uicheck/`（根目录，8.5 MB / 30 张）是 `webapp/deploy/uicheck/`（67 张）的**严格子集**（diff 确认根目录无独有文件）→ 可删。

---

## C 档：需论证（~1.5 GB）

- `output_crosscam/analysis/fusion/` 322 MB — 含 `.npz/.npy` 嵌入缓存（`bridge_emb_cache.npz`、`cradio_emb_cache.npz` 等）。这些是**重算成本高**的中间件，但若实验已定稿且结论写入报告，可归档到外部盘而非删除。
- `tmp/precision_check/` 256 MB、`tmp/gt_variant/` 255 MB — 精度核对与 GT 变体，尚有关联任务（`gt_variant` 有 a3 抽样表）。**活跃，别动**。
- `tmp/texbin/` 70 MB — tectonic.exe 与 zip。若 PDF 编译仍需，保留；zip 可删。
- `tmp/v04/ pkg/ verify6/ verify5/ v6_eval_main/ pre_v6/ ...` 等 60+ 个小目录，累计约 60 MB — 都是单次实验脚本与快照，可按"最后一次修改时间"统一归档。

---

## D 档：不要删
`datasets/`、`media-resource/`（39 GB）、`models/`（25 GB）、`src/`、`tools/`、`webapp/src`、`docs/commit-015/` 的三个外部评审包、`output_crosscam/analysis/gil/store_v1/`（含 `manifest.json`，是 GIL 对账基准）。

---

## 关于 datasets/

681,140 个文件（几乎全是 jpg），`du` 扫描超时未能出总大小。从项目结构看是抽帧结果缓存 —— 这类数据的冗余度通常最高（同一视频多次抽帧、低置信度框裁剪）。但**必须按视频源+抽帧参数去重**才能判断，不能按文件时间批删。若要处理，需要单独一轮建模：以 `datasets/*/meta*.json` 的 `source_video + stride + scale` 为键分组，同组只留最新一代。**这一步需要你明确授权后再动。**

---

## 建议执行顺序

1. 先做 A1 + A4（冲突文件 + 三个旧 output_*，约 14 MB），零风险，顺带修 `.stignore`。
2. A2（`cargo clean` + 清 tests-pure 产物）释放 2 GB，需要时重建。
3. A3 逐项确认后删，主要是 `data.bak_1014_1755`（116 MB）要你拍板。
4. B/C 档建议**归档到外部盘**而非删除，尤其是 `fusion/` 的 npz 缓存。

以上均未执行任何写操作。需要我按哪一档动手，请明确指定范围。
