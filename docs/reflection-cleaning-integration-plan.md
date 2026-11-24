# 反光清洗（玻璃掩膜 + 地板倒影运动学）融入现有流程 —— 集成方案

日期：2026-10-13 ｜ 关联：`analysis/reflection_cleanup_audit_2026-10-13.md`（现状审计）

## 一、问题定位（一句话）

现有清洗成果是"离线工具 + 成品文件"，**没有接入任何脚本的加载边界**；下游 7 处仍各自硬编码读原始
`{cam}_detections.jsonl`。因此集成要做的不是"再造方法"，而是**把清洗提升为数据加载的唯一入口**。

## 二、需要接入的 7 个输入点（已逐一核对）

| # | 脚本 | 位置 | 现状路径 | 运行环境 |
|---|---|---|---|---|
| 1 | `cradio_scale_remote.py` | L180/310/340 | `{DATA}/inference-out/{cam}_detections.jsonl` | 远程 |
| 2 | `dinov3_bridge_remote.py` | L185/334/375/428 | 同上 | 远程 |
| 3 | `crosscam_relay.py` `load_foot_trajs` | L38 | `output_crosscam/{cam}_detections.jsonl` | 本地 |
| 4 | `make_splits_v3.py` | L38/174 | 同上 | 本地 |
| 5 | `global_reid_demo.py` | L64 | 同上（仅元数据） | 本地 |
| 6 | `relay_evidence.py` | L38 | 同上（仅出图） | 本地 |
| 7 | `encode_tracks.py` | `--jsonl` | 由调用侧传入 | 远程 |

> 唯一已接清洗的是 P2 拼接链（`apply_p2_corrections.py` / `track_stitching.py` / `stitch_evidence.py`），
> 产物实体图 `net_tracks_{cam}.json` → 因此**接力链（E 号）本就是净的**，接入后属"回归确认"而非改动。

## 三、清洗资产的覆盖度（决定各机位接哪一档）

| 机位 | 玻璃掩膜（墙镜） | 运动学清洗（地板倒影） | 可用的最净档 |
|---|---|---|---|
| 316 | ✅ enabled（右侧整面玻璃墙，u≥0.60） | ✅ 18 条 | **dualclean** `316_masked_cleaned.jsonl` |
| 323 | ✅ enabled（走廊玻璃窄带） | ✅ 33 条 | **dualclean** `323_masked_cleaned.jsonl` |
| 343 | ⚠️ **disabled**（低置信草稿，霓虹反射条纹） | ✅ 4 条 | clean（仅去地板倒影） |
| 305 | ✖ 无区域 | ✅ 1 条 | clean |
| 318 | ✖ 无区域（窗外玻璃反光弱） | ✅ 4 条 | clean |
| 320 | ✖ 无区域 | ✅ 0 条 | clean（= 原始） |

**关键取舍点**：343 的玻璃掩膜是草稿且未启用。若论文要走"反光治理"叙事，建议**先请用户圈改确认 343**，
把 6 路统一到"该有的都覆盖"，否则 343 的墙镜残留会成为审稿人质疑点。

## 四、推荐方案 A：统一"清洗视图" + 单一开关

### 设计原则
清洗只在**数据加载边界**发生一次；所有消费者共享同一份清洗集 + 同一开关，杜绝"各扫门前雪"。

### A1. 数据准备（离线，一次性）
- 生成规范化清洗集到统一目录（本地 `output_crosscam/cleanset/`，远程 `{DATA}/inference-out-clean/`）：
  - 316/323 → dualclean；其余 4 路 → 运动学清洗版。
- 同时产出 `clean_manifest.json`：规则版本、掩膜版本、**各机位被删轨迹 ID 列表**、删除框数、
  源文件 md5 与产物 md5。作用：可审计 + 可断言 + 版本可追溯。
- **前置校验**：先比对远程 `inference-out/{cam}_detections.jsonl` 与本地清洗所用源的 md5；
  不一致则轨迹号错位，必须先对齐（这是 A 方案的唯一硬前提）。

### A2. 加载边界统一（改动 ≈ 7 行）
新增极简模块 `tools/det_view.py`：
```
DET_VIEW=raw|clean（env，默认 raw，向后兼容）
local_det(cam)  -> output_crosscam[/cleanset]/{cam}_detections.jsonl
remote_det(cam) -> {DATA}/{inference-out|inference-out-clean}/{cam}_detections.jsonl
removed_tids(cam) -> set()   # 读 clean_manifest，用于断言/审计
```
各脚本改为经该模块取路径（远程脚本用 `DET_DIR` env，本地脚本用 `local_det`）。
`encode_tracks.py` 无需改（调用侧传清洗路径）。

### A3. 版本纪律
- 版本 **v3 → v4**：重跑 `collect/solve/encode` 得新 W；重建全局库；重建 splits（`load_all_tracks` 改清洗后自洽）；
  重算冻结哈希。**v3 全部数字随之失效，不可与 v4 混用**。
- 评估集修正：`318#53` / `318#98`（清洗判定的地板倒影）用其源轨 `318#51` / `318#110` 替换，或直接从
  NEG30 剔除后重建划分（版本递增）。

### A4. 自动验收断言
1. 清洗版嵌入集 ∩ 倒影集合 = ∅（6 路）；
2. 全局库条数 = 667（727 − 60）；
3. 冻结测试集不含任何倒影轨，哈希更新；
4. 新 W 在冻结测试集上 AUC 与 v3 对比——**预期接近**；若显著下降，说明倒影轨此前在"提供信息"，
   需专门分析（这本身是个有价值的发现，可写入论文）。

## 五、替代方案 B：运行时过滤（保留原始 jsonl）

不改数据文件，把 `{cam}_mirrorflags.json`（轨迹→源、perr）上传远程，各 loader 读入后按 flags 跳过分镜轨迹。

- 优点：单一数据源、无版本漂移风险；对"只关心嵌入"的场景最省事。
- 缺点：过滤逻辑要挂在**每个** loader 上（7 处），下游 store/splits 仍需各自过滤，**易漏**；
  且掩膜（脚点级）无法用"整轨 flags"表达，会丢掉玻璃掩膜能力，等于只接入地板倒影一半。
- 结论：**只在 A 的 md5 前置校验失败、短期无法对齐数据时才退而用之**。

## 六、方案对比与推荐

| 维度 | A 统一清洗视图 | B 运行时过滤 |
|---|---|---|
| 改动量 | 7 行路径 + 1 数据准备 | 7 处 loader 挂过滤器 |
| 覆盖能力 | 玻璃掩膜 + 地板倒影（完整） | 仅整轨倒影（掩膜丢失） |
| 审计/可复现 | manifest + 哈希，强 | 弱（逻辑分散） |
| 版本漂移风险 | 依赖 md5 前置校验 | 无 |
| 纪律契合度 | 高（单一视图） | 中 |

**推荐 A**。它把"反光治理"从"一次性实验"升级为"流程内的常驻数据层"——这恰好也是论文里
"reflection-aware pipeline"最干净的说法。

## 七、执行步骤（A 方案，预计远程约 1h）

1. （前置）远程 vs 本地 detections md5 校验；确认 343 掩膜是否圈改；
2. 生成 `cleanset/` + `clean_manifest.json`（复用 `track_cleaning.py` / `glass_mask.py`，不改其算法）；
3. 新增 `tools/det_view.py`，改造 7 处输入点（纯路径替换，不动算法）；
4. 上传清洗集到 `{DATA}/inference-out-clean/`；`DET_VIEW=clean` 重跑 cradio collect/solve/encode（v4）；
5. 重建全局库、重建 splits（含评估集修正）、重算冻结哈希；
6. 跑 A4 四条断言 + 出 v4 对比表（v3 vs v4）。

## 八、遗留与局限（如实）

1. **部分污染无解**：跟踪器把真人 id 串接到倒影（同一 track id 前段真人、后段倒影）时，整轨剔除无效
   → 记为局限，长期解法是**把玻璃掩膜前置到"检测后、跟踪前"**，从源头抑制镜像轨生成（需重跑推理）。
2. 掩膜会打断轨迹、改变 8 帧采样点与轨迹时长，属预期副作用，需在报告注明。
3. 1012 长视频接入时，清洗规则需在新数据上重跑并重建 manifest（规则可复用，参数待校准）。
