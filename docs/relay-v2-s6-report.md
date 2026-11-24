# Relay v2 / S6 实施报告：弱事件候选通道 + 预算挤出显式化（55 判增量对账）

日期：2026-10-14 ｜ 状态：**已完成并验证（Go）** ｜ 上游：`docs/relay-v2-s3s4-report.md` §5.5（55 判增量对账暴露的 3 条漏接线索）、`docs/cam-handover-neighbor-research.md`
纪律依据：`docs/research-agenda/evaluation-protocol.md`（TG 规则）；边缘口径：cradio v5（`cradio_scale/{cam}_emb_scale.npz`，3 锚点逐位复现自证）

---

## 0. 结论（TL;DR）

55 判暴露的 3 条漏接线索，根因全部定量定位（3 类），其中 2 类已修复、1 类证实为独立限制：

| 漏接对（人工判同人） | 根因 | 归宿 | 关键证据 |
|---|---|---|---|
| **343:E010 ↔ 316:E014** | **候选生成漏**：E014 事件 `moving=False`（span 0.045 < 0.05）被 `links --only-moving` 过滤，候选池从未包含该对 | ✅ **弱通道回升**：weak→audit→F4 `active/human-assisted` | 补池后 cradio sim=**0.9584**（≥0.943）、dt=2.2s |
| **343:E009 ↔ 316:E025** | **预算静默挤出**：sim=0.9446 落在 [0.943, 0.95) 且被 K_a/K_b 挤出 → 旧逻辑不记录 | ✅ **blocked 显式层**（复核可见，非静默） | sim=0.9446、score=0.9249（dt=54.4 长尾被 dt 先验压位） |
| **343:E009 ↔ 316:E024** | **外观证据不足**（独立限制）：E024 单帧碎片（dur=0.0s），补池后 sim=**0.8560** < 0.943 | ❌ 本轮不可恢复（需运动/上下文证据） | 候选已生成（补池已含）、阈值下 |

**总量结果**：正样本归宿 **12/13**（v2 active/audit 11 + blocked 1；F4-active 11）；负样本 17 条 **active/audit 零混入**（blocked 复核层 1 条）；全部 ≥阈值候选**零静默淘汰**（主池 179=47+100+32；弱池 56=35+21，均为硬断言）。
**覆盖率（c0 v6 试算，182 边）**：exit **0.301→0.522**、entry **0.286→0.595**；**305 门首次非零**（0.0→0.1）。

---

## 1. 触发与根因（含口径纠正纪事）

### 1.1 触发
上轮增量对账（55 判）列出 3 条未覆盖同人线索：E010↔E014、E009↔E024、E009↔E025。其中 E010↔E014 被初步标注为「候选生成漏（独立根因）」。

### 1.2 根因三段论（本轮实测）
1. **moving 过滤**（候选池缺口）：`crosscam_relay.py links` 默认 `--only-moving`，6 路事件表按 `moving = span≥0.05 且 dur≥1.0s` 过滤后生成候选。
   - E014（316 侧）：dur=1.0s、**span=0.045**（距阈值仅差 0.005 归一化单位）→ 被过滤；
   - E024（316 侧）：**单帧事件**（dur=0.0s）→ 被过滤；
   - 结果：E010→E014、E009→E024 两对从未进入 2618 候选池（`relay_candidates.json only_moving=True` 已核）。
2. **预算静默挤出**（记录缺口）：`relay_expand` 旧逻辑仅把 `sim≥0.95` 的挤出候选写入 conflict_group；`sim∈[0.943,0.95)` 被挤出者**不记录**。E009→E025（sim=0.9446）落入该盲区（另有 24 条同盲区）。
3. **外观证据不足**（独立限制）：补池后 E009→E024 的 cradio sim=0.8560——单帧碎片的嵌入质量不足，属外观通道的边界，非候选生成问题。

### 1.3 口径纠正纪事（重要）
排查初期曾用 d3b（DINOv3-B）嵌入快速复算，得 E010→E014 sim=0.6348、E009→E025 sim=0.6596——与管线记录不符。查明：**管线打分口径为 cradio v5**（`relay_rescore.py`，`cradio_scale/{cam}_emb_scale.npz`）。改用正确口径并以 3 个已知对复现锚点自证后（0.9741 / 0.9446 / 0.9545 逐位复现），才启动改动。锚点已写入 `relay_weak_candidates.py`（每次运行强校验，失败即退出）。

---

## 2. 改动清单（全部增量，v5 产物只读）

| 模块 | 改动 | 产物 |
|---|---|---|
| **`tools/relay_weak_candidates.py`（新）** | 弱事件候选通道：生成（≥1 侧 moving、排除 v5 池、dt 窗 [-1,60] 同 v5）→ cradio 打分 → 冻结 `dist_v1` 插值 w_dt/score（不重拟合） | `relay_v2/weak_scored.json`（6271 打分、缺嵌入 0、≥0.943 共 56 条） |
| **`tools/relay_expand.py`** | ① blocked **全量**显式化（消除 [0.943,0.95) 盲区；含"零静默"硬断言）② 弱通道第二相（主候选分配后填充剩余预算；弱证据不争抢主预算）③ schema v2（`tier` 字段、version `/2`、`weak_accepted` 输出） | `relay_v2/edges_v2.json`（v2） |
| **`tools/relay_states.py`** | ① 判定合并消费：labeled.csv ∪ labels.json（**19→55 判**，后者优先）② blocked 纳入（已判 或 主池高置信）③ 弱层 `require_human`（自动证据不单独放行，MUST 人工确认）④ selftest 11/11（新增 T7 弱层） | `relay_v2/edge_states.json` |
| **`tools/handover_metrics.py`** | 判定合并 + blocked 口径列（FN"显露"视图、FP blocked 列）+ tier 标签修复（`load_edges` 补 tier 字段） | `relay_v2/metrics_v1.json`（v2） |
| **`tools/relay_v2_labels_check.py`** | 判定合并列 + blocked 覆盖列 + **F4 状态列**（消费 edge_states.json） | `relay_v2/labels_incremental.json`（v2） |
| **`c0_baseline.py --edges v6`（重跑）** | 182 边（47+100+35）重算 | `c0/c0_baseline_v6.json` 更新 |

## 3. 验证结果（L0–L2 全绿）

- **L0**：`relay_states --selftest` **11/11**（T1–T7 含弱层 2 断言）；weak 脚本口径锚点 3/3 逐位复现。
- **L1（兼容硬约束）**：
  - primary **47 边逐位一致**（内置断言 vs v5 triage）；
  - main-secondary **100 边零漂移**（集合级 diff：`s_new == s_old` ✔）——弱通道只利用余量、不扰动既定分配；
  - c0 v5 自证 **0.301/0.286 逐位复现**。
- **L2（55 判对账，Go 判据）**：
  - 恢复边 **5 条全 active**（4 条排他竞争 + E010↔E014 弱通道），交叉验证 ✓；
  - 已判异人 **17 条：active 0 / audit 0**（blocked 1 条 = E042→E050，显式复核层、非接受；E049 sim=0.9272 < 阈值，正确落阈值拒绝）；
  - 正样本归宿 12/13（唯一缺口 E009↔E024，note 已写明"补池已含、外观不足"）。

## 4. 数字总览

**候选归宿（零静默，全部 ≥0.943 有账）**
- 主池：**179 = primary 47 + secondary 100 + blocked 32**（8 旧记录 + 24 新撤销静默）
- 弱池：**56/6271 = accepted 35 + blocked 21**

**F4 状态（221 条）**：active **61**（47 primary + 7 auto + **7 human-assisted**）/ candidate 143 / suspect 17（含 16 条池外负例重建 + E050）
- 本轮新增人工晋升（55 判消费）：E011↔E012、E009↔E011、E010↔E014（human-assisted 4→7）
- **延迟提交 1 条**：`305:E044→320:E010`（E=0.70 双自动证据、无人工 → 依纪律不晋升）——弱层人工门的首个实例

**c0 v6 试算（182 边，上界口径）**
- 全局：exit **0.522**（71/136）｜ entry **0.595**（75/126）
- 分机位 exit：316 **0.781** / 343 **0.786** / 318 0.577 / 323 0.457 / 320 0.222 / **305 0.1（首次非零）**
- 链：26→**37** 条；实体覆盖 19.2%→**35.5%**（146/411）；318×320 GT 核验不受影响（8/8）

**复核新增量（如实列出）**：弱通道接纳 35 条中 34 条未复核（E010↔E014 已判）；blocked 53 条（main 32 + weak 21，其中 high_sim 22）；F3 top-10 清单已含 2 条 weak 边。

## 5. 附带发现

1. **E010 簇结构印证多对多设计**：E010 有 4 条边，其中 **E010↔E015（primary, 0.974）与 E010↔E014（weak, 0.958）双双人工判同人**——"一人多次通行"真实存在，v5 单边保留、v2 双保留（后者经弱通道回升）。E014 同时连向 318:E032（弱通道 sim 0.961）与 318:E038——同一人跨 316/318 的分支结构待复核页确认。
2. **305 门（原 exit 覆盖 0.000 的最严重缺口）首现候选**：弱通道产生 4 条 305 侧边（E044/E037→320:E010、E055→320:E012、E056→323:E051，sim 0.949–0.972），其中 E044→320:E010 已达双自动证据（延迟提交）。这是 305 出门问题第一次有"外观+时序"证据的候选（此前 20 次出门零接力）。
3. **thresh 阈值下的负例照常拒绝**：E049（0.9272）、E009→E024（0.8560）等低于 0.943 的判同人对保持"阈值拒绝"——弱通道没有降低接受标准，只修复了"该进池没进池 / 该记录没记录"两类工程缺损。

## 6. 兼容与回退

- **C1** primary 逐位 ✓｜**C2** v5 产物只读 ✓｜**C4** c0 v5 数字不漂移 ✓｜判定 CSV 只读 ✓
- **产物级回退**：`_pre_s6/` 快照（edges_v2 / edge_states / metrics / labels_incremental）直接覆盖恢复；弱通道为 **opt-in**（`relay_expand.py` 不带 `--weak` 即回主池形态）。
- **已知边界**：弱通道未单独调高接受阈值（统一 0.943，避免二次拍板）；如复核后需收紧，可按 dt 主带约束（35→21）或提高弱层门槛——参数位已就绪，留待 L3 校准。

## 7. 遗留与下一步

1. **E009↔E024**（唯一缺口）：候选已生成、外观不足——恢复须运动/上下文证据（论文 H3 线或 316 门区语义）；本报告已固化其状态（`edge_states.gaps`）。
2. **34 条弱边 + 53 条 blocked 待复核**（webapp 复核页数据包未重建——并行会话保护；重建即见）。
3. **L3（241 全表）**：判定推进后 `relay_weak_candidates → relay_expand(--weak) → relay_states → handover_metrics → labels_check → c0 v6` 一键重跑（全链本地 CPU 分钟级）。
4. 弱层"延迟提交"队列（当前 1 条）随判定/复核增长——F4 已就绪。

---
**复现**：
```bash
python tools/relay_weak_candidates.py                      # 弱候选生成+打分（含口径锚点强校验）
python tools/relay_expand.py --scored output_crosscam/analysis/relay_v2/scored_v2.json \
    --sort-key score --k-a 4 --k-b 2 --weak output_crosscam/analysis/relay_v2/weak_scored.json
python tools/relay_states.py && python tools/handover_metrics.py && python tools/relay_v2_labels_check.py
DET_VIEW=clean2 python tools/c0_baseline.py --edges v6
```
