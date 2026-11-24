# Relay v2 实施报告（S3/S4）——F3 handover 指标 + F4 图边状态机

日期：2026-10-14 ｜ 状态：**S0–S4 完成**（S5 闭环后置）｜ 上游：`docs/relay-v2-plan.md`（§5 F3/F4）、`docs/relay-v2-s1s2-report.md`

本轮新增：`tools/handover_metrics.py`（F3）、`tools/relay_states.py`（F4）；
新产物：`output_crosscam/analysis/relay_v2/metrics_v1.json`、`edge_states.json`。
所有数字可在本地复算（`python tools/handover_metrics.py` / `python tools/relay_states.py`，秒级~分钟级）。

---

## 1. F3 handover 评估协议（metrics_v1.json）

### 1.1 ① 边级 FN / FP（判定口径，19 行）

| 口径 | FN（判同人缺边）| 说明 |
|---|---|---|
| v5（47 边） | **5/9** | 基线（与历史一致） |
| v2-active（47 primary） | 5/9 | primary 与 v5 逐位一致（预期） |
| v2-active+audit（147 边） | **1/9** | 仅剩 E010↔E014（候选生成漏，独立根因） |

FP：active=0、audit=0（已判负样本 2 条均未被任何边集纳入）。分母提示：有 active 边且已判样本 4 条（全部判同人）；负样本 2 条（样本不足，待 241 扩充）。

### 1.2 ② crossing / returning 近似分类（accepted = 147 边）

- **crossing 候选：44 个 exit 有 ≥2 条 accepted 边**（多对多分支结构，非错误本身）。最大分支 `343:E042` 5 条（3 条已判同人 + 2 条未判）、`323:E047` 5 条（全未判）、`318:E027` 4 条。**含判异人边：0 个**——已判负样本没有进入任何 accepted 集合（符合预期）。
- **returning 候选：44 个 entry 被 ≥2 条边回联**（K_b=2 上限内）。典型：`316:E053` 被 3 条认领（343:E042 判同人 + 318:E079 + 323:E074 未判）。
- 语义说明（对齐 Ristani'16）：crossing/returning 在文献中是**交接错误类型**；在本口径下它们是"多对多结构的候选观察池"，只有当其中的边被人工判异人时才升级为缺陷（当前 0）。这些候选正是 L3 评估时的复核对象。
- JSON 中存前 30 条（控制体积）；全量可从 `edges_v2.json` 秒级重算。

### 1.3 ③ 覆盖率重算（c0 口径，★★ 核心数字）★

| 口径 | exit continuity | entry attribution |
|---|---|---|
| v5（47 边） | 41/136 = **0.301** | 36/126 = **0.286** |
| v2（147 边：primary+audit） | 64/136 = **0.471** | 67/126 = **0.532** |

- **v5 自证：0.301 / 0.286 与 `c0_baseline_v5.json` 逐位一致**（同口径重算，回归零漂移）。
- **v2 试算（c0 v6 预览）：exit +17.0pp、entry +24.6pp**——这是 audit 池（100 条待复核边）的**上界性质**收益（audit 含未复核边；最终值取决于 241 判定后的复核结果）。已过 F4 晋升（11 条）的边是其确定性下界部分。
- 分机位（v5 → v2）：

| 机位 | exit | entry |
|---|---|---|
| 305 | 0.0 → 0.0 | 0.148 → 0.259 |
| 316 | 0.531 → 0.781 | 0.333 → 0.708 |
| 318 | 0.385 → 0.577 | 0.56 → 0.8 |
| 320 | 0.222 → 0.222 | 0.125 → 0.5 |
| 323 | 0.143 → 0.371 | 0.171 → 0.457 |
| 343 | 0.5 → 0.643 | 0.429 → 0.429 |

- 说明：**305 exit 仍 0.0**（最严重缺口，三因拆分是独立任务）；320 exit 被动（球机视角特殊）；其余四路 exit 全面抬升。
- **口径教训（已修正）**：本脚本首跑未设 `DET_VIEW` → 脚点落到 raw 口径，得到 0.304/0.285 的微差（分子差 1、分母差 2）。修正为 `DET_VIEW=clean2`（与 v5/v2 边集同口径）后逐位一致。**纪律：凡消费实体图/脚点的脚本必须显式声明 DET_VIEW**（已在脚本内强制 `setdefault("DET_VIEW", "clean2")`）。

### 1.4 ④ EM−ES（暂缓，列明缺口）

- 公式（Ristani ECCV'16）：EM = 单机位内匹配误差均值，ES = 跨机位误差均值，EM−ES 衡量跨机难度差。
- 缺口：需**帧级时序真值**（完整轨迹 GT）方可计算——当前仅有边级判定（19/241）。待独立短窗真值任务（审计报告 P0 差距项）完成后启用，本轮不硬算。

### 1.5 ⑤ top-10 待复核 + 存疑 8 对裁决就绪表

- **top-10 前 7 条恰好 = F4 自动晋升的 7 条边**（`318:E036→316:E021`、`323:E068→318:E069`、`316:E041→318:E069`、`323:E023→343:E015`、`316:E053→323:E085`、`323:E035→318:E052`、`318:E079→316:E048`）——"系统最自信但未复核"的边，复核性价比最高，优先送人工。
- 存疑 8 对池内状态：**active 2**（`E037↔E040`、`E024↔E029`，均在 v5 链上）、**audit 2**（`E037↔E044`、`E031↔E040`）、**pending 1**、**not_in_pool 3**（`E009↔E025`、`E003↔E009`、`E030↔E040`——低于 sim 线或超预算）。裁决表已写入 `metrics_v1.json.query_pairs`，供 webapp `#/truth` 逐对过。

---

## 2. F4 图边状态机（edge_states.json）

### 2.1 机制（本轮实施版）

```
四态：candidate → active → suspect → dormant
证据分 E = clamp(0.5 + Σ证据, [0.05, 1.0])       （Sentinel 式正下界，永不为 0）
  human_pass +0.15   人工判定 CSV 通过（真值级）
  human_fail −0.30   人工判定 CSV 否决 → 强制 suspect，永不 active
  sim_p50    +0.10   sim ≥ primary 边 sim 中位数（0.9619；"外观不弱于半数生产边"）
  dt_peak    +0.10   dt ∈ [−2, primary dt p75]（[-2, 15.5]；"时序落在生产边常见带"）
晋升：candidate 且 E ≥ 0.7 且无 human_fail → active（0.75 含人工 → human-assisted；0.70 纯自动 → auto）
降级：suspect 逾期 14 天无复验 → dormant（TG-06：不判死，保留再激活；本轮仅机制不调度）
```

门槛校准依据（非拍脑袋）：sim/dt 两档阈值均取 primary 边分布分位（p50 / p75），在 `params` 中显式记录、可复算；权重使"人工+一路自动证据"与"两路自动证据"分别恰好命中 0.75 / 0.70。

### 2.2 实例化结果（162 条边 = 47 primary ∪ 100 secondary ∪ 13 pending ∪ 2 重建负例）

| 状态 | 计数 | 构成 |
|---|---|---|
| active | **58** | 47 primary + 11 晋升（7 auto + 4 human-assisted） |
| candidate | 102 | 89 secondary + 13 pending（等证据，延迟提交） |
| suspect | 2 | `E042→E050`、`E042→E049`（人工判异人；池外重建留痕） |

- **4 条排他竞争恢复边全部晋升 human-assisted（E=0.75）**：`E042→E046`（human+dt）、`E042→E048`（human+sim）、`E038→E043`（human+dt）、`E011→E013`（human+dt）——每条都恰好"人工 + 一路自动"。
- **7 条 auto 晋升（E=0.70）**：双自动证据（高 sim + dt 主峰）且未被人工否决——这是"延迟提交"框架下的快速通道，在 F3 复核清单中排最前（§1.5），供优先人工复验（复核后升档 human-assisted 或反证降级）。
- **交叉验证 ✓**：排他竞争 4 条恢复边全 active、已判异人 2 条全 suspect——**与 L2 对账结论完全一致**（`summary.crosscheck_ok=true`）。
- **缺口记录**：`E010↔E014`（候选生成漏）列入 `summary.gaps`——状态机视角下"不存在的边"被显式追踪，不静默。

### 2.3 L0 自检（`--selftest`，9 断言全过）

candidate+2自动→active；人工否决→suspect；单支持停留 candidate（延迟提交不冒进）；primary 直通；人工+自动→human-assisted；clamp 上下界；suspect 逾期→dormant / 未逾期保持。

---

## 3. 兼容性与回归

| 约束 | 状态 |
|---|---|
| C1 primary 47 边逐位一致 | ✓（F3/F4 均只读 edges_v2；未触碰 `relay_expand.py`） |
| C2 v5 产物只读 | ✓（新增产物全部落 `relay_v2/`） |
| C4 c0 v5 数字不漂移 | ✓（F3 重算 0.301/0.286 逐位复现；`c0_baseline.py` 仅加 v6 分支为纯增量，v5 路径经 `--tag v5chk` 复跑零漂移） |
| 判定 CSV 只读 | ✓（F3/F4 均只读；未写任何既有 schema） |
| 幂等/可复算 | ✓（两脚本确定性；evidence 无 wall-time，重跑同输出） |
| 回退 | 删除 `relay_v2/{metrics_v1,edge_states}.json` 即回退，无副作用 |

## 4. 遗留与下一步

1. **L3 全表**：241 判定完成后一键重跑 F3（`handover_metrics.py`）得边级 FN/FP 全表 + 覆盖率终值；F4 重跑自动消费新判定（晋升/降级随之更新）。
2. **EM−ES**：待帧级真值；公式与缺口已在 `metrics_v1.json.em_es` 记录。
3. **F4 展示/调度**：webapp 角标（二次迭代）；dormant 的逾期调度（需时间数据，机制已实现待接线）。
4. **S5（F5 闭环）**：后置，由 H1/H2 实验计划驱动。

## 5. 消费端适配（plan §6 落实，2026-10-14 补）

### 5.1 三项适配

| 消费端 | 改造 | 验证 |
|---|---|---|
| `tools/c0_baseline.py` | 加 `--edges v6` 分支（= `edges_v2.json` 的 active+audit 147 边，试算口径）+ DET_VIEW 防御警告（不改默认行为） | **v6 出数：exit 64/136=0.471、entry 67/126=0.532**；链 26 条/实体覆盖 27.3%（v5 19.2%）；GT 318×320 仍 8/8 召回 1.000 精度；`--tag v5chk` 复跑 **v5 覆盖/链 全等零漂移 ✓**；产物 `c0/c0_baseline_v6.json` |
| `tools/portal_truth_candidates.py` | pair 生成加 `in_relay_v2` 列（active/audit/blocked/""；旧列不动）+ README 模板说明 + **口径约定固化**（见 5.3） | 重跑后 pair 241 条：in_relay_v2 非空 15（active 6 + audit 9）；241/241 thumb 非空（见 5.2 恢复纪事） |
| `webapp/dataprep/build_web_data.py` | 数据源参数化：`--relay {v5,v2}`、`--c0 {v5,v6}`、`--out-dir`（**default v5/v5 行为不变**）；v2 模式链由"与 c0 同构的贪心路径分解"重算 | **v5 回归：chains/entities/cams 字节级相同 ✓**（metrics 仅多 `c0_tag`/`relay_tag` 标注键，数据段零变化）；**v2+v6 预览：147 边（47 active+100 audit）、26 链、链长分布与 c0 v6 完全一致**，产物 `output_crosscam/_webdata_v2check/` |

### 5.2 truth 底稿恢复纪事（重要，如实记录）

**事件**：给 pair 候选补列后，为验证主脚本改动"顺手重跑"了 `portal_truth_candidates.py`——重跑输出与已判底稿**不一致**（pair 241→72、events 92/122→60/74），且 pair_id 格式变为 `343_OUT_...`（底稿为 `343_E_...`）。

**根因**：脚本在底稿生成（09:0x）之后被改动过 3 处、未同步底稿——① event_id 格式 `E`→`IN/OUT`（**远端证据**：真值数据链 `build_truth_data.py` 硬编码 `eid.split("_E_")`，IN/OUT 格式会直接崩）；② `--min-obs` 默认 0→8（剔碎片实体）；③ `--topk` 默认 0→3（每 exit 每保留 3 候选）。

**恢复**：① pair 从 `backup_pre_v2col/`（补列脚本的首跑备份）恢复；② events 靠 `webapp/backend/data/truth/state.json`（10:52 部署数据包，含 107 事件全字段）交叉核对；③ 脚本三处默认值改回底稿口径并固化 docstring 约定；④ 重跑 + 逐项验证。

**最终验证（全部通过）**：
- pair 241/241：`pair_id` 集合全等 **且 t_a/t_b/dt 逐字段全等**（0/241 差异）；thumb 列 241/241 非空；
- events：107 唯一事件 × 9 字段（t/u/v/score/n_obs/dur_s/kind/edge/cam）与 state.json **逐字段一致（0 不匹配）**；
- labeled.csv（人工判定）241 行 / 19 判完好、in_relay_v2 列就位；
- 结构说明：events 表每实体 2 行（entry+exit 共享 event_id），state.json 按 `setdefault` 取首行（entry）——两侧自洽。

### 5.3 固化约定（写入脚本 docstring）

1. 事件 id = `f"{cam}_E_{eid}"`（"E" 固定字面量，真值链路按 `_E_` 解析；kind 由列区分）；
2. 默认 `--min-obs 0` / `--topk 0` = 底稿口径（241 对 / 107 事件逐位一致）；
3. **纪律**：真值底稿类产物，改动脚本后重跑前先对账 241 对底稿；覆盖前先备份。

### 5.4 遗留

- `state.json` 数据包**未重建**（本轮不动进行中的判定页；`in_relay_v2` 已在本地 CSV 就绪，重建 `build_truth_data.py` + 上传即可让页面显示 v2 状态，属可选二次迭代）；
- `_alt_minobs8/`（留档）：min_obs=8 口径的 events + 缩略图（非底稿口径，仅追溯用）。

### 5.5 判定回流增量对账（55 判，2026-10-14 11:2x 快照）

判定已由 19 条推进至 **55 条**（labeled.csv 早期 19 + `remote/labels.json` 38；13 同人 / 17 异人 / 25 看不清）。
`tools/relay_v2_labels_check.py`（只读消费，不写公共判定文件）对 v2 增量对账：

| 口径 | 判同人覆盖 | 判异人误纳 |
|---|---|---|
| v5（47 边） | 4/13 = 30.8% | 0/17 |
| v2-active（58 边） | 4/13 | 0/17 |
| **v2-active+audit（147 边）** | **10/13 = 76.9%** | **0/17（audit 亦 0）** |

- **新增验证**：E011↔E012、E009↔E011 两条新判同人由 v2 恢复（audit）；17 条新判异人**全部零误纳**——比 19 判时代（2 条异人）强得多的安全检验。
- **剩余 3 条未覆盖（新线索）**：
  - `E009↔E025`：候选存在（sim=0.9446）但被**静默淘汰**——sim∈[0.943, 0.95) 超预算不进 blocked 显式化（conflict_sim=0.95 门槛外的挤出）→ L3 时应评估调低 conflict_sim 或加审计；
  - `E009↔E024`：**完全无候选**（候选生成漏 #2，与 E010↔E014 同类，独立根因）；
  - `E010↔E014`：已知候选漏 #1。
- **待复验 5 条**（判"看不清"但 v2 有边）：`E024↔E029`、`E037↔E040`（均 active——v5 既有边）、`E015↔E027`、`E031↔E040`、`E037↔E044`（audit）——判定页复核时的最高优先级对象。

> 说明：本节数字为**时点快照**（判定仍在进行，仅 55/241）；§1 的 19 判口径数字保留为历史版本。判定全表（241）完成后重跑本节脚本 + F3/F4 一键更新。
