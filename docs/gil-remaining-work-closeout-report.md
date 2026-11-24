# 遗留工作收尾报告（B / A / C / D）

> 2026-10-14 · 承接 `docs/gil-human-adjudication-ingest-report.md` §7.5④ 与当日日志中标记的「待决策 / 待吸收」
> 前提：**已确认无并行任务占用**；每项均带备份与幂等/回退路径。

---

## 0. 摘要

| # | 事项 | 状态 | 结果 |
|---|---|---|---|
| **B** | 4 条误报倒影的 `obs.flags` 撤销是否落主库 | ✅ 完成 | flags 判定倒影 **12 → 8**；4 条标注 `reflection/wall_mirror → person`；annotation 总数 **809 守恒** |
| **A** | GIL relay 镜像是否含人工裁决 | ✅ 完成 | GIL 由 `47 active/135 audit` → **36 active / 126 audit / 10 confirmed / 10 rejected / 14 gate**，与磁盘边集**逐条一致**；幂等 |
| **C** | 人脸 must-link 能否吸收进 gperson 重建 | ✅ 完成（**负结果**） | 5 簇中 5 组对与**时间互斥**冲突 → 只能部分合并；人数 158→159、人工真值 TP 16→15（F1 0.653→**0.625**）→ **不落库** |
| **D** | `render.py` 导出视频仍按 tid 画框 | ✅ 完成 | 新增 `arm=A\|B\|H`：按身份口径去重（同一人只画责任机位框）+ 标签换 `g####`；316 机位采样 **88→78 框** |

---

## 1. B：撤销 4 条误报倒影（落主库）

```bash
python tools/gil_adjudication_ingest.py --apply --apply-entities
```

- 修正 5 条观测：`316|32 / 323|10 / 323|4 / 323|50 / 323|153`
- `obs.flags`：清除 `reflection / explained_by / wall_mirror / wall_mirror_src / wall_mirror_u_axis`，
  保留 `reflection_score`（作反向证据）并加 `human_revoked_reflection: true`
- `annotation`：`obs/reflection 49→48`、`obs/wall_mirror 24→21`、`obs/person 528→532`（**总数 809 不变**）
- 其他表（obs/gperson/membership/edge/visit/owner）**零改动**
- 备份 `gil.sqlite.bak_pre_entities_<ts>`

**意义**：主库 `--drop-reflections` 现在**不再需要 `--verdicts` 就能得到 8 条**（与人工裁决自洽）。

---

## 2. A：relay 镜像同步到 GIL

```bash
python tools/gil_relay_mirror_sync.py [--apply]      # 新工具，默认 dry-run
```

| | 改前 | 改后 |
|---|---|---|
| active | 47 | **36** |
| audit | 135 | **126** |
| confirmed | — | **10** |
| rejected | — | **10** |
| relay_gate/gate | 14 | 14 |
| 合计 | 196 | 196（不删记录） |

- 20 条变更 **全部命中**（0 未匹配）；`evidence` 写入 `prev_status` / `human_verdict` / `human_sync_ts`（**可回退**）
- 幂等（重跑 0 变更）；GIL 冒烟 **9/9**（悬空端点 0）
- 备份 `gil.sqlite.bak_pre_relaysync_<ts>`
- ★ 消费端影响评估：**C0 读的是磁盘边集**（`edges_v2_gated_human.json`），**不读 GIL**；
  GIL relay 表仅冒烟/诊断/分析工具在用 → 同步**不影响任何生产链路**。

---

## 3. C：人脸 must-link 吸收 —— 负结果（重要发现）

工具扩展：`gil_gperson.py --constraints face_mustlink_v1.json`（读 must_link 簇 → 强制合并；cannot_link → 禁止；**内存生效**）。

**同序列 A/B（cap12 + 人工裁决 + 冻结 T_same=0.975）**：

| | 基线 | + 人脸约束 |
|---|---|---|
| gperson | 158 | **159** |
| 人工真值 | TP 16 / FP 1 / FN 16 → P 0.941 / R 0.500 / **F1 0.653** | TP 15 / FP 1 / FN 17 → P 0.938 / R 0.469 / **F1 0.625** |

**根因：5 个 must-link 簇里有 5 组对与「时间互斥」硬约束冲突**（同机位时间重叠 → 物理上不可能是同一人）：

| 簇 | 冲突对 | 时间区间 | 结果 |
|---|---|---|---|
| A | `305\|29` × `305\|42` / `305\|33` | `[80.4,190.8]` 覆盖 `[149,184]`、`[83.4,147.8]` | 部分合并（gid 0 + 19） |
| B | `305\|68` × `305\|84` | `[270,499.8]` × `[474.2,499.6]` | 部分合并（gid 18 + 45） |
| C | `323\|149` × `323\|151` | `[357.2,391.0]` **含** `[365.8,369.4]` | **完全未合并** |
| D | — | — | ✅ 本来同簇 |
| E | `343\|30` × `343\|31` | `[301.6,304.0]` × `[301.8,306.2]`（几乎完全重叠） | **完全未合并** |

**判读**：代码正确跳过了冲突对（`incompat` 优先），所以只能部分吸收。
★ 簇 C/E 的时间几乎完全重叠 → **更像"GIL 把同一个人拆成了两条轨迹"（分割问题），而非人脸误判** ——
这两者必须靠人工看视频区分，`must-link` 与 `time-exclusive` 谁对，数据本身判不了。

**结论：不落库。** 诊断产物 `analysis/gil/face_constraint_absorption_v1.json`（含逐簇冲突明细）。
→ **建议**：把这 5 组冲突对做成前端待裁决项（同一人被分割 / 人脸误判），而不是直接施加约束。

---

## 4. D：`render.py` 身份臂画框

`webapp/backend/render.py` + `main.py`（**已上传并重启服务**，md5 本地↔远程一致）：

- 新增 `load_arm()` / `_cam_delay()` / `_apply_arm()` / `_arm_labels()`
- `render_frame(..., arm=)` 与 `export_video(..., arm=)` 支持 `A|B|H`：
  - **去重**：某 obs 的 gid 在该时刻的 owner 机位 ≠ 当前机位 → 不画该框
  - **标签**：`#tid` → **`g####`**（全局人编号）
  - 轨迹线同步去重（与框一致）
- 前端 `api.ts: frameUrl(..., arm)` 与导出 API 同步支持（**默认不传 → 行为不变**）
- 载荷位置自动查找：`frontend/public/` → `frontend/dist/`

**验证**（远程采样 75 帧 × 5 机位）：

| 机位 | 原始框 | A | B | H |
|---|---|---|---|---|
| 316 | 88 | **79** | **78** | **78** |
| 320 | 32 | 30 | 30 | 30 |
| 318 | 303 | 301 | 301 | 301 |
| 323 | 258 | 257 | 256 | 256 |
| 343 | 35 | 33 | 33 | 33 |
| 305 | 476 | 476 | 476 | 476 |

→ 过滤生效（316 最强，符合"重叠视野被降为副机位"）；305 为强机位故无剔除。
目视确认：`webapp/deploy/_a_noarm.jpg`（`#9 #29 #33 #4 #27 #2`）↔ `_b_armH.jpg`（`g0038 g0006 g0020 g0025 g0016 g0014`）。

**验收**：`verify_arms.cjs` **18/18**；`smoke_faces.cjs` **19/19**（确认未伤及人脸页）；`shot_ui.cjs` **ERRORS 0**。

---

## 5. 备份与回退

| 环节 | 备份 |
|---|---|
| B | `gil.sqlite.bak_pre_entities_<ts>` |
| A | `gil.sqlite.bak_pre_relaysync_<ts>`（+ `evidence.prev_status` 可逐条还原） |
| D | 远程 `render.py.bak.pre_arm.<ts>` / `main.py.bak.pre_arm.<ts>`；前端 `dist.bak.arm3.<ts>` |

---

## 6. 仍未做 / 建议

1. **C 的 5 组时间冲突对** → 建议做成前端待裁决（这是当前唯一"数据判不了、必须人看"的点）。
2. `reflection_score` 仍未接进 `build_selector_v2.py` 本体（需一次同序列冻结的 A/B）。
3. `visit` 门户语义（entry/exit 配对接 relay 状态机）未动。
4. D 的 `arm` 目前只在 API 层可用（前端无导出按钮）；需要时可在可视化页加「导出该口径视频」按钮。
