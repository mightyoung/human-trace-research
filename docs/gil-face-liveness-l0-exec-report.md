# L0 活体规则 · 执行层落地报告（2026-10-14）

> 前序：`docs/gil-face-liveness-l0l2-report.md`（L0/L1/L2 验证层：4 静态簇 + MiniFASNet 口径校准 + rPPG 负结果）。
> 本报告 = **L0 从"验证分析"推进到"执行"**：per-shot → 池（tid）→ 人（gperson）三级 flag 正式落地 + 控制台消费端集成。
> 纪律：全程只读产物、只增不改（不改 face_shot/face_best、不改 GIL）；flag 不进主链。

---

## TL;DR

- **执行工具** `tools/gil_face_liveness_apply.py` → `face_live_flags_v0.json`：全量 4491 帧 → **826 条 flagged shot**
  （R1 静态无主 254 / R2 静态低质 420 / 倒影 101 / 可用无轨迹 65，含 14 条 R2∩倒影），**26 条轨迹、21 个 gperson** 涉及。
- **人工目检定性**（12 条池内可比对条目 + 倒影）：**唯一"最佳脸本身被污染"= `323|2`（后脑勺，警示级）**；
  8 条可用条目为"轨迹含低质帧、top1 干净"（提示级）；4 条倒影已隔离（p0c 已移出人层）。
- **控制台人脸页集成**：KPI 质检卡 + 人卡徽标 + 详情质检框 + L0 质检表（13 行）+「只看质检」筛选；
  Playwright 验收 **17/17**、控制台 0 错误。
- **顺带修复**：人脸页"受限 flex 列压缩"缺陷（原有 section 亦被压至 ~2px 不可见），修复后页面恢复完整滚动
  （内容高 936px → 3454px），已加防回归断言。

---

## 1. 执行产物：三级 flag

`face_live_flags_v0.json`（output_crosscam/analysis/face/）：

| 结构 | 内容 |
|---|---|
| `summary` | 4491 帧 / 826 flagged / by_rule（R1 254、R2 420、倒影 101、无轨迹 65）/ 26 轨迹 / 21 人 / tiers |
| `per_shot` | 826 条明细（cam/frame/tid/flags/q/usable/face） |
| `tid_flags` | 26 条：flags、n_flagged、top1 状态（frame/q/usable/flags）、gperson、tier、审计注记 |
| `gperson_flags` | 21 人汇总（含 tier 归属） |
| `review` | 四级清单（见下） |

**分级定义**（执行口径）：

| tier | 定义 | 数量 | 行动建议 |
|---|---|---|---|
| `warn` 警示 | 最佳脸来源本身不可靠 | **1**（`323|2` 后脑勺） | 人审后剔除/降权该池条目 |
| `hint` 提示 | top1 干净，轨迹含低质帧（被吸收关联） | 8 | 无需行动，供人审参考 |
| `isolated` 已隔离 | 倒影实体（p0c 已移出人层，gp=None） | 4 | 池消费端应跳过 |
| `ignore` 忽略 | 非 usable 低质误检轨迹 | 13 | 无需行动 |

## 2. 池内条目定性（人工目检 + 证据留档）

关键背景：静态簇并非"孤立静物"——**R2 簇 90%+ 成员带 tid**（纹理误检被最近邻关联吸收进真实轨迹的个别帧）；
318 窗框区（R1）亦有 4 条真实轨迹的个别帧落入。

| 轨迹 | tier | 定性（目检结论） | 证据 |
|---|---|---|---|
| **323\|2** | **warn** | ★ top1（f560, q=0.562）是**后脑勺**（NPD 检错进池）；142/197 帧为背影/后脑勺低质检出；L1 spoof=0.026 未检出、L0 zone 未命中 top1 → 仅人审可判 | `crops/tid_323_2_f560_q0.56.jpg` |
| 323\|1 | hint | 75 帧门区背影/后脑勺低质检出（均非 usable）；top1（f3460, q=0.719 低头侧脸）可用 | debug_assoc/liveness/flagshot_323_1_* |
| 323\|132 | hint | 12 帧门区背影低质检出；top1（q=0.737）可用 | — |
| 318\|1 | hint | 3 帧窗框纹理误检被吸收（q≤0.61 非 usable）；top1（f40, q=0.765）真人在别位 | z8_318_1_f1250.jpg（纯窗框纹理） |
| 318\|120 / 318\|139 | hint | 1 / 2 帧窗框误检（非 usable）；top1 干净（q=0.828 / 0.875） | z8_318_139_f13800.jpg |
| 318\|128 | hint | 4 帧窗框区检出（含窗边局部模糊帧）；top1（q=0.892）干净 | z8_318_128_f13170.jpg |
| 305\|40 / 305\|53 | hint | 3 / 1 帧墙面纹理误检吸收（非 usable）；top1（q=0.708 侧脸 / 0.762 走动模糊）可用 | — |
| 316\|32、323\|91、323\|122 | isolated | **倒影 tid**：top1 即倒影帧（q 最高 0.964）；p0c 重链接已移出人层 | `crops/tid_316_32_f4760_q0.96.jpg` |
| 323\|80 | isolated | 倒影（1 帧，非 usable） | — |

> 口径说明：9 条 flagged 轨迹可用且挂人；其中 **5 条恰为对应 gperson 的最终最佳脸来源**
> （g0026/g0094/g0117/g0124/g0129）→ 控制台标 flag；另 4 条的对应人最佳脸来自其他更优轨迹（如 318|120 的
> g0053 best 来自 316|116）→ 不标（精确口径，避免误伤）。

## 3. 消费端落地（控制台人脸页）

- **数据**：`gil_face_web.py` 注入 —— `persons[].lf`（tier/flags/note）、`stats.liveness`、顶层 `liveness{tids}`；重新导出 45 人数据包。
- **页面**（`FacesView.vue`）：
  - KPI 第 6 卡「L0 质检注记 5/45 人」；
  - 人卡质检徽标（警示红 / 提示黄，hover 显示注记）；
  - 详情弹窗红色/黄色质检框（含完整定性 + 建议）；
  - **L0 质检表**（13 行：轨迹/分级/命中规则/最佳脸状态/目检说明）+ 忽略类计数说明；
  - 「只看质检」筛选（5 人）。
- **部署**：`dist.l0d.tgz` 仅换远程 dist（前端热生效，后端未动）；备份链完整（远程 `dist.bak.*` 共 12 份）。
- **验收**：`webapp/deploy/smoke_faces.cjs` **17/17（含 4 项 L0 新断言 + 1 项防回归）**、控制台错误 0；截图 `uicheck_faces/l0_{section,modal}.png`。

## 4. 顺带修复：受限 flex 列压缩缺陷（重要）

- **症状**：L0 落地验收时发现 L0 section 实际高度仅 **2px**；进一步度量：`.faces` 滚动容器内容总高被压至 936px（应为 3454px）——**页面上所有 section（多帧合成/候选合并等原有区块）同样被压至不可见**。
- **根因**：`.faces` 为受限高度的 flex 列容器，子项默认 `flex-shrink:1`；`.sect{overflow:hidden}` 使其 min-height 解析为 0 → 全部 section 在压缩分配中被压没（其他页均有「受限 flex 列子项 `flex:none`」保护，人脸页缺失）。
- **修复**：`FacesView.vue` 加 `.faces > * { flex: none; }`（与 Visualize/Cluster 页同模式）；修复后 section 高度恢复（L0 区 668px）、容器恢复正常滚动。
- **防回归**：smoke 增加断言「各 `.sect` 可见高度 > 80px」（当前 209/1474/668/298）。

## 5. 边界与后续建议

- 规则阈值仍为**经验初值**（R1/R2 参数、40px 网格），待 val 校准；定性为单批人工目检（非盲检）。
- `323|2` 的**裁决未执行**（建议人审后剔除/降权）——flag 仅为建议，不自动清理。
- L1（MiniFASNet）目前为独立产物（`liveness_l1_minifas_v0.json`），**未**并入本执行清单；如需要可下一轮合并为统一证据卡。
- 其他页面是否存在同类 flex 压缩问题**未做体检**（本次仅修人脸页）；建议纳入控制台巡检脚本。
- 复现：`python tools/gil_face_liveness_apply.py --tag p0c --rule-tag v0 --out-tag v0`
  → `python tools/gil_face_web.py --tag p0c --fusion-tag p0b --enh-tag p0b --embed-tag p0c`
  → 前端 `npm run build` → 换远程 dist。
