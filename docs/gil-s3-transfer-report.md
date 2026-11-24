# S3 链传导报告：同机位 frag 划分 → gperson 身份层（S3-b）

> 日期：2026-10-14 ｜ 工具：`tools/gil_gperson.py`（新增 `--s3-links` 开关）+ 审计链 `gil_s3_transfer_{audit,probe,sim}.py` / `gil_gperson_{ab_check,trace}.py` / `gil_s3_quality_audit.py`
> 产物：`analysis/gil/gperson_v2.json`、`s3_transfer_audit.json`、`s3_transfer_sim.json`
> 一句话：**把 S3（同机位碎片接续）已采纳的 268 条链传导进身份层**——gperson 264→158，同机位层 F1 val +0.12，实体一致性修复 20 处；已落库（含备份与回退路径）。

---

## 0. 结论速览

1. **修复了 EVAL-1 发现的"版本错位"**：S3 报告 §9 明确"frag 层是 gperson 的上游；需重跑 gperson"。本报告即该重跑——把 S3 的 frag 划分（268 边 / 278 frag）作为同机位层的**权威证据**注入聚类（`--s3-links`），其余约束（时间互斥 / 规模上限 / 跨机位门 0.943）不变。
2. **传导率 99.2%**（253/255 对最终同簇；255 = 268 对剔除 13 个端点为倒影的链）。
3. **gperson 264 → 158**（−40%）；frag 精化链被完整消化。
4. **实体一致性修复**：86 个多 obs 实体中被 gperson 拆分者 **47 → 27**（修复 20 个；剩余 27 个的典型 = 318-E003 跨 5 簇，待后续）。
5. **val 同机位对（共同可评 57 pos / 365 neg）**：F1 **0.406 → 0.526**（R 0.467→0.702）；对 cap12（未排倒影）对照 0.5229→**0.5263**（v2 略优：负对 FP 56→55）。
6. **人工真值（12 条可评）P 1.000 保持**；净变化 = 修复 1 对（343-E010↔316-E014）+ 破坏 1 对（343-E009↔316-E025）+ 退出 1 对（E015 涉倒影）——**全部为贪心重排效应**（与 S6-b 同性质），**不得宣称召回提升**。
7. **318×320 GT 4/8 → 5/8**；不变量违规 0；互斥审计 0。
8. **下游已重建**：visit 846→658、owner 505→491（0 违规 / 0 悬空引用）；webapp 载荷 `gil_owner.json` + `gil.json` 已更新（旧 gil.json 是"迁移初版：gperson=0"的快照，本次补齐）。

## 1. 背景与缺口诊断

- **EVAL-1（规则库）**：gperson merge 建议 37 簇共 **101 条 S3 已采纳链未传导**（版本错位）。
- **本次对账**（`gil_s3_transfer_audit.py`）：268 条链中 **73 条已同簇 / 195 条跨当前 gperson**；195 条展开为 33 个缺口连通分量（Top1 分量横跨 6 机位、91 个 gperson——链式连通上界，非"应合成一人"）。
- **entity 级**：**47/411 个实体被 gperson 拆分**（27% 的多 obs 实体）——S2 时代的同机位层（cos 阈值 T_same）无法吸纳 S3 的几何证据（`pe_adj`+`d_h`+路径覆盖）。

## 2. 方法（最小侵入实现）

`gil_gperson.py --s3-links <frag_link_v1.json>`：

- **链对注入**：268 条链中端点均存在的对 → 加入同机位候选，margin 设 ≥0.5（**高于所有普通对**，保证 frag 单元先成型）；缺失 13 对（端点为倒影 obs，被 `--drop-reflections` 排除）。
- **约束不变**：时间互斥（硬）、规模上限 `--max-size`、跨机位门（cos+dt，0.943 冻结）、318×320 豁免。
- **★ 冻结工作点铁律**：`--drop-reflections` 会改变 val 对集（435→365 负对）→ **必须 `--t-same 0.975` 冻结**（否则重演 S6-b 的"阈值漂移假变"，实测漂移值 = 0.960）。
- **`--apply` 更新**：保留 `state='non_person'` 行（gREFLECTION，由并行会话集成）；membership src 版本化（`gperson_v2:+s3links+droprefl+cap12`）；幂等（DELETE 匹配 `gperson_v1%|gperson_v2%`）。

## 3. 参数对照（A/B 全表）

| 版本 | T_same | drop | cap | 簇数 | 传导率 | 违规 | GT | 可评 | TP/FP/FN |
|---|---|---|---|---|---|---|---|---|---|
| base（现行 v1） | 0.975 | – | 8 | 264 | – | 0 | 4/8 | 13 | 7/0/6 |
| cap8 | 0.975 | – | 8 | 164 | 98.1% | 0 | 5/8 | 13 | 6/0/7 |
| cap10 | 0.975 | – | 10 | 160 | 99.3% | 0 | 5/8 | 13 | 6/0/7 |
| cap12 | 0.975 | – | 12 | 155 | 99.3% | 0 | 5/8 | 13 | 7/0/6 |
| combo（重校准0.960） | 0.960⚠ | ✓ | 12 | 133 | 99.2% | 0 | 5/8 | 12 | 6/0/6 |
| **v2（采用：冻结+drop）** | **0.975** | **✓** | **12** | **158** | **99.2%** | **0** | **5/8** | **12** | **6/0/6** |

- **cap 选择依据**：S3 最大 frag = 10 段 → cap≥10；cap12 = "10+2"（容纳链组及其跨机位成员）。cap8/10 各有 1 对因容量截断（343-E042↔316-E046）；cap12 全对通过。
- **drop 选择依据**：在**共同可评集**上（57 pos/365 neg，修正对账口径后——见 §6），drop 无代价且负对 FP 少 1；与 gREFLECTION"倒影非独立节点"原则一致。
- **重校准版（0.960）仅作对照**：它"多修复" 343-E038↔316-E043 但"丢" E042↔E046，净 TP 相同——**不可比（工作点漂移）**，弃用。

## 4. 验收明细

| 项 | base | v2 | 说明 |
|---|---|---|---|
| gperson | 264 | **158** | +1 non_person（gREFLECTION）|
| S3 链传导率 | – | **99.2%**（253/255） | 2 对因交叉阻断未同簇 |
| 多 obs 实体被拆 | 47 | **27** | 修复 20（318-E003 等 27 个待查）|
| val（共同可评） | F1 0.406 | **F1 0.526** | pos +12、neg FP +5（子集口径）|
| 人工真值 | P 1.000 / R 0.538(13条) | **P 1.000** / 12 条 TP6 | 修复1+破坏1+退出1（重排）|
| 318×320 GT | 4/8 | **5/8** | |
| 不变量（互斥） | 0 | **0** | |
| visit / owner | 846行 / 505 | **658行 / 491** | 0 违规 / 0 悬空 |
| relay 边同簇态 | 62 | **54** | 28 条变化（含 S6-b 桥接拆开；分歧清单重组）|

**门户覆盖（口径注意）**：身份级 145→126 出口、覆盖 55→41（0.379→0.3254）。**口径已变**（合并使"出口"去重、部分"跨机位后续"被吸收进同一人），不作优劣判断；305 仍为结构性低覆盖（0.147→0.067，S-1 结论不变）。

## 5. 落库与回退

- **备份**：`store_v1/gil.sqlite.bak_pre_s3v2_20261014`（backup API 一致性快照，含 gperson 265 / membership 619）。
- **落库命令**：
  ```bash
  DET_VIEW=clean2 python tools/gil_gperson.py --s3-links output_crosscam/analysis/gil/frag_link_v1.json \
      --drop-reflections --max-size 12 --t-same 0.975 --apply --out output_crosscam/analysis/gil/gperson_v2.json
  DET_VIEW=clean2 python tools/gil_visit_owner.py --apply --out output_crosscam/analysis/gil/visit_owner_v2.json
  ```
- **落库后**：gperson 159（158 candidate + 1 non_person）/ membership 607（534 obs + 73 refl）；visit 658 / owner 491；**冒烟 9/9**（前后各一次）。
- **回退**：轻 = 重跑 `gil_visit_owner.py --apply`（旧 membership）+ `gil_gperson.py --apply`（旧参数）；全 = 恢复备份库。
- **webapp 数据**：`gil_owner.json`（v2，534 节点/491 owner/329 visit）+ `gil.json`（重建：**旧版是迁移初版快照 gperson=0**）。

## 5b. EVAL-1 复跑验收（传导后，registry_v1）

`cluster_eval_registry.py --name registry_v1`（数据源改为**跟随库最新 membership**；兼容 S6B_* 裁决判读格式）：

| 项 | v0（传导前） | v1（传导后） |
|---|---|---|
| **merge_pairs（未传导链）** | **101** | **1**（版本错位基本清零）|
| **quarantine（幽灵人）** | 1（g0206：5/8 倒影） | **0**（倒影排除后消失）|
| **split（倒影污染簇）** | 5 | **0** |
| 动作分布（簇） | q1/s5/merge37/defer4/keep13 | **keep 134 / defer 22 / merge 2** |
| 55 判 | pos keep 8/21、**0 误拒**；neg reject 13/16、**0 误收** | pos keep 5/11、defer 6、**0 误拒**；neg reject 10/11、**0 误收** |
| 四态对账 | 0 hard / 13 soft | **0 hard / 13 soft**（保持）|

→ **S3 传导与倒影排除两个修复均在规则库验收上闭环**；剩余 22 defer（大簇/跨机位审计项）+ 2 merge 转入人工复核名单。

## 6. 过程记录（诚实项）

1. **统计口径 bug（已修）**：val 对账初版用 `g.get()==g.get()` 比较，被排除节点（None）与 None 相等 → 误判"同簇"；修正为"任一端无归属 → 不可评"。修正前 v2 的 val "F1 0.485" 是假数（实际 0.5263）。
2. **T_same 漂移（已改）**：drop 后重校准 → 0.960；按 S6-b 铁律冻结 0.975。
3. **库活动共享**：本工作与并行会话（倒影线：gREFLECTION / reflection_score / S6-b 裁决视图）共用 GIL 库；其 17:31 落库先于本落库，本落库保留其 non_person 行（双向兼容，互不覆盖）。
4. **重排效应**：343-E009↔E025 的"破坏"与 E010↔E014 的"修复"均为贪心合并顺序变化所致，非语义退化；与 S6-b"唯一变化对"同类（该二对处于合并边界）。

## 7. 遗留

- **剩余 27 个被拆实体**（318-E003 跨 5 簇等）：建议排查（318 为小目标机位，cos 不稳）；
- **343-E009↔E025 边界对**：进下一批人工复核（判读集补口已在并行推进：labels.json 20→45 条）；
- **316|32 / 323#212 三票分歧**：仍待裁决（S5/S6-b 遗留）；
- **relay 边 28 条重排**：审计边为"分歧清单"更新，非真值裁决。

## 8. 复现

```bash
# 只读 A/B
DET_VIEW=clean2 python tools/gil_gperson.py --s3-links output_crosscam/analysis/gil/frag_link_v1.json
DET_VIEW=clean2 python tools/gil_gperson.py --s3-links output_crosscam/analysis/gil/frag_link_v1.json --drop-reflections --max-size 12 --t-same 0.975 --out output_crosscam/analysis/gil/gperson_v2.json
# 缺口诊断 / 模拟 / 对账 / 质量审计
python tools/gil_s3_transfer_audit.py
python tools/gil_s3_transfer_sim.py
python tools/gil_gperson_ab_check.py
python tools/gil_s3_quality_audit.py
# 冒烟
DET_VIEW=clean2 python tools/gil_smoke.py     # 9/9
```
