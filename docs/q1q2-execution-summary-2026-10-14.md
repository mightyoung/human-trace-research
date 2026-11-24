# Q1/Q2 改造方案执行总结（2026-10-14）

> 指令：①检测头替换实验结项（保持 6 路并行，不再改）；②**完成 Q1、Q2 改造方案**。
> Q1 = 空间拓扑（TOP-1~4）；Q2 = 聚类/预测器自动评价（EVAL-1~5）。**9 项全部执行**，逐项状态与关键数字如下。

## 0. 总表

| # | 项 | 状态 | 一句话结果 |
|---|---|---|---|
| TOP-1 | 35min 全六路窄窗显著性 | ✅ | 3 条既有边复现增强 + **新发现 316↔305 双向显著**；305 时钟"单一偏移"假设被否，口径不改 |
| TOP-2 | 语义门户探针（G1–G4 前置） | ✅（探针级） | 两轮探针：调参后六路全检出，与 v7 门部分对齐；**辅助可用、替代早**；过检主因=玻璃隔断 |
| TOP-3 | 门户-边置信化 | ✅ | top-7 保留：pos 11/11、winner 4/4、active 47/47（基线 top1 仅 0/11、14/47）；**复合分负结果**（简单 sim 排序更优） |
| TOP-4 | 空间模型 Step-1（相对尺度） | ✅（Step-1） | 同人同时刻比 318→320 ≈ **1.13**（IQR 0.23，逐人位置波动 ±22%）；非重叠机位不可测 → 距离先验边界明确 |
| EVAL-1 | 六层评估规则库 v0 | ✅ | 同人 **0 误拒** / 异人 **0 误收**；四态对账 **0 硬冲突**；gperson 五动作清单（1 隔离+5 拆分+37 合并+4 暂缓+13 保留） |
| EVAL-2 | 学习式评审器对照 | ✅（判定不引入） | 扰动复现规则 96.8% **且+2 误拒** → 触发预注册失败条件 → **不引入** |
| EVAL-3 | 门控自训练三臂 | ✅（H1 首实验） | 门控防退化：分片一致 0.8971→(朴素)0.8841→**(门控)0.8986**；增益幅度小（+0.6pp 覆盖）→ H1 未否定、不足称增益 |
| EVAL-4 | 下一门户头 + 跟踪层验收 | ✅（含数据缺口披露） | 学习头 top1 **305 0.982 vs 运动学 0.482**、323 0.864 vs 0.558；relay 级验收因判对源全为 343（单出口）退化，列数据缺口 |
| EVAL-5 | 回放扩展 + 影子发布 | ✅ | 回放扩至 **5 面板**；影子 harness **guard PASS**（8 生产文件零变化）+ 影响面表 |

## 1. 各线关键读数索引

- **TOP-1**（`docs/top1-long35-significance-report.md`）：≤5s 窗 316→318 z=6.41 / 343→323 6.42 / 343→316 6.69；≤10s：5.16/4.87/4.68；新增 316→305 4.44、305→316 3.35、323→318 4.35、318→316 3.71（均跨 3/5/10s 一致）。阴性对照 [-60,-30] 干净；[-30,-10] 边缘（族 p=0.053）已记录。long35 检测系 10-14 00:24 已补跑（本地化+清洗后即用）。
- **TOP-3**（`docs/top3-portal-edge-confidence-report.md`）：197 zones；k 扫描表全列；k=7 全覆盖；置信分 vs 朴素排序对照（负结果）已入产物。
- **EVAL-1**（`docs/eval1-rules-registry-report.md`）：2618 候选 → 55 keep/124 defer/2439 reject（可声称池 179）；软冲突 13 条（reject vs candidate，均 sim<0.943）；74 判对中 **24 不在候选池**（上游缺口）；gperson 建议中 **101 条 S3 已采纳链未传导**（版本错位）+ **g0206 幽灵人复现**（5/8 倒影，与 S6 一致）。
- **EVAL-3**（`docs/eval3-gated-selftrain-report.md`）：三臂表 + 预注册判据讨论。
- **EVAL-4**（`docs/eval4-next-portal-report.md`）：L1 表 + L2 三个集合（pos/winner/active47）。
- **EVAL-5**（`docs/eval5-replay-shadow-report.md`）：run 命令、面板口径、护栏说明。
- **TOP-2**（`docs/top2-semantic-portal-probe-report.md`）：两轮探针数字 + G1-G4 下一步。
- **TOP-4**（`docs/top4-rel-scale-report.md`）：口径修正记录 + 数据表 + Step-2/3 衔接。

## 2. 新增文件清单

**工具（12）**：`topo_frag_narrow.py`、`build_long35_topo_data.py`、`topo_ztable_2col.py`、`topo_clock_scan_summary.py`、`cluster_eval_registry.py`、`eval_registry_diag.py`、`portal_edge_confidence.py`、`eval3_gated_selftrain.py`、`eval4_next_portal.py`、`eval2_learned_reviewer.py`、`top2_door_probe.py`、`top2_door_compare.py`、`top4_rel_scale.py`、`shadow_run.py`（共 14；replay_eval.py 扩展 5 面板）。
**产物（含目录）**：`analysis/topology/{topo_frag_narrow_v10,v35,v35_neg}.json`、`topo_ztable_2col.json`、`clock_scan/`、`long35_data/`、`portal_graph_v2.json`；`analysis/eval_registry/registry_v0.json`；`analysis/eval3/self_train_arms.json`；`analysis/eval4/next_portal.json`；`analysis/eval2/learned_reviewer.json`；`analysis/top2/*`；`analysis/top4/rel_scale.json`；`analysis/shadow/<ts>/shadow_report.json`。
**报告（8）**：本文件 + 上述七份。

## 3. 遗留与建议（按优先级）

1. **判读集补口**（EVAL-4 的验收缺口 + EVAL-1 覆盖缺口）：增加 316/305/323 出口侧的交接判读对（现 55 判源几乎全为 343）；
2. **S3 链传导修复**（EVAL-1 发现）：101 条已采纳碎片链未进当前身份层（版本错位）——建议立项核对；
3. **发布动作**（c0 v8 / relay 边更新 / 预测先验接入）：均需单独立项与授权；
4. TOP-2 走向 G2 全量（多提示词×多帧×轨迹穿越）与 TOP-4 Step-2（局部几何升级）；
5. 既有长名单不变：34 弱边+53 blocked 待复核、L3 全表等。

## 4. 纪律记录

- 全部新增产物**并列不覆盖**（196 边 / v2 边 / c0 v7 / selector_v2 逐位保留；shadow guard PASS）；
- DET_VIEW=clean2；无真值调参声明逐报告标注（阈值取既有冻结工作点）；
- 检测头替换实验已结项：训练队列停止、产物留档、生产检测口径未变。
