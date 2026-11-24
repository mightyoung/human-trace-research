# 论文数字台账（V1 登记锚定 · V2 真值与协议）

> 由 `tools/ops_claims.py` 生成；权威是 `docs/paper/EVIDENCE.md` 与 tex，本报告只反映索引状态。

## 1. claim 与锚定

| 项目 | 数值 |
|---|---|
| `paper_claim` 行数 | 238（表格行 162 / 说明行 76） |
| 已挂 tex 锚点的 claim | 91（117 个表格行） |
| 登记的文件版本 `asset_version` | 129 |
| 表格数字中 EVIDENCE 全文没有的 | **0** |
| 表格数字中有证据但本行未挂到的 | 0 |
| 整行声明“无证据行”：0 行 / 0 个数字 | **合计需补 0 个** |

## 2. 按来源等级与裁定

| 裁定 | 来源等级 | 条数 |
|---|---|---:|
| match | data | 57 |
| match | doc | 3 |
| match | none | 6 |
| pending | data | 38 |
| pending | doc | 34 |
| pending | none | 65 |
| unverifiable | data | 25 |
| unverifiable | doc | 9 |
| unverifiable | none | 1 |

## 3. 文件版本分布

| kind | 条数 | 其中弱身份 |
|---|---:|---:|
| derived | 108 | 0 |
| doc | 1 | 0 |
| labels | 1 | 0 |
| media | 6 | 0 |
| script | 13 | 0 |

## 4. 表格数字里 EVIDENCE 没有的（V2 要补行）

| 文件 | 表 | 行 | 数字 |
|---|---|---|---|

## 5. 有证据但本行未挂到的（改锚点即可）

| 文件 | 表 | 行 | 数字 |
|---|---|---|---|

## 6. 整行声明“无证据行”的（补 EVIDENCE 或改口径说明）

| 文件 | 表 | 行 | 数字 |
|---|---|---|---|

## 7. 真值集（V2）

> 条数由 `ops_claims truth` 打开文件实测；与登记值不符会在导入时直接报错。T1 与 T2 是两个总体，**不可合并计算精度**。

| 集合 | 条数 | 同 | 异 | 不决 | 抽样方式 | 文件 sha256 |
|---|---:|---:|---:|---:|---|---|
| T1 | 126 | 34 | 92 | 0 | prior-sampled：按系统认为像的对抽样（规则测 | `8d7b8745e9c1` |
| T2 | 27 | 12 | 15 | 0 | contradiction-triggered：已合并且 | `6e4fd315e528` |
| audited_groups | 15 | 8 | 7 | 0 | T2 按跨机位子集 multi 的 group 聚合；m | `6e4fd315e528` |
| gallery_double | 257 | 0 | 0 | 0 | 图库层逐轨命名，两位标注人各自独立编号后对齐（28 映射 | `11062060c0cf` |
| gallery_round1 | 257 | 0 | 0 | 0 | 图库层逐轨命名，第一版（2026-10-25）盲标对齐， | `10c500302f46` |
| gallery_round1_person | 236 | 0 | 0 | 0 | 同上，剔除 shadow/cloth 等非人物命名后的子 | `10c500302f46` |
| judged_merges | 76 | 46 | 17 | 13 | attribute-gate 精度：76 次合并逐条判读 | `0d27e01bc239` |

## 8. 评测协议（V2）

| 协议 | 状态 | 身份隔离 | 清单文件数 | 核验 |
|---|---|---|---:|---|
| package-v3 | recorded | none | 60 | ok 60 / 缺 0 / 不符 0 |

## 9. 复算（V3）

> 由 `ops_claims verify` 按 `ops/claims/verify_specs.json` 重算；「论文值」列是 EVIDENCE 的写法，**不因复算结果改动**。

| claim | 论文值 | 复算 | 判定 | 差异 |
|---|---|---|---|---|
| E01-004 | 6；(305/316/320/343)=2560×1 | cams: 6, 305, 316, 318, 320, 323, 343；resolution: 4, 2560, 1440, | **match** |  |
| E01-005 | 2099 s / 25 fps | duration: 2099, 25 | **match** |  |
| E01-006 | 1,894 | tracks: 1894 | **match** |  |
| E01-007 | 1,574 | members: 1574 | **match** |  |
| E01-008 | 443 | attr8[0] = 443 | **match** |  |
| E01-009 | 56 | crosscam: 56 | **match** |  |
| E01-010 | 67 轨 | largest: 67 | **match** |  |
| E01-011 | 445 轨 / 26 身份（中位 14 轨/身份，最 |  | **unverifiable** | 不可复算：三份 gt 副本（tmp/full/gt_8seg.json 494 轨、gt_8seg_fixed.json |
| E02-015 | 126 | T1.n_items = 126；T1.n_same+n_diff = 126 | **match** |  |
| E02-016 | 27 | T2.n_items = 27；T2.n_same+n_diff = 27 | **match** |  |
| E02-017 | 153 | T1+T2.n_items = 153 | **match** |  |
| E03-023 | 0.490 | 0.499, 0.532, 0.490, 0.350, 0.625 | **match** |  |
| E03-024 | 0.531 | 0.022, 0.015, 0.531, 0.403, 0.666 | **match** |  |
| E03-025 | 0.511 | 0.016, -0.011, 0.511, 0.397, 0.626 | **match** |  |
| E03-026 | 0.499 | 0.759, 0.777, 0.499, 0.330, 0.658 | **match** |  |
| E03-027 | 0.622 | 0.160, 0.029, 0.622, 0.470, 0.765 | **match** |  |
| E03-028 | 0.450 | -0.052, 0.012, 0.450, 0.321, 0.585 | **match** |  |
| E03-031 | 11/52 | 11, 52, 0.904 | **match** |  |
| E03-032 | 2/5 | 2, 5, 0.600 | **match** |  |
| E03-033 | 3/4 | 3, 4, 0.250 | **match** |  |
| E03-034 | 13/18 | 13, 18, 0.231 | **match** |  |
| E03-035 | 1/1 | 1, 1, 0.000 | **match** |  |
| E03-036 | 2/0, 2/0 |  | **unverifiable** | 不可复算：AUC 未定义（0 个异人）。同/异对数 2/0 与 2/0 与 emb_auc_out.txt 的分机位对块 |
| E04-039 | 150 / 1,894（7.9%） |  | **unverifiable** | 不可复算：口径并存且未与 1,894 轨全集对齐：`tmp/full/face/usable_tracks.json`（ |
| E04-042 | 52 px |  | **unverifiable** | 不可复算：现存 `tmp/full/face/face/*.jsonl` 的中位脸宽为 37 px（全部观测 1286  |
| E05-057 | 434 | base: 434, 76, 5, 34, 0, 80, 46, 63, 0.730 | **match** |  |
| E05-058 | 443 | attr8: 443, 67, 5, 34, 0, 80, 40, 43, 0.930 | **match** |  |
| E06-062 | 同段 53.8% / 同机位跨时段 42.3% /  | 同段 = 53.8；同机位·跨时段 = 42.3；跨机位 = 3.7 | **match** |  |
| E06-063 | 同段 5.5 轨/人；断口 63% >10 s |  | **unverifiable** | 不可复算：只在 docs/mainline-break-appearance-2026-11-10.md 记录，生成脚本 |
| E06-068 | 0 | s0: 56.4, 0 | **match** |  |
| E06-069 | 0 | s5: 68.2, 0 | **match** |  |
| E06-070 | 65 | s0+gt: 79.3, 65 | **match** |  |
| E06-071 | 65 | s5+gt: 83.2, 65 | **match** |  |
| E06-072 | 19 | s5+real: 69.8, 19 | **match** |  |
| E06-073 | — |  | **unverifiable** | 不可复算：上限口径未留存中间产物：该值由「当天该身份全部其他轨（中位 22 条，含下午，非因果）」重算，脚本与分簇结果均 |
| E06-082 | 27 | gt_spacetime_items.json.items = 27 | **match** |  |
| E06-087 | 12 同 / 15 异 | T2.n_same = 12；T2.n_diff = 15 | **match** |  |
| E06-088 | 15 组中 7 混 / 8 纯 | audited_groups.n_items = 15；audited_groups.n_diff = 7；audited_gr | **match** |  |
| E06-089 | 20–30 min / 27 项 |  | **unverifiable** | 不可复算：人工耗时是操作记录（20–30 min 为 2026-11-23 判读批次的实测区间），无机器可读计时产物；同 |
| E06-092 | 443 | base: 443, 67, 5, 34, 0, 80, 0.930 | **match** |  |
| E06-093 | 443 | osd: 443, 67, 7, 34, 1, 80, 0.907 | **match** |  |
| E06-094 | 444 | covis60: 444, 66, 7, 34, 1, 80, 0.905 | **match** |  |
| E06-095 | 444 | covis300: 444, 66, 7, 34, 1, 80, 0.907 | **match** |  |
| E07-108 | 450 |  | **unverifiable** | 不可复算：组数 450 是人工图着色修正后的值（verifier_split.json 的 v0 记 443，差 7 组 |
| E07-109 | 475 | v2@0.2: 475, 4, 10, 12 | **match** |  |
| E07-110 | 483 | v2@0.25: 483, 4, 10, 12 | **match** |  |
| E07-111 | 491 | v2@0.3: 491, 2, 9, 12 | **match** |  |
| E07-112 | 511 | v2@0.4: 511, 1, 7, 12 | **match** |  |
| E08-115 | 10-22 | 10-22__to__2026-10-22: 0.491, 0.904, 5.2, 523 | **match** |  |
| E08-116 | 10-23 | 10-22__to__2026-10-23: 0.490, 0.891, 10.7, 363 | **match** |  |
| E08-117 | 10-26 | 10-22__to__2026-10-26: 0.367, 0.882, 9.8, 112 | **match** |  |
| E08-118 | 10-23 | 10-23__to__2026-10-23: 0.580, 0.809, 8.8, 375 | **match** |  |
| E08-119 | 10-22 | 10-23__to__2026-10-22: 0.446, 0.775, 20.0, 70 | **match** |  |
| E08-120 | 10-26 | 10-23__to__2026-10-26: 0.367, 0.745, 36.4, 22 | **match** |  |
| E08-121 | 10-26 | 10-26__to__2026-10-26: 0.650, 0.836, 7.2, 264 | **match** |  |
| E08-122 | 10-22 | 10-26__to__2026-10-22: 0.396, 0.756, 30.0, 20 | **match** |  |
| E08-123 | 10-23 | 10-26__to__2026-10-23: 0.383, 0.730, 41.2, 17 | **match** |  |
| E08-124 | 10-22 |  | **unverifiable** | 不可复算：分簇份额向量未留存：matrix.json 与逐格 CSV 只有 coverage/same/fm/ev，H/ |
| E08-125 | 10-23 |  | **unverifiable** | 不可复算：分簇份额向量未留存：matrix.json 与逐格 CSV 只有 coverage/same/fm/ev，H/ |
| E08-126 | 10-26 |  | **unverifiable** | 不可复算：分簇份额向量未留存：matrix.json 与逐格 CSV 只有 coverage/same/fm/ev，H/ |
| E08-127 | 10-23 |  | **unverifiable** | 不可复算：分簇份额向量未留存：matrix.json 与逐格 CSV 只有 coverage/same/fm/ev，H/ |
| E08-128 | 10-22 |  | **unverifiable** | 不可复算：分簇份额向量未留存：matrix.json 与逐格 CSV 只有 coverage/same/fm/ev，H/ |
| E08-129 | 10-26 |  | **unverifiable** | 不可复算：分簇份额向量未留存：matrix.json 与逐格 CSV 只有 coverage/same/fm/ev，H/ |
| E08-130 | 10-26 |  | **unverifiable** | 不可复算：分簇份额向量未留存：matrix.json 与逐格 CSV 只有 coverage/same/fm/ev，H/ |
| E08-131 | 10-22 |  | **unverifiable** | 不可复算：分簇份额向量未留存：matrix.json 与逐格 CSV 只有 coverage/same/fm/ev，H/ |
| E08-132 | 10-23 |  | **unverifiable** | 不可复算：分簇份额向量未留存：matrix.json 与逐格 CSV 只有 coverage/same/fm/ev，H/ |
| E11-154 | 76 次合并逐条判读：34 同 / 41 异 / 1 | {'user': 'same'} = 34；{'user': 'diff'} = 41；{'user': 'unsure'} = | **match** |  |
| E11-155 | 46/63 = .730；40/43 = .930 | 46/63 = 0.7302, CP[0.6035, 0.8343]；40/43 = 0.9302, CP[0.8094, 0. | **match** |  |
| E11-157 | 无门 76 次合并/63 带判读；有门 67/43  | judged_merges.n_items = 76；judged_merges.n_items-n_unlabeled = 6 | **match** |  |
| E11-167 | 部署工作点，未在 179 条上调；0.70/0.75 |  | **unverifiable** | 不可复算：四档 τ 的召回排序可由 tmp/full/pose_pool_real.json 逐档复核，但「部署工作点未 |
| E11-174 | D1 的图库层（305/318/320/323）；正 |  | **unverifiable** | 不可复算：回表 deliverable/data-request-v2/gt/review_kappa_FILLED.c |
| E12-179 | 46/63 = 0.730 [0.60, 0.83] | 46/63 = 0.7302, CP[0.6035, 0.8343] | **match** |  |
| E12-180 | 40/43 = 0.930 [0.81, 0.99] | 40/43 = 0.9302, CP[0.8094, 0.9854] | **match** |  |
| E12-181 | 101/179 = 56.4% [48.8, 63. | 101/179 = 0.5642, CP[0.4883, 0.6381]（×100） | **match** |  |
| E12-182 | 122/179 = 68.2% [60.8, 74. | 122/179 = 0.6816, CP[0.6079, 0.7491]（×100） | **match** |  |
| E12-186 | 11 / 52 |  | **unverifiable** | 不可复算：区间由 tmp/full/emb_auc.py 的 percentile bootstrap 产出，脚本未固定 |
| E12-187 | 13 / 18 |  | **unverifiable** | 不可复算：同上：bootstrap 未固定 seed，无法逐位复现；AUC 0.231 已由 E03-034 复算一致 |
| E12-190 | 2026-10-25 |  | **unverifiable** | 不可复算：2026-10-25 是标注轮次日期，记录在 docs/kappa-a3-query-2026-10-25.m |
| E12-191 | 2026-10-26 |  | **unverifiable** | 不可复算：2026-10-26 是第二遍对齐日期，记录在 docs/kappa-a3-revision-2026-10- |
| E12-195 | 3 天（10-22、10-23、10-26）× 每日 |  | **unverifiable** | 不可复算：deliverable/e21-allday-2026-11-23/metrics_*.json 可数出 3  |
| E12-196 | 0–720 s | TRAIN_DUR = 720 | **match** |  |
| E12-197 | 720–810 s（90 s） | TRAIN_DUR = 720；TEST_START = 810；TEST_START-TRAIN_DUR = 90 | **match** |  |
| E12-198 | 自 810 s 起（10-23 至 1260 s；1 | TEST_START = 810 | **unverifiable** | 不可复算：10-23 的 1260 s 与 10-26 的 1979 s 是当日可用窗上限，只在 docs/e20-6c |
| E12-201 | 27, 376 |  | **unverifiable** | 不可复算：该行是「论文全量口径 vs matrix.json」的对照记录：矩阵两列可由 tmp/e21_xday/mat |
| E12-202 | 21, 469 |  | **unverifiable** | 不可复算：该行是「论文全量口径 vs matrix.json」的对照记录：矩阵两列可由 tmp/e21_xday/mat |
| E12-203 | 16, 110 |  | **unverifiable** | 不可复算：该行是「论文全量口径 vs matrix.json」的对照记录：矩阵两列可由 tmp/e21_xday/mat |
| E12-204 | 24, 342 |  | **unverifiable** | 不可复算：该行是「论文全量口径 vs matrix.json」的对照记录：矩阵两列可由 tmp/e21_xday/mat |
| E12-205 | 39, 555 |  | **unverifiable** | 不可复算：该行是「论文全量口径 vs matrix.json」的对照记录：矩阵两列可由 tmp/e21_xday/mat |
| E12-206 | 15, 110 |  | **unverifiable** | 不可复算：该行是「论文全量口径 vs matrix.json」的对照记录：矩阵两列可由 tmp/e21_xday/mat |
| E12-207 | 12, 303 |  | **unverifiable** | 不可复算：该行是「论文全量口径 vs matrix.json」的对照记录：矩阵两列可由 tmp/e21_xday/mat |
| E12-208 | 11, 367 |  | **unverifiable** | 不可复算：该行是「论文全量口径 vs matrix.json」的对照记录：矩阵两列可由 tmp/e21_xday/mat |
| E12-209 | 31, 195 |  | **unverifiable** | 不可复算：该行是「论文全量口径 vs matrix.json」的对照记录：矩阵两列可由 tmp/e21_xday/mat |
| E12-218 | 负 15/15、正 12/12（27 项全判读） | 负 15/15、正 12/12 | **match** |  |
| E12-221 | OSNet 0.904 / E2 0.994（318 | 0.904, 11；0.994；0.231 | **match** |  |
| E12-224 | 257 轨，κ 0.925，95% CI 0.891 | gallery_round1.n_items = 257；gallery_round1.k = 0.925；gallery_ro | **match** |  |
| E12-225 | 236 轨，κ 0.945，95% CI 0.913 | gallery_round1_person.n_items = 236；gallery_round1_person.k = 0. | **match** |  |
| E12-226 | 41 条异判 → 17 判异 / 11 改同 / 1 | {'user': 'diff'} = 41；{'user': 'diff', 'ai': 'diff'} = 17；{'user | **match** |  |
| E12-227 | 153 对 = 46 同 + 107 异；机位对并集 | T1+T2.n_items = 153；T1+T2.n_same = 46；T1+T2.n_diff = 107；机位对并集 = | **match** |  |
| E12-228 | 494 轨 / 26 身份（中位 18、最大 71） | gt_8seg.json: 494, 26, 18, 71 | **match** |  |
| E12-229 | 建库 0–720 s；隔离 720–810 s（90 | TRAIN_DUR = 720；TEST_START = 810；TEST_START-TRAIN_DUR = 90 | **match** |  |
| E12-230 | 3 天（10-22 / 10-23 / 10-26） |  | **unverifiable** | 不可复算：3 天可数；10-23 的七路含 办公区1b（同位置二次拍摄），「每日 6 路」需人工归并才成立，无脚本口径 |

已登记 **101** 条：一致 **66**、不一致 **0**、不可复算 **35**。

覆盖表格单元格 **247 / 330**（已核 **192 = 58%**、不可复算 **55**）；只统计配方里带 tex 行列位置的条目，按单元格去重。

## 10. 复算运行（eval_run / code_version）

> 设计稿 §5 约束：`verdict=match` 只能由 `exit_code=0` 的 run 写入。
> `inputs_json` 的 `role`：`read` = 检查真正打开的文件；`declared` = 该 claim 在 EVIDENCE 声明的来源（未直接读取，仅记哈希）。

| 项目 | 值 |
|---|---|
| `eval_run` 行数 | 101（exit_code=0：101） |
| `code_version` 行数 | 1 |
| 有 `read` 输入文件的 run | 50 |
| 仅有 `declared` 来源的 run | 29 |
| 无任何输入的 run（纯字面量，如 `prop` 的 k/n） | 1 |
| `match` 但无成功 run 的 claim | **0** |