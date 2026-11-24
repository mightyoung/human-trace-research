# 论文数字证据体检 V0（只读）

> 解析 `docs/paper/EVIDENCE.md` 得 **176 条 claim**；与 main.tex、supplement.tex 做数字集合比对。本报告**不执行任何复算**；“有脚本”只表示引用了存在的脚本，不表示脚本复算过这个数字。

## 1. 总览

| 指标 | 数值 |
|---|---|
| claim 总数 | 176 |
| ① 来源含数据/结果文件（非 md） | 93/176 |
| ① 来源只有文档（md） | 38/176 |
| ① 没有任何可解析来源 | 45/176 |
| ① 严格达标（引用全部存在、已记内容 sha） | 93/176 |
| ② 引用了存在的复算脚本 | 45/176 |
| ③ tex 表格内数字 / 其中 EVIDENCE 全文找不到 | 209 / **33** |
| ③ EVIDENCE 有、tex 没有的数字 | 69 |
| 已登记来源文件（去重） | 62（未记内容 sha 的 0） |

## 2. 分节

| 节 | claim | 数据文件 | 仅文档 | 无来源 | 有脚本 | 严格达标 |
|---|---:|---:|---:|---:|---:|---:|
|  | 3 | 0 | 0 | 3 | 0 | 0 |
| 一、两域与图统计（§3.1, Table 1） | 11 | 8 | 1 | 2 | 2 | 8 |
| 二、真值（§4, Appendix A） | 6 | 3 | 1 | 2 | 0 | 3 |
| 三、外观天花板与 P2（§5.2, Table 2 / Table 3） | 18 | 13 | 0 | 5 | 13 | 13 |
| 四、人脸像素预算与 P1（§5.1, Figure 2） | 12 | 2 | 9 | 1 | 0 | 2 |
| 五、服装多样性与 P3（§5.3, §6.3, Table 5） | 10 | 3 | 5 | 2 | 0 | 3 |
| 六、分级协议各级证据（§6） | 36 | 19 | 7 | 10 | 1 | 19 |
| 七、不可折叠（§7） | 17 | 13 | 0 | 4 | 9 | 13 |
| 八、跨采集验收（§8, Table 11 / Table 12） | 28 | 19 | 0 | 9 | 19 | 19 |
| 九、已知不可用 / 不要引用 | 6 | 1 | 0 | 5 | 0 | 1 |
| 十一、r2 新增/核对的事实（2026-11-14） | 29 | 12 | 15 | 2 | 1 | 12 |

## 3. 每张 tex 表里、EVIDENCE 全文找不到的数字

| 文件 | 表 | 标题 | 表内数字 | 缺 |
|---|---|---|---:|---|
| main.tex | `tab:d1` | D1 Capture and Graph Statistics}  | 23 |  |
| main.tex | `tab:agg` | Cross-Camera Discrimination on T1 (OSNet: 34 Same /  | 34 |  |
| main.tex | `tab:pair` | Per-Pair AUC on T1 (OSNet Raw). Intervals Are Hanley | 26 | 0.07 1.00 |
| main.tex | `tab:protocol` | The Protocol. Intervals Are Exact 95\% Binomial Inte | 36 | 0.81 0.83 101 122 48.8 60.8 63.8 74.9 |
| main.tex | `tab:gate` | Effect of the Attribute Gate (D1, Graph v0 Settings, | 17 |  |
| main.tex | `tab:selector` | Selector Recall at $ = 0.80$ (D2, 179 Queries)}  | 12 |  |
| main.tex | `tab:verifier` | Partition-Level Effect of the Pairwise Verifier}  | 17 |  |
| main.tex | `tab:merge` | Merge Strategies at Threshold 0.75 (D1, Attribute Ga | 16 |  |
| main.tex | `tab:xday` | Nine Gallery-Day by Query-Day Cells at $ = 0.90$ (D3 | 39 |  |
| supplement.tex | `-` | Domains} | 24 | 720 810 |
| supplement.tex | `-` | Adjudicated pairs by source (D1)} | 12 |  |
| supplement.tex | `-` | Annotation reliability by layer} | 15 | 0.913 0.925 0.945 0.958 0.972 236 |
| supplement.tex | `-` | Same cells as the nine-cell table of the article. Co | 43 | 0.275 0.296 0.371 0.516 0.643 0.653 0.877 21 23.7 40.2 50.3 67.3 76.0 83.8 … |
| supplement.tex | `-` | The form in which a system is accepted or rejected} | 21 |  |

## 4. 解析不到的引用

| 引用 | 次数 |
|---|---:|
| 无 | 0 |

## 5. §十 复现清单的脚本

| 脚本 | 存在 |
|---|---|
| `~/.workbuddy/binaries/python/envs/default/Scripts/python.exe` | 是 |
| `tmp/full/emb_auc.py` | 是 |
| `tmp/full/spacetime_audit.py` | 是 |
| `tmp/full/spacetime_ingraph.py` | 是 |
| `tmp/full/spacetime_verifier.py` | 是 |
| `tmp/full/verifier_split.py` | 是 |
| `tools/build_spacetime_review.py` | 是 |
| `tools/ingest_spacetime_review.py` | 是 |
| `tools/apply_spacetime_truth.py` | 是 |
| `tools/md2paper_html.py` | 是 |

## 6. tex 正文有、EVIDENCE 没有的数字（不含 Introduction / Related Work / 参考文献；按出现位置）

| 数字 | 出现位置 |
|---|---|
| 720 | main.tex:{D3} is three further capture days (10-22, 10- §Data and Pipeline |
| 810 | main.tex:{D3} is three further capture days (10-22, 10- §Data and Pipeline |
| 0.15 | main.tex:Detection and tracking run per camera (YOLO8,  §Data and Pipeline |
| 0.444 | main.tex:A labelled pair is correct at {pair level} if  §Evaluation Protocol |
| 200 | main.tex:{Measurement.} On a random sample of tracks (w §The Three-Probe Diagnos |
| 250 | main.tex:{Relation to standards.} We split on face-box  §The Three-Probe Diagnos |
| 2014 | main.tex:{Relation to standards.} We split on face-box  §The Three-Probe Diagnos |
| 62676 | main.tex:{Relation to standards.} We split on face-box  §The Three-Probe Diagnos |
| 1.00 | main.tex:318--320 & 11 / 52 & 0.904 & [0.78, 1.00]\\ §The Three-Probe Diagnostic |
| 0.07 | main.tex:316--343 & 13 / 18 & 0.231 & [0.07, 0.40]\\ §The Three-Probe Diagnostic |
| 17.5 | main.tex:{Reading on D1.} 673 per-camera person entitie §The Three-Probe Diagnos |
| 0.81 | main.tex:L2 & Attribute-consistent merging & P3: modal  §A Tiered Delivery Proto |
| 0.83 | main.tex:L2 & Attribute-consistent merging & P3: modal  §A Tiered Delivery Proto |
| 63.8 | main.tex:L2+ & Pose-conditioned templates, same camera  §A Tiered Delivery Proto |
| 122 | main.tex:L2+ & Pose-conditioned templates, same camera  §A Tiered Delivery Proto |
| 60.8 | main.tex:L2+ & Pose-conditioned templates, same camera  §A Tiered Delivery Proto |
| 48.8 | main.tex:L2+ & Pose-conditioned templates, same camera  §A Tiered Delivery Proto |
| 101 | main.tex:L2+ & Pose-conditioned templates, same camera  §A Tiered Delivery Proto |
| 74.9 | main.tex:L2+ & Pose-conditioned templates, same camera  §A Tiered Delivery Proto |
| 31.1 | main.tex:The gate removes 9 merge operations (Table~ )  §A Tiered Delivery Proto |
| 14.7 | main.tex:The gate removes 9 merge operations (Table~ )  §A Tiered Delivery Proto |
| 5.0 | main.tex:The gate removes 9 merge operations (Table~ )  §A Tiered Delivery Proto |
| 21 | main.tex:{The kernel.} Weighting gallery templates by $ §A Tiered Delivery Proto |
| 0.02 | main.tex:{The face-identified pool.} A same-day pool wr §A Tiered Delivery Proto |
| 3.8 | main.tex:Cluster concentration screens only when read a §Acceptance Statistics A |
| 50 | main.tex:Cluster concentration screens only when read a §Acceptance Statistics A |
| 36.1 | main.tex:Cluster concentration screens only when read a §Acceptance Statistics A |
| 0.2 | main.tex:{Stratify before thresholding.} An aggregate c §Discussion |
| 90 | main.tex:A tender that asks for ``cross-camera ReID at  §Discussion |
| 0.925 | main.tex:{One adjudicator for pair verdicts.} The 153 T §Limitations |
| 0.958 | main.tex:{One adjudicator for pair verdicts.} The 153 T §Limitations |
| 2.5 | supplement.tex:Python 3.11; {numpy}, {scipy}, {scikit-learn}, §Environment |
| 3.11 | supplement.tex:Python 3.11; {numpy}, {scipy}, {scikit-learn}, §Environment |
| 0.972 | supplement.tex:Gallery membership, person-containing subset & §Truth-set composi |
| 0.945 | supplement.tex:Gallery membership, person-containing subset & §Truth-set composi |
| 236 | supplement.tex:Gallery membership, person-containing subset & §Truth-set composi |
| 0.913 | supplement.tex:Gallery membership, person-containing subset & §Truth-set composi |
| 0.977 | supplement.tex:The gallery $ $ is computed after aligning the §Truth-set composi |
| 69 | supplement.tex:The gallery $ $ is computed after aligning the §Truth-set composi |
| 0.811 | supplement.tex:The gallery $ $ is computed after aligning the §Truth-set composi |
| 23.7 | supplement.tex:10-22 & 10-23 & 21 & 469 & 0.809 & 23.7\%\\ §Capture-change study |
| 0.877 | supplement.tex:10-22 & 10-26 & 16 & 110 & 0.877 & 20.0\%\\ §Capture-change study |
| 0.371 | supplement.tex:10-23 & 10-22 & 24 & 342 & 0.371 & 76.0\%\\ §Capture-change study |
| 76.0 | supplement.tex:10-23 & 10-22 & 24 & 342 & 0.371 & 76.0\%\\ §Capture-change study |
| 40.2 | supplement.tex:10-23 & 10-23 & 39 & 555 & 0.653 & 40.2\%\\ §Capture-change study |
| 0.653 | supplement.tex:10-23 & 10-23 & 39 & 555 & 0.653 & 40.2\%\\ §Capture-change study |
| 67.3 | supplement.tex:10-23 & 10-26 & 15 & 110 & 0.516 & 67.3\%\\ §Capture-change study |
| 0.516 | supplement.tex:10-23 & 10-26 & 15 & 110 & 0.516 & 67.3\%\\ §Capture-change study |
| 83.8 | supplement.tex:10-26 & 10-22 & 12 & 303 & 0.296 & 83.8\%\\ §Capture-change study |
| 0.296 | supplement.tex:10-26 & 10-22 & 12 & 303 & 0.296 & 83.8\%\\ §Capture-change study |
| 0.275 | supplement.tex:10-26 & 10-23 & 11 & 367 & 0.275 & 86.4\%\\ §Capture-change study |
| 86.4 | supplement.tex:10-26 & 10-23 & 11 & 367 & 0.275 & 86.4\%\\ §Capture-change study |
| 50.3 | supplement.tex:10-26 & 10-26 & 31 & 195 & 0.643 & 50.3\%\\ §Capture-change study |
| 0.643 | supplement.tex:10-26 & 10-26 & 31 & 195 & 0.643 & 50.3\%\\ §Capture-change study |
| 0.575 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 204 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.865 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.534 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.725 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.496 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.778 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 45.5 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 1953 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.613 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 13.8 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.670 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.926 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.547 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.766 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.684 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.632 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 44 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.720 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 0.796 | supplement.tex:Both populations are D1. Of the 1,473 D1 track §Gait as a candida |
| 2.05 | supplement.tex:{Adjudication.} 27 items in 20--30 minutes by  §Operating cost |

## 7. EVIDENCE 有、tex 没有的数字（69 个）

0.124 0.134 0.147 0.213 0.254 0.267 0.433 0.488 0.512 0.591 0.654 0.709 0.753 0.767 00 02 09 105 11.2 11.3 11.5 11.62 1178 13.4 14.6 141 16.7 173 181 19.2 19.8 2.1 2.35 2.6 22.9 230 24.9 25.5 26.8 28.1 3.1 3.2 3.23 3.3 31.7 33 36.3 366 37.0 389 4.5 401 496 5.3 56.0 574 6.1 6.3 6.4 6.5 64 7.1 7.3 724 896249 91 94 96 973344

## 8. 值单元数字在 tex 全文找不到的表格行

| id | 行 | 值 | 找不到 |
|---|---|---|---|
| E04-045 | 可用脸占全量轨 | 6.5%（1,178 轨） | 1178 6.5 |
| E05-052 | 平均同款人数 | 19.2 | 19.2 |
| E06-064 | 同人框高跨度 | 中位 **3.76×**，最大 11.62×；宽高比 3.23× | 11.62 3.23 |
| E06-086 | 最长同人同画 | **815 s**（305:173 ~ 320:45） | 173 |
| E07-101 | 验收 | 15/15 异人拆开、**12/12 同人保住**、旧 T1 不变（P  | 0.147 |
| E11-165 | 673 的单位 | 1,202 个逐机位实体中过 `is_person ≥ 0.5` 的 * | 56.0 |
| E11-172 | 存量偏差 | `全量` 与 `剔除无效段` 只差两格（10-22→10-23 ev 3 | 389 401 |