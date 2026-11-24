# r3 改动记录（2026-11-24）

来源：r2（`docs/paper/tbiom-r2`）+ WILDTRACK 外推实验（`docs/wildtrack-p2-region-2026-11-24.md`）。
r3 是**建议修订版**，不覆盖 r2；所有改动都围绕同一件事：把 P2 的断言从「共视 ⇒ 可匹配」收窄为「共视 ⇒ 同时可见」。

## 1. 改了什么（main.tex）

| 位置 | 原文 | 现文 | 理由 |
|---|---|---|---|
| 摘要 | 「…the one camera pair that shares a region reaches AUC 0.904.」句末即止 | 追加：在公开七视角基准上共视不预测该指标（r=0.18，21 对；把配对限制在共同地面区域使 AUC 变化中位 0.04），故 P2 认证的是同时观察、不是可匹配 | 单对 0.904 不能外推 |
| 摘要（19:35 再改） | 「Three label-free site probes … **gate** five delivery tiers.」 | 「…**inform rather than gate** five delivery tiers, each with a stated scope.」 | 作者决定：P2 不再是 L1 自动可用的闸门，gate 会被读成保证 |
| 贡献 1 | 「…one pair at 0.904 and one at 0.231」 | 追加「an external check on a public benchmark that separates observing a region from matching across it」 | 让负结果成为贡献的一部分 |
| §III 三探针 | 「Whether appearance-based identity is available depends on … whether two cameras observe a common region (P2)」 | 改为「What appearance can deliver depends on …（P1、P3、P2）」，并明确 P2「fixes where simultaneous observation—hence adjudication—is possible, not where appearance will match」 | 原句把 P2 写成可用性的决定因素，已被否证 |
| §V-B（Why the unit is a region） | 「As a guide to where appearance works, Table 3 agrees with the survey.」 | 「…is consistent with the survey, but the two factors cannot be separated here: that pair is also the one whose viewpoints are most alike.」 | 共视与视角相似在本站点混杂 |
| §V-B 末段 | 「We have not measured appearance AUC inside the shared regions… P2 predicts that L1 is available there; this prediction is untested…」 | 换成整段 External check：WILDTRACK 设计、共视与 AUC 无关、身份构成混杂与等量控制、区域级 A/B（Δ 中位 +0.04，11/17）、几何与像素均不解释、结论「P2 认证同时可见，不认证可匹配」 | 原文自认这是「最缺的实验」，现补上，结果为负 |
| §VI 表 IV L1 行 | 验收条件「P2 certifies a shared region」；证据仅 0.904 | 验收条件改为「certifies simultaneous observation of a region」；证据补「no gain from the shared region on WILDTRACK」 | 表格与正文一致 |
| §VI-B L1 小节 | 「The acceptance condition is geometric.」后即止 | 追加：外推检验未发现共享区域带来的增益，故 L1 只在「可同时观察 + 可人工裁定」范围内承诺，不作为自动档 | 避免读者按自动档采购 |
| §VI-D 路由规则 | 「P2, shared region: yes, L1 available for that region」 | 「…the region is observable by both cameras, so an adjudication entry point and a temporal anchor exist, but L1 is not certified as automatic」 | 同上 |
| §X 局限 | 「no region-level AUC for P2」 | 改为「The external check of P2 rests on one public dataset (WILDTRACK) and one encoder; its region-level test assumes upright persons standing on a plane, and it reports a null, not a calibrated threshold.」 | 局限性要写明新证据的边界 |
| 参考文献 | — | 新增 `chavdarova2018`（WILDTRACK, CVPR 2018），编号 [14] | 引用 |

## 2. 图 1（Figure_1.pdf）

图中第三块说明文字原为：

```
(i) P2 decides where L1 exists
aggregate AUC .450–.622;
the one co-visible pair reaches .904
```

改为：

```
(i) P2 gives observation, not matching
aggregate AUC .450–.622 at this site;
shared region adds .04 on WILDTRACK
```

源文件 `docs/paper/figs/fig1_pipeline.svg` 同步修改；`Figure_1.pdf` 由该 SVG 重新渲染（525.12×300 pt，与原图同尺寸、矢量）。原 SVG→PDF 的 75% 缩放关系保持不变。

## 3. 补充材料（supplement.tex）

新增 **S8 External check of P2 on WILDTRACK**：设计与口径、对级零相关（含 n_id≥30 后 r=−0.05）、身份构成混杂与等量重采样控制、区域级 A/B 表（8 行）、像素扫描表（320→16 px）、局限。

「Operating cost / Probes」补一句：外推检验在 1 855 个 crop 就绪后约需 4 分钟 GPU 时间 + 约 20 分钟标定处理。

## 4. 编译状态

| 文件 | 页数 | Overfull | 未定义引用/控制序列 |
|---|---|---|---|
| `main.pdf` | **10**（T-BIOM 限 10） | **0** | 0 |
| `supplement.pdf` | 5（r2 为 4；**作者已确认可接受，不合并 S8 两表**） | 1（**r2 既有**：表 S3，4.07 pt，`p{3.5cm}` 单元格） | 0 |

## 5. 三件事的作者决定（11-24 19:35 已定）

1. **接受把 0.904 的归因收回。** 这是本次改动中最"伤"的一处：原文用它支持 P2，现在只能作为「一对视角最相似的机位」的例子。已在 `RESPONSE-NOTE.md` §1 写好 response letter 的主动说明段（英文可粘贴），并写明撤回的是**因果读法**、P2 保留更窄的承诺。
2. **摘要已软化**：`gate` → `inform rather than gate … each with a stated scope`。P1 的范围（只约束人脸通道）**未改正文**——§P1 本来只谈人脸；限定句在补充材料 S8（`P1 is a face-channel constraint rather than a general one`）。正文加这句会变 11 页（实测），故不重复。
3. **补充材料 5 页接受，S8 两张表不合并。** 若期刊另有上限，可把区域级 A/B 表与像素扫描表并为一张（同一批 17 对的两个切面），内容不损失。

正文相对 r2 的差异最终为：摘要两处（0.904 外推说明、gate→inform）+ 九处 P2 收窄 + 新增文献 [14]。编译仍 **10 页 / 0 overfull**。

## 6. 复算入口

- 实验与全部数字：`docs/wildtrack-p2-region-2026-11-24.md`（含复算命令）
- 数据与脚本：`deliverable/wildtrack-p2-2026-11-24/`
- 远端正卡脚本：`tmp/wt/wt_sweep_gpu.py`

## 7. 外部评审意见处置（2026-11-24 20:45）

| 意见 | 判定 | 处置 |
|---|---|---|
| `\AQfalse` 会连内容一起删掉，正式声明会消失 | **成立，是缺陷** | 宏改为 `\ifAQ <红字> \else #1 \fi`；已用最小样例验证关闭后正常输出。`README.md` 重写并写明正确用法 |
| README 仍写 r2、补充材料仍写 4 页 | 成立 | `README.md` 已重写（r3 / 5 页 / 文件清单 / AQ 用法） |
| 「合成降采样 16 px 不能排除办公室成像质量」 | **成立，原句过强** | 正文改为 "synthetic resolution loss alone does not reproduce the office-floor failure"；supplement 原有 upper-bound 说明保留 |
| 「抽相同人数只控制人数、未控制构成」 | 成立，措辞不当 | "an artefact … so we equalise that set" → "a confound in who distant cameras see; standardising that set … is a sensitivity check" |
| 「未发现冲突 ≠ 已确认纯簇」 | 部分成立 | 组级定义加限定 "pure …, relative to adjudicated pairs"；§L3 原有的 "no contradiction is not purity" 保留 |
| 「共视**无效**」 | **不成立（靶子错）** | 稿中自 r3 起就是 "does not predict"（摘要）/"unrelated"（§V-B），表 IV 由 "no gain" 改 "no **reliable** gain" |
| 相机映射影响区域级比较 | 成立，属未做检验 | supplement S8 Limits 补一句：区域检验继承反推映射、未评估替代指派；21 对共享 7 机位与同一人群池，非 21 次独立场地实验 |
| 「旧实验是 YOLO8，不能写成 YOLO26-seg」 | 稿中无此问题 | 全目录无 `YOLO26` 字样；检测器自始为 YOLO8，supplement S2 明写未在站点微调 |
| 「DSATUR 不是 Dempster–Shafer」 | 稿中用法正确 | DSATUR 在图着色语境使用，与证据融合无关 |
| EmbeddingGemma-2 支持图像输入 | 稿中已是正确名称 | 补的是版本/预处理/代码，属派生结果包内容，非正文错误 |

页数影响：正文四处措辞收紧共净增约 50 字符，为守住 10 页删掉一句与前文重复的 "Table III uses the labels exactly as adjudicated"（同段已说明未重标）。编译结果 **main 10 页 / 0 overfull**；supplement 仍 5 页（1 处 r2 既有 4.07 pt）。

## 8. 派生结果包暴露的两处稿件事实错误（20:55–21:10）

为回应「逐项结果无法核对」，重跑了一次选择器实验（`tools/exp_pose_pool_real.py`，新增 `--dump-queries/--dump-pools`），结果与论文逐位一致（101/122/125/142/149、19 写 18 对），但暴露两处文本错误：

| 位置 | 原文 | 现文 | 依据 |
|---|---|---|---|
| §VI-C 人脸通道 | `similarity gate 0.80` | **`similarity gate 0.35`** | 产出 19 条写入的是 τ_face=0.35（扫描列 [0.25,0.30,0.35,0.40,0.45] 的中位）；0.80 是选择器的 τ，被误写成人脸门 |
| §IV-A 时钟偏移 | `0, +1, +2, +16, +25 and +121 s`（未指机位，读作机位序） | **`25, 16, 0, 1, 2 and 121 s for cameras 305, 316, 318, 320, 323 and 343`** | `docs/memory-full-snapshot-2026-10-24.md` §69 首帧 OSD：305 25/316 16/318 0/320 1/323 2/343 121；原升序写法会让 305 被读成 0 s |

两处均不改变任何实验结果，编译仍 **10 页 / 0 overfull**。

## 9. 派生结果包

`deliverable/derived-results-r3-2026-11-24/`：49 个文件 / 0.4 MB，`MANIFEST.csv` + `SHA256SUMS.txt` 全覆盖。

| 子包 | 关键文件 | 论文位置 |
|---|---|---|
| `10-selector/` | `selector_queries_179.csv`（179 行 × 28 列，含 S0/S1/实测池/oracle 上界的 top1、分数、命中）、`selector_pools.json`、`selector_config.json` | 表 V、§VI-C |
| `20-graph/` | `judged_merges_76.csv`（76 条重判，13 条不决标 `in_63=0`）、T1/T2、`attr_gate_runs.json`、443/450 图、四种合并策略、验证器扫描、`osd_offsets.csv` | 表 II/VI/VII、§III/§V |
| `30-wildtrack/` | `pairs_21.csv`（21 对合并表）、`pixel_sweep.csv`、`camera_mapping.csv`、`sampled_frames.csv`、5 个结果 JSON、7 个脚本 | 表 IV、§V-B、S8 |
| `40-env/` | 检测/跟踪/编码器配置、版本、确定性声明 | S2 |

校验（可复跑）：命中列求和 101/122/110/125/142/149；ΔAUC 中位 +0.0425、11/17 为正；像素扫描 0.7953→0.6709；76 = 46 同 + 17 异 + 13 不决，`in_63` 求和 63；六个偏移 25/16/0/1/2/121。

## 10. 待决定：L2+ 的 68.2% 是工作点依赖的（2026-11-24 21:45）

**不涉及数字对错，涉及"这个提升有多少是模型、多少是阈值"。**

EXP-003c（`docs/exp-003a-template-decomposition-2026-11-24.md` §10）在 D2 的 179 查询上做了三件事：

| 做法 | 召回/179 | 精度 | 与核的比较 |
|---|---:|---:|---|
| S0 @ τ=0.80（论文协议） | 101（56.4%） | .962 | — |
| S0，1 个 τ 调优（LOO） | 111（62.0%） | .965 | 核 −12 |
| S0，改「顶一减顶二」间隔规则 | 118（65.9%） | .962 | 核 −5 |
| K_h @ τ=0.80（σ 调优，LOO） | 123（68.7%） | .969 | — |
| K_h + 间隔规则同开 | 115（64.2%） | — | **负交互 −15** |

另外两条硬事实：

1. **argmax 从 142 变到 143**：核没有改善 top-1 排序，它抬的是正确身份的绝对余弦（+.022，误配仅 +.002）。收益落在"本来就排第一、却卡在 0.80 以下"的那批查询上。
2. **召回-误受曲线交叉**：误受 ≤2 时间隔规则反超（116 vs 105），≥4 时核领先（130 vs 118）。没有一方在所有精度下都更好。

**这意味着**：

- 正文 56.4% → 68.2% 的对比里，约一半可以用"换判定规则"免费拿到，而且不需要姿态信息。
- 但**数字本身不必改**：协议在 §VI-C 写明了（同机位、固定 τ=0.80、σ=0.6），Table V 的 68.2% 在该协议下是对的，r3 也从未声称它是"判别力提升"。

**可选的正文补丁**（一句话，但正文字数余量≈0，必须同时删一句）：

> Recalls in Table~\ref{tab:pool} are at a fixed $\tau=0.80$; since the pose kernel mainly lifts the similarity of the correct match rather than reordering candidates, part of the margin over the plain gallery mean is recoverable by a relative-margin rule instead.

**建议**：若审稿人未追问，保持现状（协议已写明）；若被问到"换个判定规则还成立吗"，用这段话与 §10 的表回答。是否加进正文，等作者定。

**处置（22:20，已执行）**：同样写进 **supplement §S9 第二段**，正文不动。正文里 Table V 的 68.2% 与 §VI-C 的协议描述都不动。

## 11. 待决定：Table V 的 179 条查询没有过部署的 60 px 质量门（22:15）

§VI-C 正文写「$\tau$ is the deployed selector's operating point」，但 Table V 的 179 条查询里**有 22 条低于部署选择器的 60 px 质量门**（部署端会直接拒答）。这是口径差，不是算错：数字在"无门"协议下正确。

**这个差有多大**（EXP-003e，2×2 阶乘，同一批嵌入）：

| 库 | 查询 | n | 绝对余弦 | h 核 | 核净 |
|---|---|---:|---:|---:|---:|
| 不门控 | 不门控（=Table V） | 179 | 101 | 125 | **+24** |
| 不门控 | ≥60 px | 157 | 89 | 109 | +20 |
| ≥60 px | 不门控 | 179 | 101 | 121 | +20 |
| **≥60 px** | **≥60 px（部署）** | **157** | **100** | **108** | **+8** |

**结论**：核的 +11.8 pt 里，一半以上是对**低质量库模板**的抑制；把低于 60 px 的模板从库里去掉（零成本），朴素均值就从 89 升到 100，而核只到 108。核在部署口径下仍为正（+5.1 pt），但不再是 +13.4 pt。

**同时**（EXP-003d，五协议）：把固定阈值换成「顶一减顶二」的相对间隔后，严格精度（1–2 个误受）下召回在五个协议里四个显著领先（+15～+52 查询），而在宽松精度（≥4）下核反超——两条曲线交叉，两者不可叠加（EXP-003c 的 −15 负交互复现）。

**可选的正文补丁**（一个从句，正文余量≈0）：

> The 179 queries are not filtered by the selector's 60~px quality gate; with that gate applied, the plain gallery mean reaches 100/157 and the box-height kernel adds 8.

**建议**：这一条比 §10 更值得改——它不影响数字，但影响"部署能拿到什么"的判断。若只改一处，改这里。

**处置（22:20，已执行）**：不改正文。重读后发现 §VI-C 的原句是「$\tau$ is the deployed selector's operating point」——这句**字面为真**（τ 确实是部署工作点），它并没有声称查询集等同于部署输入。所以问题在读者推断，不在稿件陈述，属于"稳健性"而非"更正经数"。这类内容的正规去处是补充材料，不是正文（正文余量≈0）。

新增 **supplement §S9**「Sensitivity of the L2+ selector to a box-height floor and to the decision rule」+ 表 S6（上面那张 2×2）。**supplement 仍 5 页，只有 r2 既有的 1 处 4.07 pt overfull；main 未改，仍 10 页。**
