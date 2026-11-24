# 第二标注人第二次交回：8 条改动的影响评估

日期：2026-10-26。对象：`gt_a3_gallery_members_FILLED (1).csv`、`gt_a3_query_gt_FILLED (1).csv`。
前一份记录：`docs/kappa-a3-query-2026-10-25.md`（第一版，κ=0.925 / 0.924）。

---

## 1 核对结论：数据可用，改动 8 条

| 检查项 | 图库 257 条 | 查询 100 条 |
|---|---|---|
| 行数 | 257（不变） | 100（不变） |
| 列与上一版逐列一致 | 是 | 是 |
| 键集合、行序与上一版一致 | 是 | 是 |
| 重复键 | 0 | 0 |
| 空 `person_id` | 1（`318_train_525#816`，`unk_reason=far`，故意） | 0 |
| 标注人 | `bzy` | `bzy` |

改动的 8 条全部是 `person_id` 一个字段，其它列未动：

| 条目 | 机位 | A1 的标注 | 第一版 | 第二版 |
|---|---|---|---|---|
| `323_train_510#2745` | 323 | male10 | person22 | **person12** |
| `323_train_510#2872` | 323 | male10 | person22 | **person12** |
| `318_test_534#839` | 318 | male16 | person23 | **person15** |
| `318_test_534#853` | 318 | male16 | person23 | **person15** |
| `318_test_534#1045` | 318 | male16 | person23 | **person15** |
| `320_test_518#916` | 320 | male23 | person20 | **person27** |
| `320_test_518#1015` | 320 | male23 | person20 | **person27** |
| `320_test_518#1169` | 320 | male23 | person20 | **person27** |

**这 8 条全部不在第一版的分歧清单里**（第一版 25 条分歧，见 `kappa_a3_by_layer.round1.json`）。
也就是说：改动没有一条落在"被系统判定为不一致"的条目上，不构成"照着 A1 的答案改"。这一点对下面的解读很重要。

## 2 改动的性质：三处身份划分修正

三组改动对应三个具体的编号问题：

1. **`person22` 原先把两个人并成了一个**。它下面 3 条：`#767`（A1=male14）、`#2745`/`#2872`（A1=male10）。
   A3 复查后把两条 male10 的移出到 `person12`（= male10），`person22` 只剩 `#767` 一条。
2. **`person23` 原先是 `person15` 的重复编号**。它的 3 条查询条目 A1 都标 male16，而 `person15` = male16。
   A3 把三条并入 `person15`，`person23` 只剩 1 条图库条目。
3. **`person20` 原先把两个人并成了一个**。它下面 5 条：2 条图库（A1=male18）+ 3 条查询（A1=male23）。
   A3 把三条 male23 的拆成新编号 `person27`，`person20` 只剩那 2 条 = male18。

三处都是**把一个编号承担的两个人分开、或把两个编号指同一个人合并**。没有一条是"这张图我看错了"式的单条改判。

## 3 结果：κ 变了，但变的原因不是这 8 条

### 3.1 分层 κ

| 层 | n | 第一版 κ | 第二版 κ | 95% CI（第二版） |
|---|---|---|---|---|
| 图库 | 257 | 0.925 | **0.937** | [0.904, 0.966] |
| 查询 | 100 | 0.924 | **0.924**（不变） | [0.861, 0.978] |
| 合并 | 357 | 0.925 | **0.934** | [0.907, 0.958] |

仅人物口径：图库 0.945 → **0.959**｜查询 1.000（不变）｜合并 0.961 → **0.970**。
合并非人标签：图库 0.946 → **0.958**｜查询 **1.000**（不变）。

### 3.2 分歧数

| | 第一版 | 第二版 |
|---|---|---|
| 图库 | 18 | **15** |
| 查询 | 7 | 7（不变） |
| 合计 | 25 | **22** |

逐条核对改动前后的判定状态（这是本文最关键的一张表）：

| 条目 | A1 | A3 第一版 | v1 判定 | A3 第二版 | v2 判定 |
|---|---|---|---|---|---|
| `320_train_514#68` | male18 | person20 | **不一致** | person20 | **一致** |
| `320_train_514#254` | male18 | person20 | **不一致** | person20 | **一致** |
| `323_train_510#767` | male14 | person22 | **不一致** | person22 | **一致** |
| `323_train_510#2745` | male10 | person22 | 一致 | person12 | 一致 |
| `323_train_510#2872` | male10 | person22 | 一致 | person12 | 一致 |
| `318_test_534#839` | male16 | person23 | 一致 | person15 | 一致 |
| `318_test_534#853` | male16 | person23 | 一致 | person15 | 一致 |
| `318_test_534#1045` | male16 | person23 | 一致 | person15 | 一致 |
| `320_test_518#916` | male23 | person20 | 一致 | person27 | 一致 |
| `320_test_518#1015` | male23 | person20 | 一致 | person27 | 一致 |
| `320_test_518#1169` | male23 | person20 | 一致 | person27 | 一致 |

**被改判的 8 条，判定状态一条都没变；翻转的 3 条，标签一个字都没改。**

原因全在映射。映射按每个 A3 编号下 A1 标签的多数派估：

- 第一版 `person20` = {male18: 2, male23: 3} → 多数派 male23 → `#68`/`#254`（male18）被判不一致。
  拆出 `person27`（3 条全 male23）后，`person20` = {male18: 2} → 2 条自动对上。
- 第一版 `person22` = {male14: 1, male10: 2} → 多数派 male10 → `#767`（male14）被判不一致。
  两条 male10 移到 `person12` 后，`person22` 只剩 1 条 male14 → 自动对上。
- `person23` 第一版已经一致（→ male16，4 条全 male16），并进 `person15` 后仍然一致。

**所以 κ 从 0.925 升到 0.937，是"编号划分变干净"带来的，不是"8 条改对了"带来的。**
这 8 条在第一版里本来就都算一致，改完仍然一致，对 κ 的边际贡献是 **0**。

这条值得写进方法学：**按编号映射算出来的 κ，同时量了两件事——逐条判对的比例，以及身份划分的一致性。**
这次是前者的一个天然对照：8 条真实改判贡献 0，另一组 8 条没动的条目贡献了 +0.012。

### 3.3 稳健性口径

| 口径 | 图库 | 查询 | 合并 |
|---|---|---|---|
| 多数派映射（in-sample，主口径） | 0.937 | 0.924 | 0.934 |
| 留一法估计映射 | 0.925 [0.888, 0.958] | 0.902 [0.837, 0.956] | 0.919 [0.890, 0.946] |
| 原 `align()` 贪心映射（敏感性对照） | 0.908 | 0.902 | 0.907 |
| 剔除只出现一次的编号 | 0.958（仅人物口径） | 1.000 | 0.970 |

孤立编号的剔除只动 0.001 量级（图库 0.9587 → 0.9581），**不是这次变化的驱动因素**，写进敏感性即可。
原 `align()` 这次挤掉 5 个编号（`cloth`、`hat1`、`person22`、`person23`、`person24`），比上一版多一个 `person23`——因为 `person23` 现在只剩 1 条。

**可辩护区间：图库 0.91–0.94、查询 0.90–0.92。不要报单点。**

## 4 分机位：320 回到 1.000，323 仍是分歧集中处

| 机位 | 人物条目 | 第一版 κ | 第二版 κ | 95% CI | 分歧/条目 |
|---|---|---|---|---|---|
| 305 | 100 | 0.977 | **0.977** | [0.942, 1.000] | 2 / 100 |
| 318 | 53 | 0.936 | **0.936** | [0.854, 1.000] | 4 / 58 |
| 320 | 14 | 0.750 | **1.000** | [1.000, 1.000] | 0 / 14 |
| 323 | 69 | 0.904 | **0.923** | [0.849, 0.981] | 9 / 85 |

- 320 的 0.750 是上一版 `person20` 混编号造成的假象，拆开后 2 条分歧清零。但 **n=14，CI 退化成一个点**，不能当"320 标注完美"的证据。
- 323 的 9 条分歧里，5 条是 `shadow` 与 `shadow1/shadow2` 的措辞差异，4 条是人物身份不同。
- 15 条图库分歧的去向：305 2 条（13%）｜318 4 条（27%）｜323 **9 条（60%）**；条目占比 39% / 23% / 5% / 33%。**323 的分歧富集约 1.8 倍。**

**四机位的 κ 两两之间没有一条显著不同**（CI 大幅重叠：323 [0.849, 0.981] 与 305 [0.942, 1.000] 重叠 0.039）。
能说的只有"323 承担了 60% 的分歧、而它只占 33% 的条目"，不能说"323 的标注质量最差"。

## 5 对稿件的影响

`manuscript.tex` 第 177 行现文（三个数字全部过期）：

> A second, independent annotator relabelled all 257 gallery tracks blind … Cohen's kappa was 0.925 (95% bootstrap interval 0.891--0.958; 0.945 on the 236 person tracks). **Agreement was perfect on cameras 318 and 320 and lowest on camera 323 (0.811), the far-field camera, where 15 of the 18 disagreements occurred.** \pending{second-annotator labels for a 100-query sample; template prepared}

建议改为（三个层次：分层 κ、分歧集中度、复查轮次；不再声称机位间有显著差异）：

> A second annotator relabelled all 257 gallery tracks blind, creating their own identity codes; after aligning codes by co-occurrence, Cohen's kappa was 0.937 (95\% bootstrap interval 0.904--0.966; 0.959 on the 236 person tracks). The same annotator labelled a stratified sample of 100 query tracks blind, giving kappa 0.924 (0.861--0.978); all seven disagreements there were differences of wording between ``no human'' and ``shadow'', and collapsing those two labels gives perfect agreement on whether a track is a person and, if so, which. Camera-level agreement did not differ significantly (305: 0.977, n=100; 318: 0.936, n=53; 320: 1.000, n=14; 323: 0.923, n=69; all intervals overlap), but camera 323, the far-field camera, accounted for 9 of the 15 gallery disagreements while holding 33\% of the annotated tracks. A final review pass with a zoom view changed eight of the 357 gallery labels; before that pass the gallery figure was 0.925.

第 400 行现文「The query layer has a single annotator so far; gallery labels have an independent second annotator.」**已不成立**——查询层现在有 0.924。建议改为：

> Both annotation layers have a second annotator (gallery kappa 0.937, query 0.924), but the query layer covers 100 of 295 tracks, and no third annotator was available to adjudicate disagreements.

第 177 行末尾的 `\pending{...}` 应删除。

**注意**：若只采纳 0.937 而不提复查前是 0.925，读者会以为复查修掉了 3 个错误；实际那 3 条标签没动（§3.2）。

## 6 这份记录不证明什么

- **改判不是在完全盲的条件下做的。** 复查发生在 A3 已经看过第一版 κ 与"318 有 4 条、323 有 10 条分歧"之后。缓解证据是 8 条改动无一落在第一版的分歧清单上；但"复查动机来自分歧统计"这件事本身无法消除，所以第一版的 0.925 必须一并报出，作为未经复查的数字。
- **κ 上升不等于标注质量上升 0.012。** 第 3.2 节已说明：8 条改判对 κ 的边际贡献是 0，上升全部来自编号划分。若只报 0.937 而不报 0.925 和这条机制，读者会以为改判修掉了 3 个错误。
- **一致不等于正确。** 两人可能犯同一个系统性错误。κ 量的是可重复性。
- 查询层只覆盖 100/295 条分层抽样，不是全部。

## 7 文件

| 文件 | 说明 |
|---|---|
| `gt/a3_gallery_members_FILLED.csv` | 第二版（257 条，现行） |
| `gt/a3_query_gt_FILLED.csv` | 第二版（100 条，现行） |
| `gt/a3_gallery_members_FILLED.round1.csv` | 第一版归档，可复现 0.925 |
| `gt/a3_query_gt_FILLED.round1.csv` | 第一版归档 |
| `results/kappa_a3_by_layer.json` | 第二版结果（现行） |
| `results/kappa_a3_by_layer.round1.json` | 第一版结果归档 |
