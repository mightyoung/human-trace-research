# 主稿同类问题审查：三类伪增益在 human-count 主稿上的对应物

**日期**：2026-11-09　｜　**对象**：`tmp/v14/.../02_manuscript/manuscript.tex`（第十四版，37 页）
**方法**：把 E21 文档识别的三个失效模式（评估集漂移 / 目标函数缺项 / 补偿性干预）改写成可在主稿既有数据上检验的形式，逐条实测。
**数据**：`deliverable/data-request-v2/results/{summary,curve,strict/precision_bgr_strict,members/*,supp_table_site}.csv`（42 配置 / 1012 四机位 / 229 可评估查询）。**本次未重跑任何流水线**，全部结论来自既有产物的重新统计。

---

## 0 结论先行

**主稿有同类问题，但都不推翻结论。** 三类里两类确认存在（选择器与门槛机制），一类不存在（没有伪正向增益）。

| # | 项 | 判定 | 严重性 |
|---|---|---|---|
| A | 同一个自校准选择器，`fm` 不进入目标函数 | **同**，且主稿未声明这一限制 | 中：需一句披露 |
| B | `fm` 的分母跨配置差 **3.05 倍**（302–922） | **同**，主稿把 42 行 `fm` 并排列出 | **高：这是本次最该修的** |
| C | `same ≥ 0.90` 门槛以零余量通过 | **同**，且是**表级现象**（20/42 余量 < 0.005，5 个精确 = 0） | 中：需一句披露 |
| D | 跨配置 ΔC 与 coverage 高度共线（**R² = 0.940**） | **同**，ΔC 主要是工作点效应 | 中：影响解读，不影响结论 |
| E | 「排名反转」是否 coverage 伪影 | **部分**：控制 coverage 后 −0.744 → **−0.500**，结论仍成立但幅度被高估 | 低：可加限定 |
| F | 是否存在「补偿性干预」式伪正向增益 | **不同**：唯一的 +4 已被 robust selection 处死，实测其余量恰为 0.0000 | 无 |

**一句话**：主稿的**结论**（簇级指标会误导、veto 有害、拒绝而非答错是瓶颈）经得起这次复查；但**支撑表格的可比性**不如它看起来的那么强——`fm` 列不可横向比、`ΔC` 列主要是覆盖率的影子、门槛擦线是常态而非个例。这三处都是"加一句披露"就能修的，不需要重跑。

---

## 1 三类失效模式在主稿上的可检验形式

| E21 的失效模式 | 主稿里的对应检验 | 数据是否够 |
|---|---|---|
| 评估集漂移：`fm` 分母随 θ 变，跨配置不可比 | 取各配置 θ\* 上的 `evaluable`，看跨配置差异幅度；再检验"比率排序"与"绝对对数排序"是否一致 | ✅ `curve.csv` 含每点 `evaluable` |
| 目标函数缺项：选择器不含 `fm` | 读主稿选择规则原文与 `d7_sweep.pick` | ✅ |
| 补偿性干预：某模块的收益由另一参数制造 | 检查主稿唯一的正向结果（+4）是否依赖孤立工作点；检查否决动机（16/39 混合簇）是否参数依赖 | ✅ / △ |
| （附加）跨配置比较的混杂 | 回归 `C ~ coverage`，看 R²；对 8 行表逐行分解 ΔC | ✅ |

---

## 2 逐项结果

### 2.1 A：同一个选择器，`fm` 不进目标函数 —— 确认

主稿 §"label-free selection" 原文：

> the rule selects the θ with maximum coverage subject to agreement ≥ 0.90 and at least 100 evaluable co-occurring pairs.

`d7_sweep.py` 的 `pick()` 可行域是 `{same ≥ 0.90, evaluable ≥ 100}`，与 E21 完全一致；`FM_BUDGET = 0.02` 只存在于 `session_selector_v1.py`，`d7_sweep` 从未引用。

主稿确实**计算并报告**了 merge rate（baseline 1.53%），但**没有约束它**，也没有说明这一点。后果与 E21 §6 相同：报告值是"最松可行 θ"上的值，即该方向上**能达到的最高**，不是被选定的目标。

> 需要澄清的是：主稿的评估工具 `eval_det_switch.py` **有** `--fm-max` 开关（默认不启用）。所以"能约束但没约束"是事实，不是能力缺失。

### 2.2 B：`fm` 的分母跨配置差 3 倍 —— 确认，且有可直接展示的实例

各配置在其 θ\* 上的 `evaluable`（即 `fm` 的分母）：

| 项 | 值 |
|---|---|
| 最小 | **302**（`C_dmax0.15`，θ=1.40） |
| 最大 | **922**（`D_split0.20_veto0.30`，θ=1.75） |
| 中位 | 806 |
| 极差比 | **3.05×** |
| Spearman(`evaluable`, θ) | +0.449 |

最直接的证据是同一张表里两行的对照：

| 配置 | 簇纯度 | θ | 分母 | `fm`（率） | **误合并对数** |
|---|---|---|---|---|---|
| `C_dmax0.15` | **1.000**（全表最高） | 1.40 | 302 | **3.64%**（全表最高） | **11**（全表最低之一） |
| `A_baseline` | 0.898 | 1.85 | 918 | 1.52% | **14** |

**簇最干净、绝对误合并最少的配置，报出的误合并率最高。** 这不是精度差异，纯粹是分母差 3 倍所致。

后果：主稿的补充表 `supp_table_site.csv` 把 `fm` 作为一列并排给出 42 个配置的值。任何按该列横向排序的读者都会得到一个与"谁更纯"相反的结论。**这一列要么补上分母，要么改用绝对对数。**

### 2.3 C：门槛擦线是表级现象 —— 确认

`same ≥ 0.90` 是硬门槛，各配置的余量（`same`@θ\* − 0.90）：

| 项 | 值 |
|---|---|
| 余量中位 | **0.0060** |
| 最小 | **0.0000** |
| 余量 < 0.005 的配置 | **20 / 42** |
| 余量 = 0.0000 的配置 | **5**：`C_dmax0.25`、`C_dmax0.26`、`D_split0.18`、`D_split0.18_veto0.30`、`D_split0.20_veto0.30` |

主稿只对其中一个（`D_split0.20_veto0.30`，+4）讨论了"孤立可行点"，处理是对的：它的余量恰好 **0.0000**，`θ−0.05` 处不可行，robust selection 下 ΔC=0。**这个个案的处理可以照搬为表级说明。**

基线本身余量 0.0079，是表内较宽的一档，所以主结果不在刀刃上；但"20/42 贴线"这个事实不写出来，读者会以为各行的 `same` 都是稳稳过线。

### 2.4 D：ΔC 与 coverage 高度共线 —— 确认

对 42 个配置回归 `C ~ coverage`：

```
C = −34.6 + 226.9 × coverage        R² = 0.940
Spearman(coverage, C) = +0.975
残差中位 +0.9，|残差| ≤ 3 的 22/42，最大偏离 −15.2（`C_dmax0.24`）
```

即：**正确答案数的变化，94% 可以由覆盖率的句号解释。** 8 行表的逐行分解：

| 配置 | θ | cov | ΔC 实测 | ΔC 若只有覆盖率变化 | 残差（干预本身） |
|---|---|---|---|---|---|
| Baseline | 1.85 | 0.664 | 0 | 0 | 0 |
| Veto dmax≥0.30 | 1.90 | 0.528 | −27 | −30.7 | **+3.7** |
| Veto dmax≥0.28 | 2.00 | 0.511 | −31 | −34.7 | **+3.7** |
| Veto dmax≥0.15 | 1.40 | 0.362 | −61 | −68.4 | **+7.4** |
| Cut edges 0.20 | 2.00 | 0.668 | 0 | +1.0 | −1.0 |
| Cut 0.20 + veto 0.30 | 1.75 | 0.699 | +4 | +7.9 | −3.9 |
| Member complete | 1.85 | 0.659 | −4 | −1.0 | −3.0 |
| Global 0.20 + veto | 2.50 | 0.568 | −16 | −21.8 | +5.8 |

读法：**每一行的实测 ΔC 都比"仅覆盖率变化"的预测值更接近 0**；三行 veto 的残差都是正的（+3.7 / +3.7 / +7.4），即**在给定覆盖率下，veto 配置答对的还略多一点**。

这不推翻"veto 有害"——veto 删簇 → 查询被拒 → 覆盖率下降 → C 下降，这条因果链正是"有害"的实现方式。但它说明两件事：

1. **表里的 ΔC 不是"干预效应"，而是"干预 + 它逼出来的工作点"的合成效应。** 读者按行比较会把 −27 全记在 veto 名称上。
2. **veto 的伤害通道是"拒绝"而非"答错"**——与主稿 "most loss came from unanswered rather than wrong queries" 一致，但主稿说的是基线整体，不是逐配置。

> 方法学边界：残差来自横截面回归，不是反事实重跑。要严格隔离，需在固定 θ 下重算各配置的 C（当前产物不支持，`members/*.csv` 已固定在各自 θ\*，见 §5）。

### 2.5 E：「排名反转」——部分成立，幅度被高估

主稿：42 配置上簇级精度与正确答案数 **Spearman ρ = −0.42**。

我用 `members/*.csv` + 真值自算的簇纯度（未加权平均簇纯度，全 42 配置）替换"inherited precision"（口径不同，绝对值不可直接比，方向可比）：

| 统计量 | 值 |
|---|---|
| Spearman(纯度, C) | **−0.744** |
| Spearman(coverage, C) | +0.975 |
| Spearman(纯度, coverage) | −0.679 |
| **偏相关(纯度, C \| coverage)** | **−0.500** |

**结论：反转不是 coverage 伪影**——控制覆盖率后仍有 −0.50 的秩偏相关，方向不变。但约三分之一（−0.744 → −0.500）的关联来自"纯度高的配置覆盖率低"这一机械耦合。

建议在主稿该处加一句限定：*the inversion is not solely a coverage artefact; controlling for coverage leaves a rank partial correlation of −0.50.* 这样反而更抗审。

补充一个刺眼的实测细节：`C_dmax0.15..0.19` 五个配置的簇纯度**精确等于 1.000**（所有含标注的簇都是纯的），而它们的 `fm` 是全表最高（3.64%）——再次说明两个指标测的不是同一件事。

### 2.6 F：没有「补偿性干预」式伪正向增益 —— 不同的地方在这里

E21 的最强发现是"C 的收益空间由 `thr_out` 制造"。主稿里有没有对应物？

- **唯一的正向结果**（`D_split0.20_veto0.30`，+4，p=0.289）主稿已自行判定为"孤立可行点"，robust selection 下 ΔC=0。我们复核：该点 `same` 余量 **0.0000**，`θ−0.05` 不可行。**主稿的处理是正确的**，这一类问题它已经自己挡掉了。
- **但动机侧有耦合未披露**：veto 的立论是"39 个簇里 16 个混合"。而混合由 `thr_in=0.15` / `thr_out=0.25` 决定。E21 实测 `thr_out` 0.25→0.20 时跨机位污染率 12.4%→8.9%（3/3 天同向）。**"16/39" 是一个参数依赖的量，不是场景固有属性。** 这一条对主稿有利（说明 veto 更没必要），但没写出来。

### 2.7 附带：42 个配置里 5 个与基线逐位相同

`C_dmax0.36` / `0.37` / `0.38` / `0.39` / `0.40` 的 `(K, θ, coverage, same, fm)` 与 `A_baseline` **完全相同**——veto 阈值高到从不触发。实际独立配置 **37 个**。

影响：主稿按 41 个非基线配置做 Bonferroni（α=0.05/41=0.0012），分母里含 5 个恒等式，校正偏保守。改成 36 个独立对照 → α=0.0014，只会让结论更容易成立，**方向安全**。仅需一句注明。

---

## 3 严重性判断：改什么、不改什么

**必须改（1 项）**

- **B**：补充表的 `fm` 列加分母，或换成"误合并对数"，并加一句"merge rates are not comparable across configurations because the evaluable pair set depends on θ"。这是唯一会让读者得出相反结论的问题。

**建议加一句披露（4 项）**

- **A**：说明 merge rate 是报告量而非约束量，其值是可行域最松端的取值。
- **C**：说明 20/42 配置的门槛余量 < 0.005，并把 `D_split0.20_veto0.30` 的个案处理上升为表级说明。
- **D**：说明 ΔC 主要由覆盖率驱动（R²=0.940），干预本身的效应量小。
- **E**：给反转加"控制覆盖率后仍为 −0.50"的限定。

**只需注明（2 项）**

- **§2.7**：42 → 37 个独立配置，Bonferroni 分母说明。
- **§2.6**："16/39 混合簇"是 `thr_in`/`thr_out` 的函数，不是场景属性。

**不需要改的**

- 主结论（簇级指标误导、veto 有害、拒绝是瓶颈、四/六条预注册假设成立）。
- 已有基线数字（K=111 / θ=1.85 / cov=0.664 / same=0.908 / C=117 / 精度 0.967）。
- `BASELINE_FREEZE`（本次未改任何文件、未改任何参数）。

---

## 4 可直接贴进 tex 的改法

**A（§label-free selection 末）**

```latex
The merge rate is reported rather than constrained: the rule optimises coverage
subject to agreement, so the value quoted above is the highest attainable on the
loose side of the trade-off, not a target we set.
```

**B（补充表表注）**

```latex
Merge rates are computed on the co-occurring query pairs that are evaluable at
each configuration's own threshold; that set ranges from 302 to 922 pairs
(3.0-fold), so the rate is not comparable across rows. Absolute merge counts are
given alongside.
```

**C（tab:configs 表注）**

```latex
Twenty of the 42 configurations satisfy the agreement gate with a margin
below 0.005, five of them exactly at the gate; the operating point of such a row
is a knife-edge and its coverage should not be read as a stable property.
```

**D（tab:configs 后一段）**

```latex
Correct answers track coverage almost deterministically across the grid
($R^2=0.94$ for $C$ on coverage): most of the variation in $\Delta C$ is the
operating point a configuration is driven to, not the intervention it names.
```

**E（§sec:inversion）**

```latex
The inversion is not solely a coverage artefact: partialling out coverage leaves a
rank partial correlation of $-0.50$ between cluster purity and correct answers.
```

---

## 5 未做 / 需要重跑才能回答的问题

1. **固定 θ 的对照未做。** `members/<cfg>.csv` 固定在各自 θ\*，无法在不重跑的前提下把某配置钉到基线的 θ=1.85 上重算 C。要严格分离"干预"与"工作点"，需给 `d7_sweep`/`site_strict_all` 加 `--theta` 直通并重跑 42 配置（预计 2–3 分钟，成本低）。
2. **`inherited precision` 未能逐位复现。** 主稿的 ρ=−0.42 用的是簇级继承精度，其定义未落在任何现有脚本里（`eval_head_clean.py` 的 `apply_inherit` 是头部裁片继承，不是这个量）。本次用自算簇纯度替代，只作方向对照，**不能声称"复现了 −0.42"**。
3. **1012 上的 `thr_out` 混合簇数未测。** §2.6 的 12.4%→8.9% 是 E21 六机位的结果，1012 未测，故"16/39 是参数依赖量"在 1012 上属**推断**。
4. 本次**未修改主稿任何文件**，也未改动 `configs/selector_deploy.json` 与冻结包。
