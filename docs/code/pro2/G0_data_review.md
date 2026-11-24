# G0 数据与统计审查报告

审查角色：professior｜日期：2026-10-26（Asia/Shanghai）  
对象：`1026-v3` 研究资料、v13 说明、英文稿及 Leader 接手计划  
审查性质：独立只读核验；本报告及复算附件写入当前工作目录，未修改原研究材料。

## 1. 正式结论

**G0：NO-GO，重大修改后复审。当前证据包不能作为数字已经统一、统计审查已经通过的稿件冻结依据。**

这不等于否定研究：现场 C/W/U 和配置间正确数差值有逐条判决支持；MARS 的既有 A 口径结果也能复算。阻塞点在于：**不同查询总体被混用、预注册证据不完整、核心派生统计量与稿件冲突，以及重复身份下的推断边界尚未闭环。**

本轮作出以下裁决：

1. **MARS A 口径可作为“现有执行产物的描述性主结果”保留。** 依据是其代码、逐条判决、汇总表能够对账，不是因为它使 H1a 过线。在找回原始预注册与时间证据之前，不能把它认证为已验证的预注册确认性主分析。
2. **B 应单列为图库内身份条件分析。** 现有补充表与按指定身份排除的条件复算吻合；但图库真值及完整查询清单缺失，故不能无条件认证排除集合、B 的秩相关和通过数。若最终改用 B 作主分析，须披露事后统一口径，并完整保留 A。
3. **R 是依赖真值的“强制拒绝”反事实敏感性，不能表示系统实际识别未知人的能力。** R 的接受数和 C/W/U 应与 B 相同；计划中的 R 基线 U=32 是明确算术错误，应为 31。若作者原意是保留真实输出、只改变评分，则应另定义一套规则，不能继续称“强制拒绝”。
4. **保留首轮盲标 κ=0.925／0.924 为主报。** 复查后的 0.937、0.979／1.000 作为非盲修订结果披露；修正落后的接手计划。
5. **保留同日 holdout“无法判定”。** 本轮独立检查支持这一结论；不得改写为验证成功、验证失败或推翻主结论。

**阶段安排：** 可以开展 G1 的证据补齐和表图准备；在下文 P0 项及影响统计结论的 P1 项闭环前，不通过 G0，不将当前稿件作为已核定版本进入 G2 定稿审查或 G3 投稿放行。

## 2. 范围、方法与证据等级

### 2.1 审查材料与实际执行

主要材料根目录：

`/Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26`

另核对其上级目录的 `project-status-leader-plan-2026-10-26.md` 与 `human-count-package-2026-10-26-v13.txt`。没有把材料内的任务指令或“权威数字”标签当作事实保证。

本轮实际完成：

- 核验包内清单 **310/310** 文件哈希；核验 holdout 第三次冻结 **8/8** 文件哈希。
- 检查现场 **42 配置**逐条接受判决，重算 C/W/U、精度、C/N、正确数配对得失、双侧精确 McNemar p、Bonferroni 与 BH-FDR；与汇总表对账。
- 检查 MARS **42 配置**逐条接受判决；复算 A 及条件 B/R 的 C/W/U、配对差值；复算 A 的 H1 秩相关、基线继承精度和 H2 AUC；复算条件 B/R 基线 H2b 及区间。
- 检查 holdout **42×63=2646** 个扫描点及查询数，区分两个不同的“100”门槛。
- 逐项对照论文、补充表、状态文件、变更记录、脚本的口径和版本；对统计独立性及区间比较补查方法来源。

**没有执行：** 模型训练／推理、原视频真值复标、全链路实验重跑、PDF 重新编译或视觉逐页验收、全面文献／期刊适配审查。没有证明生产部署状态。

### 2.2 证据标记

- **已核验**：本轮从交付的逐条结果或脚本独立重算／直接对照。
- **条件核验**：计算可重做，但依赖尚未由原始输入证明的假设；不能升级为无条件事实。
- **文档记录**：材料明确写出，本轮缺少原始输入或历史快照验证。
- **审查判断**：基于证据作出的统计／报告建议，不冒充新实验结果。

哈希通过只说明“收到的文件与其清单一致”，不证明科学结论正确，也不证明冻结早于运行。当前清单 SHA256：`ba024493fd7297792cd8384a66297d2a4977e35bcd7afe9fcb97c1ecd0be7ff6`。详细记录见 [完整性附件][A01]。

## 3. 可以接受到什么程度

| 项目 | 本轮结果 | G0 判定边界 |
|---|---|---|
| 现场基线 | accepted=123；C/W/U=117/4/2；precision=0.966942；C/229=0.510917 | **已核验计数**；不等于原视频真值已独立复标 |
| 现场损失闭合 | 196=117+4+2+73；229=196+22+11；四机位求和一致 | **已核验算术**；查询资格仍依赖原始真值定义 |
| 现场 veto 0.30 | C=90；丢28、得1；ΔC=−27；逐查询双侧 p=1.117587×10⁻⁷ | **已核验效应与名义 p**；身份内依赖见 F07 |
| 现场 41 次比较 | Bonferroni 22项、BH 24项，全部 ΔC<0 | **已核验现有逐查询计算**；不能据此认证身份聚类推断已完成 |
| 现场 H1 | 当前表及导出簇元数据均给基线74/123=0.601626、ρ=−0.425984 | **与稿件冲突**；不能继续签认0.585／−0.418，见 F04 |
| MARS A 基线 | accepted=320；C/W/U=255/33/32；precision=0.885417；C/484=0.526860 | **已核验交付判决计数**；完整查询总体与图库构成未独立重建 |
| MARS veto 0.30 | 丢19、得4；ΔC=−15；逐查询双侧 p=0.002599478 | **已核验**；条件剔除指定身份不改变这些正确数配对 |
| κ 首轮主报 | 图库0.925、查询0.924 | **文档／结果文件相互支持**；本轮未从两套原始标注复算 κ |
| 同日 holdout | 68查询；0/42可行配置；共现对可评数最高81，same≥0.90时最高74 | **已核验导出扫描**；“无法判定”成立 |

现场详细结果：[A02]、[A03]；MARS：[A05]、[A06]；holdout：[A08]。现场所有42配置的 C/W/U/accepted 与主汇总及补充表无差异；这里的“无差异”不包含正文 H1 数字、未知查询资格、原始簇投票或未经提供的输入。[E01] [E02]

## 4. MARS 口径裁决与可复算数字

### 4.1 三种口径并非同一个统计对象

| 口径 | 查询／输出处理 | 本轮证据性质 | 可使用方式 |
|---|---|---|---|
| A：include | 保留全部实际 accepted 输出；比较查询与簇身份，得到 C/W/U | 代码和 accepted 判决直接支持 | 现有执行记录的描述性主结果；尚未认证预注册地位 |
| B：exclude | 限定查询真实身份出现在图库；排除非人／图库外查询的 C/W/U | 脚本明确；缺图库GT，只能条件复算指定排除集合 | 图库内条件精度／敏感性；不得称完整开集性能 |
| R：forced rejection | 在全部查询上，把已知图库外查询人为设为拒绝，其他输出同 B | 本轮显式反事实计算；未见可运行的未知身份拒识机制 | 理想化拒绝敏感性；不得当作实际系统结果 |

A 的分支在 `public_eval_tieu.py` 先检查系统是否 accepted，再判 C/W/U；没有按图库身份过滤查询。[E03] B 的 `c1_supp_table.py` 则先构造图库内查询集。[E04] 这足以解释两套表为何不同，而不是简单四舍五入误差。

**条件前提：** 接手计划称图库缺 PID 0782、1172，共4条查询。本包没有图库／查询GT，也没有完整484条查询输出，故不能独立证明“正好这两人、正好四条、图库148人”。可直接观察的事实仅是：0782 的 `c2_q#65`、`c3_q#75` 出现在 accepted 判决中，跨42配置共48个接受实例；基线分别是 U 和 W。[E05] [A07] 1172 的两条恒拒绝查询无法从 accepted-only 文件观察到。不能把“未在预测簇身份中出现”当作“真实图库不存在”的证明。

### 4.2 基线数字与分母

以下 B/R 使用上述排除前提；480与4来自待验证的查询总体声明。

| 指标 | A：观察到的全量输出 | B：条件排除 | R：条件强制拒绝 |
|---|---:|---:|---:|
| 用于计数闭合的查询总量 | 484 | 480 | 484 |
| accepted | 320 | 318 | 318 |
| C | 255 | 255 | 255 |
| W | 33 | 32 | 32 |
| U（接受后无法归属） | 32 | 31 | **31** |
| 未答总数 | 164 | 162 | 166 |
| precision=C/(C+W) | 0.885417 | 0.888502 | 0.888502 |
| 继承精度 | 234/320=0.731250 | 233/318=0.732704 | 233/318=0.732704 |
| C/484：全部查询中的正确身份产出率 | 0.526860 | 0.526860 | 0.526860 |
| 图库内子集 C/480 | 不与全量 C/484 混用 | 0.531250 | 0.531250 |

每列均应满足 `accepted=C+W+U`。R 的166条未答包含162条图库内未答和4条图库外强制拒绝；**若 H3 的研究对象是“可答查询”，应比较162与32，而不是166与32。** A 中164与33是全体查询的未答／错答计数，不能无说明地放进“among answerable queries”谓词。

B 表沿用 C/484 是可解释的：它表示完整工作负载上的正确身份产出率。必须另标分母，不能把它误称为图库内子集召回率。B 的 precision 则只在图库内接受且可判定的输出上计算，两者并非同一个分母。也不得把 U 当作拒答，或者把 `unanswerable` 字段名直接理解为“身份不在图库”；源脚本实际用它表示 `n_query−accepted`。[E06]

### 4.3 七项假设逐项审核

这里核验的是**现有判定表与代码列出的七项规则**，尚不是对“原始预注册确有这七项”的认证。

| 假设 | A 本轮结果 | A按现有规则 | B/R 本轮可确认程度 |
|---|---|---|---|
| H1a：ρ≤−0.3 | −0.302510515，使用源码同样的四位继承精度导出值 | 达到门槛，距边界约0.00249 | **不能独立复算**各配置排除后的继承精度；计划中的约−0.289暂不签认 |
| H1b：继承精度最佳配置ΔC<0 | 最佳 `C_dmax0.16`；C=102，ΔC=−153 | 达到门槛 | 此配置的C不变；但不能据此证明它仍是B/R继承精度最佳配置 |
| H1c：继承精度<逐条精度 | 0.731250<0.885417 | 达到门槛 | 基线条件复算0.732704<0.888502，支持 |
| H2a：混人AUC≥0.80 | 0.7421875 | 未达到 | 不改变图库簇，沿用同一簇元数据仍未达到 |
| H2b：有害AUC≤0.65或CI含0.5 | 0.671131；CI [0.569779,0.774334] | 未达到 | 条件结果0.678744；CI [0.572507,0.783112]，未达到 |
| H2c：veto ΔC<0且p<0.05 | −15；p=0.002599478 | 达到名义门槛 | 条件复算不变；统计独立性限制仍适用 |
| H3：未答>错答 | 全量164>33 | 实现层比较成立，但需修正“可答”谓词 | 条件图库内162>32成立；R不要把4次正确拒绝计为可答损失 |

**结论：** 当前 A 判定表按其实现计数是 **5/7**，不是4/6。[E07] B/R 的“4/7”仍是计划中的报告值，本轮不予独立认证，因为 H1a 和 H1b 所需的全配置簇身份数据缺失。H1a 数值接近阈值不表示研究被推翻，但必须并列披露口径敏感性，禁止按过线与否选择分析集。

H2b 复算使用原脚本同类的簇重采样：`numpy.default_rng(7)`，2000次，每次在“收到接受查询的簇”内有放回抽样，剔除单一类别样本，取2.5/97.5百分位。A 的结果复现原区间；B/R 区间是本轮条件复算，不能与计划中的约[0.564,0.779]混用。它们是基于导出的 `d_max`／混人标签做的结果层复算，没有重建嵌入或簇成员。[A05] [E08]

### 4.4 为什么暂不认证“预注册外部复验”

包内只有同日 holdout 的完整预注册材料。MARS 报告引用 `datasets/VALIDATION.md`，但该文件、冻结副本与可核验的事前版本未随包提供；在相应本地原路径也未找到。[E09]

公开验证记录写明：先按 `evaluable≥100` 执行，42配置均无工作点，再改 `--min-eval 0`；稿件则称“Following the preregistered fallback”。[E10] [E11] 两者不必然矛盾——原方案可能确有预先安排的 fallback——但当前证据无法判定。也不能凭接手计划断言原始方案一定未规定 query universe。

处理要求：恢复运行前的原始方案及版本／时间依据；若无法恢复，披露“有内部方案记录，事前内容与时间不能独立核验”，删除或限定无法支持的确认性措辞。不得用本轮复算、后补哈希或 holdout 预注册代替 MARS 的原始方案。

## 5. 重大问题、定位与修改要求

优先级：**P0**阻塞当前数字／证据冻结；**P1**影响统计解释或审计完整性、相关结论发表前必须关闭；**P2**局部文档修订。每项都给出可核验的关闭条件。

### F01｜P0｜MARS主文、补充表和补表脚注混用A/B

**事实，高置信度。** 主文与 `h1_rank.csv` 基线为320/255/33/32；补充CSV及长表行为318/255/32/31。更直接的是：同一补充TeX第108行采用B，但第141行脚注仍写A的W33/U32/精度0.885。[E12] [E13] 42配置逐条A与A表一致，条件B与B补表一致，说明混用源于分析集分支，不是随机数据损坏。

**修改：** 为每张表／图明确 `analysis_policy`、查询集合、未知查询处理、分母；主表、补表、脚注、摘要、图2/图5、VERDICTS、canonical须由同一组明确标识的产物生成。B表若保留为敏感性，标题必须说明排除及C/484分母。修正“4/6”，但不能直接把所有位置机械改成未经验证的B“4/7”。

**关闭：** A/B各42行逐项对账；主文与对应主表一致；所有七项规则逐项计算；没有混合口径的脚注。

### F02｜P0｜MARS原始协议、排除集合与完整输入证据不足

**事实／未知，高置信度。** 缺原始 `VALIDATION.md`；缺图库GT、完整查询GT、每配置成员表和完整查询输出。A的accepted并集仅403条、144个查询PID，不能恢复全部484条查询或图库148个身份。[A05] 已导出的簇元数据只足以重算基线部分指标，不能恢复B全部42配置的继承精度。

**修改：** 补充脱敏的查询清单（qid、pid、图库内外、可评资格）、图库成员真值、每配置成员与所有查询决策、版本哈希及原始方案；无须为这些统计核验公开视频或嵌入。若无法补齐，降低为可追溯的描述性分析，并列清楚哪些结论无法复核。

**关闭：** 能从图库身份集合独立生成0782/1172排除名单、核实4条及148身份、复算B/R H1a及最佳配置；协议承诺能定位到运行前版本，或稿件完成明确降级披露。

### F03｜P0｜R口径不闭合，并混淆U、拒答和可答损失

**事实，高置信度。** 计划写R基线318/255/32/32，后3项和为319；并称“原已拒绝的1172，剔除只改U”。[E14] 代码在处理非accepted时直接跳过，它们不进入U。[E03] 按强制拒绝定义，R应是318/255/32/**31**。R中166条未答也不全是可答身份损失。

**修改：** 固定U为“已接受但身份不能归属”；拒答另列。R标注为依赖真值的反事实，不得并入实际系统性能；H3按可答集合统计，未知人正确拒绝单列。全量开集输出还应单报未知查询接受／误接受与未知拒绝，不能仅用排除了未知人的precision表达能力。

**关闭：** 每配置两条恒等式成立；真实输出与反事实输出分别存储；未知接受、未知拒绝与U没有交叉混计。

### F04｜P0｜现场H1核心数值与新结果文件不一致

**事实，高置信度。** 当前 `onsite_h1.csv` 基线为0.6016；从当前逐条判决的 `cluster_n_ids==1` 也得到74/123=0.601626；42配置Spearman为−0.425984。正文却仍写继承精度0.585，假设表写ρ=−0.418。[E15] [E16] [A02] 本轮重算负相关方向、最高继承配置的C=56及ΔC=−61仍成立；冲突影响精确数字和对应图表，不能因方向没变而略过。

**潜在原因只是推断：** v13重导出现场结果后派生数字未同步，或不同脚本对UNK／非人簇的严格定义不同。`site_strict_all.py` 先排除UNK成员、再以 `n_ids==1` 累计继承分子；`inh2.py` 的读入／roster处理不完全相同。[E17] 缺原始成员真值，不能仅凭当前n_ids断定哪个定义最终正确。

**修改：** 冻结严格继承精度定义（UNK是否污染纯簇、非人如何处理、分子按查询加权还是按簇）；由GT与成员表重建整数分子／分母。暂将0.601626／−0.425984标为“当前导出结果可复算值”，不要未经定义裁决直接覆盖论文。

**关闭：** 同一定义的逐簇／逐查询账本、42配置H1表、稿件、canonical、图2完全一致，并解释从旧值变到新值的原因。

### F05｜P0｜“所有标注修订均未改变U”超出证据

**事实，高置信度。** 稿件把历史36条非人复核与后期v3复核并列，声称两轮均没有改变C/W/U。[E18] 历史变更记录却写该36条修改使accepted138→124、U14→0，C/W不变。[E19]

**修改：** 分开叙述“历史非人重分类改变评估资格与U”和“后期v2→v3在固定strict规则下的C/W/U不变”。后者本轮只能确认现有文件内部一致，缺旧判决快照不能独立验证跨版本逐条零变化。不要把“42配置的C不变”推广为所有派生指标不变；F04已说明风险。

**关闭：** 提供版本转换表，分别列GT变更、评分规则变更、查询资格变化及各输出差异；正文不再作全历史不变的概括。

### F06｜P1｜κ轮次与真值版本混写，接手计划已落后

**事实，高置信度。** 新稿与变更记录已主报首轮盲标0.925/0.924；计划仍建议v2主报0.937。[E20] [E21] 记录明确第二轮是在看过κ及分歧统计后复查，不能重新命名为盲标。[E22]

**修改：** 维持首轮主报；将两次后续结果作为看过一致性结果后的修订。分别给“GT版本”和“信度测量轮次”编号；同步计划、状态表及列说明。列说明末尾把round1又写成“仅追溯”，也应删除。[E23]

**关闭：** 每个κ均注明标注轮次、盲性、样本量、身份编号映射和排除规则；提供原始双标表才可升级为独立复算通过。首轮κ可以作为报告主口径，不代表当前材料已证明全部映射无偏。

### F07｜P1｜配对不自动消除同一身份内的重复观测依赖

**事实，高置信度。** 现有McNemar脚本按query逐条累计discordant pairs；本轮完全复现其算术。[E24] 现场veto的28条损失中，female3占16、male3占6；得失分布集中在6个身份，其中一人的得失相抵。[A04] 正文也承认查询不独立，并对C/N采用身份聚类区间。[E25]

**审查判断：** 逐查询配对解决“同一query在两配置之间比较”的配对问题，不解决“同一person的多个query之间相关”的问题。现有p值应标为逐查询名义结果；Bonferroni/BH校正不修复底层独立性假设。身份聚类匹配数据需要相应的聚类推断方法，见 [Durkalski等原始方法论文][M01]、[Yang等方法研究][M02]。

正文报告veto身份重采样CI为[−53,−3]，但包内旧独立性报告使用旧计数及[−59,−7]；当前完整身份查询清单和该配对区间的明确再现入口缺失，故本轮不能签认新版区间。[E26] `derived2.py`提供的是绝对C/N区间代码，不能替代配对差值区间的验证。[E27]

**修改：** 为预先定义的关键比较报告身份层面的配对效应区间／适当检验，保存全身份组（包括零变化组）、随机种子、重复次数、估计量与样本单位；A/B同一比较使用同一身份重采样。保留名义McNemar作为附加结果，并限定41配置显著性结论。不要只因算术复现就宣称统计假设已满足。

**关闭：** 关键ΔC和区间可重算，独立单位有明确说明；多重比较家族和确认性／探索性比较区分清楚。样本不足时保留描述性效应，不强行制造“显著”结论。

### F08｜P1｜196/24分母混用与κ“无显著差异”措辞

**事实，高置信度。** 局限仍把196条可答写成24身份组；包内待办已说明24对应229条可评（23人物身份＋1非人组），196条对应19人物身份。[E28] 本轮accepted并集只覆盖18人物身份，缺完整拒答身份表，不能独立证明19/24；应按完整GT重新确认，而不是拿accepted并集替代总体。

κ修订报告以各机位CI重叠推出“无显著差异”，稿件沿用该结论。[E29] 两个区间重叠不能替代差值的检验或区间，参见 [Altman与Bland][M03]。

**修改：** 明确C/229及其身份分组；保留非人合并为一组的规则并说明理由，避免把24都称为独立人员。无直接差异检验时，把机位κ结论改为描述性区间比较。

**关闭：** 全部查询到分组的表可复核；正文分母与区间估计对象一致；没有由CI重叠推导的显著性断言。

### F09｜P1｜“每个数字均可复算”的交付能力表述过强

**事实，高置信度。** 多个脚本依赖未提供的 `deliverable/data-request-v2/gt`、`results/members`、`results/queries`、`validation/mars`、ledger或嵌入。[E17] [E27] [E03] 目前可以进行结果层审计，不能独立重建所有簇投票、绝对身份区间、现场d_max AUC、全部阈值选择或原始真值版本变化。

**修改：** 建立“指标→输入→脚本→输出→规则版本／哈希”索引；明确哪些可由公开脱敏表复算、哪些需要受控数据、哪些只有历史记录。把“只有伦理／作者材料阻塞”改为实际技术阻塞清单。[E30] 对B图表与现场H1，不得用“脚本已提供”代替输入可得性。

**关闭：** 在干净目录从已声明可用输入重建关键数字，或显式缩小可复现性主张；所有缺失依赖均有责任人和替代披露。

### F10｜P1｜计数与RGB仍未按当前口径闭环

**事实，高置信度。** 正文计数MAE1.28/1.30及RGB的118→67仍标pending；后者与当前基线117不是同一版本。[E31] 这些可以留作历史探索记录，不能与strict当前结果混为已核定的同一证据层。

**修改：** 用当前真值与评分规则复核61窗口计数及RGB C/W/U、ΔC、区间；不必重新编码模型，先复用原结果与真值。若输入无法取得，移为明确标注旧口径的探索性材料或撤下精确数字。

**关闭：** 每项均有新版输出、旧新差异解释与正文替换；正文技术pending清零，或者相关主张已删除／降级。

### F11｜P1｜holdout结果可接受，冻结时点仍不能认证

**已核验：** 8/8冻结文件匹配；2646扫描点没有满足same≥0.90且可评共现对≥100的点。另有68条可评query，因此不可能满足“可答query≥100”。这两个门槛独立，不得混写。[A08] [E32]

**未闭环：** 原方案要求开跑前外部时间戳，签署页仍为空；结果记录已承认未完成。[E33] 不能用本轮哈希通过证明事前登记，也不能把待签字写成已签字。无工作点且未标注时，不补造C/W/U。

**修改／关闭：** 保持无法判定与时间证据缺失披露；若有可取得的历史凭据则附上，若无则如实说明。将“外部事前登记已完成”类断言删除，不倒填时间。

### F12｜P2｜残余数据字典与报告解释错误

**事实，高置信度。** `README_columns.md`把coverage描述为图库轨迹分簇比例，实际稿件与输出定义为查询a/b双侧接受率；`C_over_229`在CSV中存整数117，不是比值0.511。[E23] [E01] MARS补表CSV把不可计算的fm写为0.0，但TeX显示“--”；应区分缺失与零错误率。[E12] holdout结果第106行已经更正“跨机位错开不影响同机位fm”，第116行仍保留“共现窗压缩4.3%”旧解释。[E34]

**修改／关闭：** 数据字典与实际字段一致；缺失fm用明确NA并解释；删除已被更正的旧因果解释。历史文件若保留，标明被取代范围。

## 6. 可执行修改清单与复审门槛

以下为审查后的执行建议，不表示本轮已修改研究材料。

| 顺序 | 责任建议 | 交付物 | 验收条件 | 对应问题 |
|---|---|---|---|---|
| 1 | engineer提供，professior核验，Leader确认研究对象 | 脱敏MARS查询／图库／成员表、原始方案与版本证据 | 能重建排除集合、查询总量、图库身份数；事前承诺或其不可验证性明确 | F02 |
| 2 | professior定统计口径，engineer实现 | A主执行表、B条件表、R反事实表及逐项七假设判定 | 每表独立标识；U/拒答闭合；B H1a与最佳配置可复算；H3只在其声明总体计算 | F01–F03 |
| 3 | engineer复算，professior复核 | 现场strict继承定义与整数账本、H1表 | 解释0.585→0.6016与−0.418→−0.426差异；同步表图而非只改文字 | F04 |
| 4 | Leader/标注负责人提供历史，engineer生成diff | GT版本转换表、κ轮次表、全查询身份分组 | 历史36条与后期v3分开；首轮盲标主报；196/229/19/24各有来源 | F05、F06、F08 |
| 5 | professior明确推断单位，engineer复算 | 关键比较身份层配对区间／检验及多重比较说明 | 重采样包含零变化组；固定输入、种子、估计量；区分名义query p与聚类推断 | F07 |
| 6 | engineer、Leader | 当前口径计数／RGB复核，统一数字索引 | 正文无未解决技术pending；缺失数据不能靠文字确认代替 | F09、F10 |
| 7 | Leader整理，professior复审 | 新候选包、修改响应表 | 逐条答复F01–F12；正文、补表、图、canonical一致；8/8和包哈希验证；时间缺口如实披露 | F01–F12 |

**复审所需最低材料：** 上述输出、F01–F12逐项响应、修订稿及新manifest。允许保留科学上的“无法判定”或证据访问限制，但必须调整相应主张；不要求为了让结果过线而重选窗口、调阈值或补做无目标实验。

**本次可签认范围：** 指定交付文件内可复算的计数、差值和规则实现，以及本报告列出的文件冲突。**本次不签认：** 全部原始真值、B/R完整秩相关与通过数、MARS原始预注册地位、现场新版聚类区间、全链路复现及投稿就绪。

## 7. 复算附件与复现方法

所有新文件均位于当前目录；源材料未写入。

| 附件 | 内容 |
|---|---|
| [g0_audit/site_summary.json][A02] | 现场42配置汇总核验、H1冲突值、环境版本与限制 |
| [g0_audit/site_configuration_checks.csv][A03] | 每配置C/W/U、继承分子、名义p及校正结果 |
| [g0_audit/site_veto_identity_deltas.csv][A04] | veto丢28得1的身份分布 |
| [g0_audit/mars/summary.json][A05] | A/B/R条件、基线、AUC/CI、未验证事项 |
| [g0_audit/mars/configuration_results.csv][A06] | 42配置×3口径=126行计数与配对结果 |
| [g0_audit/mars/excluded_accepted_instances.csv][A07] | 两个0782查询的48个接受实例 |
| `g0_audit/mars/clusters_A.json`、`clusters_B.json`、`clusters_R.json` | 基线逐簇AUC复算输入及条件过滤后结果 |
| [g0_audit/holdout_summary.json][A08] | 8文件冻结校验、扫描可行性和查询数 |
| [g0_audit/source_integrity.csv][A01] | 310文件逐项哈希 |

在当前工作目录复现：

```bash
python3 g0_audit/recompute_site.py
python3 g0_audit/mars/recompute_mars.py
python3 g0_audit/verify_holdout.py
```

本轮环境：Python 3.14.8、NumPy 2.4.2、SciPy 1.17.1。脚本只读取源包，覆写本目录对应审计产物。数字结果层复算不需要影像、嵌入或模型；它也不能证明这些未提供输入正确。B/R脚本中的排除PID及480/484总体是显式条件，不能把该脚本的成功运行当作原始排除集合已经验证。

## 8. 证据定位索引

以下链接指向本次实际核对的文件与起始行；对涉及多个位置的事项分别列出。源包同日文件存在跨阶段内容，因此按具体证据项引用，不按文件名“最新”或“权威”整份采信。

[A01]: /Users/muyi/Downloads/dev/human-count-professior2/g0_audit/source_integrity.csv
[A02]: /Users/muyi/Downloads/dev/human-count-professior2/g0_audit/site_summary.json
[A03]: /Users/muyi/Downloads/dev/human-count-professior2/g0_audit/site_configuration_checks.csv
[A04]: /Users/muyi/Downloads/dev/human-count-professior2/g0_audit/site_veto_identity_deltas.csv
[A05]: /Users/muyi/Downloads/dev/human-count-professior2/g0_audit/mars/summary.json
[A06]: /Users/muyi/Downloads/dev/human-count-professior2/g0_audit/mars/configuration_results.csv
[A07]: /Users/muyi/Downloads/dev/human-count-professior2/g0_audit/mars/excluded_accepted_instances.csv
[A08]: /Users/muyi/Downloads/dev/human-count-professior2/g0_audit/holdout_summary.json

- **E01**：[现场precision.csv，第2行][E01]；基线123/117/4/2；`C_over_229`为整数117。
- **E02**：[现场loss_decomposition.csv，第2行][E02]；196、229和分机位闭合。
- **E03**：[public_eval_tieu.py，第195行][E03]；A口径的接受筛选、C/W/U与继承精度（至228行）。
- **E04**：[c1_supp_table.py，第30行][E04]；图库身份集合、B查询筛选和独立传入分母（至66行）。
- **E05**：[MARS基线判决，第133行][E05]；另一条见同文件第218行。
- **E06**：[public_eval_tieu.py，第308行][E06]；H3全量损失计算，`unanswerable`实际为未答（至326行）。
- **E07**：[MARS VERDICTS，第6行][E07]；七条规则与五成立、二未成立。
- **E08**：[public_eval_tieu.py，第264行][E08]；mixed/harmful定义与AUC分析集（至285行）。
- **E09**：[public-validation报告，第3行][E09]；引用未提供的原始方案。
- **E10**：[public-validation报告，第71行][E10]；首次无可行点、后用min_eval0（至78行）。
- **E11**：[稿件，第326行][E11]；“preregistered fallback”措辞。
- **E12**：[补充MARS CSV，第13行][E12]；B基线；相应主文A见稿件第328行。
- **E13**：[supp_tables.tex，第108行][E13]；B行；第141行仍为A脚注。
- **E14**：[接手计划，第148行][E14]；R不闭合；第155行混淆原拒绝与U。
- **E15**：[onsite_h1.csv，第2行][E15]；当前0.6016。
- **E16**：[稿件，第215行][E16]；旧继承精度0.585；ρ=−0.418见第339行。
- **E17**：[site_strict_all.py，第25行][E17]；去UNK读入及第103行继承分子；另见 [inh2.py，第5行](/Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/05_scripts/inh2.py:5)。
- **E18**：[稿件，第175行][E18]；把多轮修订概括为C/W/U均不变。
- **E19**：[CHANGELOG，第295行][E19]；36条非人修改及U14→0（至299行）。
- **E20**：[CHANGELOG，第9行][E20]；首轮盲标改为主报；稿件执行见第177行。
- **E21**：[接手计划，第26行][E21]；仍建议v2主报；重复位置第166、313行。
- **E22**：[κ修订报告，第151行][E22]；看过首轮结果后的复查。
- **E23**：[README_columns，第14行][E23]；字段说明；第76行主报round1，第88行又称“仅追溯”。
- **E24**：[a8_multicomp.py，第46行][E24]；逐query配对和二项式p（至76行）。
- **E25**：[稿件，第191行][E25]；身份内不独立与C/N区间。
- **E26**：[稿件，第255行][E26]；新版veto配对区间；旧独立性报告见 [第10行](/Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/06_reports/query-independence-2026-10-23.md:10)。
- **E27**：[derived2.py，第53行][E27]；绝对C/N的身份组重采样，非配置配对区间。
- **E28**：[OPEN_ITEMS，第115行][E28]；196/24已知错误；稿件残留见第402行。
- **E29**：[κ修订报告，第128行][E29]；凭区间重叠判断机位差异；稿件第177行沿用。
- **E30**：[START_HERE，第12行][E30]；“硬阻塞只剩伦理与作者材料”。
- **E31**：[稿件，第391行][E31]；计数pending；第396行RGB pending。
- **E32**：[holdout预注册，第153行][E32]；共现对筛选；第158行可答查询门槛。
- **E33**：[holdout结果，第103行][E33]；外部时间戳未完成；签署页待填见 [第38行](/Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/03_preregistration/preeg_signature_2026-10-24.md:38)。
- **E34**：[holdout结果，第106行][E34]；fm解释更正；第116行仍残留旧因果说法。

[E01]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/04_results/site/precision.csv:2
[E02]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/04_results/site/loss_decomposition.csv:2
[E03]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/05_scripts/public_eval_tieu.py:195
[E04]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/05_scripts/c1_supp_table.py:30
[E05]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/04_results/mars/verdicts/A_baseline.csv:133
[E06]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/05_scripts/public_eval_tieu.py:308
[E07]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/04_results/mars/VERDICTS.md:6
[E08]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/05_scripts/public_eval_tieu.py:264
[E09]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/06_reports/public-validation-2026-10-23.md:3
[E10]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/06_reports/public-validation-2026-10-23.md:71
[E11]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/02_manuscript/manuscript.tex:326
[E12]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/04_results/supp_table_mars.csv:13
[E13]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/02_manuscript/supplementary/supp_tables.tex:108
[E14]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/project-status-leader-plan-2026-10-26.md:148
[E15]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/04_results/site/onsite_h1.csv:2
[E16]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/02_manuscript/manuscript.tex:215
[E17]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/05_scripts/site_strict_all.py:25
[E18]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/02_manuscript/manuscript.tex:175
[E19]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/01_status/CHANGELOG_zh.md:295
[E20]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/01_status/CHANGELOG_zh.md:9
[E21]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/project-status-leader-plan-2026-10-26.md:26
[E22]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/06_reports/kappa-a3-revision-2026-10-26.md:151
[E23]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/04_results/README_columns.md:14
[E24]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/05_scripts/a8_multicomp.py:46
[E25]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/02_manuscript/manuscript.tex:191
[E26]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/02_manuscript/manuscript.tex:255
[E27]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/05_scripts/derived2.py:53
[E28]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/01_status/OPEN_ITEMS_zh.md:115
[E29]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/06_reports/kappa-a3-revision-2026-10-26.md:128
[E30]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/00_START_HERE.md:12
[E31]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/02_manuscript/manuscript.tex:391
[E32]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/03_preregistration/PREREGISTRATION_holdout_2026-10-24.md:153
[E33]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/07_holdout/result/holdout_result_2026-10-24.md:103
[E34]: /Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26/07_holdout/result/holdout_result_2026-10-24.md:106

方法资料（本轮仅用于上述两项统计原则，不构成完整文献审查）：

- [M01：Durkalski等，Analysis of clustered matched-pair data][M01]，Statistics in Medicine，DOI 10.1002/sim.1438：聚类配对数据需要对通常的McNemar推断作相应处理。
- [M02：Yang等，A note on the tests for clustered matched-pair binary data][M02]，Biometrical Journal，DOI 10.1002/bimj.201000035：区分独立匹配对与聚类匹配对。
- [M03：Altman与Bland，Interaction revisited: the difference between two estimates][M03]，BMJ 2003，DOI 10.1136/bmj.326.7382.219：区间重叠不能替代差异检验。出版方正文抓取受限，本轮依据出版方可检索摘要／片段核验该原则，未声称全文审阅。

[M01]: https://doi.org/10.1002/sim.1438
[M02]: https://onlinelibrary.wiley.com/doi/10.1002/bimj.201000035
[M03]: https://www.bmj.com/content/326/7382/219
