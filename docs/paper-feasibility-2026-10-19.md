# 论文可行性评估与近邻研究

日期：2026-10-19 ｜ 依据：`docs/handoff-2026-10-18.md`、`docs/baseline-precision-2026-10-19.md`、`docs/cluster-purity-signal-2026-10-19.md`、`docs/camera-aware-clustering-2026-10-19.md`、`docs/recall-truth-2026-10-19.md`、`docs/neighbor-implementation-survey-2026-10-19.md`
状态：评估报告（非论文初稿）。文中标注【实测】【文档】【推断】【待验证】，数字均可在仓库复跑。

> 术语约定：本文的「近邻研究」指**相邻领域与最近邻工作的现状调研（related work / SOTA survey）**，不是 k-NN 算法本身；若指后者，本项目已实测 1-NN 多原型与簇大小门并否证，见 §2.3。

---

## 0. 结论（先看这一页）

**1. 能成文，但不能按「新方法刷 SOTA」写。**
本项目的方法组件（机位感知聚类、图库拒识、a/b 一致性）在文献里都有前身；真正稀缺的是**在真实部署条件下把「误差从哪里来」量化清楚了**，并且拿到了一个反直觉、有机制、有代价的结论。这属于**实证/系统论文（empirical field study）**，不是方法论文。

**2. 最强的一句话主张（可辩护）：**
> 在由自动轨迹构建图库、冻结编码器、无标定的跨机位身份判定中，**该被拒识的对象是图库簇，而不是 query**；评分层的六种归一化/聚合改造全部无效，因为被打分的簇本身混了人。一个簇级统计量（簇内最大成对余弦距离 `d_max`）即可识别不纯簇，用作否决后标注精度 0.643→0.806、错标 50→21 条，代价是覆盖 0.664→0.528。

这句话同时具备：反直觉（评分类方法无效）、有机制（average-linkage 的链式串联）、有代价（覆盖换精度）、可复现（profile 一键跑）。这是论文的骨架。

**3. 证据强度只够 Q2/Q3 应用刊或会议；冲 Q1（EAAI 级）必须补三件事：**
- **第二场/第二天数据复验**（当前只有 1012 一天，`veto_dmax=0.30` 在同一批 39 簇上选出）；
- **公开数据集上的受控实验**（证明 `d_max` 不是本场景特例，Market-1501/MSMT17 冻结 OSNet 即可）；
- **第二标注人 + Cohen's kappa**（当前 39 簇为单人目检，审稿人必问）。

**4. 推荐投稿顺序：**
EAAI（补齐后）→ CVIU / IVC → MVA / JVCIR → Sensors / IEEE Access（保底，快）→ 中国图象图形学报（国内路线，最快）。

**5. 不建议投：** TPAMI、TIFS、TCSVT、Pattern Recognition。方法新颖度与实验规模都不够，投出去是浪费时间。

**6. 论文成熟度评分（主观判断，非承诺）：**

| 维度 | 分数 | 说明 |
|---|---|---|
| 问题真实性 | 5/5 | 真实办公楼、4 机位、非重叠无标定、有人脸不可用等硬约束 |
| 结论反直觉性 | 4/5 | 「评分层六连否 + 簇纯度才是瓶颈」有传播力 |
| 方法新颖度 | 2/5 | 机位感知聚类 = CALR/O2CAP 的工程实例；`d_max` = 经典簇紧致度统计 |
| 实验规模 | 2/5 | 4 机位、1 天、579 库轨迹、229 query，单场单日 |
| 评测严谨性 | 4/5 | 三指标同出一次 θ 扫描；39 簇全普查而非抽样；留一交叉验证 |
| 可复现性 | 4/5 | profile 驱动、fp32 逐位一致；但数据不可公开 |
| **综合** | **3.2/5** | 现在可投 Q2/Q3；补三项后可达 Q1 应用刊的门槛 |

---

## 1. 项目成果盘点（论文可用的硬资产）

### 1.1 场景与数据

| 项 | 值 | 备注 |
|---|---|---|
| 场景 | 办公楼一层，4 个机位：305 大场景 / 318 办公区 / 320 茶水间 / 323 玻璃走廊 | 视野**不重叠**、**无标定**、无深度 |
| 采集 | 2026-10-12 单日；上午 4 段各 60 min 建库，下午 4 段各 15 min 评估 | TRAIN/TEST 严格时间隔离，防泄漏 |
| 分辨率 | 305 为 2560×1440@25fps，323 为 1920×1080@25fps | 其余两路同量级 |
| 使用量 | 约 5 机位小时（4×60 + 4×15 min）；原始可用约 21 段 09:00–17:00 | 规模是「现场研究」而非「数据集」 |
| 图库 | 579 条自动轨迹 → 100 个簇（veto 后）/ 111 个（veto 前） | 轨迹来自 YOLO+ByteTrack，不是人工片段 |
| 查询 | 229 条可评估 test 轨迹（检测到 295 条） | 逐条含 a/b 两片 |
| 共现对 | 918 对（14 对误合并）/ 805 对（6 对，改前） | 同帧共现 = 铁定不同人，天然 cannot-link |

**这一点在论文里要强调**：绝大多数 ReID 数据集给的是**人工裁剪的干净图库**；本研究的图库是**自动检测+跟踪产生的、含重复框/碎轨/头框的脏图库**。这正是工业部署的真实输入，也是「图库纯度」问题成立的前提。

### 1.2 系统管线（全部可复跑）

```
YOLO26s-seg（实例分割，stride 5/10）
  → ByteTrack（原始帧率、低激活阈值）
  → 同帧重复框去重（IoU≥0.5 丢低分框）
  → 每轨迹 8 帧 → OSNet-x1.0（MSMT17，冻结）512 维均值
  → 图库机位感知聚类：机位内 average-linkage 0.15 → 跨机位 complete-linkage 0.25（同机位强制不合并）
  → 簇纯度否决：d_max ≥ 0.30 的簇整簇弃用
  → 每身份 z 校准的簇分数 → 单一全局阈值 θ=1.85 → 接受/拒识
```

工程属性（对应用类期刊是加分项）：**纯 CPU 可跑**（YOLO 采样帧 3.3 f/s、OSNet 112–229 ms/裁剪），无训练、无 GPU、无标定；`configs/selector_deploy.json` 一键复跑；OSNet CPU 重编码与存档**逐位相同**（91/91，maxdiff 0）。

### 1.3 评测协议（这是被低估的方法学资产）

三指标同出**一次 θ 扫描**，不各自调参：

| 指标 | 定义 | 作用 | 判据 |
|---|---|---|---|
| coverage | a、b 两片都被接受的轨迹占比 | 敢不敢下结论 | 越高越好 |
| same | 两片都接受时落到同一簇的比例 | 内部一致性（无标签代理） | **≥0.90** |
| false merge (fm) | 同帧共现 ≥5 帧的轨迹对被并入同簇 | 铁定不同人，硬负例 | **≤2%** |

工作点选择（self-cal）：在 same≥0.90 且可评估共现对 ≥100 的 θ 中取 coverage 最大。**这不是统计保证，是启发式**——论文里必须这么写，并补 risk-coverage 曲线与置信区间。

配套的**全簇人工普查**是更硬的一块：152 条接受只落在 42 个簇上，39 个有定义的簇**全部审完**（不是抽样），并验证「同簇同结论」成立（10 个多成员簇 0 冲突）。这把「精度」从抽样估计变成了普查实数。

### 1.4 主结果（当前生产口径）

| 配置 | K | coverage | same | fm | 精度（排除看不清） | 正确/错标/看不清 |
|---|---|---|---|---|---|---|
| 全局聚类 0.20（改前） | 109 | 0.568 | 0.908 | 0.75% | — | — |
| 机位感知 0.15/0.25 | 111 | **0.664** | 0.908 | 1.53% | **0.643** | 90 / 50 / 8 |
| **机位感知 + veto 0.30（生产）** | **100** | **0.528** | **0.901** | **1.48%** | **0.806** | **87 / 21 / 8** |

> 口径说明：coverage 0.528 = 121/229；已完成人工判定的 116 条中 87 对 / 21 错 / 8 看不清，精度按 87/108 计（排除看不清）；其余 5 条为 a/b 分歧或重复轨迹，未纳入精度。

对外必须按三层写，不能混：
- **标注覆盖率** 152/229 = 0.664（含 15 条重复头框）→ veto 后 121/229 = 0.528；
- **轨迹级正确率** 90/229 = 0.393 → veto 后 87/229 = 0.380；
- **错标率** 50/229 = 0.218 → veto 后 21/229 = 0.092。

分机位精度（veto 前）：305 0.711｜318 0.808｜**323 0.438（占全部错误 36%）**｜320 0.167（仅 6 条）。这是论文里「误差不是均匀分布」的关键证据。

### 1.5 阴性结果矩阵（论文的另一半，价值不低于正结果）

| 方向 | 做法 | 结果 | 结论 |
|---|---|---|---|
| 表征 | 掩膜重编码（人体外置黑） | 逐裁 cosine 0.61–0.79，覆盖大跌 | OSNet 依赖整框构图，域外输入 |
| 表征 | 掩膜做检测门控 | fm 1.96%→1.18%，cov −1.8pp，p=0.41 | 不显著 |
| 评分 | T-norm / AS-norm | 覆盖 +9pp 但新增 33 条仅 42.4% 正确；AS-norm fm 2.69% | 否证 |
| 评分 | 竞争簇校正 / 每机位原型 | 与基线等价或更差 | 否证 |
| 评分 | 1-NN 多原型 / 簇大小门 | 新增正确率 44–57%，低于基线 0.643 | 否证 |
| 门控 | 轨迹质量门（n≥8 & h≥64 & 分数） | 精度 17–43%，**质量与正确性负相关** | 否证 |
| 约束 | cannot-link 事后弃判 | 同 θ 下 −1.8pp 覆盖换 fm 0.75%→0.26% | 不扩前沿 |
| 检测 | 纯检测模型替代分割 | 端到端 0.387 vs 0.568 | 否证 |
| 检测 | 混用两种检测器共享图库 | cov 0.594 < 0.664，θ 被迫抬高 | 不可混库 |
| 人脸 | 人脸作为判据 | 可用脸仅 6.4%，323 为 0%；AUC 0.880 < body 0.981 | 只能做「看不清」标签 |

**统一的机理解释**（这是论文的理论支点）：用全库轨迹级 top-1 余弦区分「判对/判错」的 **AUC 仅 0.571**；判对中位 0.624、判错中位 0.506，错误分两类——图库没这个人（低分，可挡）与**图库有极像的别人**（高分，挡不住）。所以任何对 body 余弦做单调变换的评分层方法，都不可能突破。

### 1.6 一条必须写进论文的「指标反噬」

coverage 的分母是「检测到的轨迹」，不是「真实在场的人」。
- 换召回更高的检测器（323 远场 l-seg 召回 100% vs s-seg 25%），覆盖率反而可能下降；
- 头框轨迹（21/295）得分最高、却被接受进别人身体的簇，**指标上看是「成功」**；
- 因此论文必须同时报告「按帧点数」的召回与「按簇普查」的精度，否则指标会自我欺骗。

这一条对审稿人很有说服力：它解释了为什么工业界的 ReID 报告经常「指标很好、现场不可用」。

---

## 2. 近邻研究（Related Work）

### 2.1 技术谱系：六条线

**线 1：单机位检测与跟踪（商品化组件）**
ByteTrack（ECCV 2022）、BoT-SORT、OC-SORT、StrongSORT（AFLink/GSI，arXiv 2202.13514）。这些是本项目的前端，不构成贡献；但**跟踪质量决定图库质量**这一点，文献里讨论不足（TMC/SSR-C 是例外，见线 3）。

**线 2：行人重识别表征**
OSNet（ICCV 2019）、TransReID（ICCV 2021）、CLIP-ReID（AAAI 2023）、DINOv2/v3（2023/2025）。趋势是**冻结基础模型特征 + 轻量适配**。本项目用冻结 OSNet，是合理的部署选择，但也是弱点：**OSNet 只看衣着**，同一着装的人在图库里不可分。这解释了「极像的别人」这一类错误。

**线 3：无监督 ReID 与聚类伪标签**
SpCL（NeurIPS 2020）、ClusterContrast（ACCV 2022）、ICE（ICCV 2021）、PPLR（CVPR 2023）、UTAL（TPAMI 2020）、TMC（ICCV 2021）、SSR-C（arXiv 2406.14261）。
这一线已经明确：**跟踪噪声（ID switch / 碎片）是伪标签质量的主要污染源**，对策是轨迹内分裂、子轨迹聚类、渐进合并。

**线 4：camera bias / IDs Merge（与本项目最直接）**
- CALR：*Camera-aware Label Refinement for Unsupervised Person Re-ID*，arXiv 2403.16450（2024）
- O2CAP：*Offline-Online Associated Camera-Aware Proxies*，arXiv 2201.05820（2022）
- CMLR：*Unsupervised person re-ID via camera-aware multi-level label refinement*，Neural Networks 195 (2025) 108292
- CARR：*Camera-aware reciprocal refinement for unsupervised visible-infrared ReID*，Applied Soft Computing 188 (2026) 114445
- *Cross-camera reliability modeling with soft contrastive learning*，Knowledge-Based Systems（2026）

核心观察（文献原话大意）：**同一机位的特征天然更近，全局聚类会把同机位的不同人并到一起（IDs Merge）**；对策是「先机位内聚类，再跨机位合并」+ camera-aware proxy。
**本项目的机位感知聚类就是这条思路的 training-free 工程实例**——所以它不能作为论文的主要新颖点，只能作为「已落地的前置条件」。

**线 5：跨机位/多目标跟踪（MTMC）**
- AI City Challenge 2025/2026 Track 1：3D HOTA，依赖标定/深度；2025 榜首 HOTA 69.91（ZV），在线 RGB-only 约 45.22（Sparse4D）
- GMT：*Effective Global Framework for Multi-Camera Multi-Target Tracking*，CVPR 2026（全局 tracklet 关联，DETR 式）
- Glance-MCMT（2025）：glance 初始化 + 渐进关联，验证集 HOTA 51.34
- Self-MVA（CVPR 2025）：自监督、**无标定**多视角行人关联
- Calibration-free 3D multi-camera tracking（AI City 2024，HOTA 53.13）

这一线的共同前提是：**有重叠视野/可标定/可用 3D 或深度**，或者可以训练。本项目的设定（非重叠、无标定、无深度、冻结编码器、纯外观）在这一线里是**边缘设定**，反而留下了空间。

**线 6：开集识别、拒识与风险控制**
- Open-world ReID 的 DIR/FAR 协议（Liao et al.）
- *Gallery-Distance Profiles for Open-Set Person Re-ID with Frozen Encoders in Camera Networks*，**Sensors 2026, 26(17), 5479** —— 与本项目最接近的一篇：冻结编码器、gallery-relative、开集拒识、camera network。差异：它是**query 级**风险打分（9 统计量），我们是**簇级**否决；两者互补。
- Selective classification / risk-coverage（Geifman & El-Yaniv 2017）
- Learn then Test（Angelopoulos et al., Annals of Applied Statistics 19(2), 2025）——可把「fm≤2%」从点估计升级为有限样本置信保证。

**线 7（应用）：多机位人数统计/疏散**
- *Smart building evacuation by tracking multi-camera network and explainable re-identification model*，**Engineering Applications of Artificial Intelligence 148 (2025) 110394** —— 4 机位级系统论文，报告 ReID 跟踪精度 >75%、人数统计接近 100%，自建数据集。
- 建筑人数统计（EC³ 2025）：非重叠机位 + ReID + 分区计数。

这两篇证明：**「多机位 + ReID + 人数」在应用类 Q1 刊有稳定发表通道**，且不要求刷 SOTA；要求的是真实部署、可核实的工程指标、以及对失败的诚实讨论。

### 2.2 最近邻工作逐条对照

| 工作 | 设定 | 方法 | 报告结果 | 与本项目的差异 |
|---|---|---|---|---|
| CALR (2024) | 无监督 ReID，有训练 | 机位感知伪标签精修 | Market/MSMT mAP 提升 | 需要训练；改表征；我们 training-free，改图库 |
| O2CAP (2022) | 无监督 ReID，有训练 | camera-aware proxy + 在线/离线聚类 | SOTA 级 | 同上；且我们只借结构不借训练 |
| CMLR (2025, NN) | 无监督 ReID | 簇级 + 实例级标签精修 | 优于 SOTA | 仍是训练框架；无「簇否决」概念 |
| Gallery-Distance Profiles (2026, Sensors) | 冻结编码器、开集、camera network | 9 统计量 profile + 风险门 | 开集指标提升 | **query 级 vs 簇级**；我们用 1 个统计量做簇否决 |
| AI City 2025/2026 | 有标定/深度/3D | 3D 检测 + 全局关联 | HOTA 45–70 | 我们无标定、无深度、纯 2D 外观 |
| GMT (CVPR 2026) | 多机位，有重叠 | DETR 式全局 tracklet 关联 | 超 MOTRv2 +4.8% HOTA | 需训练、需多视角同步；我们离线 gallery 匹配 |
| Self-MVA (CVPR 2025) | 无标定多视角 | 自监督同步信号 | 关联提升 | 需训练；我们零训练 |
| EAAI 2025 疏散 | 4 机位、自建数据 | YOLO + 跟踪 + 注意力 ReID + 匹配 | ReID >75%、计数≈100% | **系统相似、诊断缺失**：未分析图库纯度与错标 |
| 本项目 | 4 机位、非重叠、无标定、冻结 OSNet、自动图库 | 机位感知聚类 + 簇纯度否决 | cov 0.528 / same 0.901 / fm 1.48% / 精度 0.806 | 贡献在**误差归因**与**簇级拒识**，不在表征 |

### 2.3 如果「近邻」指 k-NN 方法：已有实测结论

- 全库 top-1 余弦：判对 p50 0.624、判错 p50 0.506，**AUC 0.571** → 不足以做拒识；
- 1-NN 多原型取 max：cov 0.7205，人工目检新增 18 条 = 8 对 / 9 错 / 1 看不清（44.4%），未达 50% 线；
- 簇大小门：size≥3 → fm 2.89%、size≥4 → 3.22%（破预算）；size≥6 → 新增正确率 57%，仍低于基线 0.643。
结论：**k-NN 类方法在本场景已否证**，论文里应作为「我们试过并排除」的一部分，而不是遗漏。

### 2.4 空白点（Gap）：五条，按可辩护性排序

1. **「图库簇纯度」没有被当作一等问题**。无监督 ReID 关心伪标签质量，但评价仍在 mAP/Rank-1；部署侧关心的是「已接受的标签里有多少是错的」。两者之间缺一座桥。本项目用 39 簇全普查把这座桥搭起来了。
2. **拒识的对象错了**。开集 ReID 拒识 query；但在自动图库场景，**该拒识的是簇**。这个视角在文献里没有明确提法（至少我们未检索到）。
3. **评分层方法的失效边界没有被量化**。文献报告 T-norm/AS-norm 在说话人识别有效；本项目给出「body 余弦在 0.6–0.75 段无区分力（AUC 0.571）」的边界，解释了为什么无效。
4. **无标定、非重叠、冻结编码器的四机位现场研究稀缺**。AI City 有标定，EAAI 疏散论文有部署但无误差归因。
5. **指标反噬（覆盖率无法奖励召回、头框被算成功）在文献里缺少系统讨论**。这是一个方法学贡献点。

---

## 3. 论文定位与三个候选骨架

### 3.1 三种写法

| | P1 实证/系统论文（推荐） | P2 方法论文 | P3 评测协议短文 |
|---|---|---|---|
| 主张 | 图库纯度是瓶颈；应拒识簇而非 query | 机位感知聚类 + 纯度否决构成新框架 | 无标签条件下评估开集跨机位关联的协议 |
| 新颖度 | 中（归因 + 视角） | 低-中（组件多为已有） | 中（协议 + 风险控制） |
| 需要补的实验 | 第二场、公开数据、双标注 | 公开数据 SOTA 对比、消融、第二场 | 形式化 + 与 DIR/FAR 对齐 |
| 适合期刊 | EAAI / CVIU / MVA / JVCIR / IEEE Access / Sensors | Neurocomputing / CVIU / IVC / MTAP | AVSS / ICDSC / ICIP / Journal of Imaging |
| 风险 | 低（诚实写即可） | 高（审稿人会问「比 CALR 好在哪」） | 中（贡献偏薄，需强写作） |
| 建议 | **主投** | 作为 P1 的 §5 方法节，不单独成文 | 可作会议短文/ workshop |

### 3.2 推荐 P1：**Reject the gallery, not the query**

**标题候选**
- *Reject the Gallery, Not the Query: Cluster-Level Purity Veto for Training-Free Cross-Camera Identity Association*
- *The Gallery Is the Bottleneck: A Field Study of Training-Free Cross-Camera Re-Identification in a Four-Camera Office*
- *When Score Normalization Cannot Help: Diagnosing Gallery Impurity in Cross-Camera Person Association*

**四段式贡献（每条都要有数字）**
1. **一个无标签评测协议 + 全簇普查**：coverage / same / fm 同出一次 θ 扫描；39 个有定义的簇全部人工审完，得到「标注覆盖率 0.664、轨迹级正确率 0.393、错标率 0.218」的完整分解；验证同簇同结论（10/10 多成员簇 0 冲突）。
2. **一张阴性结果矩阵**：六种评分层改造（T-norm、AS-norm、竞争簇校正、每机位原型、1-NN 多原型、簇大小门）全部低于基线正确数 90；并给出机制解释（top-1 余弦 AUC 0.571）。
3. **一个簇级诊断与干预**：13/39 簇不纯，承载 50 条错标；`d_max` AUC 0.769；veto 0.30 使精度 0.643→0.806、错标 −58%，代价覆盖 −13.6pp；留一交叉验证切点稳定在 0.29–0.30。
4. **一组现场失效模式**：头框轨迹 21/295（15/15 落进别人身体的簇）、323 远场召回 25%（l-seg 100%）、人脸可用率 6.4%（323 为 0%）、混检测器共享图库导致 θ 抬高。这些是「指标好看但现场不可用」的具体成因。

**结构（8 节，约 10–12 页）**
1. Introduction：工业痛点（多路计数相加偏高）→ 反直觉发现 → 贡献
2. Related Work：线 3/4/5/6，收敛到 gap
3. Deployment & Protocol：场景、管线、三指标、全簇普查、时间隔离
4. Where the Errors Come From：39 簇普查、分机位、坏簇画像（size / 跨机位 / d_max）
5. Why Score Normalization Fails：六种变体 + 余弦 AUC 0.571 + 两类错误
6. Gallery Purity Veto：d_max、机制、留一、真实流水线代价-收益、误伤分析
7. Field Failure Modes：头框、远场召回、人脸、混库、指标反噬
8. Discussion & Limitations：可迁移性、隐私、单场单日、单标注人

**图表清单**
- Fig.1 管线与三个判定点（a/b 切分、簇构建、veto）
- Fig.2 229 条 query 的 100% 去向分解（堆叠条）
- Fig.3 精度-覆盖前沿（veto 前后 + 六种评分变体散点）
- Fig.4 `d_max` 好坏簇分布 + 留一选点
- Fig.5 分机位精度与错误占比（323 突出）
- Table 1 场景与数据统计
- Table 2 生产配置与三指标
- Table 3 六种评分层变体（阴性结果）
- Table 4 veto 消融（阈值、代价、误伤）
- Table 5 与最近邻工作的设定对比

### 3.3 不能主张的七条（写进去会被拒）

1. 不能说 0.664 是准确率——它是标注覆盖率；
2. 不能说「达到 SOTA」——没有公开基准对比；
3. 不能说「可推广到任意场景」——单场单日；
4. 不能说 `d_max` 是新指标——它是簇紧致度/最长边一类经典统计；
5. 不能说人脸融合有效——可用率 6.4%，AUC 低于 body；
6. 不能说实时——CPU 采样 3.3 f/s，是离线/准离线；
7. 不能说「人数统计准确率」——当前没有人级 MAE，只有身份级覆盖。

### 3.4 英文摘要草稿（可直接改写）

> Deploying cross-camera person re-identification (ReID) in real buildings rarely starts from a clean, manually curated gallery. It starts from tracklets produced by a detector and a tracker, which contain duplicate boxes, fragmented tracks, and body parts misdetected as persons. We study a four-camera, non-overlapping, uncalibrated office deployment with a frozen OSNet encoder and a gallery built entirely from automatic tracklets (579 gallery / 229 query tracklets). Using a label-free protocol in which coverage, split-half consistency, and co-occurrence-based false merges are produced by a single threshold sweep, followed by a full census of all 39 defined gallery clusters, we find that end-to-end precision is limited by gallery-cluster purity rather than by score calibration. Thirteen of 39 clusters (33%) are impure and carry all 50 wrong labels; six score-normalization and aggregation variants, including T-norm, AS-norm, per-camera prototypes, and 1-NN multi-prototype matching, fail to improve the number of correct labels, consistent with a top-1 cosine AUC of only 0.571 between correct and incorrect matches. We then show that a single cluster-level statistic—the maximum pairwise cosine distance within a cluster—separates impure from pure clusters (AUC 0.769), and that vetoing clusters above 0.30 raises label precision from 0.643 to 0.806 and reduces wrong labels by 58%, at the cost of 13.6 points of coverage. We argue that, when the gallery is built automatically, the right object to reject is the cluster, not the query.

---

## 4. 审稿人必问的九个问题

| # | 问题 | 现有证据 | 需要补什么 |
|---|---|---|---|
| 1 | 单场单日，如何 generalization？ | 无 | **第二场/第二天**复验 veto 0.30（最高优先） |
| 2 | 阈值 0.30 是否在同一批数据上选的？ | 留一交叉验证每折 0.29–0.30；留出精度 0.796 | 预注册 + 第二场；嵌套 CV |
| 3 | 人工标注只有一个人？ | 39 簇全审、同簇同结论 | **第二标注人 + Cohen's kappa**（≥0.8） |
| 4 | 为什么不用 IDF1/HOTA？ | 无 ID 真值 | 补 DIR/FAR 映射 + risk-coverage 曲线；说明为何不可得 |
| 5 | 与 SOTA 比呢？ | 六种评分变体 + k-NN 已比 | 补 CALR/O2CAP 风格聚类基线、DBSCAN/HDBSCAN |
| 6 | 为什么用 OSNet 而不是 CLIP/DINOv3？ | 318×320 上 DINOv3-B AUC 0.983 vs DINOv2-S 0.869（30 对） | 同协议下换 DINOv3-B 重跑全链路（1–2 天） |
| 7 | `d_max` 不就是 Dunn index？ | 是同类统计 | 明确承认；novelty 放在「机制 + 簇级否决 + 现场代价」 |
| 8 | 覆盖掉 13.6pp，业务能接受吗？ | 错标 50→21；错标比未标更有害 | **补人级计数 MAE**：veto 前后每 15 min 唯一人数误差 |
| 9 | 数据能公开吗？ | 不能（人脸/隐私） | 提供协议、代码、参数；说明伦理审批与脱敏 |

---

## 5. 期刊与会议建议

> 分区与 IF 每年变动，下表为 2026-10 检索到的公开信息与经验判断，**投稿前以当年 JCR/中科院分区表为准**。

### 5.1 分级表

| 层级 | 期刊/会议 | 出版方 | 分区/IF（约） | 匹配点 | 门槛/风险 | 建议 |
|---|---|---|---|---|---|---|
| A（补齐后冲） | **Engineering Applications of Artificial Intelligence** | Elsevier | JCR Q1，IF 约 8–9，中科院 1 区（工程综合）/TOP | 真实部署 + 自建数据 + 工程指标；2025 年有 4 机位疏散 ReID 先例 | 有公开数据集验证会显著提高送审概率（据投稿经验，非官方规则）；审稿 3–9 个月 | **首选**（补三项后） |
| A- | **Computer Vision and Image Understanding** | Elsevier | Q2，IF 约 4–5 | 欢迎分析型/实证型论文 | 要求可推广的洞见，不接受纯工程报告 | 次选 |
| A- | **Image and Vision Computing** | Elsevier | Q2，IF 约 4 | 系统 + 分析 | 同上，篇幅较短 | 次选 |
| B+ | **Neurocomputing** | Elsevier | Q1/Q2，IF 约 6 | 方法向 | 需要更强的方法新颖度与消融 | 若把 P2 写厚可投 |
| B+ | **Machine Vision and Applications** | Springer | Q2/Q3，IF 约 2.5 | 应用系统论文 | 对「真实部署 + 失败分析」友好 | **性价比高** |
| B | **Journal of Visual Communication and Image Representation** | Elsevier | Q2/Q3，IF 约 3 | 视频监控/关联 | 创新要求中等 | 可投 |
| B | **Multimedia Tools and Applications** | Springer | Q2/Q3，IF 约 3 | 应用 | 审稿快、门槛中低 | 保底 |
| B | **IEEE Access** | IEEE | Q2，IF 约 3.4 | 跨学科工程 | 快（1–3 月），APC 约 2,000 USD | 保底/赶时间 |
| B | **Sensors**（Sensing and Imaging） | MDPI | Q2，IF 约 3.5 | 相机网络 + 开集 ReID；2026 有直接先例 | 快（1–2 月），APC 约 2,600 CHF | 保底/快速见刊 |
| C+ | **Journal of Imaging** | MDPI | Q3，IF 约 2.7 | 成像系统 | 门槛低 | 仅保底 |
| C+ | **IET Computer Vision** | Wiley | Q3 | 视觉应用 | 门槛中低 | 备选 |
| 国内 | **中国图象图形学报** | 中科院空天院 | EI/CSCD/中文核心 | 监控 + ReID + 系统 | 周期 6–12 月，重工程价值 | 国内路线首选 |
| 国内 | 计算机应用 / 计算机工程与应用 | — | 中文核心 | 工程实现 | 快 | 备选 |
| 会议 | **AVSS**（IEEE Advanced Video and Signal-based Surveillance） | IEEE | 约 50% | 监控视频分析，主题最贴 | 8 页，需强演示 | 可作短文首发 |
| 会议 | **ICDSC**（Distributed Smart Cameras） | ACM/IEEE | 较小众 | 相机网络 | 6 页 | 很贴题 |
| 会议 | ICIP / ICME | IEEE | 约 50% | 通用视觉 | 4–6 页，创新要求中等 | 可投 |
| 不推荐 | TPAMI / TIFS / TCSVT / Pattern Recognition | — | Q1 顶刊 | — | 需要方法 SOTA + 大规模公开基准 | **不建议** |

### 5.2 三条投稿路线（按目标选）

**路线 1：学术优先（6–10 周补实验）**
EAAI 主投 → 被拒转 CVIU → 再转 MVA。
适合：需要 Q1/TOP 成果认定、有第二场数据、能安排第二标注人。

**路线 2：时间优先（2–3 周成稿）**
Sensors 或 IEEE Access 主投（用现有数据，充分披露单场单日局限）→ 国内期刊并行。
适合：项目结题/验收/职称，先拿到发表记录。

**路线 3：国内优先（4–6 周）**
中国图象图形学报 主投（中文写作快、对工程系统友好）→ 同时投 AVSS 短文。
适合：不希望承担 APC、审稿周期可接受。

### 5.3 周期与成本（经验值，非承诺）

| 路线 | 投稿到一审 | 接受概率（主观） | APC |
|---|---|---|---|
| EAAI（补齐后） | 3–6 月 | 40–55% | 订阅制免费（OA 可选） |
| CVIU / IVC | 2–4 月 | 45–60% | 订阅制免费 |
| MVA | 2–4 月 | 50–65% | 订阅制免费 |
| Sensors | 4–8 周 | 60–75% | 约 2,600 CHF |
| IEEE Access | 4–8 周 | 55–70% | 约 2,000 USD |
| 中国图象图形学报 | 2–4 月 | 50–65% | 免费/少量版面费 |

**当前证据原样投 vs 补齐后投**：原样投 EAAI 约 15–25%，补齐后 40–55%；原样投 Sensors 约 55–65%。这是「补三项」值不值得的直接答案。

---

## 6. 补实验计划（8 周，按性价比排序）

| 优先级 | 实验 | 目的 | 成本 | 通过判据 |
|---|---|---|---|---|
| P0 | **第二场/第二天数据复验** | 证明 veto 0.30 不是单场拟合 | 数据采集 1 天 + 计算 1 天 | 精度仍 ≥0.75，覆盖下降 ≤15pp |
| P0 | **第二标注人对 39 簇复核** | 排除单人主观偏差 | 1–2 天 | Cohen's kappa ≥0.80 |
| P0 | **公开数据集受控实验**（Market-1501/MSMT17，冻结 OSNet，按机位切伪轨迹构库） | 证明 `d_max` 信号可迁移 | 2–4 天（CPU） | `d_max` 对不纯簇 AUC ≥0.70；veto 提升精度 |
| P1 | **人级计数 MAE**（每 15 min 唯一人数 vs 人工点数） | 回答「覆盖掉了，业务损失多少」 | 2–3 天标注 | veto 后 MAE 不劣于 veto 前 |
| P1 | **聚类基线对比**（全局 average、DBSCAN、HDBSCAN、CALR 风格） | 回应「与 SOTA 比」 | 3–5 天 | 机位感知 + veto 的组合最优或持平 |
| P1 | **风险控制**（bootstrap CI + LTT 认证 fm≤2%） | 把点估计升级为置信保证 | 2 天（纯分析） | 给出 same 的 95% 下界与 fm 的上界 |
| P2 | **DINOv3-B 换编码器全链路** | 回应「为什么不用基础模型」 | 1–2 天 | 精度/覆盖不劣于 OSNet，或给出成本-收益 |
| P2 | **l-seg 323 结果并入**（预计 21:50 出） | 323 精度 0.438 的补丁 | 已在进行 | 323 精度显著 >0.438 |

**最小可投组合**：P0 三项 + P1 的计数 MAE。约 4–6 周。

---

## 7. 风险与止损

| 风险 | 触发条件 | 应对 |
|---|---|---|
| 拿不到第二场数据 | 用户/现场不配合 | 降级投 Sensors/IEEE Access/国内刊；论文标题改为「a field study」并明确单场局限 |
| 公开数据实验不成立 | `d_max` 在 Market/MSMT 上 AUC <0.65 | 不写「可迁移」，只写「本场景机制」；改投应用刊 |
| 双标注一致性低 | kappa <0.6 | 重新定义判定标准（对/错/看不清），必要时只报「可判簇」子集 |
| 323 l-seg 失败 | 覆盖不升或 fm 破线 | 论文保留 323 作为「最差机位」案例，不影响主线 |
| 隐私合规 | 数据不可脱敏 | 不发布数据；提供协议与代码；说明伦理审批 |
| 审稿人认定「工程报告」 | 缺少可推广洞见 | 强化 §5 机制（AUC 0.571）与 §6 通用性讨论；把「拒识簇而非 query」上升为设计原则 |

---

## 附录 A：与「近邻研究」相关的可核对文献

| # | 文献 | 类型 | 链接/标识 |
|---|---|---|---|
| 1 | ByteTrack: Multi-Object Tracking by Associating Every Detection Box | 会议（ECCV 2022） | arXiv:2110.06864 |
| 2 | StrongSORT: Make DeepSORT Great Again（AFLink/GSI） | 期刊/预印本 | arXiv:2202.13514 |
| 3 | OSNet: Omni-Scale Feature Learning for Person Re-ID | 会议（ICCV 2019） | arXiv:1905.00953 |
| 4 | TransReID: Transformer-based Object Re-ID | 会议（ICCV 2021） | arXiv:2102.04378 |
| 5 | CLIP-ReID: Exploiting Vision-Language Model for Re-ID | 会议（AAAI 2023） | — |
| 6 | SpCL: Self-paced Contrastive Learning | 会议（NeurIPS 2020） | arXiv:2006.10637 |
| 7 | ICE: Inter-instance Contrastive Encoding | 会议（ICCV 2021） | — |
| 8 | Cluster Contrast for Unsupervised Person Re-ID | 会议（ACCV 2022） | — |
| 9 | PPLR: Part-based Pseudo Label Refinement | 会议（CVPR 2022/2023） | — |
| 10 | UTAL: Unsupervised Tracklet Person Re-ID | 期刊（TPAMI 2020） | — |
| 11 | TMC: Learning from Automatically Generated Noisy Tracklets | 会议（ICCV 2021） | — |
| 12 | SSR-C: Unleashing the Potential of Tracklets | 预印本（2024） | arXiv:2406.14261 |
| 13 | CALR: Camera-aware Label Refinement | 预印本（2024） | arXiv:2403.16450 |
| 14 | O2CAP: Offline-Online Associated Camera-Aware Proxies | 预印本（2022） | arXiv:2201.05820 |
| 15 | CMLR: Camera-aware Multi-level Label Refinement | 期刊（Neural Networks 2025） | doi:10.1016/j.neunet.2025.108292 |
| 16 | CARR: Camera-aware Reciprocal Refinement | 期刊（Applied Soft Computing 2026） | doi:10.1016/j.asoc.2025.114445 |
| 17 | Cross-camera Reliability Modeling with Soft Contrastive Learning | 期刊（Knowledge-Based Systems 2026） | S0950705126011299 |
| 18 | Gallery-Distance Profiles for Open-Set Person Re-ID with Frozen Encoders | 期刊（Sensors 2026, 26(17):5479） | mdpi.com/1424-8220/26/17/5479 |
| 19 | GMT: Effective Global Framework for MCMT | 会议（CVPR 2026） | — |
| 20 | Glance-MCMT: Glance Initialization + Progressive Association | 预印本（2025） | — |
| 21 | Self-MVA: Self-Supervised Uncalibrated Multi-View Person Association | 会议（CVPR 2025） | github.com/CAMMA-public/Self-MVA |
| 22 | AI City Challenge 2025/2026 Track 1（3D HOTA） | 赛事 | aicitychallenge.org |
| 23 | Smart Building Evacuation by Tracking Multi-Camera Network | 期刊（EAAI 2025, 148:110394） | doi:10.1016/j.engappai.2025.110394 |
| 24 | Selective Classification（risk-coverage） | 会议/期刊（2010/2017） | Geifman & El-Yaniv |
| 25 | Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control | 期刊（AoAS 2025） | arXiv:2110.01052 |

## 附录 B：本项目可复跑入口

```bash
cd /d/human-count
export PYTHONPATH=D:/human-count/pylibs TMPDIR=D:/human-count/tmp TEMP=D:/human-count/tmp
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"

# 生产口径（含 veto 0.30）
"$PY" tools/eval_det_switch.py --profile configs/selector_deploy.json

# 回滚口径（删 veto）
# 将 configs/selector_deploy.json 中 cluster.veto_dmax 删除后重跑
```

> 注：`PY` 路径请以本机实际环境为准（仓库内文档用的是 `C:/Users/muyizWindows/...`）。

---

## 一句话总结

**这是一篇「把工业部署里最贵的那类错误讲清楚」的论文：不是我们提出了更强的人体特征，而是我们证明了在自动图库场景里，最该被拒识的是簇而不是 query；证据是 39 簇全普查 + 六种评分层方法的系统性否证 + 一个可解释的簇级否决信号。它够得着 Q2/Q3，补齐第二场数据与公开数据验证后够得着 EAAI 级 Q1。**
