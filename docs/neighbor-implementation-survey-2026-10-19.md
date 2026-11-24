# 近邻实现调研：重复框、人脸/人体并行、跨机位身份匹配

日期：2026-10-19 ｜ 目的：为我们当前的三个卡点找**可参考的具体实现**（不是综述式罗列）
范围：公开论文 + 开源实现 + 专利/厂商文档，全部给出可核对链接
状态：调研结论；每条都标注「文献做法 / 我们的差异 / 能否直接借」

---

## 0. 结论摘要（先看这五条）

1. **「一个人两个框（头肩 + 全身）」在拥挤行人检测里是成熟问题，标准解法是「可见框 / 部分框」做 NMS**，
   不是降低全身框的 IoU 阈值。代表工作：VFG（visible + full 双框，NMS 只跑在可见框上）、vg-NMS（像素级可见性）、
   R2NMS（Full-Full IoU + Visible-to-Full IoU 双判据）。**我们的掩膜包含判据 = vg-NMS 的实例掩膜版**，方向被文献支持。
2. **头/身匹配绝对不能用 IoU**：人头面积通常不到人体 1/8，亲生的头-身 IoU 只有约 0.1。工程界的做法是 **IoA
   （交集 ÷ 头面积）**，头有 80% 面积落在身体框内即算匹配。**这与我们实测的小框/大框框 IoU 0.14–0.22 完全吻合**，
   也解释了为什么 iou=0.5 的 NMS 对这类重复框完全无效。
3. **人脸/人体并行有工业级参考架构**：专利 US20260045091A1 把 face / head / body 当三条独立模态，
   定义「背对时用头+身、身体被挡时用头做锚点、脸出现后回填关联」的 fallback 与 retro-link 策略；
   CN117542101A 用双分支 YOLO（脸框 + 人头人体框）+ ArcFace/OSNet + 匈牙利匹配（阈值 0.35）。
   **我们的「并行检测 + 几何归属」与它们一致，缺的是 fallback 语义与回填关联**。
4. **图库把不同人并到一起，在无监督 ReID 里有专门名字：camera bias / IDs Merge**——同机位特征天然更近，
   跨机位才该收紧。CALR / O2CAP 的做法是「**先机位内聚类，再跨机位合并**」，并用 camera-aware proxy 表示同一人不同机位的子簇。
   我们 2.19% 的图库共现对被并，正是这个失效模式，**这条可以零成本试**。
5. **轨迹拼接（AFLink，1.7ms/帧）救不了我们的重复轨迹**：它拼接的是**时间不重叠**的碎片；
   我们的头肩/全身重复轨迹是**同时存在**的，时间上重叠，拼接模型会直接判为不可连。→ 必须在检测级解决。

---

## 1. 我们的三个卡点

| 卡点 | 现象（本项目实测） | 文献里的对应问题名 |
|---|---|---|
| A 一个人两个框 | 305 中 13–21% 轨迹是「头肩框」，与全身框掩膜包含率 0.99 | crowded pedestrian detection / duplicate proposals / part-whole association |
| B 人脸与人体并行 | 设计并行，已实现 3×3 网格 SCRFD + 几何归属；但可用脸仅 6.4% | face-body association, multi-modal person tracking |
| C 图库纯度 / 跨机位 | train 共现对 53/2417 = 2.19% 被并进同一簇 | camera bias / IDs Merge（无监督 ReID） |

---

## 2. 卡点 A：一个人两个框，文献怎么做

### 2.1 Visible Feature Guidance（VFG）

- 出处：*Visible Feature Guidance for Crowd Pedestrian Detection*，arXiv:2008.09993（Aibee）
  链接：https://arxiv.org/html/2008.09993v2
- 做法：训练时同时回归 **visible box** 与 **full box**；**推理时只在 visible box 上做 NMS**，再取出与之绑定的 full box。
  理由：两个被遮挡的人 full box 可能 IoU>0.7，但 visible box 几乎不重叠；反过来，同一个人的「头肩框」与「全身框」
  在 visible 意义上高度重叠 → 该被抑制。
- 效果：两阶段与一阶段检测器 mAP / AP50 稳定 +2~3%，在 CrowdHuman / CityPersons / Caltech / KITTI 上验证。
- 对我们的意义：**直接印证「用掩膜重叠做抑制」是对的**——我们的掩膜就是 visible 区域的精确版本。
  差异是他们把 visible 作为网络输出，我们用实例掩膜事后计算，不需要重训。

### 2.2 vg-NMS（Visibility Guided NMS）

- 出处：*Visibility Guided NMS: Efficient Boosting of Amodal Object Detection in Crowded Traffic Scenes*
  链接：https://www.openreview.net/forum?id=EIpMxsmO0n
- 做法：利用像素级可见性（amodal 检测范式）作为抑制依据，替代手调的 IoU 阈值；计算开销很小；在 KITTI / VIPER / Synscapes 上优于当时的 NMS。
- 对我们的意义：文献明确指出「标准 NMS 的阈值是手调超参，在拥挤场景必然失败」，与我们的观察一致：
  iou 0.5 能压掉 22% 的重复帧，但 0.2–0.5 这一档压不掉。

### 2.3 R2NMS：双判据（Full-Full IoU + Visible-to-Full IoU）

- 出处：*NMS by Representative Region: Towards Crowded Pedestrian Detection by Proposal Pairing*（OpenReview）
  链接：https://openreview.net/forum?id=Lbjuc9_Fm2
- 做法：Paired-Box Model 同时预测 full / visible 框；抑制时同时看两个量：
  **只有 Full-Full IoU 超主阈值、且 A 的 visible 框对 B 的 full 框 IoU 也超辅阈值时才抑制**。
  论文给的例子：两行人 full-full IoU 0.63，但 visible-to-full 只有 0.18 → 判为两个人，都保留。
- 对我们的意义：**这正是我们需要的双判据**——我们的「框 IoU 0.2–0.5 + 掩膜包含 ≥0.9」结构相同，
  区别是我们用掩膜而非学习出来的 visible 框。

### 2.4 CrowdHuman Double Anchor / Joint NMS（端到端头-人绑定）

- 出处：CrowdHuman 头部-人体联合检测方案（Double Anchor RPN + Joint NMS），中文工程解读：
  https://bbs.plcjs.com/forum.php?mod=viewthread&tid=550831
- 做法：RPN 阶段用「头框正样本 anchor」同时回归**配对的人框**，得到成对 proposal；后处理用 Joint NMS 成对抑制。
  对比「后绑定」（先跑行人检测再跑人头检测，然后用 IoU 规则配对）——**后绑定正确率低**，因为一个身体框里可能有多个头，
  且人体召回常低于人头召回。
- 对我们的意义：我们是「后绑定」的极端情形（同一个网络出两个框，没有配对信息），所以只能靠**掩膜/几何**补，
  不可能靠「配对头」来解决。这条也说明：**指望在轨迹层把两个框区分开，信息已经丢了**。

### 2.5 工程坑（可直接抄的经验）

中文工程总结（https://m.toutiao.com/article/7629325284130488872）列的三个坑，与我们遇到的一模一样：

| 坑 | 现象 | 解法 |
|---|---|---|
| 用 IoU 配对头与身 | 头面积 < 人体 1/8，IoU 永远约 0.1，配对全失败 | **改用 IoA（交集 ÷ 头面积）**，头 80% 在身内即匹配 |
| 大个子「偷走」小孩的头 | 多个头都 100% 包含在同一身体框里，argmax 随便挑 | 加横向/纵向约束 + 贪心一对一（我们 L2 规则已有） |
| 头身框独立输出 | 后处理阶段两批张量互不相识 | 端到端配对（Double Anchor）或按 IoA 归属 |

我们实测：小框/大框框 IoU 中位 0.22、高占比 0.41–0.47 → **完全落在「必须用 IoA/包含率」的场景**。

### 2.6 与我们的差异 / 能不能借

| 文献做法 | 我们能否直接借 | 说明 |
|---|---|---|
| visible/full 双框头 | ❌ 不能 | 需要重训检测器 + 可见框标注；我们已有实例掩膜，等价信息免费 |
| 掩膜/可见性做 NMS | ✅ 已实现 | `--dedup-mask 0.9` 就是它的实例掩膜版；正在 305 一对段做验证 |
| 双判据（框 IoU + 掩膜包含） | ✅ 建议 | 我们目前只有「掩膜包含」，可加「框 IoU ≥0.15」做保守门，降低误删 |
| IoA 而非 IoU 做头身归属 | ✅ 可借 | 若以后接入人头检测（SCRFD 只出脸，不出头），归属规则要写 IoA |

---

## 3. 卡点 B：人脸与人体并行后怎么融合

### 3.1 专利 US20260045091A1（Human Subject Tracking in Secure Environment）

链接：https://patents.google.com/patent/US20260045091A1/en

它把 **face / head / body** 当三条独立检测模态，明确写了四种运行状态（这段几乎就是我们的设计说明书）：

| 场景 | 行为 |
|---|---|
| 背对镜头 | 只用 head + body 建轨迹 |
| 身体被遮挡 | **继续用头作为独立锚点**跟踪 |
| 脸重新可见 | 把新检出的脸**回填关联**到先前的头/身轨迹（retroactive linking） |
| 三者同时可见 | 三模态联合关联，稳定身份 |

另外它用 **semantic center**（人体语义中心）+ 从头到身体其他部位的**方向向量**做「外观指纹」，跨帧跨机位不变；
并在边缘设备做 part-to-whole association，中心侧只做跨机位汇聚。

**对我们的意义**：我们的设计停在「并行检测 + 几何归属 + 可用性统计」，缺的是
①**fallback 策略**（脸缺失时用什么锚点继续）、②**回填关联**（脸出现后把之前无脸的轨迹补上人脸证据）。
这两件事不需要新模型，只需要在数据结构上加一层 `face_evidence_by_track` 的时序拼接。

### 3.2 专利 CN117542101A（跨相机多目标跟踪的人像视角选择）

链接：https://patents.google.com/patent/CN117542101A/zh

- 双分支 YOLO 同时出 **人脸框** 与 **人头人体框**，先匹配「同一目标的人脸框 + 人头人体框」；
- 人脸走 ArcFace、人体走 OSNet，各自出特征；
- 跟踪用卡尔曼滤波，状态量是**人头人体框的中心点、宽、高、朝向**；
- 当前帧与历史轨迹匹配：用人脸+人体特征建代价矩阵，**阈值 0.35**，超过置无穷；用**匈牙利算法**求全局最优；
  失败再退回 IoU 匹配；匹配失败的历史轨迹删除。
- 设计意图写得很直白：**人脸解决「衣着相似的不同人」，人体解决「人脸遮挡/模糊」**。

**对我们的意义**：这条给了两个可直接抄的参数/结构——
① **代价矩阵 + 匈牙利 + 阈值 0.35**（我们现在是 argmax + 全局 θ，没有一对一约束）；
② **失败退回 IoU 匹配**的分层策略（我们现在是直接弃判）。

### 3.3 MDPI 2024：面部→身体级联（旅游景区）

链接：https://www.mdpi.com/2220-9964/13/12/423

- RetinaFace 检脸 + ArcFace 认脸，YOLOX 检人；**人脸与人体用匈牙利匹配，IoU≥95% 才保留配对**；
- 认不出脸时退回 Fast-ReID 人体特征（ResNet34 + 聚合模块）；
- 后处理用 **k-reciprocal 重排 + Query Expansion**。

**对我们的意义**：人脸→人体的**级联**（不是投票）是他们选择的融合方式；
另外 k-reciprocal / QE 我们从未试过，是低成本可验证项。

### 3.4 Sci Rep 2026：动态人脸库 / 人体库互更新（工程实训课）

链接：https://www.clearskyscience.com/en/10.1038/s41598-026-45964-6

- 开局签到：脸对得上就**同时存一张人体图**，形成初始人脸库 + 人体库；
- 运行中：**脸匹配成功但身体匹配差 → 把当前身体视图加入人体库**；
  **身体匹配好且位移小、但脸临时缺失 → 把当前人脸视图加入人脸库**；
- 效果：三人（纯脸 / 纯体 / 融合）对比，融合在登录与全程都最好；仍受强侧脸限制。

**对我们的意义**：这就是「证据互更新」的最简实现——用高置信模态去扩充低置信模态的库。
我们已有 a/b 两片结构，可以直接用它做**样本增广**（不重训模型，只扩 gallery 原型）。

### 3.5 与我们的差异

| 项 | 文献常见设定 | 我们 |
|---|---|---|
| 人脸可用率 | 门禁/签到场景 >90% | **6.4%**（19/295），323 机位 0% |
| 人脸角色 | 主判据或与 body 平权 | 只能做**拒识理由 / 佐证**（AUC 0.880 < body 0.981） |
| 归属方式 | 匈牙利 + 阈值 / IoA | 覆盖率≥0.6 + 纵向/横向约束 + 贪心（等价，参数未调） |
| 回填关联 | 有 | **没有**（这是我们真正缺的一块） |

---

## 4. 卡点 C：图库纯度与跨机位匹配

### 4.1 NVIDIA DeepStream 的 ReID 与重关联（工业基线）

链接：https://developer.ridgerun.com/wiki/index.php?diff=prev&oldid=73329

- 目标维护一个 **feature gallery**（按 `reidExtractionInterval` 抽帧提特征存入）；
- 新检测与目标的相似度 = **与 gallery 中最近邻的余弦**（不是与均值比）——这点和我们的「簇内最大余弦」一致；
- 目标丢失后进入 **shadow tracking**（`maxShadowTrackingAge` 控制保留时长），回来时先在丢失目标里做重关联，
  再考虑新 ID；
- 跨机位**不 out-of-the-box**，需要应用层做：用同一套余弦相似度比较不同机位输出的特征张量。

**对我们的意义**：① 「最近邻而非均值」我们已经在用，说明不是问题所在；
② **shadow tracking 的「丢失-重关联」窗口**我们没有——轨迹断掉就直接成新轨迹，这可能是「34 条单侧」的一部分成因。

### 4.2 CALR / O2CAP：camera bias 与 IDs Merge（直接对应我们的图库污染）

- CALR：*Camera-aware Label Refinement for Unsupervised Person Re-identification*，arXiv:2403.16450
  链接：https://www.alphaxiv.org/audio/2403.16450
- O2CAP：*Offline-Online Associated Camera-Aware Proxies*，arXiv:2201.05820
  链接：https://alphaxiv.org/abs/2201.5820

核心观察（原文表述）：**同一机位的特征天然更接近，global 聚类会把同机位的不同人并到一起（IDs Merge）**；
他们的对策是**先机位内聚类得到可靠局部伪标签，再做跨机位聚类精修**，并把每个全局簇按机位拆成多个 **camera-aware proxy**。

**对我们的意义（最值得试的一条）**：我们的图库是 4 个机位混在一起做一次 average-linkage 聚类（cosine 0.2），
而我们的实测问题正是「同一机位、相似工装的人被并」——**这就是 IDs Merge**。
零成本改法：改成「机位内先聚（阈值紧）→ 跨机位合并（阈值松）」的两段式，或对每个簇按机位拆原型。
这不需要重训任何模型，只要改 `session_selector_v1.py` 的聚类入口。

### 4.3 AFLink / GSI（轨迹拼接与补洞）

- 出处：*StrongSORT: Make DeepSORT Great Again*，arXiv:2202.13514；代码 https://github.com/dyhBUPT/StrongSORT
- AFLink：**无外观**的轨迹拼接网络，输入两条轨迹的时空序列（帧号 + 位置），7×1 时序卷积 + 1×3 融合 + MLP 二分类，
  推理时用时空约束过滤不合理对，再解线性分配；**1.7 ms/帧**。
- GSI：线性插值补洞 + 高斯过程回归平滑轨迹（4 个坐标分别回归），**7.1 ms/帧**；插值间隔与平滑参数 `tau` 可调。

**对我们的意义**：
- **AFLink 救不了我们的重复轨迹**：它拼接的是时间不重叠的碎片；我们的头肩/全身两条轨迹**同时存在**，
  时空约束会直接判为不可连（推断，未实测）。
- 但 AFLink 的**思路**可以借给「34 条单侧 + 12 条异簇」——那批是轨迹碎片问题，不是重复框问题。
- GSI 的补洞可以改善 323 远场轨迹的观测稀疏（如果后续做轨迹级特征聚合）。

### 4.4 评估口径

- AI City Challenge 的 MTMC 用的是 **IDF1 / HOTA**（需要真值 ID）；我们无真值，用 coverage/same/fm 三个代理。
  链接：https://www.aicitychallenge.org/
- 我们已补的「按帧人数真值」正是补 IDF1 里 **IDR（召回）** 那一半——因为 coverage 的分母不含漏检。

---

## 5. 可以直接落地的五条（按成本排序）

| # | 做法 | 依据 | 成本 | 现状 |
|---|---|---|---|---|
| 1 | 掩膜包含判据做重复框抑制（框 IoU + 掩膜包含双判据） | VFG / vg-NMS / R2NMS | 已实现，验证中 | `--dedup-mask 0.9`，305 验证排队中 |
| 2 | 图库改为「机位内紧聚类 → 跨机位合并」 | CALR / O2CAP 的 camera bias | 0.5d，纯 offline | 未做，**建议下一个做** |
| 3 | 人脸 fallback + 回填关联（P0 拒识语义） | 专利 US20260045091A1 | 0.5–1d | 未做 |
| 4 | 归属规则改用 IoA（覆盖率）而非 IoU；加一对一匈牙利 | Double Anchor 工程坑 + CN117542101A | 0.2d | 部分已有（覆盖率≥0.6），缺匈牙利 |
| 5 | k-reciprocal 重排 + QE | MDPI 2024 | 0.5d | 未试过，与 our 的 same 约束可能冲突，需实测 |

## 6. 不建议照搬的三条

| 做法 | 为什么不适合我们 |
|---|---|
| 重训带 visible 双框的检测器 | 需要可见框标注；我们已有实例掩膜，信息等价且零成本 |
| 3D / 标定 / BEV 关联（NVIDIA Mv3DT） | 我们四个机位**视野不重叠且无标定**，外观是唯一可用的跨机位线索 |
| 无监督 ReID 的迭代聚类训练（O2CAP 全套） | 需要重训编码器；我们只借它的**聚类策略**，不借训练流程 |

## 7. 参考清单

| # | 资料 | 类型 | 链接 |
|---|---|---|---|
| 1 | Visible Feature Guidance for Crowd Pedestrian Detection | 论文 | https://arxiv.org/html/2008.09993v2 |
| 2 | Visibility Guided NMS | 论文 | https://www.openreview.net/forum?id=EIpMxsmO0n |
| 3 | NMS by Representative Region（R2NMS） | 论文 | https://openreview.net/forum?id=Lbjuc9_Fm2 |
| 4 | CrowdHuman Double Anchor / Joint NMS 工程解读 | 博客 | https://bbs.plcjs.com/forum.php?mod=viewthread&tid=550831 |
| 5 | 头身匹配三大坑（IoA 而非 IoU） | 博客 | https://m.toutiao.com/article/7629325284130488872 |
| 6 | US20260045091A1 人脸/头/身三模态 tracking | 专利 | https://patents.google.com/patent/US20260045091A1/en |
| 7 | CN117542101A 双分支 YOLO + 匈牙利 0.35 | 专利 | https://patents.google.com/patent/CN117542101A/zh |
| 8 | 景区游客 face→body 级联 + k-reciprocal | 论文 | https://www.mdpi.com/2220-9964/13/12/423 |
| 9 | 实训课 动态人脸库/人体库互更新 | 论文 | https://www.clearskyscience.com/en/10.1038/s41598-026-45964-6 |
| 10 | DeepStream ReID / shadow tracking / gallery | 开发者文档（RidgeRun） | https://developer.ridgerun.com/wiki/index.php?diff=prev&oldid=73329 |
| 11 | NVIDIA Mv3DT 多机位 3D 跟踪 | 厂商文档 | https://developer.nvidia.com/blog/build-a-multi-camera-3d-tracking-application-with-nvidia-deepstream-9-1-skills |
| 12 | CALR（camera-aware label refinement） | 论文 | https://www.alphaxiv.org/audio/2403.16450 |
| 13 | O2CAP（camera-aware proxies） | 论文 | https://alphaxiv.org/abs/2201.5820 |
| 14 | StrongSORT / AFLink / GSI | 论文+代码 | https://github.com/dyhBUPT/StrongSORT |
| 15 | AI City Challenge（IDF1/HOTA 口径） | 赛事 | https://www.aicitychallenge.org/ |

> 标注说明：论文结论与参数来自摘要/正文摘录；「对我们的意义」一栏是我的判断，其中
> 「AFLink 救不了同时存在的重复轨迹」是**推断**，未实测；其余可直接用我们的产物验证。
