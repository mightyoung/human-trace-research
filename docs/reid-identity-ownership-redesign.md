# 遮挡接续 · 全局身份卡 · 跨画面去重 · 镜像倒影 —— 现状 / 近邻 / 改造

日期：2026-10-14 ｜ 口径：`DET_VIEW=clean2`，实体图 `analysis/stitch_clean2/`
目的：针对用户提出的四个问题，先固化现状（带证据），再做近邻研究（带阅读深度标注），最后给改造方案。
证据来源一律标 `文件:行` 或产物路径，可复核。

---

## 0. 结论先行

**四个问题不是四件事，是同一个架构缺口的四种表现：我们没有"全局身份层"。**

现在系统里同时存在 **四套互不相通的"身份"记号**：

| # | 记号 | 粒度 | 谁在生成 | 谁在消费 |
|---|---|---|---|---|
| A | `tid`（track id） | 单机位碎片 | ByteTrack | **展示视频**（`render.py`）、`tracks/{cam}.json` |
| B | `eid`（entity） | 单机位实体（碎片已拼） | `track_stitching.py` / `rebuild_clean2_entities.py` | 真值、relay、C0 |
| C | `chain`（跨机接力链） | 跨机位路径 | `crosscam_relay.py` mutual-best | ReID 页"跨镜头统一 ReID"、JourneyPanel |
| D | `cid`（聚簇 id） | 跨机位外观簇 | `build_algo_data.py` mutual-kNN | 聚类工作台 |

**没有一个是"人"。** 四问分别落在缺口的四面：

1. **遮挡断 ID** = 记号 A 的生成器（ByteTrack）**完全不用外观**，且丢了就是新号，且 B 的缝合是 mutual-best 的事后补丁（只补回了一部分）。
2. **身份卡不一致** = C/D 是两条独立线，且都不是唯一人，页面上"C 是统一"只是**一次通行的路径**，不是人。
3. **跨画面没去重 / 没责任划分** = 展示视频读的是记号 A（每机位各自编号）；"责任划分"在文献里有正式名字叫 **ownership**，我们**完全没有实现**。
4. **镜像倒影** = 掩膜方案在**记号 A 的层面逐帧删框**，必然打断轨迹（实测缺嵌入 1181 条、误删 7 条真人轨），所以用户拒绝它是对的；而外观判据根本不可用（见 §1.4）。

**改造主干（推荐）**：建一个 **全局身份层（Global Identity Layer, GIL）**，把 A→B→`gperson`→`visit`→`chain` 收敛成一条链，四问全部挂在它上面。理由、替代与取舍见 §3.6。

---

## 1. 现状总结

### 1.1 问题一：遮挡后 ReID 续不上，被重分配

**事实**

- 检测/跟踪 = **GroundingDINO + `supervision.ByteTrack`**，唯一推理入口 `tools/infer_supervision.py`（由 `tools/run_1012_detect.sh` 调度）。
  口径：`--stride 5 --box-threshold 0.30 --track-activation 0.10 --containment-threshold 0.85`。
- `track id` 由 ByteTrack 内部生成，逐帧写进 jsonl 的 `track_ids`（`infer_supervision.py:210,260`）。
- **没有任何"续 id"逻辑**。唯一的"中断"机制是丢失缓冲超时：
  `tools/infer_supervision.py:46` `--lost-buffer 90`；`:118` `frame_rate=float(fps)`；`:124` `max_time_lost = fps/30*90`。
  ⚠️ 注意 `frame_rate` 传的是**原始视频 fps** 而 `stride=5` → 实际容忍 **≈15 秒**无检测（25fps × stride 5 × 75 次 update）。
  **所以"缓冲太短"不是主因**（这一点后面决定改造优先级）。
- **每条 track 一个嵌入向量**（采样 8 帧 → 编码 → 均值 → L2 归一）：`tools/embed_server.py:174-178`、`tools/encode_tracks.py:110-113`。
- **同机位碎片拼接已有实现**（这是问题核心）：
  `tools/track_stitching.py` docstring 明确写"同一镜头、同一人的碎片轨迹（**遮挡/出画/短暂丢失导致 track ID 切换**）合并"；
  连接构造 = **mutual-best（相互最优）+ 禁止传递闭包**，门限 `link_cos=0.80 / link_pe=0.08`（`:200-218`）；
  `tools/rebuild_clean2_entities.py:44-46,113-142` 用**同一套门限**在 clean2 上重建实体，Union-Find 合并成实体。

**证据（碎片规模）** —— `analysis/stitch_clean2/net_tracks_*.json` 实测：

| 机位 | 实体 | 成员轨 | 多段实体（`n_frag>1`） | 占比 |
|---|---|---|---|---|
| 305 | 81 | 107 | 22 | 27% |
| 316 | 61 | 68 | 5 | 8% |
| 318 | 105 | 151 | 20 | 19% |
| 320 | 20 | 24 | 3 | 15% |
| 323 | 98 | 138 | 32 | 33% |
| 343 | 46 | 53 | 4 | 9% |
| **合计** | **411** | **541** | **86** | **20.9%** |

→ 541 条轨被拆成 411 个实体（合并掉 130 段）。**这 20.9% 是"已经拼回的"下界**；325 个单段实体里仍混着未拼回的碎片（无人核对过）。

**为什么现在做不到"接续"**

1. **跟踪器层面零外观**：ByteTrack 两阶段匹配全靠 IoU，低分框只用于"维持已有轨迹"，**不用于重识别**；一旦轨迹被删（超缓冲），新出现的同人拿新号。
2. **拼接是事后 mutual-best 补丁**：只保留"互为最优"的边，无法处理**一人多次通行 / 一对多竞争**（这个失败模式我们在 relay v5 上已经量化过：47 条 primary 边里 3 对因排他竞争被误排）。
3. **阈值是全局常数**（cos 0.80 / pe 0.08），而我们自己的数据已经证明**跨机位/跨机位外观分布不同**，per-cam 校准是有效的（选择器 v2 三档 0.972/0.977/1.000）。

---

### 1.2 问题二：不同画面的 ReID 身份卡不一致，需统一为全局

**事实**

- 全局 ReID 库：`output_crosscam/analysis/global_reid/store_cradio_v4/`（`emb.npy` + `emb_mrl.npy` + `meta.sqlite` + `version.json`）。
  `version.json`：`{"model_version":"cradio_gemma2_bridge_v4","dim":768,"mrl_dim":256,"n":539}`。
- ★ **键是 `f"{date}|{cam}|{tid}"`**（`tools/global_reid_store.py:42,92`）→ **库里没有"全局人 id"这一列**，539 条 = 539 条**轨迹**，不是 539 个人。
- **有聚簇**：`webapp/backend/data/clusters.json`（`mutual-knn, k=3, thr=0.95, n=539, n_clusters=219`），由 `webapp/dataprep/build_algo_data.py:117` 生成；选择器线另有簇 id（`tools/selector.py:99-113`，`-1` 表示 unknown）。
- **但 ReID 页的"身份卡"不是按簇分组的，是按接力链分组的**：
  `webapp/frontend/src/views/ReidView.vue:104-108` `chains = store.chains.chains`，`chainMembers(cid)` 经 `store.entChain` 反查；
  `webapp/frontend/src/store.ts:142` `store.entChain["cam:eid"] = ch.id`。

**证据（链规模）**

- `analysis/relay_clean2/relay_chains_cradio_v5.json`：**47 边 / 32 链**，`thr=0.943`。
- `analysis/relay_v2/edges_v2.json`：182 边（`status`: active 47 / audit 135）。

**卡在哪里**

- "链"= **一次通路**（节点序列），不是"人"。一个人在同一段素材里多次通行会产生多条链 —— 我们已经在真值上量到：241 个跨机配对候选里**外推同人约 41 个（17.1%）**，而 13 条已判"同人"只落在 **7 个不同的 343 事件**上（**互斥违规 46%**，见 `analysis/truth/TRUTH-FINDINGS-2026-10-14.md`）。
- 唯一化用的是 **mutual-best（出/入度 ≤1）**，对"一人多次通行"结构性失败。
- 簇（D）和链（C）**互不引用**：聚类工作台改了簇，不会影响 ReID 页的链；ReID 页的卡也不会进视频。

---

### 1.3 问题三：多画面同一人未去重，展示视频未实现"责任目标划分"

**事实**

- **渲染按每机位各自的 track id 画框**：`webapp/backend/render.py:41-42` `_track_color(tid)=_PALETTE[tid%N]`；`:136-143` 用 `tracks/{cam}.json` 的 `tracker_id` 出框和标签 `#tid`；`:289` 导出视频同样。
  `_filter_view`（`:210-222`）只按 `keep/reflection/masked` 过滤，**没有任何跨机去重**。
- 前端有 `colorBy: 'chain' | 'status' | 'entity' | 'track'`（`store.ts:42,157-166`）—— 按链着色只是**视觉提示**，不改变计数、不改变渲染源。
- **没有运行时 owner / 在线交接**。`docs/cam-handover-audit.md:12,43` 明确："运行时与更新未实现…全部产物为离线批处理…无在线服务消费 relay 图"。
- 现有 relay 边/链**可以直接支撑去重人次统计**，但**只被离线用于可视化**：事件 `relay_clean2/events_{cam}.json`（`{eid,t_first,t_last,u0,v0,u1,v1,h_med,moving}`）、边、链三段产物齐全。
- 前端页面：`/`(VisualizeView + ChainPanel + JourneyPanel)、`/annotate`、`/reid`、`/cluster`、`/truth`、`/config`（`main.ts:16-21`）。

**结论**：去重与责任划分**缺的不是算法，是"消费端"** —— 算法产物（链）已经存在，但没有任何一处把它当作"人次真值"来用。

---

### 1.4 问题四：镜像/倒影依然严重，且不能再用玻璃掩膜

**事实：玻璃掩膜做了什么、为什么该弃用**

- `tools/glass_mask.py`：按**归一化多边形**（每机位手画）判断**落脚点**是否落在区域内，命中就**逐帧删框**（`:116-117`）；`--apply` 后统计"完全消失的轨迹"（`:130-135`）。
- 调用点：`tools/1012_post_detect.py:56-76`（只在 323 上），以及 `tools/build_cleanset.py:27,81` 的 `dualclean` 机位（**316 / 323**）。
- ★ **代价（实测，不是推测）**：
  1. **打断轨迹**：帧级删框 → 轨迹被截断，8 帧采样点与轨迹时长都变（`docs/reflection-cleaning-integration-plan.md:111`）；
  2. **缺嵌入 1181 条**：v4 接力里正是这些被掩膜删掉的轨迹（`docs/clean2-rebuild-report.md:47`）；
  3. **误删真人 7 条**：`316#32, 323#193, 318#53/82/98/135, 305#12`（`build_cleanset_v2.py:28:` `FP_RESTORE`），v2 才恢复。
- → 用户"不能再用玻璃掩膜"的判断**完全成立**，且理由是可量化的：**它在错误的层面上动手（逐帧删检测），代价落在轨迹链上**。

**事实：替代方案的现状与硬约束**

- 镜像运动学：`tools/mirror_kinematics.py:84-108,201-205` —— 倒影中心关于触地线镜像 `pred = 2*vs - vc_s`（`:84-85`），输出 `du_med / hratio / offset_ratio / pred_err / 逐帧差分锁步`；组合规则 `|du_med|<0.06 and pred_err≤0.10 and 0.35≤offset_ratio≤0.60`。全量版 `tools/track_cleaning.py:32` `RULE = dict(du_max=0.06, perr_max=0.10, off_lo=0.35, off_hi=0.60, n_min=4)`。
- 深度：MiDaS 试过，**地板倒影无解**（AUC ≤ 0.65），只作 tie-break。
- ★★ **最关键的既有事实（决定改造方向）**：真值实测 **同人 cos 0.948–0.966 vs 异人 0.9466–0.9478，间隔仅 0.0002**（`TRUTH-FINDINGS-2026-10-14.md`）。倒影与真人是**同一外观的近副本**，所以
  - **"像不像真人"这条路走不通**（外观分类器分不开）；
  - 唯二可用的信号是**几何关系（镜像耦合）** 与 **上下文（在反光面/触地线以下）**。

**结论**：掩膜被否定后，现在的 316/323 是 `dualclean`（掩膜 + 运动学），其余是 `clean`（仅运动学）—— **口径已分裂**，且倒影问题在 305/318/343 基本没治。需要一套**不动轨迹**的判据。

---

## 2. 近邻研究

阅读深度标注：`[全]`=读过正文、`[摘]`=摘要/官方介绍、`[二]`=可靠二手综述。按本项目的证据纪律，**`[摘]/[二]` 不足以立"采纳"**，实施前需补正文（见 §2.5）。

### 2.1 问题一（遮挡后身份续接）

| 工作 | 机制要点 | 对我们可借鉴 | 不可迁移 |
|---|---|---|---|
| ByteTrack (ECCV'22) `[二]` | 两阶段匹配、复用低分框维持轨迹 | **我们已在用**；确认其"低分框不初始化新轨迹"是关键 | 无外观，长时遮挡必换号 |
| **BoT-SORT** (2022) `[二]` | ByteTrack + **ReID 外观代价** + 相机运动补偿(GMC)，IoU/外观自适应融合 | 关联代价里**加外观**是最直接的一步；我们机位固定可省 GMC | 需重新调参；Ultralytics 默认 `with_reid:False`，要显式开 |
| **OC-SORT** (CVPR'23) `[摘]` | **观测中心**更新（ORU/OCM）：遮挡结束后用**真实观测回填**被污染的预测 | 我们的 stride=5 离线批处理天然适合做后验回填 | 需改跟踪器内部，非配置项 |
| **Deep-OC-SORT** (ICIP'23) `[二]` | OC-SORT + ReID，**只信清晰/高分帧**建外观记忆 + 自适应权重 | ★"**只信清晰帧**"正好治我们的反光/模糊污染；与我们的"质量门控"主线同构 | 需 GPU 重跑 |
| **StrongSORT** (TMM'23) `[二]` | **EMA 外观更新**（替代特征库，抗单帧噪声）+ NSA 卡尔曼 + **AFLink 离线片段连接** + **GSI 高斯插值补空洞** | ★★ 三个都可直接用：EMA 嵌入、AFLink 式**离线拼接**（正是我们 `track_stitching` 的升级方向）、GSI 补断帧 | 仍需按序列校准 |
| **TrackTrack** (CVPR'25) `[二]` | 反过来：**每个轨迹主动认领**最优框；新框与已跟踪目标重叠时**拒绝新建 ID** | ★ 直接治"重复/闪烁 ID"；与我们"禁止传递闭包"的动机一致 | 最重、最新，集成风险高 |
| **Lifted Disjoint Paths** (2020) `[摘]` | 在 min-cost flow 上加 **lifted edges（长程边）** 作连通先验，ILP 求解；MOT17 IDF1 **52.3→65.6** | ★★ 把 `track_stitching` 的 mutual-best 升级为**全局最优 + 长程边**，是"接续"的理论正确形式 | 离线（需未来帧）；ILP 求解约 26 min/序列（我们 10min×6 路可接受） |
| MOTIP `[二]` | 保留多帧 object embedding 记忆 + 专用"新 ID"token | 思路参考 | 需训练，代价高 |
| ★ **反证**：《Does Re-ID Really Help in MOT?》(SCCAI'25) `[全/官网 PDF]` | Re-ID 在 BoT-SORT/Deep-OC-SORT 上的增益**温和**，且**部分序列会变差** | ★ **纪律**：任何"加外观"的改造必须做**同序列 A/B**，不得只看均值 | — |

**要点**：问题一的正确解法是**三层**——①关联代价加外观（BoT-SORT/Deep-OC-SORT 思路）；②**离线全局长程关联**（LDP/AFLink，这是"接续"的主战场）；③补空洞（GSI）。缓冲长度**不是**主因（已是 ~15s）。

### 2.2 问题二（全局身份统一）

| 工作 | 机制要点 | 可借鉴 |
|---|---|---|
| **Hsu et al., TIP'21** (arXiv 2105.01213) `[摘+二手详述]` | **TCLM**（由拓扑自动建入口/出口 zone + 转移时间）+ **层次聚类**联合做**全局 ID 指派**；CityFlow IDF1 76.77 | ★★ "全局 ID = 约束下的层次聚类"，而不是 mutual-best 一对一；**zone + 转移时间**正是我们已有资产 |
| **Specker et al., CVPRW'22**（AI City 2022 Track1）`[全/PDF]` | zone 约束 + **两轮层次聚类**（先按转移分、再按相邻机位）+ 距离取"均值距离 × 最强单帧对距离"之积 + **track completion** | ★ 距离度量的那个**乘法**很关键（既看整体也看最强证据）；track completion 对应我们的"补漏接" |
| **Glance-MCMT** (arXiv 2507.10115, 2025) `[摘]` | **渐进式全局 ID 指派**（先世界坐标轨迹相似、后外观）+ **SNMS 顺序 NMS 合并重复/碎片 tracklet** + **代表性特征**（取轨迹内主簇特征） | ★ SNMS 治"同机位重复/碎片"；"代表性特征"治"轨迹内被遮挡帧污染"——与我们"只信清晰帧"一致 |
| 9th AI City Challenge (arXiv 2508.13564) `[摘]` | 2025 Track1 = 3D MTMC；冠军 ZV 用**几何优先离线管线 + 3D 点云 + 全局离线关联**；SKKU 用 **Tracklet-Cluster Mapping (TCM)**；普遍用 **ViT-ReID + 层次聚类** | 确认"**离线全局关联 + 聚类**"是当前 SOTA 范式；也确认**依赖标定/深度**（我们无标定 → 需替代） |
| Do et al., ICCVW'25 (arXiv 2509.09946) `[摘]` | **局部 ID 一致性**达标即**晋升为全局 ID** | ★ 一个极便宜的在线基线：稳定窗口内不变的 track 直接升格 |

**要点**：全局身份 = **`外观 + 门户 zone/转移时间 + 互斥` 约束下的聚类**，输出 `gperson_id`；**"链(visit)"与"人(gperson)"必须分开**。

### 2.3 问题三（去重 + 责任划分）

| 工作 | 机制要点 | 可借鉴 |
|---|---|---|
| **Kettnaker & Zabih, ICMS'99** `[全/PDF]` | 多机位**人数去重**开山：外观匹配 + **相机间互内容约束**（由楼层拓扑推可行转移，如"C1 消失后只可能出现在 C2 或 C4"） | ★★ "去重"从一开始就是"外观 + 拓扑约束"的联合推断；我们的 relay 图就是这个约束的实现 |
| **Zhang et al.** (arXiv 2104.04662, 清华) `[全/PDF]` | 非重叠室内多机位：**随机转移矩阵 × 外观**联合提升跨机 ReID 排序；并给噪声抑制的理论解释 | ★ 转移矩阵与外观是**乘性**结合（与我们的 `cos + λ·ln w_dt` 同族，但我们只用了对数线性） |
| ★★ **Esterle et al., ICDSC'11 / TAAS'15** `[全/PDF]` | **ownership** 的正式定义：**任一时刻只有一台摄像机对某目标负责**；`O_i` = 机位 i "拥有"的目标集；**vision graph** = 重叠 FOV 的边 **+ 非重叠但"若干时间内可出现"的边**；效用 = 可见性/置信度；Vickrey 拍卖式**交接**；交接是**局部机制**（只看邻居） | ★★★ **"责任划分"就是这个概念**。我们的 relay 图 = vision graph；relay v2 的 F4 状态机 = 交接状态机；缺的是 **owner 判定（效用函数）+ 时间线** |
| 商业化口径（Isarsoft / Plumerai / FootfallCam）`[二]` | ReID 用于**跨机只计一次**、把多次出现**合并为一个 event**、拼接时间线、**员工/路过者排除** | ★ 产品化的验收口径："同一人跨机只计一次" + "事件合并" |

**要点**：问题三的算法**我们几乎全有**（relay 图 + 转移分布 + 状态机），缺的是
①**owner/效用函数**（谁负责，按可见性质量 argmax）；②**把链当人次真值消费**（渲染、计数、去重视图）。

### 2.4 问题四（不用掩膜的镜像/倒影）

| 工作 | 机制要点 | 可借鉴 |
|---|---|---|
| ★★ **Detecting Reflections by Combining Semantic and Instance Segmentation** `[全/二手详述]` | 监控场景"ghost 检测"：Mask R-CNN 因 ROI 裁剪**丢失全局上下文**而分不开倒影；用 **UPerNet 语义分割（大感受野）+ 实例分割融合**，按"人 mask 内平均语义分"**判真伪**。precision **71%→83%**（误报 502→251），**recall 基本不掉**；微调后 80%→84% | ★★★ 最贴近的"**不用区域掩膜、按检测判真伪**"的方案；关键教训：**判别力来自全局上下文**，不是局部外观（正好解释我们的 0.0002 间隔） |
| **RODD** (arXiv 2407.05575) `[摘]` | 首个**反射目标检测基准**：21,059 图 / 10 类，标注"real vs reflected"；现有 SOTA 检测器在此上**明显退化** | ★ 可作为**预训练/评测**集；证明"反射检测"是独立任务 |
| 镜像检测线：MirrorNet (ICCV'19) / PMDNet (2020) / SANet (2022) / **SATNet** (AAAI'23) / VCNet（视觉手性）/ Mirror-YOLO / 轻量边缘引导 CNN (WSCG'23) `[二/综述]` | 分割**镜面区域**；SATNet 用"**松对称**"（内外内容语义/亮度一致但位置不严格对称） | 输出的是**语义镜面 mask**（≠ 手画多边形），可作**软先验**；但与我们"地板倒影"不完全同类 |
| SIRR 去反射线：ERRNet / Location-aware SIRR / RDNet(LowAux-RDNet 2026) `[二]` | 单图去反射，恢复透射层（`I ≈ W∘T + R`）；location-aware 版用 **Laplacian 特征回归反射置信图 RCMap** | 可作**预处理**改善检测/ReID crop；但问题病态、可能引入伪影，不宜直接改原图喂检测 |
| **视觉手性 / 翻转**：FlipReID (2021)、MPGA-Net（HFNI：把水平翻转图当**新身份**）`[全/二手]` | 翻转是**手性**变换；模型可学"翻转不变/手性判别" | ★ 理论支撑：倒影 = **手性翻转副本** → 可用**翻转一致性**做几何判据（我们 `mirror_flip_test.py` 已验过"倒影=源镜像翻转"） |
| 专利 CN111898553B `[全/专利]` | 用"人脸**远近程度指标** vs 该二维位置**允许的指标区间**"判虚像，**不新增红外** | ★ 思想=**几何可行域检验**（某位置只允许某些深度/尺度）—— 与我们"倒影比真人更远/尺度异常"同构，且明确"不需要额外硬件" |
| ❌ 不可用路线 | 把"倒影 vs 真人"当**外观分类**问题 | **我们的数据已证伪**：cos 间隔 0.0002 |

**要点**：不用掩膜有三条腿——①**关系型**（镜像配对残差 + 配对唯一性，我们已有 `mirror_kinematics`）；②**上下文型**（语义+实例融合式的"不在反光面里"软分，可借 RODD 预训）；③**几何可行域**（深度/尺度一致性，弱、仅 tie-break）。**三条都不该逐帧删框**。

### 2.5 阅读深度与待补

按本项目证据纪律，下面这些**决定改造可行性**的工作目前只到 `[摘]/[二]`，**实施前必须读正文**：
BoT-SORT / Deep-OC-SORT / StrongSORT(AFLink, GSI) 的关联代价与离线连接细节；LDP 的 lifted edge 代价构造；Hsu'21 的 TCLM 构造与层次聚类距离；Specker'22 的两轮聚类与 track completion；Esterle 的效用函数与交接触发条件；语义+实例融合那篇的阈值与 recall 代价。

---

## 3. 改造方案

### 3.1 共同地基：全局身份层（GIL）

**数据模型（新增一层，不动已有口径）**

```
obs(t, cam, tid, box, foot, emb)            # 已有（tracks/{cam}.json）
  └─ frag   = 同机位片段（= 现 tid）
  └─ entity = 同机位实体（现 eid，碎片已拼）
        └─ gperson = 全局人  ★新增，唯一人 id（可跨时段、可多次通行）
              └─ visit  = 一次通行（≈ 现 chain，但归属到 gperson）
                    └─ owner(t) = 某时刻哪个机位对该 gperson 负责  ★新增
```

**`gperson` 的产生（问题二的解）**：在
①**门户可行转移**（复用 `analysis/relay_v2/dist_v1.json` 的 Parzen 过渡时间 + relay 图拓扑）、
②**时间互斥**（同一时刻一个 gperson 在同一机位至多一个实体）、
③**外观**（实体级嵌入，扩展到 411 实体）
三重约束下做**层次聚类**（Hsu'21 / Specker'22 口径），而不是 mutual-best 一对一。

**产出**：`analysis/gil/gil_v1.json`（`gpersons / visits / owners / provenance`）+ 冻结哈希；消费端显式声明 `GIL=v1`（沿用 `DET_VIEW` 的纪律）。

### 3.2 问题一改造：同机位接续（三层，可独立上线）

| 层 | 做法 | 复用/新建 | 验收 |
|---|---|---|---|
| **L1-a 关联加外观** | 跟踪阶段加外观代价（BoT-SORT/Deep-OC-SORT 思路）：**只信高分帧**建外观记忆；低分框仍只用于维持 | 新建 `tools/track_with_reid.py`（或给 `infer_supervision.py` 加 `--reid` 路径） | 同机位 **ID 切换率**（316/323 各抽 20 段人工）↓；A/B 同序列 |
| **L1-b 离线全局长程关联** ★主战场 | 把 `track_stitching.py` 的 mutual-best 升级为 **min-cost flow / ILP + lifted edges**（长程边允许跨 gap）+ **GSI 插值补空洞** | 改造 `tools/track_stitching.py` → `track_assoc_global.py` | 实体总数↓、`n_frag` 分布右移、**误合率不升**（用真值互斥违规守） |
| **L1-c 阈值按机位自适应** | 用选择器 v2 已有的 **per-cam 分位校准**替换全局常数 `cos 0.80 / pe 0.08` | 复用 `selector_v2.npz` 的 per-cam thr | 每机位独立扫点，报告"接续↑ / 误合"取舍曲线 |

⚠️ 缓冲长度**不用动**（已 ~15s）；重点在"丢了之后能不能认回来"。

### 3.3 问题二改造：全局身份卡统一

1. 建 `gperson`（§3.1）；
2. **把"链"和"人"分层**：ReID 页"身份卡"改为 **gperson 卡**，卡内按 `visit`（通行）分组，卡上标"跨 N 个机位 / M 次通行 / 总时长"；
3. **簇（D）与 gperson 合并**：聚类工作台的编辑回流到 GIL（现在簇改动不回流，是死路）；
4. **嵌入回退**：修 `316-E015` 类缺失（补嵌入或用 relay 相似度回退）—— 这个坑已在真值上暴露。

**验收**：以已判 13 条"同人"为正样本，测 `gperson` 聚类的 **pairwise precision/recall**，且**互斥违规率必须显著低于外观单阈值的 46%**。

### 3.4 问题三改造：去重 + 责任划分（owner）

1. **效用函数**：`u(cam, gperson, t) = 可见性质量`（复用选择器 per-cam 校准分 + 反光/遮挡 flag + 脚点是否在边界）；
2. **owner 判定**：`owner(t) = argmax_cam u`，并列时按门户拓扑/zone 决；
3. **交接**：把 relay v2 **F4 状态机**从"边的状态"升到"`gperson` 的状态 + owner 转移时间线"，产出可回放的 `owner_timeline.json`；
4. **消费端**（这才是用户看到的）：
   - 渲染新增 `colorBy: 'gperson'` 与 **「去重视图」**（同一 gperson 全片只出现一个"当前框"+ 标注 owner 机位；重叠对 318×320 合并）；
   - **计数改为 `gperson × visit`**，不再按 track；
   - ChainPanel / JourneyPanel 改为读 GIL。

**验收**：① 任一时刻同一 `gperson` 的 owner **唯一（0 违反）**；② 去重后人次与**事件页人工账本**（107 事件已填 `human_person_id`）对账；③ 给出 305 出门覆盖 0.0 的修复前后对比。

### 3.5 问题四改造：不用掩膜的镜像/倒影

**核心改写**：**倒影 = 另一个观测的镜像副本**，因此把它做成**指派问题**，而不是"区域过滤"：

```
对每个候选倒影观测 r，必须存在一个真人观测 s 使其被"解释"：
  cost(s,r) = 镜像残差(du_med, pred_err, offset_ratio) + 尺度/深度不一致 + 门户几何
求全局指派（真人 ← 倒影 一对至多一），只有被解释的观测才可能是倒影。
```

三条腿：

1. **关系型（主）**：现有 `mirror_kinematics` 的残差**从"逐轨规则"升级为"全局指派约束"**；加**配对唯一性**（一个真人最多解释一个倒影）——这一条同时解决"误删真人"（真人没有配对被解释）。
2. **上下文型（辅，软分）**：学一个"是否处于反光面内"的判据（**语义+实例融合**思路，可借 RODD 预训）；**只作软分，不作硬删**。
3. **几何可行域（辅）**：位置→允许的深度/尺度区间（专利 CN111898553B 思路），MiDaS 仅 tie-break。

**硬纪律（针对掩膜的教训）**
- ❌ 不逐帧删框 → ✅ **只对实体/轨道下判定 + 打 flag**，原始检测**一个字都不改**；
- 单机位**不再新增手画多边形**；需要反光面先验时用**学习式语义区域**作软先验；
- 判据上线前必须在**已审计的倒影短名单**上测 **precision/recall，且以"不误删真人"为第一约束**；报告"识别出多少真倒影 / 误删几个真人"。

**验收**：316/323（现有 dualclean）与 305/318/343（现在基本没治）分别给"倒影剔除率 / 真人误删率"；目标是**在误删率不升的前提下**，把 316/323 的掩膜替换掉（口径统一为 `clean3`）。

### 3.6 排期、替代与取舍

**推荐（主线）**：先建 **GIL**（§3.1），因为
- 它**同时**解题 2 和题 3（去重与责任划分本就是同一层）；
- 它为题 1（实体归属）和题 4（倒影只降权不删框）提供**共享容器**；
- 我们的资产（411 实体、relay 图、转移分布、per-cam 校准、真值集）**全部可直接复用**，边际成本最低。
并行推进 `L1-a`（跟踪加外观）与 `Q4 关系型指派`（两者互不依赖 GIL）。

**替代**：只做表面修补（视频换 gperson 着色、加长 buffer、继续用掩膜）——**便宜但治不了 2 和 4**，且掩膜已被证伪。

**取舍**：GIL 要一次性把**四套身份记号合并**，改动面覆盖所有消费端（webapp/relay/truth/渲染），**一次性成本高**；但这是唯一能同时满足四问的路。若不做 GIL，四问会各自长出一套补丁，版本链会再次分裂（这正是 `stitch` vs `stitch_clean2` 的历史）。

### 3.7 风险与停止条件

| 风险 | 说明 | 停止/转向条件 |
|---|---|---|
| **外观不可分** | 已证 cos 间隔 0.0002（真值） | GIL 的 `gperson` 主要靠**拓扑+互斥**；若门户先验不达标则 gperson 不可靠 |
| **305 出门覆盖 0.0** | 305 的门户先验缺失（最严重缺口） | ★ **先修 305，否则 GIL 在 305 段不可信** —— 这是前置条件 |
| **重跑检测会改哈希** | 改跟踪器 → 全部下游产物变化 | 冻结旧版、版本化（`det_v2`），消费端显式声明 |
| **Re-ID 增益有限**（SCCAI'25 反证） | 加外观**不保证**全面变好 | ★ 必须**同序列 A/B**；只看均值不算通过 |
| 倒影判据误删真人 | 历史上已发生（7 条） | 误删率**不得上升**；否则回退为只降权 |

---

## 4. 检索记录（本次）

| 意图 | 查询 | 日期 |
|---|---|---|
| 遮挡/ID 续接 | OC-SORT/BoT-SORT/Deep-OC-SORT/survey 2025；tracklet association min-cost flow long-term occlusion | 2026-10-14 |
| 全局身份 | MTMC global identity assignment clustering AI City 2024/2025；hierarchical clustering global ID CityFlow | 2026-10-14 |
| 去重/责任划分 | cross-camera person counting deduplication non-overlapping；track ownership camera handoff single owner | 2026-10-14 |
| 镜像/倒影 | mirror reflection detection without mask reflection-aware ReID；mirror detection MirrorNet/PMD/SATNet；SIRR ERRNet/Location-aware；reflection-aware ReID / chirality | 2026-10-14 |
| 强基线反证 | StrongSORT EMA/AFLink/GSI | 2026-10-14 |

来源：网络检索（含 arXiv/出版社/官方 PDF）。**未穷尽**；命中数未逐条统计；`[摘]` 项按纪律不得单独作为采纳依据。

---

## 5. 建议的下一步（需你拍板一件事）

四问的改造可以并行，但**顺序取决于一件事**：GIL 走哪条路线。

- **A（推荐）**：把 GIL 作为主干，四问挂在它上面 —— 一次性解决 2/3，并为 1/4 提供容器；成本高、改动面大。
- **B（替代）**：只做"能立刻看见效果"的两件 —— 先出**去重视图（gperson 着色 + owner 标注）**，再修 **305 门户**；不动跟踪器、不动倒影判据。见效快、但 1/4 仍未解。
- **C（最小**）：先只修 **305 出门覆盖 0.0** —— 因为它是所有方案的**前置条件**（门户先验不达标，GIL 在 305 段不可信）。

**推荐 A**，但**先做 C 的 305 修**作为第一步（它便宜、且是 A 的前置）。
