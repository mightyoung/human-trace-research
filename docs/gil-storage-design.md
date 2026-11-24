# GIL 统一存储层设计（向量缓存 + 数据库）

> 版本 v1 · 2026-10-14 · 对应决策：**取 A 方案（全局身份层 GIL 主干）+ 统一存储**
> 交付物：`tools/gil/{schema.sql,store.py,vectors.py}`、`tools/gil_migrate.py`、`tools/gil_smoke.py`
> 产物：`output_crosscam/analysis/gil/store_v1/`

---

## 0. 结论（TL;DR）

**三层存储，一个都不多：**

| 层 | 技术 | 管什么 | 为什么 |
|---|---|---|---|
| **真相层**（文件） | `.npz` 分块矩阵 + `manifest.json` | 向量 | 可哈希、可复现、可 diff、零依赖 |
| **结构真相层** | **SQLite 单文件** | 元数据 / 图 / 状态机 / 标注 | 事务一致、关系+时间查询天然、单文件备份 |
| **缓存层**（进程内） | **numpy 常驻 f32** | 检索 | 实测比 sqlite-vec **快 11–57 倍** |

**明确不引入**：向量数据库扩展（sqlite-vec）、ANN 索引（FAISS/HNSW）、Redis —— **都不是现在的瓶颈，而是负收益**（见 §5 触发条件）。

⚠️ 一个必须记住的反直觉实测：**f16/memmap 每次查询才转换 = 灾难（10 万规模 400–660 ms）**。f16 只能"落盘省空间、启动时一次性转 f32 常驻"。

---

## 1. 需求与访问模式（选型的真正依据）

选型不是看"哪个库更火"，而是看我们的**四件事实**：

| 事实 | 数值 | 含义 |
|---|---|---|
| 向量规模 | 539 轨（现）→ 10³–10⁵（1012 全长 + 片段级展开） | **远未到需要 ANN 的量级** |
| 查询形态 | 全是**强约束检索**：排除同机位 + 时间互斥窗(±3s) + 门户可达 | 过滤选择率常在 **0.2–3%** |
| 写入形态 | **离线批重建**（每次改口径全量重建），不是高频流式 | 事务吞吐不是瓶颈 |
| 复现要求 | 逐框契约、冻结哈希、版本链 | **必须是可哈希的文件真相**，不能只有内存态 |

★ 关键：我们的瓶颈是**约束**，不是**相似度**。

---

## 2. 成熟实践：业界怎么做

查了 4 个真实系统的存储架构，**结论高度一致**：

1. **一个真实行人 ReID 系统**：`FAISS VectorDB（向量） + SQLite（元数据） + Redis（缓存）` —— 三种数据三种存储，各司其职。
2. **FaceNet 工程实践**（positioniseverything）：
   > *"For a **small gallery**, storing embeddings in PostgreSQL, SQLite, Redis, or even an in-memory structure can be sufficient. For **larger deployments**, use a vector database or ANN index."*
   > *"**Store metadata separately from embeddings**: user ID, display name, enrollment date, image source, **embedding model version**, and consent status. **Keeping the model version with each embedding** helps when you upgrade... since embeddings from different pipelines may not be directly comparable."*
3. **ArcFace 生产化实战**：总结为"**分离特征存储与元数据管理**"——
   `特征向量层（FAISS / 内存缓存 / .npy 容灾备份）+ 元数据层（SQLite，含录入时间/版本/日志）+ 原图层`。
   并明确：小型系统用 `.npy` 足够。
4. **Face Recognition Attendance**（DeepWiki）：`SQLite(记录) + FAISS(index.bin) + pickle(元数据) + 文件系统(原图)` 四件套。

**三条可迁移的共识**：
- ✅ **元数据与向量分离存储**（我们完全一致）
- ✅ **模型版本随嵌入存储**（我们已有 `model_version` / `version.json`）
- ✅ **小规模不需要向量库**，`.npy` + 关系库就够
- ⚠️ 差异：它们的检索是"以图搜人"（纯相似度），我们是"**带时空约束的身份指派**" → 这让我们比它们**更不该**用 post-filter 的 ANN（见 §4）

---

## 3. 实测基准（远程 48 核 CPU / Xeon，768 维）

`tools/bench_vector_store.py`，两种方法、三种查询形态：

| 规模 | 方法 | 纯 KNN | 排除同机位 | 时间窗 ±3s | 建库耗时 | 磁盘 |
|---|---|---|---|---|---|---|
| **539 轨** | numpy 精确 | **0.096 ms** | — | — | — | — |
| | sqlite-vec v0.1.9 | 5.47 ms | 4.45 ms | 4.72 ms | 0.1 s | 19 MB |
| **10 万** | numpy 精确 | **17.4 ms** | 14.4 ms | **0.03 ms** ⚡ | — | 307 MB(f32) |
| | sqlite-vec | 190.6 ms | 158.6 ms | 80.9 ms | **31 s** | 325 MB |
| **100 万** | numpy 精确 | 136 ms | 1279 ms ⚠️ | — | — | 3.1 GB(f32) |
| | sqlite-vec | 未测（brute force 已不可用） | | | | |

**读法**：
- sqlite-vec 在**每个规模、每种查询**上都慢 11–57 倍（它是逐行算距离，无法用 BLAS 多线程矩阵乘）。
- **强过滤是 numpy 的碾压性优势**：候选 180 条时 **0.03 ms**（比 sqlite-vec 快 2700 倍）。这恰是 GIL 的核心查询。
- ⚠️ **10⁶ 的坑**：`E[mask]` 拷贝一次 940 ms，导致"排除机位"从 136 ms 劣化到 1279 ms。→ 解法：**按机位分块**（排除机位 = 不遍历该块，零拷贝），已实现。

**f16 存储实测**（10 万 × 768）：

| 方式 | 每查耗时 | 结论 |
|---|---|---|
| f16 落盘 → 每查转 f32 | **400 ms** | ❌ 灾难 |
| f16 落盘 → 启动转 f32 常驻 | **17.6 ms**（与 f32 同） | ✅ **采用** |
| memmap f16 → 每查转 f32 | **660 ms** | ❌ 灾难 |

→ 策略：**f16 落盘（省 50% 磁盘）+ 启动时一次性转 f32 常驻**。

---

## 4. 文献依据：为什么"约束检索"不该用 ANN

业界对"带元数据过滤的 ANN"（Filtered ANN）有共识性难点：

- **pre-filter**：过滤后子图碎片化 → HNSW 贪心搜索迷路 → **召回崩塌**；
- **post-filter**：ANN 先取 top-K 再筛 → 强选择率下**返回不足 K 甚至 0 条**；
- 定量：选择率 1% 时，候选池 40 条平均只剩 4 条，**recall@10 = 0.41**；选择率 2% 且候选 40 时 **recall@10 = 0.07**（90/300 查询返回空）。

而文献里一句关键判词：
> *"Filtering difficulty is a cost of approximation — **it dissolves under an exact scan**."*

**我们的选择率常在 0.2–3%**（时间窗 ±3s，候选 ~180/10 万）→ 正是 ANN 的"召回悬崖"区。而 10 万规模的**精确扫描只要 17 ms**。

**结论**：在本项目规模下，**精确扫描 + 预过滤**不仅更快，而且是**唯一不会掉召回的**选择。

---

## 5. 选型结论与升级触发条件

### 采用
`numpy 分块常驻 f32（检索） + SQLite 单文件（结构真相） + npz + manifest（可复现）`

### 不采用，以及"什么时候才该用"

| 候选 | 现在不用的理由 | **升级触发条件**（写死，避免过早上复杂度） |
|---|---|---|
| **sqlite-vec** | 实测慢 11–57×；我们的写入是离线批处理，用不上它的增量优势 | 若将来要求"**单条 SQL 跨表 JOIN + 向量检索**"且规模 > 10⁶ 且写入高频 |
| **FAISS / HNSW** | 10⁵ 以内暴力更快；ANN 在强过滤下掉召回 | **单机位块 > 3×10⁵** → 该块换 `faiss.IndexIVFPQ`（`pip install faiss-cpu` 可装） |
| **Redis** | 规模差 2–3 个数量级；要求单文件可复现真相；Redis 8 为 **AGPLv3**（发布需注意） | **L6 在线闭环**：多路实时流 + 跨进程低延迟共享 + 高频写 |
| **服务化（HTTP 向量服务）** | 单机单进程用不上；多进程各自 mmap 更简单 | 多消费方（webapp + worker + 训练）**并发读且频繁重建**时 |

### 许可备注（若将来发布）
- Redis 8 CE 为 **AGPLv3**（网络 copyleft）；`sqlite-vec` MIT/Apache-2.0；FAISS MIT；**SQLite 公有领域**。

---

## 6. 架构

```
output_crosscam/analysis/gil/store_v1/
  gil.sqlite              ← 全部结构化真相（一张库，一个事务）
  skipped_edges.json      ← ★ 零静默：无法导入的记录必须落盘
  manifest.json           ← 版本链：每个文件的 sha1_16 + 各表行数 + meta
  gil_vectors/
    305.npz 316.npz ...   ← 按机位分块；{"emb": f16[n,768], "ids": int64[n]}
```

### 为什么按机位分块
1. "排除同机位" = **不遍历该块**（零拷贝；避免 10⁶ 规模的 940 ms 拷贝）；
2. 冻结/重建粒度细（改一个机位的编码只重写一个块）；
3. 与项目"按机位真实分辨率/按机位校准"的既有纪律一致。

### 表清单与设计意图

| 表 | 承载 | 对应问题 |
|---|---|---|
| `obs` | 最细单元（一条 track 的一段连续出现）；**id 即向量行号，append-only** | 基础 |
| `frag` | 同机位连续片段 = **q1 遮挡接续的单位** | **Q1** |
| `entity` | 同机位已拼接实体（= 现 `net_tracks` 口径） | Q1 |
| `gperson` | ★ **全局人**（跨机位唯一身份） | **Q2 / Q3** |
| `membership` | 节点→全局人（层次聚类产物，**允许候选多归属**） | Q2 |
| `edge` | 边（`stitch` 同机位 / `relay` 跨机位 v2 / `relay_v5` / **`relay_gate`** 门户驱动通道），统一状态机 | Q1/Q3 |
| `visit` | 门户通行事件（entry/exit + 方向 + 边距） | Q3 |
| `owner` | ★ **责任时间线**（任一时刻单机位负责一目标） | **Q3** |
| `annotation` | **统一人工真值**（替代散落的 csv/json） | 全局 |
| `mutex` | 时间互斥 / 禁止同人 约束 | Q2 |
| `camera` / `gil_meta` | 相机参数（含 OSD 时间偏移）；**视图(clean2)与版本显式记录** | 防口径漂移 |

★ **Q4 的处理方式（写死在 schema 里）**：镜像/倒影**不删框**，只作为观测属性存在
`obs.flags = {"reflection": 0.8, "mirror_of": 123}` + `obs.quality`。
理由：玻璃掩膜逐帧删框已证伪（缺嵌入 1181、误删真人 7 轨、打断轨迹）；而倒影与真人
外观 cos 间隔仅 **0.0002**，外观分类不可行 → 只能靠**几何关系 + 指派**，因此必须**保住观测**。

---

## 7. 迁移结果与验证（对既有资产）

`python tools/gil_migrate.py`（**默认整库重建；重复运行结果逐位一致**）：

```
obs=546            （539 主库 + 7 条 clean2 补缺；分块 305:107 316:68 318:151 320:29 323:138 343:53）
frag=541  entity=411
同机位拼接边=130    （多 frag 实体 86 个 = 20.9% → ★ 与既有口径精确对上）
relay 边：v2=182 / v5=47 / gate=14     （gate = 门户驱动通道，见 docs/s1-305-egress-findings.md）
跳过 0 条（S1 补齐 7 条嵌入后端点全部可解析）
annotation=55      （seed + 用户判定）
不变量：向量总数 546 == obs_alive 546   ✅
```

### 数据缺口（S0 时发现 → S1 已闭环）

S0 首次迁移时 **10 条接力边因端点实体缺 ReID 嵌入被跳过**（`318-E042` 3 条、`316-E015` 2 条、
`323-E083`、`316-E027`）。根因：**ReID 主库按 clean(v1) 口径编码 539 轨，而实体图来自 clean2**；
clean2 多保留的 **7 条轨**（`305|12` `316|32` `318|53/82/98/135` `323|193`）无嵌入。

S1 用 `cradio_scale_v4_extra/`（与主库**逐位同源** cos=1.000000）补齐后：obs 539→546、
实体 404→411、v2 边 176→182、v5 边 43→47、**跳过边 10→0**。

> ★ 补源纪律：**必须用 v4 空间**（`cradio_scale_v4_extra`）；`cradio_scale/`（v5）虽覆盖全集但
> cos 只有 0.9985 —— 本项目判定阈值间隔仅 **0.0002**，混用会致命。

### 冒烟 `python tools/gil_smoke.py`：**9/9 全绿**

```
obs 数 == 向量总数 (546)              ✅
向量 id 全在 obs 中                    ✅
接力边悬空端点 = 0（回归防线）          ✅
跨机位 top-10 与裸 numpy 逐条一致 200/200 ✅  ← 正确性
Q1 跨机位检索 P50 0.162 ms / Q3 带时间窗 P50 0.060 ms（本地 546，功能验证）
gperson 时间线可查 / owner@t 可查       ✅
manifest 含 7 个文件 sha1_16           ✅
```

> 远程 48 核实测性能见 §3（本地 Windows 只是功能验证）。

---

## 8. GIL 主干落地路线（A 方案）

存储层是地基，算法层按此顺序：

| 阶段 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| **S-1** | ★ **305 出门覆盖诊断 + 门户驱动通道**（结论：305 覆盖低主因是**结构性**，非缺陷；通道独立复现全部人工 relay 命中） | 无 | **✅ 完成**（报告 `docs/s1-305-egress-findings.md`） |
| S0 | **统一存储层**（本文档） | 无 | ✅ **完成** |
| S1 | 补 7 条缺嵌入轨（§7）→ 接力边 100% 入库、跳过边归零 | S0 | ✅ **完成** |
| **P0** | 门户通道并入 relay 候选（独立 `channel=gate`，不覆盖 v2）→ 196 边 | S-1 | ✅ **完成**（`tools/merge_relay_channels.py`） |
| **P1** | ★ **C0 分母修正**（剔除"室内消失点"）→ `c0_v7`，与 v6 并列 | P0 | ✅ **完成**（`c0_baseline.py --edges v7`） |
| S2 | `gperson` = 「门户转移先验(dist_v1) + 时间互斥 + 外观」约束下的**层次聚类**（替代 mutual-best） | S0/S-1 | ✅ **完成**（报告 `docs/gil-gperson-s2-report.md`；264 gperson，人工真值 P 1.000/R 0.538） |
| S3 | Q1：关联加外观（**只信清晰帧**）+ 离线 LDP 长程关联 | S2 | **下一步**（S2 已证：外观无法判同机位碎片，需运动学/几何） |
| S4 | Q3：效用函数 → `owner=argmax(可见性)` → 时间线 → **去重视图** + 计数改 `gperson×visit` | S2 | 待做 |
| S5 | Q4：倒影改写为**指派问题**（倒影须被真人"解释"）+ 实体降权打 flag | S2 | 待做 |

### 记住三条纪律
1. **同序列 A/B**：加外观的改造必须同序列对比（文献反证：Re-ID 增益温和，部分序列会更差）；
2. **零静默**：任何跳过/丢弃都必须落盘可查（`skipped_edges.json` 是模板）；
3. **不逐帧删框**：只对实体降权/打 flag（保住轨迹）。

---

## 9. 复现命令

```bash
# 重建整库（幂等）
python tools/gil_migrate.py
# 验证
python tools/gil_smoke.py
# 查看统计/基准
python tools/gil/vectors.py --root output_crosscam/analysis/gil/store_v1/gil_vectors --bench
# 选型基准（远程）
python tools/bench_vector_store.py --real webapp/backend/data/reid/store_emb.npy --sizes 100000
```
