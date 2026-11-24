# Table I（E2 实验）时序回放与逐条查询配对得失 CSV 对账单规范

本规范与 `evaluate_e2_replay.py` 及论文初稿 `main-ieee-r5.tex` 的公式 (2)–(10)、Section IV-B/C 一一对应，确保在不泄露测试集真值的前提下，一次性产出 Table I 的六个主实验格、两项补充归因对照（固定写入换选择器、Oracle 理想写入上限）以及 E3 分层统计表。

## 1. 核心协议铁律（防泄漏与公平对照约束）

1. **严格因果时序（Pre-Write Scoring）**：在回放阶段（$t > T$），当同一时刻既有查询又有写入事件时，队列优先级强制设定为 `query (0) < write (1)`，任何查询仅针对前一时刻图库状态 $\mathcal{G}_j(t^-)$ 打分，杜绝“用刚写入的帧识别自身”。
2. **查询池与写入源彻底隔离（Track & Near-Duplicate Isolation）**：凡包含可用配对人脸的回放轨迹一律划入写入机会池（每条单机位轨迹至多取人脸质量 $q_t$ 最高的 1 帧）；主评测查询池 `eligible_queries` 严格限定为**全轨迹无可用人脸**，且剔除与同机位写入轨迹时间间隔小于 `near_dup_isolation_sec`（默认 `15 s`）的紧邻切分片段。
3. **跨条件分母锁定**：全部 `2 × 5 = 10` 组条件（2 种历史建库 × 5 种运行策略）共享完全相同的 `query_id` 集合、相同的写入人脸观测机会与相同的单身份模板容量上限 $M$。某次人脸观测落在不确定区间 $[\tau_{\mathrm{new}}, \tau_{\mathrm{face}})$ 未能触发写入时，仍在 `ledger_e2_writes.csv` 记为 `unresolved_band` 写入机会，不从分母中剔除。
4. **建库期固定真值映射（Enrollment-Fixed GT Mapping）**：匿名图库标签到人工参考身份的映射在截止时刻 $t = T$ 根据建库模板纯度（默认 `gt_purity_threshold = 0.80`）一次性锁定为 `clean`、`mixed` 或 `unresolved`；若查询命中 `mixed` 混杂簇，单列记为 `mixed_group_error`（归入 Table I 的 `wrong` 计数），绝不在回放期动态涂改身份真值。

## 2. 输入数据规范（`observations.csv` + `embeddings.npz`）

评测脚本接收一个元数据表 `observations.csv` 与一个特征压缩包 `embeddings.npz`（包含浮点矩阵 `body_embs` 形状 `[N_b, 512]` 及可选 `face_embs` 形状 `[N_f, D_f]`，脚本加载时自动做 $\ell_2$ 归一化）。

| 字段名 | 类型 | 必填 | 对应公式与物理含义 |
| :--- | :--- | :--- | :--- |
| `obs_id` | `str` | 是 | 单张人体裁剪框全局唯一 ID |
| `track_id` | `str` | 是 | 单机位 ByteTrack 轨迹 ID $\ell_n$ |
| `camera_id` | `str` | 是 | 摄像头流标识 $c_i$ |
| `timestamp` | `float` | 是 | 采集时间戳 $t_i$（秒；$t_i \le T$ 进入历史建库，$t_i > T$ 进入回放） |
| `bbox_h` | `float` | 是 | 人体检测框像素高度 $h_i > 0$，用于公式 (2) 门控与公式 (7) 尺度加权 |
| `det_score` | `float` | 是 | 人体检测置信度 $s_n$，对应公式 (2) 阈值 $\eta_{\mathrm{det}}$ |
| `max_concurrent_iou` | `float` | 是 | 同帧同机位其他人体框最大 IoU，对应公式 (2) 遮挡阈值 $\gamma_{\mathrm{occ}}$ |
| `border_margin_px` | `float` | 是 | 检测框距画面四边最小像素边距，对应公式 (2) 截断门控 $\beta_{\mathrm{edge}}$ |
| `body_emb_idx` | `int` | 是 | 对应 `embeddings.npz['body_embs']` 的行下标 |
| `has_raw_face` | `bool` | 否 | 当前帧是否检出候选人脸（`0` 或 `1`） |
| `face_conf` | `float` | 否 | 人脸检测置信度，对应 $\eta_{\mathrm{face}}$ |
| `face_quality` | `float` | 否 | 人脸清晰度/关键点质量分 $q_m$，对应 $q_{\min}$ |
| `face_iof` | `float` | 否 | 人脸框落入当前人体框的面积占比 $\mathrm{Area}(f_m \cap b_i)/\mathrm{Area}(f_m) \ge \gamma_{\mathrm{in}}$ |
| `face_rel_y` | `float` | 否 | 人脸中心在人体框内的相对纵向位置（`0` 为框顶，需 $\le \alpha_{\mathrm{head}}$） |
| `face_overlaps_other_body` | `bool` | 否 | 该人脸是否同时与同帧另一人体框相交（`1` 则按 M2 规则直接丢弃） |
| `face_emb_idx` | `int` | 否 | 对应 `embeddings.npz['face_embs']` 的行下标（无脸填 `-1`） |
| `query_id` | `str` | 否 | 主评测查询标识（历史帧或非评测背景帧留空） |
| `population_type` | `str` | 否 | 取值 `in_gallery`（库内可回答）、`absent_from_gallery`（库外人）或 `non_person`（非人误检） |
| `gt_person_id` | `str` | 否 | 独立人工标注的真实人员 ID（仅评测阶段对账使用，算法全程不可见） |
| `outfit_id` | `str` | 否 | 独立核实的衣着标识（如 `coat1`、`coat2`），用于自动派生 E3 换装分层 |
| `posture_label` | `str` | 否 | 独立核实的姿态标签（`standing`、`sitting`、`unknown`），用于 E3 姿态分层 |

## 3. 输出对账单规范（六份标准产物）

运行 `evaluate_e2_replay.py` 后，输出目录自动生成以下文件，可直接支撑正文填表与补充材料溯源：

### 3.1 逐条查询主账本 `ledger_e2_queries.csv`

对标原稿四机位审查的 `ledger_query_42.csv`，记录每个条件在每条查询上的原始打分与判定明细。

| 字段名 | 取值范围 / 示例 | 说明 |
| :--- | :--- | :--- |
| `condition` | `face_linked__proposed` | `<historical_build>__<replay_rule>` 组合键 |
| `historical_build` | `body_only` / `face_linked` | 对应 Table I 的两列历史建库方式 |
| `replay_rule` | `frozen` / `geff_max` / `proposed` / `proposed_max_sel` / `oracle_proposed` | 对应 Table I 的三行运行策略及两项归因消融 |
| `query_id` | `Q_001` | 跨条件严格对齐的查询主键 |
| `population_type` | `in_gallery` / `absent_from_gallery` / `non_person` | 区分库内主指标分母与开放集误接受分母 |
| `gt_person_id` | `P_03` / `NON_PERSON` | 独立真实身份标签（用于事后核对与按人聚类自助法） |
| `is_cross_camera` | `yes` / `no` / `na` | 该人员此前在建库期是否从未在该 `camera_id` 出现 |
| `is_clothes_changed` | `yes` / `no` / `unknown` | 查询衣着 `outfit_id` 是否异于建库期已登记衣着 |
| `posture_label` | `standing` / `sitting` / `unknown` | 独立姿态分层标签（不以框高代替真值标签） |
| `predicted_anon_id` | `J_0003` / `REFUSED` | 公式 (9) 输出的匿名图库 ID 或拒识标记 |
| `mapped_gt_person_id` | `P_03` / `<mixed>` / `REFUSED` | 由建库期锁定映射表转换出的参考身份 |
| `top1_score` / `top2_score` | `-1.0` 至 `1.0` | 最优与次优候选身份得分 $s_{(1)}, s_{(2)}$ |
| `score_margin` | `float` | 裕度 $s_{(1)} - s_{(2)}$，用于核查 $\delta_{\mathrm{body}}$ 门控 |
| `verdict` | `correct` / `wrong_id` / `mixed_group_error` / `refused` / `false_accept_absent` / `false_accept_nonperson` / `correct_refusal` | 细粒度裁定结果 |

### 3.2 成对得失流转明细 `ledger_e2_paired_diffs.csv` 与汇总表 `summary_e2_paired_gains.csv`

对任意基准配置 $A$ 与对照配置 $B$（例如 `face_linked__frozen` 对比 `face_linked__proposed`），按 `query_id` 一一配对拆解净变化：

| 汇总字段名 | 计算逻辑与统计意义 |
| :--- | :--- |
| `newly_correct` | 原来在 $A$ 中为 `wrong_id`、`mixed_group_error` 或 `refused`，在 $B$ 中变为 `correct` 的条数（救回数） |
| `ceased_correct` | 原来在 $A$ 中为 `correct`，在 $B$ 中退化为错误或拒识的条数（新增损失数） |
| `net_change` | 恒等于 `newly_correct - ceased_correct`，防止净增益掩盖类似四机位审查中 `-28` 误拒的退化 |
| `refused_to_correct` / `wrong_to_correct` | 将救回来源拆分为“从拒识救回”与“从错认救回” |
| `correct_to_refused` / `correct_to_wrong` | 将新增损失拆分为“过度保守变拒识”与“模板污染致错认” |
| `person_bootstrap_ci95_lo` / `hi` | 以 `gt_person_id` 为聚类抽样单元做 2,000 次重采样所得 `net_change` 的 95% 置信区间 |

### 3.3 写入机会与归属审计表 `ledger_e2_writes.csv` 及分层表 `summary_e3_strata.csv`

- `ledger_e2_writes.csv`：逐条记录回放期每个有脸轨迹的匹配分 $s_{j_1}^{\mathrm{face}}$、裕度、最终状态（`written`、`near_dup_refreshed`、`unresolved_band`、`staged_new`、`spawned_new`）及写入身份归属是否正确（`is_attribution_correct`），直接提供类似 D2“写入 19 条、正确 18 条”的精确统计。
- `table1_e2_cells.csv` 与 `table1_e2_populated.tex`：直接生成填好 `correct / wrong / refused` 三元组的 `IEEEtran` LaTeX 表格源码。
- `summary_e3_strata.csv`：按 `cross_camera`、`clothes_changed`、`posture` 三个正交维度输出各子层的查询数、独立人数与得失分布，直接支撑 E3。

## 4. 运行方式

```bash
# 1. 运行内置多机位换装/坐姿合成数据自检，验证全部公式与输出表结构
python3 evaluate_e2_replay.py --selftest --out-dir ./e2_selftest_out

# 2. 在真实特征与标注集上正式运行 E2 评测并生成 Table I LaTeX 源码
python3 evaluate_e2_replay.py \
  --observations-csv ./data/site_observations.csv \
  --embeddings-npz ./data/site_embeddings.npz \
  --config-json ./data/frozen_e2_config.json \
  --out-dir ./results/e2_forward_replay
```
