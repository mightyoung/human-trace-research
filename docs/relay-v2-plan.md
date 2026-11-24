# 跨摄像头责任目标转移改造方案（Relay v2 实施规划）

日期：2026-10-14 ｜ 状态：**S0–S4 + S6 已完成**（实施报告：`docs/relay-v2-s1s2-report.md`、`docs/relay-v2-s3s4-report.md`、`docs/relay-v2-s6-report.md`；S5 闭环后置）｜ 上游：`docs/cam-handover-neighbor-research.md`（近邻研究+设计方向）、`docs/cam-handover-audit.md`（现状盘点）
纪律依据：`docs/research-agenda/evaluation-protocol.md`（TG 规则）、charter H2 三限定

---

## 0. 结论先行

1. **改造范围**：5 个功能点（F1 多对多匹配 / F2 过渡时间分布 / F3 handover 指标 / F4 状态机 / F5 闭环纲要），**全部增量式**——现有 47 边/32 链生产链逐位兼容，新能力隔离在新产物目录 `output_crosscam/analysis/relay_v2/`。
2. **涉及模块**：改造 3 个既有文件（`c0_baseline.py` 加 v6 分支、`portal_truth_candidates.py` 加新列、`build_web_data.py` 加数据源开关）+ 新建 4 个脚本 + 1 个对账脚本；**不改** `crosscam_relay.py` / `relay_rescore.py` / `relay_triage.py` 的默认行为。
3. **约束条件**：口径冻结（clean2 + 统一 OSD 时间轴 + 事件级定义）、TG-05 留出纪律（分布拟合与评估分离）、primary 47 边逐位一致、本地 CPU 可跑（无 GPU 需求）、241 判定为 L3 验证前置。
4. **验证方式**：四级门（L0 单元自检 → L1 回归逐位一致 → L2 19 对账 → L3 241 真值全表），每级有明确 Go/No-Go 判据；**S1 的 Go 标准 = 恢复 ≥3/5 漏接（排他竞争 3 对必须恢复）且已判异人 2 对不进 active**。
5. **排期**：S0–S4 五步（S5 闭环后置）；S1（F1）与 S2（F2）为主干，S3/S4 可并行。

---

## 1. 目标范围

### 1.1 In scope（本次改造什么）

| # | 功能点 | 一句话 | 优先级 |
|---|---|---|---|
| F1 | 预算约束多对多匹配 | mutual-best 之外新增"扩展分配"遍：被排掉的高分候选以受控多对多进入审计池（secondary 边） | P0 |
| F2 | per-direction 过渡时间分布 + 联合评分 | 用裁定真值拟合非参分布（Parzen），候选分改为 `sim + λ·log w_dt`，候选窗按分布自适应 | P0 |
| F3 | handover 评估协议 | 边级 FP/FN + crossing/returning 分类 + 覆盖率，直接消费人工判定 CSV | P1 |
| F4 | 图边状态机 + 延迟提交 | 边证据分 + 四态生命周期（candidate→active→suspect→dormant），pending/secondary 走晋升通道 | P1 |
| F5 | 选择器↔relay 闭环 | 两条单向数据隔离通道的**设计纲要**（实施后置，本方案只锁定接口与纪律） | P2 |

### 1.2 Out of scope（明确不改什么）

- **不改** clean2 实体图/版本链（`stitch_clean2`、`relay_clean2` 只读消费）；
- **不改**选择器线任何已交付资产（`selector_v2.npz`、`selector.py` 接口冻结）；
- **不改**意图预测线；
- **不碰** 305 出门 0.000 的三因拆分（时钟/外观/真无人——独立任务，见审计报告 P0，本方案不耦合）；
- **不引入**标定/几何/深度依赖（三限定不变）；
- **不重跑**检测与嵌入（复用 `cradio_scale_v4c` 现有嵌入）。

### 1.3 前置依赖

| 依赖 | 现状 | 影响 |
|---|---|---|
| 343×316 人工判定 | 19/241 | L2 对账可用 19 对；L3 全表需 241 完成 |
| 真值标注页 | 已就绪（webapp `#/truth`） | 判定可持续补充 |
| 现有嵌入 | `cradio_scale_v4c`（6 相机 npz） | F1/F2 直接复用，无需重算 |

---

## 2. 当前系统基线（改造前的准确快照）

### 2.1 生产链 7 环节与模块映射

```
[L1 事件]  crosscam_relay.py events
           → relay_clean2/events_{6 路}.json（411 事件；字段 eid/t_first/t_last/u0/v0/u1/v1/h_med/moving）
[L2 候选]  crosscam_relay.py links  --dt-min -1 --dt-max 60 --only-moving
           → relay_candidates.json（2618 对；字段 a/b/dir/dt/a_h/b_h/a_end/b_start/a_tlast/b_tfirst）
[L3 打分]  relay_rescore.py score --emb-dir cradio_scale_v4c --emb-pat "{cam}_emb_scale.npz" --tag cradio_v5 --sim-thr 0.943
           → relay_scored_cradio_v5.json（末碎片⇄首碎片 cos；n_missing=0）
[L4 筛选]  relay_triage.py --scored ...（mutual_best_flags：全池上 a/b 双 best）
           → relay_triage_cradio_v5.json（confirmed 47 / pending 13 / rejected 2558）
[L5 唯一化+链] relay_rescore.py mutual（sim≥0.943 → global mutual-best → 链拼接）
           → relay_mutual_global_cradio_v5.json + relay_chains_cradio_v5.json（47 边/32 链）
[L6 度量]  c0_baseline.py（load_edges "v5" 读 chains；entities 读 stitch_clean2）
           → c0_baseline_v5.json（exit continuity 0.301 / entry attribution 0.286）
[展示]     webapp/dataprep/build_web_data.py（读 chains + mutual）→ chains.json → 控制台
[真值]     portal_truth_candidates.py（读 chains v5 生成 in_relay_v5 列）→ truth/*.csv → webapp #/truth 标注 → labels.json/CSV
```

### 2.2 关键参数（现状值，改造须对齐）

| 参数 | 现值 | 位置 |
|---|---|---|
| dt 窗 | `[-1.0, 60.0]s`（全局固定） | `crosscam_relay.py links` |
| 生产 sim 阈值 | `0.943`（hi）/ `0.9`（lo） | `relay_triage.py` |
| 唯一化 | global mutual-best（出/入度≤1） | `relay_rescore.py do_mutual` / `relay_triage.py` |
| 时间基准 | T0 偏移 {305:-5, 316:+4, 318:+20, 320:+19, 323:+18, 343:+19}，FPS=25 | `crosscam_relay.py` |
| 实体口径 | `DET_VIEW=clean2` → `stitch_clean2/net_tracks_{cam}.json` | `det_view.net_dir()` |
| 本轮新锚点 | 高 sim 淘汰 132/179；确认边 dt p25=1.2s/med=4.8s/max=59s | `relay_v2/anchor_stats.json` |

### 2.3 数据结构（真值侧，供 F3/F4 消费）

- `truth/pair_candidates.csv`：`pair_id, cam_a, event_a, t_a, cam_b, event_b, t_b, dt, in_relay_v5, thumb_a, thumb_b, human_same_person, human_confidence, human_note`
- `webapp/backend/data/labels.json`：`{version, pair_decisions, mirror_decisions, ...}`（标注落盘位置）

---

## 3. 涉及模块清单

### 3.1 改造的既有模块

| 文件 | 改动 | 兼容策略 |
|---|---|---|
| `tools/c0_baseline.py` | `load_edges()` 增加 `"v6"` 分支：读 `relay_v2/edges_v2.json`（primary=active 集合）；其余逻辑不动 | 默认 tag 仍 `v5`；`--tag v6` 显式选 |
| `tools/portal_truth_candidates.py` | `pair_candidates.csv` 增加列 `in_relay_v2`（与 `in_relay_v5` 并列）；旧列保留 | 列只增不减；旧 CSV 仍可读 |
| `webapp/dataprep/build_web_data.py` | 数据源参数化：`SOURCES = {"v5": relay_chains_cradio_v5.json, "v2": relay_v2/edges_v2.json}`；default 仍 `v5` | 默认不变；切换需显式传参 |
| `tools/relay_triage.py` | **不动**（其 `mutual_best_flags` 作为 F1 的 primary 基准直接复用进口） | — |
| `tools/relay_rescore.py` | **不动**（do_mutual 逻辑作为回归基准） | — |
| `tools/crosscam_relay.py` | **不动**（F2 的窗模式若启用，通过新脚本产出替换候选 json 路径，不改原 CLI 默认） | — |

### 3.2 新建模块

| 文件 | 职责 | 输入 → 输出 |
|---|---|---|
| `tools/relay_expand.py` | **F1**：primary（复用 triage 逻辑）+ secondary 扩展分配（预算约束） | `relay_scored_cradio_v5.json` + 对账真值 → `relay_v2/edges_v2.json`（primary + secondary/audit） |
| `tools/relay_transition_dist.py` | **F2**：Parzen 分布拟合 + 联合评分 + 自适应窗 | `relay_scored` + 判定真值（留出切分） → `relay_v2/dist_v1.json`（分布+λ+窗） + `relay_v2/scored_v2.json`（联合分） |
| `tools/handover_metrics.py` | **F3**：边级 FP/FN、crossing/returning、覆盖率、对账 | `edges_v2/v5` + `pair_candidates_labeled.csv` → `relay_v2/metrics_v1.json` + 可读报告 |
| `tools/relay_states.py` | **F4**：证据分 + 四态状态机 + 晋升/降级 | `edges_v2.json` + 复核记录 → `relay_v2/edge_states.json` |
| `tools/relay_v2_selftest.py` | **L0**：合成案例自检（一人多次通行/毫厘差/回穿三类构造） | 内置 → 断言通过/失败码 |
| `tools/relay_v2_reconcile.py` | **L2**：19 对账（可扩展 241）——v5 vs v2 的逐对差异表 | 判定 CSV + edges → `relay_v2/reconcile.json` |

> 全部新脚本遵循：纯 CPU（numpy/scipy，本地 venv）、无远程依赖、输出 JSON（ensure_ascii=False, indent=1）、CLI 参数有默认值且默认不改变既有产物。

### 3.3 不在本方案内但受影响的消费端

- **webapp 展示**：F4 后可选展示边状态角标（二次迭代；本方案只预留 `edges_v2.json` 的 `status` 字段）；
- **报告/论文**：F3 的指标表直接对接 `docs/evaluation-protocol.md` 的"主约束：误接率"。

---

## 4. 约束条件

### 4.1 兼容性硬约束（回归不可破）

| # | 约束 | 验证方式 |
|---|---|---|
| C1 | primary 47 边与 v5**逐位一致**（a/b/dir/dt/sim 全同） | `relay_v2_reconcile.py` 断言 |
| C2 | `relay_clean2/` 与 `stitch_clean2/` 产物**只读**，不重写 | 文件哈希清单比对 |
| C3 | 既有脚本默认 CLI 行为不变（跑不带新参数的旧命令，产物字节一致） | 重跑对照 |
| C4 | `c0_baseline_v5.json` 数字不漂移（重跑 diff=0） | 重跑对照 |
| C5 | webapp 默认数据源仍 v5（控制台数字不变） | 冒烟对比 |

### 4.2 方法论约束（TG 规则，不可绕过）

| 规则 | 在本方案中的落点 |
|---|---|
| **TG-05 留出验证** | 分布拟合（F2）与评估（F3）的样本必须分离：拟合集 = 判定真值的"训练时段"；评估集 = 留出对。primary 边若用于拟合，则该边的效果必须从评估中剔除（在 `dist_v1.json` 中记录 `fit_provenance`） |
| **TG-06 dormant 不判死** | F4：无流量边降置信休眠，不删除；保留再激活路径 |
| **TG-08 防闭环** | F5 闭环三向数据隔离（训练/更新/验证）；F1 的 secondary 边**不直接进生产链**（只进审计池），防止早期错误自我强化 |
| **TG-02 三类关系分开** | F1/F2 只处理"门户转移关系"；不带入视觉重叠（318×320 走 stitch 支路）与物理可达假设 |
| **禁止自证** | 同一批已裁定对不得同时用于：拟合分布、调 λ、又证明效果。`dist_v1.json` 与 `metrics_v1.json` 的样本清单必须显式列出去重逻辑 |
| **阈值侧纪律** | K_a/K_b/λ/高 cos 冲突阈值一律在**留出侧**校准，测试集只读 |

### 4.3 数据依赖约束

- **241 判定**：L3 全表的前置；F1 的 Go 判据（L2）先用 19 对 — 判定在持续补充中（webapp `#/truth`）；
- **判定质量**：18 个"存疑"对不得作为正/负样本用于自动校准（只作审计线索）；
- **多判冲突**：E010/E042 类"一 event 多判"必须先经互斥裁决（webapp 已有 ⚠ 角标）才能入对账集。

### 4.4 资源与工程约束

- 全链本地 CPU（venv `C:/Users/might/.workbuddy/binaries/python/envs/default`）；预计单脚本 <10s（2618 候选规模）；
- **C 盘空间 97%**（曾瞬时 ENOSPC）——所有新产物写 `D:/dev/human-count` 工作区，不碰 C 盘；开工前查空间；
- 时间基准沿用现有 T0 偏移与 OSD 口径，**不重估时钟**（305 时钟问题独立任务）。

---

## 5. 功能点改造清单（核心）

### F1 预算约束多对多匹配（P0）

**现状**：`relay_triage.py:mutual_best_flags` + `relay_rescore.py:do_mutual` 双实现 mutual-best；179 高 sim 中 132 被淘汰；53 exit 多候选竞争；判定样本 5 漏接中 3 对根因于此。

**改造**（`tools/relay_expand.py`）：
```
输入：relay_scored_cradio_v5.json（2618 候选，含 sim/dt/dir）
Step 1 primary：直接复用 relay_triage.mutual_best_flags（import，不重写）
   → 47 条 primary（与 v5 逐位一致，C1 断言）
Step 2 候选池 E = {被 primary 淘汰的候选}，过滤 sim ≥ 0.943
Step 3 secondary 分配（预算约束）：
   约束：K_a ≤ 3（同一 exit 最多 3 条 secondary）
        K_b ≤ 2（同一 entry 最多 2 条 secondary）
   目标：在预算内最大化 Σ sim（加权二部图 k-匹配；规模小，贪心+匈牙利即可）
   特殊规则：sim ≥ 0.95 且被预算挤出的 → 标记 conflict_group（不淘汰，显式冲突）
Step 4 输出 edges_v2.json：
   每条边 {a,b,dir,dt,sim, edge_type: primary|secondary,
           budget_rank, status: active|audit, conflict_group?}
   primary → status=active（生产链）；secondary → status=audit（审计池）
```
**接口变化**：新产物 `relay_v2/edges_v2.json`（结构见 §附 A）；不改任何旧接口。
**兼容性**：primary 只在输出侧复制（不动 v5 文件）；`c0_baseline.py --tag v6` 消费时默认只计 active（=primary），`--include-audit` 才并入 secondary 试算。
**验证**：L0 合成（构造 1 对 3 通行 + 毫厘差场景断言恢复）+ L1（C1）+ L2（19 对账）。

### F2 per-direction 过渡时间分布 + 联合评分（P0）

**现状**：dt 全局硬窗 [-1,60]s；确认边 dt 双模态（p25=1.2s/med=4.8s/max=59s）；候选人排序纯 cos。

**改造**（`tools/relay_transition_dist.py`）：
```
Step 1 样本集（显式 provenance）：
   fit 集 = 判定"同人"对（19 对中 9 对，留出 2~3 对作 L2）+ primary 边的留出时段子集
   （按时间切：t < 480s 的边入 fit，其余留出——记录于 dist_v1.json.fit_provenance）
Step 2 层级 fallback 估计（Parzen 高斯核，带宽 h=5s）：
   gate-pair（样本≥20）→ direction（≥20）→ 全局（兜底均匀）
   输出：每个层级 f(δ) 的离散 CDF + P5/P50/P95
Step 3 联合评分：
   w_dt = max(f_level(dt), ε) / max(f_level)      # 归一化 [ε,1]
   score = sim + λ · log(w_dt)                    # λ 留出侧网格校准（0~0.08）
   → 全部 2618 候选重排序输出 scored_v2.json
Step 4 自适应窗（Cho 公式，供候选生成可选）：
   [T_L,T_U] = P5~P95；T = (T_U−T_L)/(1−E)，E=分布拟合残差率
   输出 dist_v1.json.window_{dir}（默认关：只写文件不启用到 L2 生成）
```
**接口变化**：新产物 `dist_v1.json` + `scored_v2.json`；F1 的 Step 2 过滤线可引用 scored_v2 的 `score ≥ 阈值`（版本化）。
**兼容性**：原 scored/links 不动；窗模式通过新脚本产出的候选路径传递（不改 `crosscam_relay.py` CLI）。
**验证**：L0（已知分布的合成恢复测试：给一组 Gamma 混合样本，断言 Parzen 估计 P50 偏差 <10%）+ L1（不改变已确认边排序前 47 的相对关系——**仅当不引入排序翻转时**；否则记录变更清单）+ L2（0.0006 毫厘差对是否升入 top2）。

### F3 handover 评估协议（P1）

**现状**：continuity/attribution（事件级 2 项）；无边级 FP/FN；无 crossing/returning 分类。

**改造**（`tools/handover_metrics.py`）：
```
输入：edges（v5 或 v2）+ pair_candidates_labeled.csv（判定）
输出 metrics_v1.json：
  ① 边级 FN = 判定同人但无边 / 判定同人对        （当前 5/9）
  ② 边级 FP = 有 active 边但判定异人 / 有边且已判对（当前样本不足，标注 n）
  ③ crossing/returning 近似分类：
     crossing = FP 边与另一条边冲突（同 a 不同 b 都被接受）
     returning = 同一实体被两条以上边回联（重复计数）
  ④ 覆盖率重算（active 集合口径，与 c0 v6 对齐）
  ⑤ EM−ES：**事件级近似暂缓**（需帧级真值；列出公式与差距，不硬算）
  ⑥ 人工抽检支持：输出 top-10 待复核边清单（供 truth 页逐条过）
```
**接口变化**：纯新增脚本；`pair_candidates_labeled.csv` 只读。
**验证**：L2（对 19 对逐对可复算，与人工记录一致）+ 复杂样本人工抽检 10 条。

### F4 图边状态机 + 延迟提交（P1）

**现状**：TG-10 状态机设计就绪未运行；pending 13 无晋升路径。

**改造**（`tools/relay_states.py`）：
```
状态文件 edge_states.json：{key: {state, evidence: [{src, val, t}], E_score, last_change}}
四态：candidate（默认）→ active（primary 或复核通过）→ suspect（冲突/反证）→ dormant（长期无支持）
证据分 E（Sentinel 式正下界）：
   E0 = 0.5；新支持 +0.10；新冲突 −0.15；clamp [0.05, 1.0]（永不为 0：保持最小警戒）
   支持来源：人工复核通过 / 链一致性（第三机位）/ 重复通行模式统计
   冲突来源：人工判异人 / 与其它边互斥冲突（webapp ⚠ 角标对接）
状态迁移：
   candidate → active：primary 直接进；secondary/pending 需 E ≥ 0.7 且无未决冲突
   active → suspect：冲突入账 或 人工判异人
   suspect → dormant：审计过期（默认 2 周）无复验
   active → dormant：无流量 + 衰减（TG-06：不判死，保留再激活）
```
**接口变化**：新产物 `edge_states.json`；webapp 可选角标（二次迭代）。
**验证**：L0（构造 pending 升级/降级/冲突三类场景断言）+ 与 truth 页"互斥冲突"角标的对账。

### F5 选择器↔relay 闭环（P2，纲要）

**只锁定接口与纪律，本方案不实施**：
- 通道 1（relay→选择器）：复核通过边产出的跨机同人对 → 下一轮选择器更新候选（不进当前轮评估）；
- 通道 2（选择器→relay）：`selector_v2.npz` 的 `predict(..., margin="prec")` 输出作为边的第三方证据源（F4 的 evidence 项之一）；
- 纪律：三向数据隔离 + 轮换（每次闭环消耗的样本从验证池剔除）；失败判定：闭环后 F3 指标恶化 >2pp → 回滚闭环通道。

---

## 6. 兼容性处理汇总

| 层 | 处理 |
|---|---|
| **产物层** | 新目录 `relay_v2/`；v5/v4/v3 产物只读；`edges_v2.json` 携带 `based_on: relay_scored_cradio_v5.json` 与输入哈希（版本链惯例） |
| **代码层** | 既有脚本零行为变更（C3/C4 回归）；新逻辑全部在新脚本；跨脚本复用通过 import（如 `relay_expand.py` import `relay_triage.mutual_best_flags`） |
| **消费端** | `c0_baseline.py` 加 v6 分支（opt-in）；`build_web_data.py` 数据源参数化（default v5）；`portal_truth_candidates.py` 加列（旧列不动）→ **✅ 三项全部完成（S3/S4 报告 §5；v5 回归零漂移，v2/v6 预览验证通过）** |
| **数据层** | 判定 CSV 只读；labels.json 结构不变（F4 的复核记录用新文件，避免污染既有 schema） |
| **回退** | 任一功能点可独立回退：删除 `relay_v2/` 对应产物 + 还原参数即回到 v5 行为（c0/webapp 默认即 v5） |

---

## 7. 验证方式（四级门）

### 7.1 L0 单元自检（`tools/relay_v2_selftest.py`）

| 案例 | 构造 | 断言 |
|---|---|---|
| 一人多次通行 | 1 个 a 端 + 3 个同人 b 端（0.96/0.95/0.94）+ 竞争干扰 | F1 恢复全部 3 条（budget=3 内） |
| 毫厘差 | 两条边 0.9520 vs 0.9526 竞争同一 b 端 | F2 联合分下位置合理者胜出且低者入 conflict_group |
| 回穿 | 同一 b 端被两条高 sim 边回联 | F3 标记 returning 候选；F1 K_b=2 拦截第三条 |
| 分布恢复 | Gamma(2,8) 合成 dt 样本 200 条 | Parzen P50 相对偏差 <10% |
| 状态机 | candidate + 2 支持 → active；+ 冲突 → suspect | 迁移与 E 值断言 |

### 7.2 L1 回归（每次运行强制）

- C1：primary 47 边与 v5 逐位一致（diff=0）；
- C3：`relay_triage.py --scored relay_clean2/relay_scored_cradio_v5.json --out ...` 输出与既有 `relay_triage_cradio_v5.json` 一致；
- C4：`c0_baseline.py --tag v5` 重跑数字与 `c0_baseline_v5.json` 一致。

### 7.3 L2 对账（19 对判定 → 可扩展）

`tools/relay_v2_reconcile.py` 输出逐对差异表：

| 判据 | Go | No-Go → 行动 |
|---|---|---|
| 漏接恢复 | **≥3/5**（排他竞争 3 对 E042↔E046/E048、E038↔E043 必须恢复；毫厘差 1 对期待恢复） | <3 → 检查预算/联合分实现 |
| 误纳入 | 2 对已判异人**不得**进 active（进 audit 可接受） | 违例 → 收紧 sim 线/预算 |
| 候选漏 1 对（E010↔E014） | 期望被 F2 窗修复（若根因是窗截断） | 未修复 → 保留为独立根因线索，不阻塞 |

### 7.4 L3 真值全表（241 判定完成后）

| 指标 | 目标 | 备注 |
|---|---|---|
| 边级 FN | 相对 v5 下降 ≥30% | 主目标 |
| 边级 FP | ≤5%（有边且判异人占比） | 主约束（对齐 evaluation-protocol） |
| exit continuity（c0 v6 含 audit 试算） | 0.301 → 0.35–0.45（预期） | 数字口径与 c0 对齐 |
| 一致性 | v5/v2 差异全部可解释（每条新增边有 provenance） | 审计就绪 |

### 7.5 验证纪律

- 每次改动**先跑 L0+L1**（秒级）再跑 L2（分钟级）；
- 全部验证产物落 `relay_v2/`（含 `selftest.log`、`reconcile.json`），可复算；
- 判定集更新后 L2/L3 一键重跑（脚本幂等）。

---

## 8. 实施排期与里程碑

| 步 | 内容 | 产出 | 完成判据 |
|---|---|---|---|
| **S0** | 目录与骨架：`relay_v2/` 约定、自检脚本框架、对账脚本框架 | `relay_v2_selftest.py`、`relay_v2_reconcile.py` 跑通空数据 | 空跑退出码 0 |
| **S1** | F1 多对多扩展（核心） | `relay_expand.py`、`edges_v2.json` | L0 全过 + L1 C1 + L2 漏接恢复 ≥3/5 |
| **S2** | F2 分布+联合分 | `relay_transition_dist.py`、`dist_v1.json`、`scored_v2.json` | L0 分布恢复 + L2 毫厘差对改善 |
| **S3** | F3 指标 | `handover_metrics.py`、`metrics_v1.json` | 19 对逐对可复算 → **✅ 完成（S3/S4 报告 §1；v5 自证 0.301/0.286 逐位复现）** |
| **S4** | F4 状态机 | `relay_states.py`、`edge_states.json` | L0 状态迁移断言 → **✅ 完成（S3/S4 报告 §2；9 断言全过、与 L2 交叉验证 ✓）** |
| **S6** | 弱事件候选通道 + 预算挤出显式化（55 判增补，计划外增量） | `relay_weak_candidates.py`、`weak_scored.json`；edges/states/metrics/labels 全链更新 | L0 锚点+断言 / L1 零漂移 / L2 恢复 5/5、异人零混入 → **✅ Go（S6 报告）** |
| **S5** | F5 闭环（后置） | 设计→实施（另行排期） | 由 H1/H2 实验计划驱动（未开始） |

依赖图：S0 → S1 → S2（S2 的 fit 需要 S1 的对账产出）→ S3/S4（可并行）→ S5。
S1 不依赖 241 判定（用 19 对即可启动）；L3 全表在判定完成后补跑。

---

## 9. 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|
| 多对多预算放宽 → FP 上升 | 中 | 误接率约束被破 | K 保守起步（3/2）+ 高 cos 冲突显式化 + audit 隔离（不进生产链）+ L2 硬门 |
| 分布样本少（9 对 + 留出边）→ 估计不稳 | 高 | 联合分误导排序 | 层级 fallback + 均匀兜底 + λ 上限 0.08 + 全程 provenance |
| "多判冲突"（E010/E042）未裁决污染对账 | 中 | L2 判据失真 | 先过互斥裁决（webapp ⚠）再入对账集；冲突对单独分组报告 |
| 判定补充缓慢（19→241） | 中 | L3 延迟 | S1–S4 全链不依赖 241；L2 判据先行为准 |
| C 盘空间 97% | 中 | 运行失败 | 全部产物在 D 盘工作区；开工前查空间 |
| 早期错误自我强化（TG-08） | 低（已设防护） | 闭环放大 | secondary 只进 audit；S5 三向隔离；保留探索通道 |

---

## 附 A：`edges_v2.json` 结构草案（版本 v1）

```json
{
  "version": "relay_v2/edges/1",
  "based_on": {"scored": "relay_clean2/relay_scored_cradio_v5.json",
                "hash": "<sha256 前 16>"},
  "params": {"sim_thr": 0.943, "K_a": 3, "K_b": 2, "conflict_sim": 0.95, "lambda": 0.05},
  "edges": [
    {"a": "316:316-E059", "b": "318:318-E099", "dir": "316->318", "dt": 1.4,
     "sim": 0.9826, "edge_type": "primary", "budget_rank": 0,
     "w_dt": 0.92, "score": 0.9871, "status": "active"},
    {"a": "343:343-E042", "b": "316:316-E048", "dir": "343->316", "dt": 41.6,
     "sim": 0.965, "edge_type": "secondary", "budget_rank": 1,
     "w_dt": 0.31, "score": 0.9492, "status": "audit", "conflict_group": "G042"}
  ],
  "summary": {"n_primary": 47, "n_secondary": "?", "n_conflict_groups": "?"}
}
```

## 附 B：本方案与上游文档的对应关系

| 上游点 | 本方案落点 |
|---|---|
| 近邻研究 §4.1（P0-1 多对多） | F1（§5） |
| 近邻研究 §4.2（P0-2 分布） | F2（§5） |
| 近邻研究 §4.3（P1-1 指标） | F3（§5） |
| 近邻研究 §4.4（P1-2 状态机） | F4（§5） |
| 近邻研究 §4.5（P2 闭环） | F5（§5，纲要） |
| 审计报告 P0（343×316 判定/互斥裁决） | §1.3 前置 + §7.3 对账 |
| 审计报告 P0（305 三因拆分） | Out of scope（§1.2） |
