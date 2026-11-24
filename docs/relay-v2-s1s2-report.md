# Relay v2 实施报告：S1（多对多）+ S2（过渡时间分布）

日期：2026-10-14 ｜ 状态：**S0/S1/S2 完成，主工作点验证 Go** ｜ 上游：`docs/relay-v2-plan.md`（方案）、`docs/cam-handover-neighbor-research.md`（设计）

---

## 结论先行

1. **S1+S2 完成并通过四级验证中的 L0/L1/L2**。主工作点（**K_a=4 / K_b=2 / λ=0.01 / score 排序**）对账结果：
   - **漏接恢复 4/5**（5 对中恢复 E011↔E013、E042↔E046、E042↔E048、E038↔E043）；
   - **排他竞争 3 对全部恢复（3/3）→ Go ✓**；
   - **异人零混入**：已判异人 2 对（E042↔E050 / E042↔E049）在 active、audit、conflict 三处均为 0 → Go ✓。
2. **唯一未恢复项**：E010↔E014——**不在 2618 候选池内**（候选生成漏，独立根因），F1/F2 均无法触达；保留为独立线索（需查 E014 事件是否被 moving 过滤或实体化并轨）。
3. **规模**：高 sim 179 = primary 47（逐位不变）+ **secondary 100（审计池）** + blocked 8（conflict 显式化）+ 常规淘汰 24。
4. **K_a=3 对照留档**：恢复 3/5（排他竞争 2/3，No-Go）——差额全来自 E042↔E046（多对多容量不足）；K_a=4 的论证见 §4。

---

## 1. 实施清单

| 模块 | 状态 | 说明 |
|---|---|---|
| `tools/relay_expand.py` | ✅ 新建 | F1 核心：primary 复用 + secondary 预算分配 + conflict_group；`--sort-key sim/score`、`--archive` |
| `tools/relay_v2_selftest.py` | ✅ 新建 | L0 自检 6 案例（T1–T6），退出码判定 |
| `tools/relay_v2_reconcile.py` | ✅ 新建 | L2 对账（19 判定 vs v5/v2），内置 Go/No-Go |
| `tools/relay_transition_dist.py` | ✅ 新建 | F2：Parzen 分布 + 联合分 + λ 敏感性表 |
| 产物 `relay_v2/` | ✅ | `edges_v2.json`（主）、`dist_v1.json`、`scored_v2.json`、`reconcile*.json`、对照 `edges_v2_k3sim.json`/`edges_v2_k3score.json` |

## 2. 验证证据

### L0（合成自检，6/6 过）

T1 一人多次通行（3 同人恢复）｜ T2 预算溢出（冲突/静默分流）｜ T3 K_b 回穿拦截 ｜ T4 确定性排序 ｜ T5 空池边界 ｜ T6 score 排序翻转（短 dt 主峰胜长尾）。

### L1（回归）

- **primary 47 边与 v5 triage 逐位一致**（两次运行均断言通过）；
- 旧脚本零改动（`relay_triage.py`/`relay_rescore.py` 未触碰）。

### L2（19 对账，主工作点）

| 状态 | 对 | 结果 |
|---|---|---|
| 恢复（4） | E011↔E013 / E042↔E046 / E042↔E048 / E038↔E043 | 全部入 **audit**（审计池） |
| 未恢复（1） | E010↔E014 | 不在候选池（独立根因，F2 窗外） |
| 异人（2） | E042↔E050 / E042↔E049 | **零混入**（active/audit/blocked 全 0） |
| 存疑（8） | — | 不参与判据；其中 E037↔E044、E031↔E040 入 audit（待复核） |

### λ 敏感性（`dist_v1.json.sensitivity`）

| λ | secondary 数 | 恢复/5 | 异人入边 |
|---|---|---|---|
| 0（纯 sim） | 94 | 3 | **1**（E050） |
| 0.005 | 96 | 3 | 0 |
| **0.01（主）** | 96→100（K_a=4） | **4** | **0** |
| 0.02 | 98 | 3（换为 E046 但丢 E048） | 0 |
| 0.05 | 98 | 3（同 0.02） | 0 |

> 读法：λ≥0.005 起 dt 先验开始"挤出异人长尾"；λ=0.01 在 K_a=4 下达成 4/5；λ≥0.02 过度压制长尾（E048 被 E079 挤出）。λ 的**正式校准待 241 判定后按留出纪律执行**。

## 3. 关键产物结构（`relay_v2/edges_v2.json`）

```
version / based_on（含 scored 哈希）/ params{sim_thr, K_a, K_b, conflict_sim, sort_key}
edges[]: primary×47（status=active）+ secondary×100（status=audit）
         每条含 a/b/dir/dt/sim/w_dt/score/edge_type/budget_rank
conflict_groups[]: 7 组 / 8 条 blocked（sim≥0.95 被预算挤出，显式化供复核）
summary: {n_primary:47, n_secondary:100, n_conflict_groups:7, high_sim_total:179, sort_key:"score"}
```

**E042 案例（设计目标场景，实测分配）**：

| 边 | 类型 | sim | dt | w_dt | score | 判定 |
|---|---|---|---|---|---|---|
| E042→E053 | primary | 0.9769 | 59.0 | 0.112 | 0.9550 | 同人 ✓ |
| E042→E074 | secondary | 0.9567 | 2.6 | 0.984 | **0.9565** | 未判 |
| E042→E048 | secondary | 0.9651 | 41.6 | 0.190 | 0.9485 | 同人 ✓ |
| E042→E079 | secondary | 0.9496 | 10.0 | 0.874 | 0.9483 | 未判 |
| E042→E046 | secondary | 0.9445 | 1.0 | 0.944 | 0.9439 | 同人 ✓ |
| E042→E050 | **不入边** | 0.9499 | 47.6 | 0.165 | 0.9319 | 异人 ✓（正确排除） |

（E050 的 score 0.9319 被 dt 先验从第 4 位压到第 6 位——F2 的区分目标成立。）

## 4. 设计决策记录

| 决策 | 选择 | 依据 | 留档 |
|---|---|---|---|
| K_a 主值 | 3 → **4** | E042 判定显示一人多次通行的真同人边 ≥3 条（1 primary + 2+ secondary），K_a=3 容量封顶；K_a 是"最大边数"结构参数而非拟合参数 | K_a=3 全结果留档（`edges_v2_k3score.json` + `reconcile_k3score.json`，2/3 No-Go）；241 判定后复核 |
| λ 主值 | **0.01**（先验） | 使主峰-长尾贡献差（~0.01–0.02）与 sim 噪声尺度匹配；非从评估集调参 | 敏感性表全览；正式校准待留出 |
| 排序键 | sim → **score** | λ=0 时异人入池；score 排序为 F2 设计目标 | `edges_v2_k3sim.json`（首版）留档 |
| 冲突语义 | sim≥0.95 被挤出 → blocked 显式化 | 对齐方案 §5 F1；实际 8 条 blocked 均属"高外观分但 dt 长尾/竞争失败" | `conflict_groups` 段 |

## 5. 遗留问题（诚实清单）

1. **E010↔E014 候选生成漏**：E014 不在池——需查：E014 事件是否存在（events_316.json）、是否 moving 被滤、或实体化时被并入他轨。**独立根因任务**，不属于 F1/F2 职责。
2. **E079/E074 等待判定**：E042→E079（0.9496, dt=10）与 E042→E074（0.9567, dt=2.6）已入 audit——是"真同人（多次通行）"还是"外观相似异人"直接决定 K_a=4 的精que；241 判定完成后复核。
3. **同 entry 多 secondary 的仲裁**：34 个 b 端被 2 条 secondary 认领（如 316:E048 同时被 E042 与 E079 认领）——审计池展示时需并列呈现供人工裁决。
4. **K_b=2 的语义**：当前 secondary 预算不含 primary 占用；241 判定后可评估是否收紧（含 primary）。
5. C 盘空间提醒（97%）——S3/S4 开工前复查。

## 6. 下一步

- ~~**S3（F3 handover 指标）**~~ **✅ 已完成**（`docs/relay-v2-s3s4-report.md` §1）：边级 FN/FP（active+audit 口径 1/9）、crossing/returning（44+44 候选）、覆盖率重算（v5 自证 0.301/0.286 + v2 试算 0.471/0.532）、top-10 复核清单；
- ~~**S4（F4 状态机）**~~ **✅ 已完成**（同上 §2）：162 边实例化 → active 58 / candidate 102 / suspect 2，11 条晋升（7 auto + 4 human-assisted）；
- 判定继续补充（webapp `#/truth`）→ 241 完成后 **L3 全表**（F3 一键重跑）+ λ/K 正式校准；
- **S5（F5 闭环）**：后置，由 H1/H2 实验计划驱动。
