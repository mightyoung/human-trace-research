# EVAL-5：回放评估器扩展（三→五面板）+ 影子发布流程 报告

日期：2026-10-14 ｜ 性质：真实执行（工具扩展 + 影子运行示例）｜ 工具：`tools/replay_eval.py`（扩展）、`tools/shadow_run.py`（新）

## 0. 结论速览

1. **回放评估器扩到五面板**（原 mirror / match30 / relay 三个基线全部回归一致）：
   - **`registry`（新）**：EVAL-1 规则库阈值回放——在冻结的六层快照上重算动作（sim_hi / sim_reject / w_dt_min 可扫）；
   - **`zone`（新）**：TOP-3 zone-pair top-k 覆盖率回放（k 可扫）。
2. **影子发布 harness 建成**（`shadow_run.py`）：候选变更在 `analysis/shadow/<ts>/` 沙盒执行，运行前后对 **8 个生产文件做 md5 护栏**（PASS 才允许进入发布评审），产出 `shadow_report.json`（含影响面表）。
3. **示例影子运行全绿**：registry+zone 重跑 + 两组参数扫描（sim_hi、zone.k）→ **guard PASS（生产文件零变化）**；
   - 影响面示例①（registry.sim_hi 扫描）：0.945→0.96 时 同人 keep 6→5、**误拒恒为 0**——为"提精度不动误拒"提供一键依据；
   - 影响面示例②（zone.k 扫描）：k=3→7 pos 7→11/11、active 35→47/47。
4. 至此评测协议 §6 的流程链条（冻结窗口 → 候选 → **回放** → **离线影子** → 发布/回滚报告）**全部具备工具支撑**；"发布/回滚"动作本身仍按纪律单独立项。

## 1. 用法（可复述）

```bash
# 单变更回放（参数覆盖 / 扫描）
python tools/replay_eval.py --set relay.sim_thr=0.95
python tools/replay_eval.py --sweep registry.sim_hi --values 0.945,0.95,0.955,0.96
python tools/replay_eval.py --sweep zone.k --values 3,5,7,8

# 影子运行（沙盒 + 护栏 + 影响面）
python tools/shadow_run.py --job registry --job zone \
    --job "replay:registry.sim_hi=0.945,0.95,0.955,0.96"
# → analysis/shadow/<ts>/shadow_report.json
```

## 2. 面板口径

| 面板 | 参数 | 指标 | 数据 |
|---|---|---|---|
| mirror | du_max / pred_err_max / offset | TP/FP/FN（29 真对+408 对照） | 既有 |
| match30 | sim_thr | TP/FP/FN（8 正 22 负） | 既有 |
| relay | sim_thr×参数 | 候选→过门→边→链、锚点、增删边 | 既有 |
| **registry** | sim_hi / sim_reject / w_dt_min | 同人 keep/误拒、异人拒绝/误收、组裁决 | EVAL-1 六层快照（冻结） |
| **zone** | k | pos / active / audit top-k 覆盖 | TOP-3 排序表（冻结） |

## 3. 影子运行保证（护栏）

- 覆盖 8 个关键生产文件：edges_v2_gated / edge_states / scored_v2 / c0 v7 / gil.sqlite / portal_graph_v2 / registry_v0 / selector_v2；
- 任一 job 若污染生产路径 → report.guard=FAIL + 差异清单（本例 PASS）；
- 所有 job 输出在沙盒内，含命令、状态、日志尾部、影响面摘要。

## 4. 局限与边界

- 影子示例未包含"回到生产"的发布步骤（按纪律不自动执行）；
- registry/zone 面板基于冻结快照重算，不含候选生成重跑（需重跑时用 shadow 的 registry job 变体 + `--cand`）；
- replay 的 relay 面板为简化重放（49 边口径），与 relay v2 主产物（47 active/196 边）体系并列不混用。
