# 交接：工作台 K0（通用流程层）— 2026-11-24 23:45

## 1. 这一期解决什么

用户指出工作台「不该只为这个项目的研究做，而应围绕从实验卡贯穿始终的流程」。核对后确认偏差：
v2.1 把主语定成了「论文数字」，`paper_claim` 认 EVIDENCE.md 行号、核验器认 tex 表格，
实验卡 5 张却与任何证据无连接。K0 把主语改回流程，论文降为一种交付物。

## 2. 版本与文件

| 文件 | 状态 | 说明 |
|---|---|---|
| `docs/workbench-flow-model-2026-11-24.md` | 新建 | 流程模型：七环节、四条守恒、三层分离、术语映射、分期 |
| `ops/schema.sql` | 修改 | 新增 `study`/`claim`/`claim_anchor`/`deliverable`/`deliverable_ref`；删去 `paper_claim` 建表 |
| `ops/views/paper_tbiom.sql` | 新建 | 适配器投影：通用 claim → 老列名，供老工具读 |
| `tools/ledger_migrate.py` | 新建 | paper_claim → claim+anchor；老表改名 `paper_claim_v1`；幂等、自动备份 |
| `tools/ledger_flow.py` | 新建 | C1–C4 诊断 → `ops/reports/ledger/FLOW.md` |
| `tools/adapters/demo_csv.py` + `examples/demo-csv/` | 新建 | 第二适配器（推荐系统离线评估），通用性冒烟 |
| `tools/ops_claims.py` | 修改 | 写入改 claim/claim_anchor，读仍走视图；其余逻辑未动 |
| `tools/ops_db.py` | 修改 | 去掉已失效的 `paper_claim.kind` ALTER |
| `webapp/DESIGN.md` | 修改 | 新增 §8.7 流程层（定位、四守恒、K2 落点） |
| `docs/ledger-v2-design-2026-11-24.md` | 修改 | 开头声明降级为适配器实现说明 |

## 3. 已验证（可复跑）

| 项 | 命令 | 结果 |
|---|---|---|
| 本项目回归 | `ops_claims.py all --apply` | 238 claim / 91 挂锚 / 117 表行 / 101 复算（66 match、35 unverifiable、0 mismatch）/ 247-330 覆盖率 —— **与迁移前逐项一致** |
| 视图可读 | `ops_claims.py status` | `claim 238`、`paper_claim（视图）238` |
| 迁移幂等 | `ledger_migrate.py --apply` 连跑 | 第二次 `claims 0 / anchor 0`，无重复 |
| 通用性 | `tools/adapters/demo_csv.py` | 4 claim / 4 anchor / 4 run(exit0) / **4 match，内核零改动** |
| 隔离 | 同上后查生产库 | 生产 `claim` 仍 238，demo 只写 `tmp/demo-ledger.db` |
| 库自检 | `ops_claims.py doctor` | assets 129（缺 0 变 0）、run 输入 87（缺 0 变 0）、cv_stale 0 |

## 4. 未解决

1. **C1 = 0/238（建议已出，待确认）**：`ops/claims/card_map.csv` 28 行容器、`claim_candidates.csv` 91 条逐条；
   在副本上预演的结果是 **primary 104 / n/a 76 / 待判 58**（表级映射又吃掉了 33 条）。**还没有写生产库**，
   写库命令 `tools/ledger_assign.py --apply --yes`，执行前必须人工过一遍这两份 CSV。
2. **C2 = 101/238**：137 条是说明/日期类，本身无可执行核验。Q2 已定三态（card / n/a / orphan），
   但 C2 的 `n/a` 口径还没落到 verify（现在 `verdict` 只有 match/mismatch/unverifiable/pending）。
3. **K1 未做**：`ops_claims.py` 1436 行仍是「内核+适配器」混装，算子未分层（`_pool_cell`/`_matrix_cell`/`_gate_run` 属项目特化）。
   目录结构与注册机制已在 Q10/Q11 定稿，见第 8 节。
4. **K2 未做**：前端没有「证据链」入口，orphan 归属没有界面（本轮用 CSV 代替）。

## 8. Q7–Q12 已定稿的实现约定（2026-11-25）

| 项 | 决定 | 落点 |
|---|---|---|
| Q7 建议表单位 | 两级容器（12 section + 14 表），表优先 | `card_map.csv` 28 行（含 supporting 拆行） |
| Q8 候选生成 | 脚本出候选 + 预填 decision，人改错的 | `ledger_assign.py --suggest` |
| Q9 归属基数 | 多对多 `claim_card`，`role` primary/supporting，primary 部分唯一索引 | `ops/schema.sql` |
| Q10 K1 结构 | `tools/ledger/` 包（core/operators/flow/adapters），`ops_claims.py` 留 30 行 wrapper，命令不变 | 待做 |
| Q11 算子注册 | `@operator("name")` 装饰器 + dict；adapter 在 `setup()` 时注册 | 待做 |
| Q12 引用校验 | `deliverable_ref.anchor_id` 可空外键；C4 分「锚定/文本」两列 | 已回填 117/117 |

**优先级**（Q1）：① 建议表 → ② 人工确认后写库 + 改 schema → ③ K1 拆包。当前在 ① 完成、② 待确认。

## 5. 陷阱

- 写入**只能**走 `claim`/`claim_anchor`；`paper_claim` 是视图，写会报 "cannot modify ... because it is a view"。
- 视图里 `json_group_array` 空集返回 `[]` 不是 NULL，必须 `NULLIF(...,'[]')`，否则「已挂锚点的 claim」会算成全部。
- `paper_claim_v1` 是迁移时点的**静态快照**，不要当数据源；以 `claim` 为准。
- 改 `tools/ops_claims.py` 会让 `code_version` 多一行，清旧行：`DELETE FROM code_version WHERE id NOT IN (SELECT DISTINCT code_version_id FROM eval_run WHERE code_version_id IS NOT NULL)`。
- 跑 verify 必须用 `~/.workbuddy/binaries/python/envs/default/Scripts/python.exe`（带 scipy）。

## 6. 下一步（按优先级）

1. **C1 归属**：给 238 条 orphan 出「按容器 → 候选卡」建议表（不写库），人确认后批量挂。
2. **K1 拆包**：`ledger/core.py`（调度/报告/doctor）+ `adapters/paper_tbiom.py`（EVIDENCE/tex/项目算子）+ `ledger/operators.py`（通用算子）。判定：拆分前后报告逐字节一致。
3. **K2 界面**：`#/workbench` 卡详情加「证据链」页签；orphan 待归属入口按容器分组。

## 7. 授权边界

- 已授权并在本期执行：新增表、迁移、脚本、文档、demo。
- **未授权**：改 `main.tex`/`supplement.tex` 任何数字；自动给 claim 挂卡；删除 `paper_claim_v1`；r3 的三件事（445 派生比例重算、`tab:pair` 方法名、κ 下界 .913→.912）。
