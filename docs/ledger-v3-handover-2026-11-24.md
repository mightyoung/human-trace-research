# 论文数字台账 V3 交接（2026-11-24 21:20）

> 阶段结束交接：目标 / 版本 / 已验证 / 未解决 / 下一步 / 授权边界。设计稿 `docs/ledger-v2-design-2026-11-24.md`，权威文件 `docs/paper/EVIDENCE.md`，报告 `ops/reports/paper_claims_v1/`。

## 1. 目标与结论

把「论文里的每个数字」变成有出处、可复算、有裁定的索引，替代按实验聚合的旧台账。**V1 登记 → V2 真值与协议 → V3 复算**三段已完成，V4/V5 按设计降级为可选。

**结论：V3 通过。** 可验率（复算一致或已标不可复算+原因）100%；正文 9 张表 + 补充材料 5 张表共 14 张表的每一行都有裁定。覆盖率 75%，缺口已分类入档，**r2 按现状投**。

## 2. 版本与文件

| 对象 | 状态 | 说明 |
|---|---|---|
| `docs/paper/tbiom-r3/main.tex` | **未改**（本轮只读） | r2 按现状投；r3 是另一条线 |
| `docs/paper/tbiom-r3/supplement.tex` | **未改** | 补充材料 5 表 27 行全部登记 |
| `docs/paper/EVIDENCE.md` | 505 行 | 追加 §12.9（7 行）；§12.1–§12.8 为前几轮 |
| `tools/ops_claims.py` | 1222 行 | 新增 6 类检查 + `kappa_layer` 点路径 |
| `ops/claims/*.json` | — | `tex_anchor` 91 claim/117 行；`verify_specs` 101 条；`label_sets` 7 个 |
| `ops/ops.db` | 写入 | `paper_claim` 238、`asset_version` 128、`label_set` 7、`eval_protocol` 1、**`eval_run` 101、`code_version` 1** |

## 3. 已验证（可复现）

- **锚定**：14 表全部命中，`gaps=0`、`errors=0`；表格数字里 EVIDENCE 全文没有的 **0 个**。
- **复算**：101 条配方 → **一致 66 / 不可复算 35 / 不一致 0**。
- **复算留痕（eval_run）**：每条配方一行 run，含输入文件 sha256、`exit_code`、`code_version`；**101/101 `exit_code=0`**，101/101 claim 已挂 run，`match` 而无成功 run 的 **0** 条。输入侧：50 条有 `read` 文件、29 条仅 `declared` 来源、1 条纯字面量（E06-089 人工耗时）。
- **正文 9 表全绿**：`tab:d1` `agg` `pair` `protocol` `gate` `selector` `verifier` `merge` `xday`。
- **补充 5 表 27 行全裁定**：16 可复算或部分可复算、11 完全不可复算、0 待裁。
- **负向对照**：故意翻转 3 个期望值（0.731 / 0.946 / 91）→ 3/3 判 `mismatch`，真库与 tex 均未被改动。
- **协议包**：`package-v3` 60/60 文件 sha 全对。
- **台账体检（`python tools/ops_claims.py doctor`，只读）**：资产 129 行 / 115 路径，缺失 0、内容变化 0（4 个 >200 MB 视频只验存在性）；run 输入 87 个路径全在且未变；`code_version` 过期 0。报告 `ops/reports/paper_claims_v1/DOCTOR.md`。

## 4. 未解决（三条，全部只需在 r3 处理）

1. **D2 的 445 轨口径**：现行 `tmp/full/gt_8seg.json` 是 494 轨、26 身份、中位 18、最大 71；论文 445 / 14 / 62。445 只存在于文档。若 r3 要保留 54/445（12.1%）、141/445（31.7%），**必须按现行 gt 重算**。
2. **`tab:pair` 标题的方法名**：写 Hanley–McNeil 近似区间，实际脚本是 percentile bootstrap（`tmp/full/emb_auc.py`）。
3. **κ 仅人物 CI 下界**：论文 0.913，两个存活产物 0.9124（→0.912）；κ 值本身 0.945 逐位吻合。

## 5. 已知陷阱（别重复踩）

- 跑 `verify`/`truth` 必须用 `~/.workbuddy/binaries/python/envs/default/Scripts/python.exe`；PATH 里的 3.13.12 没有 scipy，`prop` 会被记成**假 mismatch**。
- 改 EVIDENCE 后要 `anchor --rebind`，否则报「绑定的是另一版」。
- tex 解析盲点：`0.891--0.958` 的第二数、`0--720` 的 720 会被 `-` 的 lookbehind 丢掉 → 该行单元格少算；`tex_idx` 不能超范围；改解析就要重报 330 这个分母。
- 覆盖率 330 的分母与 247 的分子是两套 key（表名 vs 表题），只能比总数，不能逐格相减——本轮已用表索引重建过一次（见 `COVERAGE-GAPS.md` 的说明）。

## 6. 下一步（按代价排序）

1. **停在这里**（推荐）：r2 投出，V3 入档。
2. r3 排版时改两处不涉及数字的地方：`tab:pair` 标题方法名、κ 下界 0.913 → 0.912。**需另行授权**（正文本轮锁定）。
3. 若要用 445 的派生比例：先按现行 gt 重算 D2 全部下游数字，再改正文两处 —— 这是一次**数字变更**，不是排版。
4. 覆盖率补到 100%：约 83 格，其中 59 格是重复出现、21 格是机位号/分母；补了不增证据，不建议。

## 7. 授权边界

- 本轮**只写**：EVIDENCE、工具、`ops/` 下 JSON/DB/报告、设计稿、记忆。**未改任何 tex**。
- `main.tex` / `supplement.tex` 的修改需要当次授权；发布、删除同样。
- 台账的写操作一律 `--apply`，写前自动备份到 `ops/db_backup/`。
