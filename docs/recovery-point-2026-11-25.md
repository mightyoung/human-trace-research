# 统一恢复点（recovery point）

> 2026-11-25。响应外部审查第 3 条第 3 点：「模板文件还在，并不保证恢复后的算法行为与中断前相同」。
> 工具 `tools/recovery.py`；验收实验 `tmp/derived/rp_resume_test.py`；结果 `tmp/derived/rp_resume_test.json`。

## 1. 问题是什么

此前有四处持久化，彼此不知道对方存在：

| 存在哪 | 存了什么 | 谁能读 |
|---|---|---|
| `selector_pool.json` | 模板条目 + `params` | 选择器（但**旧版只写不读**） |
| `store_meta.json` | 轨元数据 + `version`（模型/维度） | 选择器 |
| 身份图 `graph_450_v1.json` | 簇划分 | 离线脚本 |
| 各段 `*_summary.json` | 处理了多少帧、stride、fps | 没人汇总 |

后果有两层。浅层是已知缺陷：`_save_pool` 写参数、`_load_pool` 不读，重启后 σ 从 0.42 回到 0.60（已修）。
深层是结构性的：**即使参数恢复了，你也不知道「现在这份状态」是不是「上次那份状态」**——
嵌入换了模型没有、身份图改了版本没有、数据只跑了半段没有，都没有一处能回答。

## 2. 一个恢复点记什么

`python tools/recovery.py snapshot --tag <短标签> --note "<做了什么>"` 采集十项：

| 项 | 内容 | 可整体回退 | 缺失/异常时 |
|---|---|---|---|
| `code` | git rev、脏文件数、跟踪文件数 | 否（走 `git checkout`） | 必填 |
| `pack` | `data/.pack_tag`（如 `long35`） | **是** | 必填 |
| `gallery` | `store_meta.json` 的 `version`（模型/维度/n）+ `entity_index.json` | 否（需重算嵌入） | 必填 |
| `pool` | 模板条目 + `params`（σ/k/max/min_h） | **是** | 空池是合法状态 |
| `graph` | `graph_450_v1.json`、`attr_filter_ingraph.json`、`attr_gate_runs.json` | 否（走 diff 确认） | 必填 |
| `embeddings` | `store_emb.npy` + `embs_osnet/*.npz` 逐个 sha256 | 否（需重跑编码） | 必填 |
| `tracks_range` | 各机位 `frames_total / frames_processed / stride / fps / 分辨率` | 否（需重跑） | 必填 |
| `labels` | `gt_long35.json`、`recheck_ai_qc.json` | 否（改真值等于改基准） | 必填 |
| `training` | 优化器/调度器/rng/采样进度 | — | **当前不存在，如实记 absent** |
| `ledger` | `ops/ops.db` sha + runs/claims/cards 计数 | 否 | 必填 |

**只有 `pack` 和 `pool` 被标为可回退**，因为只有它们是「配置态」且体积小。其余要么必须重跑，
要么改动等于改评价基准。这个区分是刻意的：把不可回退的东西标成可回退，比不提供恢复更危险。

采集后落 `ops/recovery/<rp_id>/manifest.json`，可回退项另存 `files/` 副本。一个点约 **17 KB**。

## 3. 三个命令

```bash
python tools/recovery.py snapshot --tag v1 --note "EXP-003b 跑完"
python tools/recovery.py verify  rp_20261125_094441_baseline     # 与当前比，报漂移
python tools/recovery.py restore rp_20261125_094441_baseline     # 只回可回退项
python tools/recovery.py list                                    # 所有点
```

`verify` 的比对粒度分三级：整体 sha → 分件 sha → 值（JSON）。**有漂移返回 1**，可直接进 CI 或 `&&` 链。
`restore` 对不可回退项不静默跳过，而是打印「需重跑」及对应脚本名。

```
$ python tools/recovery.py verify rp_20261125_094441_baseline
项             结果       细节
code           OK        sha 一致
pool           已变      sha 变 471a2bb9→9c31e0a2
...
漂移项 1 个
```

## 4. 验收：中断 → 恢复是否等价

**这是本轮唯一通过实际执行验证的结论，不是静态检查。** 脚本 `tmp/derived/rp_resume_test.py`：

1. 采集恢复点；
2. 取探针 A：固定 15 条查询（`sorted(by_key)[:15]`），记 `top_lib`/`top_both` 的身份与 9 位分数；
3. **改坏状态**：σ 0.6→0.42、池 20→17 条、`pool_max` 8→3，模拟「上次会话停在别处」；
4. **错位恢复**：只回 `pack`，不回 `pool` —— 这是负向对照，用来证明探针确实能察觉差异，
   而不是「恢复函数返回 0 就算过」；
5. 全量恢复，要求探针 D 与 A **逐位一致**；
6. 再 `verify` 一次，要求漂移 0。

实测结果（`tmp/derived/rp_resume_test.json`）：

| 检查 | 期望 | 实测 |
|---|---|---|
| 改坏后探针变化（实验有鉴别力） | True | **True**（σ 0.42 / 池 17 → 摘要不同） |
| 只回 pack 不够（对照） | True | **True**（仍 σ 0.42 / 池 17） |
| 全量恢复后与 A 逐位一致 | True | **True**（σ 0.60 / 池 20，0 条差异） |
| 恢复后 verify 干净 | True | **True**（漂移 0） |

另核对 `selector_pool.json` 恢复后与测试前备份 **sha256 相同**（`471a2bb9…`），即逐字节还原，不是「参数看起来对了」。

## 5. 它现在能保证什么、不能保证什么

**能保证**

- 「参数 + 模板 + 数据包」这三件在中断后可以恢复到与中断前**行为等价**（已实测逐位一致）。
- 任何一次状态漂移都能在一条命令内被指出，且指出是哪一项、哪种漂移（sha / 分件 / 值）。
- 嵌入、图、真值、数据范围的版本被同时固定，事后能回答「这次的数是用哪套东西算的」。

**不能保证**

1. **嵌入与身份图不能回退**。它们只能校验、不能还原——还原意味着重跑编码（远端 GPU）或从交付包取回。
   所以正确用法是：**改这些之前先 snapshot**，出事时知道回到哪个版本去重跑。
2. **训练态不存在**。项目里没有 optimizer/scheduler/rng/采样进度的落盘，因此「周期训练中断续跑」
   目前**无法验证**，恢复点对它是 `absent` 而不是「已支持」。这是审查第 3 条里唯一没有实体的部分。
3. **不覆盖「人做过什么」**。标注进度在 `ops.db`（`ledger` 项只记 sha 与计数），不在恢复点里。

## 6. 与 `ops_run` 的分工

| | 记什么 | 何时用 |
|---|---|---|
| `ops_run` | **一次运行**：命令、参数、环境、git、日志、产物 | 跑实验时 |
| `recovery` | **一个状态**：模型/图/池/参数/数据范围/真值 | 改配置、换数据、收工前 |

两者互补：`ops_run` 回答「这个数字是怎么算出来的」，`recovery` 回答「算它的时候用的是哪套东西」。
审查说「能重新克隆仓库 ≠ 能恢复上周的实验状态」——`ops_run` 的 `git_rev` 只解决代码那一半，
本工具补上另外九项的一半（可校验），并明确标出剩下那一半（需重跑）。

## 7. 建议习惯

- 每完成一批实验、或改动 `pool`/图/嵌入之前：`snapshot --tag <批次名>`。
- 长时间中断后回到项目，第一条命令：`verify <最后一次的 rp>`，先知道变了什么再动手。
- `snapshot` 不做自动清理；`ops/recovery/` 每点约 17 KB，留 20 个点也不到 0.5 MB。

## 8. 一个自己踩到的坑（已修）

第一版 `code` 项用 `git status --porcelain` 的原样行数当 `dirty_files`，而快照目录 `ops/recovery/<rp_id>/`
本身是未跟踪的——**每次 snapshot 都会把自己算成"脏"**，紧接着 `verify` 就报 `code 已变`，
但 rev 一模一样（`7d6a318→7d6a318`）。这类"工具污染自己的信号"比不检测更糟：它会让使用者
养成忽略告警的习惯。修法是把 `ops/recovery/` 前缀从脏文件计数里排除，修完立刻 `verify` 得到 0 漂移。

教训值得记：**任何自检指标都要先问「它会不会被自检动作本身改变」**。同类风险还有日志目录、
临时输出、运行中改写的时间戳——本工具的 `tracks_range` 刻意只记 summary 里的整型水位而不是 mtime，
就是同一个考虑。
