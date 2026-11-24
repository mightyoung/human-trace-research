# BASELINE_FREEZE 重建记录（2026-11-09）

## 为什么重建

`BASELINE_FREEZE` 是 M0-1 里程碑（2026-10-23）产出的硬锚，规定「M1/M2 任何对照开跑前必须先过 `--check` 校验」。

2026-11-09 执行 `tools/freeze_baseline.py --check` 报 4 个 `changed`：

```
changed  configs/selector_deploy.json
changed  deliverable/data-request-v2/gt/gt_query_gt_FILLED.csv
changed  deliverable/data-request-v2/gt/gt_gallery_members_FILLED.csv
changed  tools/session_selector_v1.py
```

前 3 项在本次之前就已漂移，第 1 项是 11-09 写入 `cluster.thr_out_review` / `thr_default_mismatch` 两条注释性字段所致（不改变行为）。锚已失效，需统一重建。

**但直接重建是危险的**：等于把「漂移后的状态」冒充「10-23 的锚」。重建前必须先证明漂移不影响指标。

## 诊断：用 sha256 反查冻结时的版本

冻结包里只存哈希、不存内容，所以无法直接 diff。但可以用哈希反查「磁盘上哪个副本才是冻结时的版本」：

```python
want = {a['path'].split('/')[-1]: a['sha256'] for a in freeze['artifacts'] if a['role'] == 'gt'}
# 对同目录下所有候选副本算 sha256，找匹配项
```

结果：**`.v2.csv` 精确匹配冻结哈希**，可以直接拿来 diff。

| 文件 | 冻结哈希匹配的副本 | 当前文件 |
|---|---|---|
| `gt_query_gt_FILLED.csv` | `gt_query_gt_FILLED.v2.csv`（31204 B） | 31205 B |
| `gt_gallery_members_FILLED.csv` | `gt_gallery_members_FILLED.v2.csv`（30738 B） | 32363 B |
| `clusters_gt.csv` | 自身（未漂移） | 同 |

## 漂移的实质内容

按主键对齐后，把「标注元数据」与「实质真值」分开统计（元数据列 = `annotator` / `time_spent_s` / `note` / `prompt_jpg`）：

| 文件 | 行数 | 主键增删 | `annotator` 变更 | **实质真值变更** |
|---|---|---|---|---|
| `gt_query_gt_FILLED.csv` | 295 → 295 | 0 | 295 | **2 处** |
| `gt_gallery_members_FILLED.csv` | 257 → 257 | 0 | 257 | **22 处** |

实质变更明细：

**query（2 处）**
- `318_test_534#1398`: `male8` → `male9`
- `318_test_534#1343`: `male5` → `no-human`

**gallery（22 处，节选）**
- `323_train_510#*`：`shadow1/2/3` → `no-human`（16 条）
- `318_train_525#8374`: `male17` → `male20`
- 其余同类修正

`annotator` 字段整体从 `checker` / 空 变为 `marker1`，属标注流程改名，不影响判定。

这 22 条 `shadow*` → `no-human` 与 10-23 那次「323 远场 31 条 + 318 簇 61 的 5 条全部标 `no-human`」是同源修正（见 `BASELINE_FREEZE.md` §3 口径禁令第 1 条）。

## 验证：漂移是否影响指标

三项独立验证，全部零差异。

### 1 精度表重算（42 个配置）

```bash
python tools/eval_precision.py --deliv deliverable/data-request-v2 --run results \
    --out-dir tmp/freeze_rebuild/eval_now
```

`--out-dir` 是必须的防呆参数：换真值却不换输出目录会覆盖冻结结果（`eval_precision.py` 第 127 行的注释记录 2026-10-24 踩过一次）。

| 对比 | 结果 |
|---|---|
| 键集合 | 42 = 42，完全一致 |
| 单元格差异 | **0 处** |
| 文件 md5 | **完全相同**（`73b94be2…`） |

→ 真值修正对全部 42 个配置的 C / W / U / precision / C-229 **零影响**。

### 2 损失分解重算

```bash
python tools/analyze_loss_decomp.py      # 先备份输出，它原地覆盖
```

| 对比 | 结果 |
|---|---|
| 字段总数 | 76 = 76 |
| 差异 | **0 处** |

`ALL` 组：queries 295 / evaluable 229 / answerable 196 / accepted 123 / correct 118 / wrong 5 / unanswered 73（`ua_both_low` 44 · `ua_one_pass` 18 · `ua_ab_split` 11）。

### 3 基线复现（文档承诺的命令）

```bash
python tools/eval_det_switch.py \
   --variant yolo8dd=output_crosscam_local/1012_yolo8_dd_emb \
   --cooc-map yolo8dd=output_crosscam_local/1012_yolo8_dd \
   --cluster-mode camera --thr-in 0.15 --thr-out 0.25 \
   --scorer proto --same-min 0.90 --min-eval 100
```

得到 `tr=579 te=229 K=111 pairs=1120 | cov=0.664 same=0.908 fm=1.53%`，与冻结值（θ=1.85 / cov=0.66376 / same=0.90789 / fm=1.525%）**逐位一致**。

## 重建

`tools/freeze_baseline.py` 原先把 `frozen_at` 硬编码为 `'2026-10-23'`，重建会写出错误日期。本次加三个 CLI 参数：

```
--frozen-at      冻结日期（默认 2026-10-23，保持历史行为）
--supersedes     被取代的旧冻结日期
--rebuild-note   重建原因与验证结论（写进 JSON 与 MD）
```

```bash
python tools/freeze_baseline.py \
    --frozen-at 2026-11-09 --supersedes 2026-10-23 \
    --rebuild-note "..."          # 完整文本见 BASELINE_FREEZE.md
```

旧包备份为 `BASELINE_FREEZE.{json,md}.bak_20261023`。

## 结果

```
python tools/freeze_baseline.py --check
→ OK 基线一致（15 个产物 + 嵌入目录）
   指标：precision=0.9516 C/229=118 θ=1.85 K=111
```

新旧包对比：

| 项 | 结果 |
|---|---|
| `metrics` | **完全一致** |
| `baseline.cluster` | 完全一致（thr_in 0.15 / thr_out 0.25 / veto null） |
| `baseline.selected` | 完全一致（θ 1.85 / K 111） |
| `loss_decomposition` | 完全一致 |
| 产物哈希变化 | 4 / 15 —— 正是那 4 个漂移文件 |

**基线指标一个都没变**，本次重建只是把磁盘状态、文档哈希与真实情况对齐。

## 复用要点

1. **不要直接重建**。先证明漂移无害，否则等于把错误固化成锚。
2. **用 sha256 反查历史版本**。冻结包只存哈希，但哈希可以定位「哪个副本才是当时那份」。
3. **元数据列与实质真值列要分开统计**。`annotator` 改名会让 100% 的行都"变化"，但这对指标毫无意义。
4. **重算必须用 `--out-dir` 写到临时目录**，比对完再决定是否回写。
5. **重建要留痕**：`supersedes` + `rebuild_note` 写进包内，而不是只写在提交说明里。
