# 第二次冻结（v6）复核记录

日期：2026-10-24。对象：`holdout-refreeze-v6.zip`（8 个替换文件）。本文只记**我独立复算过**的结论；未复算的明确标注。

> **已被取代（2026-10-24 17:30）**：本文记录的哈希属**第二次冻结**，仅供追溯。揭盲前核对接缝后发现两处切片偏移不准（305_hgal_p2 13→0 s、320_hgal_p2 0→4 s），已做**第三次冻结**，现行清单与哈希以 `docs/preeg_freeze_2026-10-24.sha256` 为准，本文的哈希值不再用于校验。第二次清单归档为 `preeg_freeze_2026-10-24.sha256.superseded-2nd-freeze`；改动依据见 `docs/holdout-seam-check-2026-10-24.md`，正文记录见预注册 §2.1.3。

## 1 结论

**8 个文件的参考哈希 8/8 与实际文件一致**；替换后用 §9 命令生成的清单 `sha256sum -c` **8/8 OK**；签署页 8 个锚点与清单、文件三方一致。

## 2 逐项复核

### 2.1 主实验图库时段改为 08:03–10:28（原写 08:00–09:35）

从 `docs/material-audit-2026-10-20.json` 的 `used` 字段独立重算，四个机位各自的 60 min 窗口是错开的：

| 机位 | train 段 | 起止 |
|---|---|---|
| 320 | `320_train_514` | 08:03–09:03 |
| 318 | `318_train_525` | 08:22–09:22 |
| 323 | `323_train_510` | 08:35–09:35 |
| 305 | `305_train_529` | 09:28–10:28 |
| **并集** | — | **08:03–10:28** |

**改动正确**：旧写法 08:00–09:35 只对应 323 一路，且起点比最早的 320 晚 3 分钟。同期核对：test 并集 13:45–15:18（与原文一致）、val 并集 12:15–12:54。

### 2.2 段名必须以机位号开头

在下游代码里确认了这条依赖，旧名确实会把四个机位解析成同一个：

| 位置 | 代码 |
|---|---|
| `tools/d7_sweep.py:67` | `cams = np.array([k[0].split("_")[0] for k in ktr])` |
| `tools/d7_sweep.py:199` | `"cam": k[0].split("_")[0]` |
| `tools/session_selector_v1.py:46` | `cam = s.split("_")[0]` |
| `tools/eval_selector_1012test.py:103` | `cams = [s.split("_")[0] for s in seg_of]` |

旧名 `H_gal_305_p1` 解析为 `H`，10 个片段全落到同一"机位"，机位感知聚类（机位内 average、跨机位 complete）会失效。**这是替换前会静默出错的一处，改得对。**

### 2.3 分析代码的 6 个依赖哈希

`§5.1` 表列出的 6 个哈希，与仓库文件、导出包存档副本**三者逐位相同**：

| 文件 | 前 16 位 | 仓库=存档=§5.1 |
|---|---|---|
| `d7_sweep.py` | `bfb1b2650d756d06` | 是 |
| `session_selector_v1.py` | `3a344cbc59a7d27c` | 是 |
| `eval_mask_variants.py` | `549c12f622145627` | 是 |
| `cluster_camera_aware.py` | `85eb338b8ad48b43` | 是 |
| `eval_selector_1012test.py` | `854314b9d5be0ee4` | 是 |
| `selector.py` | `57e83fb7477db10a` | 是 |

### 2.4 判定脚本在主实验数据上复现主结果

`holdout_eval.py --pkg deliverable/data-request-v2`（主实验 42 配置、229 可评查询）：

| 项 | 复算值 | 主实验已发表值 | 一致 |
|---|---|---|---|
| 可答 / 覆盖 | 196 / 0.664 | 196 / 0.664 | 是 |
| C / W / U | 117 / 4 / 2 | 117 / 4 / 2 | 是 |
| 精度 | 0.9669 | 0.9669 | 是 |
| C/N | 0.5109 | 0.511 | 是 |
| H1a ρ | −0.418（42 配置） | −0.418 | 是 |
| H1b | `C_dmax0.15`（继承 0.908）ΔC −61，p 6.8e-16 | −61 | 是 |
| H1c | 0.585 vs 0.967 | 0.585 vs 0.967 | 是 |
| H2c | −27（1 增 / 28 减），p 1.12e-7 | −27，p 1.1e-7 | 是 |
| H3 | 73 vs 4 | 73 vs 4 | 是 |
| 多重比较 | m=41，Bonferroni 显著 22，正向 0 | 22 个全负 | 是 |

C/N 的 95% CI 复算为 [0.310, 0.713]（24 个身份组），与稿件 0.306–0.704 的差异来自 bootstrap 随机种子，区间宽度一致（约 40 pp）。

### 2.5 标注模板的范围规则

`holdout_make_templates.py --run deliverable/data-request-v2/results`：

- 图库模板 **257 条**，与主实验 `gt_gallery_members_FILLED.csv` 的 257 条**集合完全相同**（无多、无缺）。
- 查询模板 **229 条**。主实验 gt 查询文件有 295 行，差在 66 条不可评查询（前后半片未同时可编码）——按 §3.1 的范围定义（"全部可评查询"）这是预期结果，不是遗漏。

### 2.6 命令行与门禁

| 检查 | 结果 |
|---|---|
| `run_holdout.sh --print-plan` | 10 条命令，段名 10/10 解析到正确机位；偏移与时长与 JSON 逐项相同 |
| `run_holdout_analysis.sh --print-plan` | 4 条命令，`--train-segs` 6 段 / `--test-segs` 4 段；打印里的 `\,` 是 `printf %q` 的显示转义，真实值 65 字符、无转义 |
| `run_holdout_analysis.sh --stage verify` | 6/6 OK，退出码 0 |
| 人为篡改 `d7_sweep.py` 后重跑 | **拦截**，`FAIL tools/d7_sweep.py`，退出码 2；恢复后哈希回到 `bfb1b265…`，再跑退出码 0 |
| 缺人工标注时 `--stage eval` | `[ABORT] 缺少人工标注 …`，退出码 3 |
| 缺编码文件时 `--stage sweep` | `[ABORT] 缺少编码 …`，退出码 3 |
| `verify_preeg_hashes.py` | 结论"全部一致，未偏离"，退出码 0（§7 的 5 个哈希在新正文上照常解析） |

## 3 冻结结果

`docs/preeg_freeze_2026-10-24.sha256`（8 个文件），`sha256sum -c` **8/8 OK**：

| 文件 | 前 16 位 |
|---|---|
| `docs/PREREGISTRATION_holdout_2026-10-24.md` | `e74ef7ef40bccd32` |
| `docs/preregistration_holdout_2026-10-24.json` | `376c25d5ed9651af` |
| `tools/prereg_holdout_window.py` | `41a3e181e8465830` |
| `tools/run_holdout.sh` | `1e15fb2ea6072543` |
| `tools/verify_preeg_hashes.py` | `4bbd6b6ca839dc37` |
| `tools/run_holdout_analysis.sh` | `3d257ea19722f44d` |
| `tools/holdout_eval.py` | `d43fa3e8c0892feb` |
| `tools/holdout_make_templates.py` | `031c29265d6ba859` |

第一次冻结的 5 文件清单归档为 `docs/preeg_freeze_2026-10-24.sha256.superseded-5files`。它对其原文件仍然有效，但被冻结的文件已变，**不要用它校验当前文件**。

## 4 本记录不证明什么

- **sweep 阶段没有真跑**：需要留出段的编码文件（`output_crosscam_local/holdout/emb/*.npz`），只能在检测+编码后产生。`d7_sweep` 的 42 配置扫描路径未经端到端验证。
- **判定脚本只在主实验数据上验证过**：它复现了主实验，但留出段规模更小、可能触发"无法判定"分支，该分支只做了静态阅读，没有用真实留出数据走过。
- **冻结时间尚未外部锚定**：签署页 §2 的"外部时间戳"三行仍是待填。哈希证明内容，不证明时间。
- **未验证全量检测的跨卷行为**：305 的 529→517、320 的 540→543（41 s 缺口）在完整长度下的表现仍未测。
