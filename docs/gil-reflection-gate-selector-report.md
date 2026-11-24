# S7 倒影门接入选择器本体（同序列冻结 A/B）

> 日期：2026-10-14　产物：`output_crosscam/analysis/eval3/reflection_gate_ab.json`
> 上游：`docs/gil-reflection-cluster-s7-report.md`（判别器，GIL 留一机位 AUC 0.758）

## 1. 问题

S7 训练了倒影判别器（收缩 LDA 方向，768d），把 `obs.flags.reflection_score`（0..1）打到 GIL 全部
obs。但**选择器本体一直没用它** —— `IdentitySelector.predict` 只吃 ReID 向量。本次把它接进本体，
并用一次**同序列冻结的 A/B**回答：接进去之后，身份分配发生什么变化？

## 2. 接入点（默认全部关闭 → 生产行为零变化）

| 文件 | 改动 |
|---|---|
| `tools/selector.py` | `predict()` 新增 `refl / refl_mode / refl_tau / refl_base / refl_alpha`。默认 `refl=None` → 与旧版**逐位一致** |
| `tools/build_selector_v2.py` | 新增 `--reflection-gate`（把判别器与操作点写进 npz `meta.reflection_gate`）、`--reflection-tau/base/alpha`、`--reflection-drop`（危险，默认只报数不删） |
| `tools/reflection_gate_ab.py` | **新增**：同序列冻结 A/B（复用 EVAL-3 的度量函数） |

**两种门控语义**：
- `soft`（推荐）：`eff = best − α·max(0, refl − base)`，**只作用于"接受判定"、不改簇归属**（低于
  基线的样本不受罚）。
- `hard`：`refl ≥ τ → unknown`。

> ★ 首版把 `soft` 写成"簇级 score 乘 (1−refl)"，实测**直接失效**（覆盖被砍到 0）—— 因为 selector
> 的阈值很紧（卡在 score 分布上），乘性降权会整体翻转到阈值之下。故改为**偏移式、且只动接受判定**。

## 3. 回归自检

| 检查 | 结果 |
|---|---|
| `refl=None` vs `refl=全零(soft)` 逐位一致 | **OK** |
| 重建后 `protos / thr_unknown / proto_cluster / thr_loose / margin_*` vs 旧库 | **逐位相同**（仅 `meta` 多 `reflection_gate`） |
| `[align]` slot 双射 + single 行序 cos min | **1.000000** |
| `[regress]` new-single(loose) vs 旧行为 pred_diff | **0 / 1395** |

→ 接入对既有 selector 是**纯增量**。

## 4. 同序列冻结 A/B（1012，除门控外全同）

操作点在 **val 段**按「覆盖损失 ≤ 5pp 预算，取预算内最激进者」校准（禁 test-oracle）：
`alpha* = 0.02`，`tau* = 0.70`，`base = val 中位 0.313`。

| 臂 | 覆盖 | 分片一致 ab_same | 跨段保持 ret_same | 误接代理 viol |
|---|---|---|---|---|
| A off（现状） | 0.5201 | 0.8971 | 69/107 | 0 |
| B soft (α=0.02) | 0.4454 (−7.5pp) | **0.9123 (+1.5pp)** | 61/107 (−8) | 0 |
| C hard (τ=0.70) | 0.4799 (−4.0pp) | **0.9091 (+1.2pp)** | 64/107 (−5) | 0 |

**门控曲线**（test）：
- hard τ：0.4→unknown 0.733、0.5→0.624、0.6→0.558、0.7→0.520、0.8→0.483、0.9→0.480
- soft α：0.02→0.555、0.05→0.655、0.10→0.721、0.20→0.759、0.35→0.799、0.50→0.819、1.0→0.836

**读法**：门控**用覆盖换分片一致性**（ab_same ↑1.2~1.5pp），但**跨段保持下降**（把一些本可正确保持的
对砍成 unknown）。`viol` 在本数据集恒为 0 → **该指标无区分度**。

## 5. GIL 侧支撑证据（绝对阈值不可用）

GIL 546 条 obs 的 `reflection_score`：mean 0.438 / p50 0.420 / p95 0.826。
人工判定倒影 8 条（全在 323）：**0.4796 ~ 0.7643（中位约第 82 百分位）**。

| τ | 捕获人工倒影 | 全体命中（含真人） |
|---|---|---|
| 0.5 | 7/8 | 218/546（**39.9%**） |
| 0.7 | 1/8 | 73/546（13.4%） |
| 0.9 | 0/8 | 13/546（2.4%） |

→ **排序有效、绝对阈值不可用**（真人基数大、分数分布重叠）。判别器 `eval.camera_only_acc=0.8665`
也提示分数里含**相机效应**，留一机位 AUC 0.758 是其泛化上界。

## 6. 结论

1. **接入完成且零风险**：接口默认关闭、生产库逐位不变；开启后操作点写进 `meta.reflection_gate`。
2. **在 1012 上无净收益证据**（无倒影标签 → 只能给权衡曲线）：门控是"覆盖换一致性"，且跨段保持下降。
3. **不建议默认开启**；若启用：只用 `soft`、`α ≤ 0.02`、**按相机**（先 323 玻璃门区）开启。
4. **不可替代几何**：S5 地板倒影 F1 0.9515 远高于判别的排序能力；门控只作**降权提示**。
5. 要拿到"净收益"，需要**有倒影标签的评测集** —— 唯一现成的是 GIL 的 8~73 条，样本太小；
   建议把 S5/S6 的人工判定（含本次新增的 5 项人脸冲突对）作为小样本评测，逐步累积。

## 7. 复现

```bash
PY="/c/Users/might/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
$PY tools/reflection_gate_ab.py                      # 同序列 A/B（只读）
$PY tools/build_selector_v2.py --reflection-gate \
     --reflection-base 0.3130 --reflection-alpha 0.02 --reflection-tau 0.7   # 刷新 npz meta
```

备份：`selector_v2.npz.bak_pre_reflgate`（接入前）。
