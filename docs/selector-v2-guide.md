# 选择器 v2 部署指南（改造范围 / 建议要点 / 预期效果）

日期：2026-10-14 ｜ 资产：`selector_v2.npz`（主）/ `selector_v2_single.npz`（回退）｜ 代码：`tools/selector.py`
研究报告：`docs/selector-upgrade-study.md`（实验细节与全量数字）

---

## 一、改造范围（本次做了什么）

| # | 改造项 | 内容 | 状态 |
|---|---|---|---|
| 1 | **多原型（PCLHD 借鉴）** | 每簇 1 均值原型 → 簇内 kmeans 分裂 ≤8 子原型（412 个）；簇级 score = 子原型 max 聚合 | ✅ 已落地 |
| 2 | **per-camera 开集阈值（折中方案 F）** | 全局单阈值 → 每摄像头分别校准（selfcal 档 + loose 档） | ✅ 已落地 |
| 3 | **一致性权重（N-ULC 借鉴）** | 新增 margin 拒识（top1−top2 簇分差）；**per-camera 分位校准**（保留率语义） | ✅ 已落地 |
| 4 | **簇 ID 兼容** | 新库簇 ID 与旧 `selector.npz` 一一对齐（下游引用不破坏） | ✅ 已验证 |
| 5 | **接口升级（向后兼容）** | `predict(v, cams, loose, margin)` / `sims_all` 簇级语义；旧 npz 行为逐位不变 | ✅ 已回归 |
| 6 | **软标签聚合（N-ULC 软标签）** | soft3/5（簇内 top-k 均值）实验后**未采纳**（弱于多原型；soft 叠加有害） | ❌ 实验证否 |
| 7 | **323 复核** | val 独立时段复核 → 结论修正为"时段波动"（非系统性缺陷） | ✅ 已闭环 |
| 8 | 多原型 × 软聚合混合 | 未做（边际收益低的证据充分） | ➖ 未做（有意） |

## 二、遵循的建议要点（来源 → 落实）

| 建议要点 | 来源 | 落实方式 |
|---|---|---|
| 全局聚类不动、开集阈值按摄像头分别校准 | `camgroup-selector-study.md`（方案 F，上轮推荐） | selfcal/loose 两档 per-cam thr 入库，`predict(v, cams)` 自动按摄像头取 |
| 补 hard 原型、提高簇表达（PCLHD） | 近邻方法调研（14 篇） | multi4/8/16 对照实验 → **multi8** 采纳 |
| 近邻软标签 + 一致性权重（N-ULC） | 同上 | 软标签实验证否；**一致性权重（margin）采纳并 per-cam 校准** |
| 硬隔离（按摄像头分组聚类）无收益 | `camgroup-selector-study.md` | 不采用分组；保持全局单一 ID 空间 |
| 阈值校准不得用 test | 数据纪律（项目铁律） | 所有 thr/δ 校准全部在 train（val 仅用于复核观测） |
| 基座可复现 | kmeans 环境敏感性发现 | 基聚类固定 mutual-knn；重建走内建回归流程 |

## 三、预期效果（test 348 轨 + 跨段 107 对，实测）

### 3.1 三种工作档位（按用途声明）

| 档位 | 调用 | coverage | same-known | coll-known | 跨段保持 |
|---|---|---|---|---|---|
| **默认保守档** | `predict(v, cams)` | 0.391 | 0.897 | 0.032 | 0.972（71/107） |
| **覆盖档** | `predict(v, cams, loose=True)` | 0.621 | 0.880 | 0.034 | 0.977（88/107） |
| **精确档** | `predict(v, cams, margin="prec")` | 0.230 | 0.988 | 0.039 | **1.000（49/107，零错误）** |

### 3.2 相对旧版（v1 单原型 + 全局 thr，同口径）

| 指标 | v1 | v2（对应档位） | 增益 |
|---|---|---|---|
| 同覆盖 same（固定 cov 0.4） | 0.856 | 0.894 | **+3.8pp** |
| 跨段身份保持 | 0.890 | 0.972 / 0.977 / 1.000 | **+8.2 ~ +11pp** |
| 公平子集 same | 0.863 | 0.912 | +4.9pp |
| train 共现 F1 | 0.8357 | 0.8698 | +3.4pp |

### 3.3 per-camera 预期（默认档）

305: cov 0.55/same 0.96 · 318: 0.24/0.84 · 320: 0.11/1.0（旧版 cov=0）· 323: 0.50/0.81（时段波动监控项）

## 四、使用方式

```python
import sys
sys.path.insert(0, "tools")          # 或把 selector.py 与 npz 放同目录
from selector import IdentitySelector

sel = IdentitySelector.load("output_crosscam/analysis/selector_1012/selector_v2.npz")

ids, sims = sel.predict(vec, cams=["305"])                    # 保守档（推荐默认）
ids, sims = sel.predict(vec, cams=["305"], loose=True)        # 覆盖档
ids, sims = sel.predict(vec, cams=["305"], margin="prec")     # 精确档
ids, sims = sel.predict(vec)                                  # 无摄像头标签：全局 thr（兼容旧用法）
```

- `ids`：簇 ID（-1 = unknown）；**簇 ID 已与旧库对齐**，可直接与历史输出对照；
- 阈值来源：`selfcal`（默认）/ `thr_loose_*`（覆盖档）/ `margin_prec_per_cam`（精确档）——全部 train 侧校准，入库可审计；
- 远程副本：`~/human-count/analysis/selector_1012/`（含独立 `selector.py`，rfdetr 环境零依赖可直接跑）。

## 五、边界与注意事项

1. **档位选择是业务决策**：coverage↔精度权衡按用途声明（如"夜间安防只要高精度"→ prec；"日间点名要覆盖"→ loose）；
2. **小样本摄像头（320/323）时段波动**：单时段指标可能偏移 ±5pp 量级，关键决策建议两时段复核；
3. **margin 的时序漂移未测**：长时间部署建议周期性重校准（复用重建流程，1 分钟级成本）；
4. **重建流程**（唯一入口）：`python tools/build_selector_v2.py`——内建"对齐旧库槽位 + 回归自检（含逐位一致检查）"，任何重训/数据更新后走此流程；
5. **不要用 kmeans 做基聚类**（不可复现）；**不要用 test 校准任何参数**（项目铁律）。

## 六、验证与复现清单

| 验证 | 命令/产物 | 结果 |
|---|---|---|
| 实验对照（8 变体） | `tools/selector_upgrade_study.py` → `upgrade/study.json` | ✅ |
| margin per-cam 校准 | `tools/selector_margin_percam.py` → `upgrade/margin_calib2.json`（含 split-consistency 分布） | ✅ |
| 323 独立复核 | `upgrade/val323_recheck.json` | ✅ 时段波动 |
| 回归 1（旧库+新代码） | `upgrade/regress_before/after.json` diff=0 | ✅ |
| 回归 2（旧库跨段） | `upgrade/crossseg_old.json` vs 历史三档一致 | ✅ |
| 回归 3（新 single=旧行为） | 1395/1395 逐位一致（build 内建自检） | ✅ |
| 跨段三档 | `upgrade/crossseg_v2_{selfcal,loose,prec}.json` | ✅ 0.972 / 0.977 / 1.000 |
