# GIL S7：倒影单一聚类 + 选择器学习「镜像向量特征」

> 需求（原话）：**"将所有阴影放到一个聚类，让选择器学习阴影镜像的向量特征"**。
> 产物：`tools/gil_reflection_probe.py` → `analysis/gil/reflection_cluster_v1.json`；
> 落库 `gperson(gREFLECTION, state=non_person)` + `membership(node_kind='refl', 73 条)`
> + `obs.flags.reflection_score`(546 条)。

---

## 0. 结论先行

| 问题 | 答案 |
|---|---|
| 能不能把"所有阴影"聚成一类？ | **能**。S5（地板倒影 50）+ S6（墙镜倒影 24）并集 **73 条**已归入单一聚类 `gREFLECTION` |
| 会不会污染既有身份层？ | **不会**。成员用 `node_kind='refl'`（**不占 obs 节点**）→ `gil_gperson`/`gil_visit_owner` 只读 `node_kind='obs'`，S4 仍报 **264** 人 |
| 选择器能学到镜像向量特征吗？ | **能，但只能当软特征**。留一机位 AUC ≈ **0.76**（跨机位可迁移）；**远弱于几何判据（S5 F1 0.95）** → 不可替代 S5/S6 |
| 是"位置捷径"吗？ | **不是**。位置对照（只用 u,v、跨机位）AUC **0.52 / 0.17** → 绝对位置不可迁移，排除了区域捷径 |

---

## 1. 聚类：73 条倒影 → 一个 gperson

- 清单来源：`reflection_assign_v1.json`（S5 地板 50）+ `wallmirror_assign_v1.json`（S6 墙镜 24），
  并集去重后 **73 条**（316:29 / 323:40 / 343:4；重叠 1 条）。
- GIL 落库：
  - `gperson(id=265, label='gREFLECTION', state='non_person')`——★ `non_person` 明确标注它**不是人**；
  - `membership(node_kind='refl', node_id=0..72)` **73 条**，`src` 里带 `cam|tid:来源:源tid`（视图无关）；
  - `obs.flags.reflection_score` 写到**全部 546 条 obs**。

★ **设计选择（为什么不写 annotation）**：`annotation` 表的 `UNIQUE(target_kind,target_id)`
会让写入**覆盖**同一 obs 上 S5/S6 已写的 `wall_mirror` / `reflection` 标注（本次踩坑：
曾覆盖 11 条 wall_mirror + 1 条 reflection，已按原顺序 S5→S6 重跑恢复，779 条分布逐项一致）。
故聚类信息只放 `membership.src`，**annotation 保持 S5/S6 原文不动**。

★ 另一处防护：`gil_gperson.py --apply` 的 `DELETE FROM gperson` 已改为
`WHERE state != 'non_person'`，避免将来重建 gperson 时抹掉倒影簇。

---

## 2. 选择器学习：外观里到底有没有"倒影签名"

**特征空间**：C-RADIOv4 + EmbeddingGemma2（`cradio_scale_v3`，与 GIL/v4 空间 cos 中位 **0.999**，
故可用于学习；v4 只覆盖 clean2 的 137 条，正样本不够）。
**方向**：类均值差（收缩 LDA）—— 高维小样本下最稳，无需训练循环。
**样本**：73 正 / 474 负（负样本池**包含倒影的源轨迹**，外观几乎相同 → 结果是**保守下界**）。

| 指标 | 值 | 读法 |
|---|---|---|
| K 折 AUC（全体） | **0.870** | ⚠ 受**机位构成混杂**（见下） |
| 留一机位 **316**（训练 323+343） | **0.788** | ★ 真正的跨机位泛化 |
| 留一机位 **323**（训练 316+343） | **0.728** | ★ 同上 |
| 留一机位 343 | (1.0, n=4 不可信) | 已从均值中剔除 |
| 留一机位 平均（n_pos≥10） | **0.758** | 结论依据 |
| 打乱标签对照（同 K 折） | **0.602**（max 0.648） | ★ **≠0.5** → 汇总 AUC 不可信 |
| 位置对照（只用 u,v，跨机位） | **0.524 / 0.168** | ★ 位置不可迁移 → 排除"区域捷径" |
| 仅用机位先验的准确率 | 0.8665 = 基准率 | 机位本身无判别力 |

### 两个必须说清的坑
1. **不要用汇总 AUC**：把 3 个机位的样本汇在一起算 AUC 会被机位构成混杂——
   打乱标签的汇总对照就有 0.60（应≈0.5）。**只用"留一机位"这类跨机位指标**。
2. **位置对照是决定性的**：如果分数只是"画面区域"的代理，那么把方向学到别的机位后
   在该机位应当失效（镜面位置各机位不同）。实测 AUC 0.52 / 0.17（≈随机甚至反向）
   → **外观探针学到的是可迁移的成像签名，不是位置**。

### 判定
**外观里确有可迁移的"倒影签名"（留一机位 AUC≈0.76）**，但：
- 距离"可用硬门"很远（AUC 0.76 意味着在 24% 的对照上会判错）；
- 远弱于几何判据（S5 F1 0.9515 / S6 目视确认）；
- 负样本含源轨迹、且 GIL 内仅 12 条倒影有向量。

→ **结论：作为选择器的"软门控/降权特征"，不作为删除依据**。与项目既有发现
（倒影与源的 cos 间隔 0.0002）并不矛盾：那条说的是**倒影↔其源**不可分，
本表说的是**倒影↔人群**有可分的成像差异（低对比/模糊/镜像翻转）。

---

## 3. 给选择器的接口（integration contract）

```
特征：obs.flags.reflection_score   # 0..1，越大越像倒影；已写 546 条
来源：analysis/gil/reflection_cluster_v1.json → probe.weights（768 维）+ score_norm
用法：作为**降权因子**乘进外观相似度（例如 w = 1 − α·score），α 在 val 上校准
禁令：① 不得作为硬门（会误伤"倒影的源"这条真轨迹）；
      ② 不得用 test 调阈值（项目铁律：禁 test-oracle）；
      ③ 不得替代几何判据（S5/S6 才是主判据）。
复现：python tools/gil_reflection_probe.py [--apply]
```

---

## 4. 复现 & 回归

```bash
python tools/gil_reflection_probe.py            # 只读：训练 + 三组评估
python tools/gil_reflection_probe.py --apply    # 落库（幂等）
python tools/gil_smoke.py                       # 9/9
python tools/gil_visit_owner.py                 # 仍报 gperson 264（确认零污染）
```

**落库后计数**：`obs 546 / frag 278 / entity 411 / gperson 265(264 person + 1 non_person) /
membership 619(546 obs + 73 refl) / edge 641 / visit 846 / owner 505 / annotation 779`。
冒烟 **9/9**，`--apply` 幂等，S4 消费端数字**不变**。

---

## 5. 未做 / 下一步

- **未把 `reflection_score` 接进选择器本体**：`build_selector_v2.py` 的分数融合需要动
  选择器的 A/B（同序列冻结工作点），属独立一步。
- **未在控制台可视化倒影簇**：目前只有「倒影降权」开关视图（数据层已就绪）。
- **GIL 内只有 12 条倒影有向量**（其余被掩膜剔出 clean2）——若要更强的探针，需先把
  73 条倒影全部编码进 GIL 向量库（走 `embed_server`，需上传裁剪图 + 一轮 GPU 编码）。
