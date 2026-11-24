# clean2 口径重建报告：实体 / 接力 / C0 版本链闭环

日期：2026-10-14 　范围：仅数据口径与评测口径修复，**无模型训练、不触碰 splits 冻结哈希**

---

## 一、结论先行

外部评审指出的"评测与版本一致性没做实"，其**根因比摘要所述更深**：不是"v1/v2 字段没同步"这一层，而是**实体图建立在与生产口径不同的数据上**——316/323 两个"dualclean"机位，实体图用的是**无掩膜**轨迹（127/205 条），而生产/评测口径是**掩膜后**的（67/137 条）。结果是：

- 旧实体图 **492 个实体中有 88 个**（316:46、323:42）引用了 clean 口径中**不存在的轨迹**；
- 由此，旧接力图 v3 有 **8 条悬空边**，v4 在 clean2 口径下有 **14 个悬空节点**。

本次已在 **clean2 口径**上重建实体图与接力图，并修订 C0 评价口径。重建后：

| 指标 | 旧链（v4/stitch-旧） | 新链（v5/stitch_clean2） |
|---|---|---|
| 实体成员在 clean2 中缺失的实体 | **88 / 492** | **0 / 411** |
| 接力边悬空引用 | v4 在 clean2 下 14 节点 | **0** |
| 接力图 | 40 边 / 30 链，**缺嵌入 1181** | **47 边 / 32 链，缺嵌入 0** |
| C0 出门连续性 / 进门归属率 | 0.261 / 0.244 | **0.301 / 0.286** |
| C0 重叠支路口径 | 8/9 且把 1 条 MODERATE 记作"误报" | 8 CONF + 1 MODERATE（**未决**）+ 0 REJ |

> 注：以上 C0 数字均为**同一 clean2 实体口径**下的可比结果（旧版 v3/v4 的 0.291/0.261 是在旧口径下得到的）。

---

## 二、问题定位（实测）

### 2.1 口径分裂

| 机位 | raw | `*_cleaned.jsonl`（旧实体图输入） | clean（v1） | clean2 |
|---|---|---|---|---|
| 305 | 107 | 106 | 106 | 107 |
| **316** | 145 | **127（无掩膜）** | 67 | **68** |
| 318 | 151 | 147 | 147 | 151 |
| 320 | 29 | 29 | 29 | 29 |
| **323** | 238 | **205（无掩膜）** | 137 | **138** |
| 343 | 57 | 53 | 53 | 53 |

`apply_p2_corrections.py` 读的是 `analysis/track_cleaning/{cam}_cleaned.jsonl`，而 `clean_manifest.json` 明确记录 316/323 的 `source_used` 是 `*_masked_cleaned.jsonl`（dualclean）。两套口径由此分裂。

### 2.2 后果量化

- 旧 `net_tracks`：**88/492 实体**含 clean2 缺失轨迹（316:46、323:42）；
- 旧拼接链：316 的 21 条链有 14 条、323 的 67 条链有 26 条，端点落在 clean2 之外；
- v4 接力：`relay_scored_cradio_v4.json` **缺嵌入 1181**（正是被掩膜删除的轨迹）。

### 2.3 修复引入的改善（v4 → v5）

被误删的 7 条真人轨（316#32、323#193、318#53/82/98/135、305#12）在重建后**重新进入实体与接力链**，例如新增边 `316[32]→305[52]`、`316[78,92,102]→323[193]`——修复了原先的漏接。

---

## 三、执行步骤（可复现）

1. **实体重建** `tools/rebuild_clean2_entities.py`
   在 clean2 轨迹集合上重算 mutual-best 拼接（复用 `stitch_*.csv` 缓存分数，门限 cos≥0.80 / pe_adj≤0.08，禁传递闭包），应用原人工修正（318 黑名单与白衣人合并、320 非人剔除与 18-23 合并）→ `analysis/stitch_clean2/net_tracks_*.json` + manifest。
2. **接力重建**（`crosscam_relay.py` / `relay_rescore.py` / `relay_triage.py` 已加环境变量与参数支持，默认向后兼容）
   `events → links → score/mutual/chains`，sim_thr=0.943，tag=`cradio_v5` → `analysis/relay_clean2/`。
3. **C0 修订并重跑** `tools/c0_baseline.py`（新增 `--edges v5`、`--nets`）
   - 重叠支路改**按 label 列**评价（CONF/MODERATE/REJ/未标注），不再"非 GT 即 FP"；
   - 身份键改为带相机命名空间的 `((318,tid_a),(320,tid_b))`；
   - 重跑 v3/v4/v5 → `analysis/c0/c0_baseline_{v3_clean2,v4_clean2,v5}.json`。
4. **审计闭环** `tools/audit_data_chain.py`（扩展 L1–L4 版本链）
   → `analysis/audit/data_chain_audit.json`。

---

## 四、结果

### 4.1 版本链（审计输出摘要）

```
L1 检测层（逐框契约违规 count/scores/tracks）: raw 0/0/0 | clean 4039/4039/0 | clean2 0/0/0
L2 实体层:
  stitch_old   : 实体成员在 clean2 中缺失 88/492  {316:46, 323:42}
  stitch_clean2: 实体成员在 clean2 中缺失 0/411
L3 接力层: v3(旧实体,悬空9节点) / v4(旧实体,悬空0) / v5(clean2,悬空0)
L4 评测层: c0_v5 边源=v5(47边/32链) 实体=stitch_clean2 ; c0_v4/v3 均绑定旧实体
```

### 4.2 C0（同口径对比）

| 版本 | 边 | 链 | 出门连续性 | 进门归属率 | 实体覆盖 | 悬空 |
|---|---|---|---|---|---|---|
| v3 | 49 | 34 | 0.291 | 0.268 | 83/404 | 8 边 |
| v4 | 40 | 30 | 0.261 | 0.244 | 70/404 | 14 节点 |
| **v5** | **47** | **32** | **0.301** | **0.286** | 79/411 | **0** |

### 4.3 C0-A 重叠对（318×320）

修订后：AUTO 档 9 条 = **8 CONF（与 8 个人工 GT 完全一致）+ 1 MODERATE（318#128↔320#21，未决）+ 0 REJ**。
- 召回（与 GT 交集）8/8 = 1.000；精度（仅在有标签集上）8/8 = 1.000；
- 明确：**MODERATE 计为"未决"，不是已确认误报**。

---

## 五、变更清单

**新增工具**：`tools/rebuild_clean2_entities.py`
**修改工具**（默认行为不变，向后兼容）：
- `tools/crosscam_relay.py`（新增 `CROSSCAM_NET_DIR` / `CROSSCAM_RELAY_OUT` 环境变量）
- `tools/relay_rescore.py`（同上）
- `tools/relay_triage.py`（新增 `--scored`）
- `tools/c0_baseline.py`（新增 `--edges v5` / `--nets` / `--tag`；重叠支路按 label 评价；身份键带命名空间）
- `tools/audit_data_chain.py`（扩展 L1–L4 版本链与 L2 存活率）

**新增产物**：
- `analysis/stitch_clean2/`（net_tracks_*.json + manifest.json）
- `analysis/relay_clean2/`（events/links/scored/mutual/chains/triage, tag=cradio_v5）
- `analysis/c0/c0_baseline_{v5,v4_clean2,v3_clean2}.{json,png}` + `fig_clean2_version_compare.png`
- `analysis/audit/data_chain_audit.json`（覆盖更新）

**旧产物全部保留**（`analysis/stitch/`、`analysis/relay/*_v4`、`analysis/c0/c0_baseline_{v3,v4}.json`），可回退。

---

## 六、局限与下一步

**本次未做（诚实标注）**：
1. 拼接链的相似度分数**复用**自旧 `stitch_*.csv`（在无掩膜数据上算），未在新口径上重跑 DINOv3 嵌入与拼接；被掩膜删除轨迹的链接已丢弃，但**边界处的竞争结构未重算**。
2. 被恢复的 7 条 FP 轨不在任何 stitch 候选 CSV 中，只作为**孤立实体**加入（已知，见 `rebuild_clean2_entities.py` 局限）。
3. `splits_v3/v4` 未重建（其 `splits_v4.meta` 自称 clean 实为 v1 的问题仍在）。
4. 覆盖率是**代理指标**（无独立身份真值），不能当作"未丢目标率"。

**下一步（建议顺序）**：
1. 把消费端默认指向 clean2：`apply_p2_corrections` / `make_splits_v3` / `relay_missing_recheck` / `unsup_reid_experiment` / `build_handoff_pkg` 五处的 `net_tracks` 目录切换；
2. 在 clean2 上重建 `splits`（三集键显式带相机命名空间），并冻结新哈希；
3. 独立短窗真值 + 出入口事件审计（评审主线），再谈深度蒸馏。
