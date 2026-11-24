# S6-b 人工裁决回灌报告（30 项）

> 输入：控制台「S6-b 待裁决」30 项人工判定（`truth/labels.json`，前缀 `S6B_`）
> 工具：`tools/gil_adjudication_ingest.py`（默认 dry-run）
> 台账：`output_crosscam/analysis/gil/human_adjudication_v1.json`

---

## 1. 判定分布（30/30 全部判完）

| 分组 | 问法 | 判定 | 条数 |
|---|---|---|---|
| ① relay 冲突边 | 是否同一人 | **同人 5** / 不是 **7** / 看不清 4 | 16 |
| ② 倒影搭桥合并 | 是否同一人 | **该留 1** / **该拆 1** / 看不清 1 | 3 |
| ③ 倒影实体 | 是倒影还是真人 | **是倒影 7** / **是真人 4** / 看不清 0 | 11 |

---

## 2. 已落库：裁决层（GIL `annotation`）

- `target_kind='human_verdict'`，`target_id='<S6B id>'`，**独立命名空间**；
- `annotation` **779 → 809**（+30）；分布：`confirm 5 / reject 7 / defer 5 / keep 1 / split 1 / confirm_reflection 7 / revoke_reflection 4`；
- **幂等**（重跑仍 809）；`obs / edge / gperson / membership / visit / owner` **零改动**；
- 备份：`store_v1/gil.sqlite.bak_pre_human_adj_<ts>`。

选用独立命名空间的原因：`annotation` 有 `UNIQUE(target_kind,target_id)`，写入既有 `obs` 命名空间会覆盖 S5/S6 的 `reflection / wall_mirror` 标注（曾踩坑）。

---

## 3. 影响面与关键发现

### ① relay 冲突边（最大收获：active 边零污染）

- **7 条判为假链的边全部是 `audit` 状态，`active` 0 条**（`#198 #241 #216 #221 #220 #187 #228`）
  → 生产高置信边集（47 条 active）**未被人工否证任何一条**。
- **5 条判为真**，其中 **3 条是 `active`**：`#135 343-E010→316-E015`、`#158 323-E003→318-E015`、`#171 316-E015→305-E038`
  → 人工**为 active 边提供支持证据**（另 2 条 `#238 #202` 为 audit）。
- 4 条看不清：`305-E056→323-E051`、`318-E054→323-E051`、`323-E051→316-E040`、`316-E022→323-E035`。
  ⚠ 其中 `316-E022→323-E035` 在 GIL `edge` 表**查无此边**（存在于 v2 边集，未落 GIL）—— 口径留痕。
- 结论：**剔除 7 条假链只影响 audit 池，不改变生产 active 边集** → relay v2 的 active 精度获得人工背书。

### ② 倒影搭桥合并

| A 口径簇 | 判定 | 现行动作 |
|---|---|---|
| `g0206`（5 条倒影搭桥 323\|198 / 343\|17 / 343\|35） | **该拆** | 与 B 臂拆法一致（`g0080`+`g0098`） |
| `g0167`（318\|62 / 323\|151,154,155） | **该留** | B 臂拆分被人工**否决** |
| `g0109`（316\|62 / 343\|36） | 看不清 | 维持现状 |

### ③ 倒影实体（S5/S6 的实体级误报）

**确认倒影 7**：`323-E034 / E035 / E038 / E051 / E065 / E074 / E090`
**误报 4（实为真人）**：`316-E015`、`323-E003`、`323-E025`、`323-E066`

- `316-E015`（成员 `316|32`）是 **S5 地板倒影**，`reflection=0.9045` **高置信却误报**；
  且同一实体在 relay 裁决中被判「同人」→ 两条独立结论互相印证它**是真人**。
- `323-E025 / E066` 的 `wall_mirror_src` 分别指向 `66 / 152`，而 **`323|152` 本身属于被发现为倒影的 `323-E065`**
  → 323 玻璃门区存在**链式镜像混淆**（源自己就是倒影）。
- `323-E003` 用户备注「B 为倒影」：A 判为真人，但其 `wall_mirror_src=8` 才是倒影 → **判定锚在错误源上**。

---

## 4. ⚠ 口径漂移警告（需你知晓）

在本次回灌期间，**另一个进行中的任务改写了 GIL 主库**：`store_v1/` 出现 `gil.sqlite.bak_pre_s3v2_20261014`、`gperson_v2.json`、`visit_owner_v2.json`（17:43–17:45）。

| 口径 | 参数 | 节点 | 人数 |
|---|---|---|---|
| S6-b A/B（16:19） | `max_size=8` | 546 / 534 | **264 / 262** |
| 现行主库（17:44） | `max_size=12` + S3 链传导 | 534 | **158** |

**后果**：控制台显示的「264 人」与 GIL 主库现值（158）已不一致；用户所判 3 处搭桥基于 **A 口径**，而现行划分下这三组**已全部拆开**：

- `g0206` 的 `323|198+343|17` → `g0057`、`343|35` → `g0070` → **与「该拆」一致** ✓
- `g0109` 的 `316|62` → `g0060`、`343|36` → `g0072` → 默认已拆
- `g0167` 的 `318|62` → `g0076`、`323|151/154/155` → `g0133` → **与「该留」相反** ✗

→ 30 项裁决中 **27 项（relay 16 + 倒影 11）不受口径影响**（entity/obs 级）；仅 3 处 bridge 需在新口径下重新对齐。

---

## 5. 未执行（等你决定）

结构化生效**默认不执行**（脚本开关已就绪）：

| 开关 | 动作 | 影响 |
|---|---|---|
| `--apply-edges` | 7 条假链 `edge.status → rejected` | 只动 audit 池，active 47 条不变 |
| `--apply-entities` | 4 条误报倒影：清 `obs.flags`（reflection/wall_mirror）+ 标注 `person` | 影响倒影降权臂 |
| （bridge） | `g0206` 拆 / `g0167` 留 | gperson 人数 ±1 |

**推荐**：在**统一口径（cap12）**下重跑一次「人工裁决版」划分与边集（把 30 项裁决作为冻结输入），而不是在已失效的 A/B(cap8) 口径上打补丁 —— 否则新旧口径混用。

---

## 6. 复现

```bash
python tools/gil_adjudication_ingest.py                                        # dry-run + 台账
python tools/gil_adjudication_ingest.py --apply                                # 裁决层（已执行）
python tools/gil_adjudication_ingest.py --apply --apply-edges                   # 可选：假链 rejected
python tools/gil_adjudication_ingest.py --apply --apply-entities                # 可选：撤销误报倒影
```

---

# 附：按推荐执行 —— 「人工裁决版」统一口径重跑（cap12）

> 决策：**在统一口径（cap12）下重跑一次人工裁决版划分与边集**，把 30 项裁决当冻结输入，
> 而不是在已失效的 A/B(cap8) 口径上打补丁。**主库零改动**（裁决以 `--verdicts` 外挂注入）。

## 7.1 做法（不改主库）

给 `gil_gperson.py` 增加 `--verdicts <human_adjudication_v1.json>`（**默认关闭**，不影响既有行为）：

1. `revoke_reflection` → 从倒影排除集合中剔除（视为真人）。撤销 4 个误报实体 → 命中 **4 条**带标记的观测（`316|32 / 323|4 / 323|50 / 323|153`；`323|10` 本就无标记）。
2. `bridge keep` → 对 `b_clusters` 两组之间施加**强制合并**（margin≥0.65）；`split` → **强制禁止**。
3. 全部在内存生效，**不写 GIL 的 flags/annotation**。

```bash
PY=<managed>/python
$PY tools/gil_gperson.py --max-size 12 --drop-reflections \
    --s3-links output_crosscam/analysis/gil/frag_link_v1.json \
    --verdicts output_crosscam/analysis/gil/human_adjudication_v1.json \
    --t-same 0.975 --dump-nodes --out output_crosscam/analysis/gil/gperson_human_v1.json
```

★ **两臂冻结同一 `T_same=0.975`**（撤销节点会改变 val 正负对，否则差异来自阈值漂移）。

## 7.2 结果：同序列 A/B

| 指标 | Baseline（cap12） | **人工裁决版** | Δ |
|---|---|---|---|
| 节点 | 534 | **538** | +4（撤销的倒影回归真人） |
| **gperson 人数** | 158 | **158** | 0 |
| 合并次数 | 376 | 380 | +4 |
| 候选对 | 2424 | 2482 | +58 |
| 不变量违规 | 0 | **0** | — |
| 318×320 GT | 5/8，7 簇 | 5/8，7 簇 | — |
| **人工真值 P / R / F1** | **1.000 / 0.500 / 0.667** | **1.000 / 0.538 / 0.700** | **R +3.8pp，FP 0→0** |
| owner 区间 / 通行段 | 491 / 329 | **495** / 329 | +4 |
| 门户覆盖（gperson 级） | 41/126 (0.3254) | **42/129 (0.3256)** | — |

**读法**：撤销 4 条误报倒影后，它们**并未成为独立的人**（人数仍 158）—— 说明这 4 个实体本就是真人，只是此前被误判为镜像副本而遭降权；纠正后它们**正确归入真人簇**，因此**人工真值召回 0.500→0.538，精度保持 1.000、误接 0**。→ 裁决的价值在于**成员构成正确**而非人数变化。

（baseline 产物 `gperson_v2_check.json` / `visit_owner_v2_check.json` 保留作对照。）

## 7.3 裁决版 relay 边集

`tools/relay_human_adjudicate.py --write` → `analysis/relay_v2/edges_v2_gated_human.json`

| 状态 | 原（196） | 裁决版 |
|---|---|---|
| active | 47 | 44 |
| **confirmed**（人工背书） | — | **5**（含 3 条原 active） |
| audit | 135 | 126 |
| **rejected**（假链） | — | **7** |
| gate | 14 | 14 |
| 合计 | 196 | 196（保留记录，不删除） |

★ **7 条假链全部出自 `audit` 池，`active` 边一条未被否证** —— 生产高置信边集保持不变。

## 7.4 控制台载荷（已生成，未部署）

`webapp/frontend/public/gil_owner_human.json`（人工裁决版 owner 载荷，538 obs / 158 人 / 495 owner / 329 通行）。
→ 想让控制台可见，需在「倒影降权」开关旁再加一档「人工裁决版」并重新部署 dist。

## 7.5 落地：主库 apply + 控制台第三档（已完成）

### ① GIL 主库已切到人工裁决版

```bash
$PY tools/gil_gperson.py --max-size 12 --drop-reflections --s3-links …/frag_link_v1.json \
    --verdicts …/human_adjudication_v1.json --t-same 0.975 --dump-nodes \
    --out …/gperson_human_v1.json --apply
$PY tools/gil_visit_owner.py --gperson-map …/gperson_human_v1.json \
    --out …/visit_owner_human_v1.json --web-out public/gil_owner_human.json --apply
```

| 表 | apply 前 | apply 后 |
|---|---|---|
| gperson | 159（158+S6 幽灵） | **159（158 + gREFLECTION）** |
| membership | 611 | **611（obs 538 + refl 73）** |
| visit / owner | 658 / 491 | **658（329 段）/ 495** |
| **obs / frag / entity / edge / annotation** | — | **零改动**（546 / 278 / 411 / 641 / 809） |
| owner 不变量违规 | 0 | **0** |

备份：`gil.sqlite.bak_pre_human_gperson_<ts>`。

### ② 控制台三档身份口径（已上线）

`http://192.168.11.16:8099/#/` → 打开 **「去重视图」** → 工具条 **「身份口径」** 三档：

| 档 | 载荷 | obs | 人 | owner |
|---|---|---|---|---|
| **现状**（A） | `gil_owner_A.json` | 546 | **155** | 503 |
| **倒影降权**（B） | `gil_owner_deduprefl.json` | 534 | **158** | 491 |
| **人工裁决版**（H） | `gil_owner_human.json` | 538 | **158** | 495 |

★ 三档**统一 cap12 口径**（原 A/B 的 264/262 是 cap8，已随口径升级失效）。
★ 切换同时改变框标签/着色/责任机位所用的 gperson 划分；右栏面板给出三档人数 / owner / 通行对照。

**验收** `webapp/deploy/verify_arms.cjs`：**18/18 通过、控制台 0 错误**；
标准回归 `shot_ui.cjs`（4 视口 × 6 页 × 双主题）**ERRORS 0**。

### ③ ⚠ 并行冲突与规避

落地期间发现**另一进行中的任务也在写同一批文件**：18:01:51 它产出了 `visit_owner_v3.json` 并把
`public/gil_owner.json` 覆盖成了裁决版内容（导致 A 档与 H 档重复）。

→ 规避方式：A 档改用**独立文件名 `gil_owner_A.json`**，前端 `gilOwner()` 指向它，
**不再触碰公共的 `gil_owner.json`**（留给其它任务使用），从而零争用。

### ④ 仍未做

- 4 条误报倒影的 `obs.flags` 撤销（`--apply-entities`）未落主库 —— 裁决版划分已通过外挂覆盖
  实现等效效果；落主库属"写入层"决定，留待确认。
- `render.py` 导出视频仍按 tid 画框（三档只在浏览器叠加层生效）。
