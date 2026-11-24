# S4 报告：Q3 责任划分 —— `visit`（通行段）+ `owner`（责任时间线）+ 去重计数

> 2026-10-14 · 工具 `tools/gil_visit_owner.py` · 产物 `analysis/gil/visit_owner_v1.json` + GIL（`visit` 846 行 / `owner` 505 行）
> 上游：S2 `gperson`（264 簇）、S3 `frag`、`analysis/c0/c0_baseline_v7.json`（门户覆盖参照）
> 一句话：**给每个"人"一条唯一责任时间线** —— 任一时刻同一 gperson 只有一个 owner（0 违反），重叠视野天然去重；并把 C0 的门户覆盖从**边级**降到**身份级**重测。

---

## 0. 结论先行

| # | 结论 | 证据 |
|---|---|---|
| 1 | **去重成立**：546 条轨迹 → 411 机位内实体 → **264 gperson**（去重 35.8%） | `gperson` + `membership` |
| 2 | ★ **owner 硬不变量：0 违反** —— 任一时刻同一 gperson 的 owner 唯一 | 独立复核 505 个区间 |
| 3 | 重叠视野去重生效但**量很小**：`reason=visibility`（多机位同时可见后择一）仅 **4** 段 | 同人被 2 个机位同时看到的时间占 **0.65%**（1 机位占 99.35%） |
| 4 | ★★ **互斥审计 0 例**：obs 级跨机位同时可见（排除 318×320）**不存在** → S2 的时间互斥约束真实成立 | 阈值：重叠 > 0.6s（留 0.2s 量化 + 偏移余量） |
| 5 | ⚠ **口径陷阱（已修）**：`visit` 是"gap≤30s 聚合"的**包络**，当"可见区间"用会造出假并发 | 用包络时出现 6 对非重叠机位"同时可见"（316×318 等）；改逐 obs 判定后 **0 例** |
| 6 | ★ **身份级门户覆盖 0.379**（+30s 时间门 **0.290**），对照 **C0 v7 边级 0.626** | 移动门户 exit 145 → 覆盖 55（C0：123 → 77） |
| 7 | 分机位：343 **0.579** / 316 **0.548** / 318 0.500 / 323 0.323 / 320 0.200 / **305 0.147**（最差，与 S-1 的"结构性"结论一致） | 见下表 |
| 8 | GIL：`visit` 423 段（846 行）+ `owner` 505 行；`--apply` 幂等；冒烟 **9/9** | 仅新增两表，`obs/frag/entity/gperson` 未动 |

---

## 1. 为什么做（Q3 原话）

> "多画面中的同一个人没有去重（目前的展示视频还没有实现责任目标的划分）"

现状（`docs/reid-identity-ownership-redesign.md` §1.3）：渲染按**每机位各自的 track id** 画框
（`webapp/backend/render.py`：`_track_color(tid)=_PALETTE[tid%N]`），**没有任何跨机去重**；
也没有"某时刻哪个机位负责谁"。算法产物（链）存在，但**没有任何一处把它当"人次真值"消费**。

→ 缺的不是算法，是**消费层**。S4 补 `visit` + `owner` 两个表，使：

```
gperson（谁）
  └── visit（在这个机位的一次通行；entry/exit 各一行，带脚点/航向/贴边距离）
        └── owner（t 时刻由哪个机位负责 —— 唯一）
```

---

## 2. 定义与算法（可复述）

**visit**：把一个 gperson 在**同一机位**的成员 obs 按时间排序，间隔 > `VISIT_GAP=30s` 处切开；
每段产 **两行**（`kind='entry'` / `'exit'`，符合 schema 语义），带：
归一化脚点 `(u,v)`、航向 `dir`（°，由头/尾速度）、到画面边界最小归一化距离 `edge_margin`、`h/H`。

**owner**：把该 gperson 全部 obs 的 `[t_first,t_last]` 端点排序成基本区间，逐区间求
```
u(cam,g,t) = 归一化框高 h/H  ×  (0.5 + 0.5·min(1, edge_margin/0.05))
owner(t)   = argmax_cam u
```
相邻同机位区间合并；`reason` 三态 **`visibility`（多机位同时可见→按效用择一，即去重）/ `handoff`（换机位且间隙 ≤30s）/ `sole`**。

★ **关键实现纪律**：`u` 的"活跃"判定**必须逐 obs**。`visit` 的包络区间（gap≤30s）会把
"同一机位两段相隔 25s 的碎片"连成一个 30s 长区间，从而在**非重叠机位对**上造出"同时可见"假象
（实测 316×318 ×3、316×343 ×1、318×323 ×1、323×343 ×1）；改成逐 obs 后这些全部消失。

**门户覆盖（gperson 级）**：与 C0 v7 同口径 —— 只算**移动**段（脚点跨度 ≥0.05 且时长 ≥1.0s）、
端点按 `border / portal / transient` 分类（弱门 + 长段 = `interior_vanishing` → 剔出分母）；
`covered` = 同一 gperson 在**另一机位**有 visit，`+30s` 严格版还要求 `0 ≤ t_其他 − t_本段 ≤ 30s`。

---

## 3. 结果

### 3.1 规模

```
obs 546 → 机位内实体 411 → gperson 264            （去重 35.8%，相对实体）
visit 423 段（移动 153；多 obs 段 79）            按机位 305:84 316:55 318:110 320:28 323:96 343:50
通行段时长 p10 0.2s / p50 6.6s / p90 64.4s / max 599.8s
owner 505 区间（涉及 235 个 gperson；29 个 gperson 全是零长单点观测 → 无区间）
reason：sole 437 / handoff 64 / visibility 4
```

### 3.2 去重的量化（"未去重"到底有多严重）

```
同一 gperson 同时被 K 个机位看到的时间占比：K=1 → 99.35% ；K=2 → 0.65% ；K≥3 → 0
```

★ **诚实结论**：本数据里"同一人被多机位同时看到"的情况**很少**（0.65%），
所以**"去重视图"的收益不在于消除同屏重复框**，而在于：
① **人次计数**从 411 实体降为 264 人；② **每段通行有唯一 owner**（跨机位的"谁跟着谁"不再各自为政）。

### 3.3 门户覆盖：边级 vs 身份级（C0 对照）

| 机位 | exit 门户（移动） | 覆盖 | 覆盖率 | +30s 时间门 |
|---|---|---|---|---|
| 305 | 34 | 5 | **0.147** | 0.147 |
| 316 | 31 | 17 | **0.548** | 0.452 |
| 318 | 20 | 10 | 0.500 | 0.300 |
| 320 | 10 | 2 | 0.200 | 0.200 |
| 323 | 31 | 10 | 0.323 | 0.194 |
| 343 | 19 | 11 | **0.579** | 0.474 |
| **合计** | **145** | **55** | **0.379** | **0.290** |
| 对照 C0 v7（**实体级/边级**） | 123 | 77 | **0.626** | — |

**读法**：
- C0 的 0.626 判据是"**该实体有任一出边**"（图论邻接，不看对端是谁）；
- 本表的 0.379 判据是"**同一个 gperson 真的在别的机位出现**"（身份级）；
- 两者差 **0.247** —— 这部分就是"**图上有边、但身份层不承认是同一人**"的量，
  与 S2 的发现（196 条 relay 边中仅 62 条落进同一 gperson）同源；
- **加 30s 时间门后再降 0.089** → 即约 1/4 的"覆盖"其实对端出现得太晚，不是可信交接。

→ ★ **可操作结论**：进一步提覆盖的杠杆在 **gperson 召回（S2 的 R 0.538）**，不在门户几何。
305 的 0.147 与 S-1 的"出口多通向无摄像头区域"结论一致，**不是本层能修的**。

---

## 4. 与既有资产的关系 / 验收对照

| 设计文档 §3.4 验收项 | 状态 |
|---|---|
| ① 任一时刻同一 gperson 的 owner 唯一（0 违反） | ✅ **0 违反**（505 区间独立复核） |
| ② 去重后人次与事件页人工账本对账 | ⚠ **不可做**：`truth/events_all.csv` 214 行**尚未填 `human_person_id`**（0 行有值）→ 只能对**结构**（107 次通行）；本层 visit 对 316/343 分别给出 55/50 段（含静止段），移动段口径待人工账本补齐后再对 |
| ③ 305 出门覆盖修复前后对比 | ✅ 已给：边级 0.626 → **身份级 0.379**；305 = 0.147（结构性，非本层可修） |

| 资产 | 关系 |
|---|---|
| `gperson` / `membership` | 只读消费；**未修改** |
| `frag`（S3）/ `entity` | **未修改**（owner/visit 是独立新表） |
| `relay v2` / C0 | 只读参照（`c0_baseline_v7.json` 的 gates + 覆盖数字） |
| `render.py`（渲染） | **未改**；去重视图仍需 S4-b |

---

## 5. S4-b 消费端：控制台「去重视图」（已完成）

**交付方式**：**纯前端**（零后端改动、零服务重启）——
数据由 `tools/gil_visit_owner.py --web-out` 导出到 `webapp/frontend/public/gil_owner.json`（39 KB），
随 `dist` 静态托管；前端 `fetch('/gil_owner.json')`。

| 改动 | 文件 | 说明 |
|---|---|---|
| 载荷 + API | `webapp/frontend/src/api.ts` | `gilOwner()` |
| 去重逻辑 | `src/store.ts` | `gidOf / ownerAt / personsAt`；`tracksAt()` 增 `dedup` 过滤 + `{raw:true}` 对照；着色新增 `gperson` |
| 框标签 | `src/components/CamPanel.vue` | 去重/按人着色时显示 `g0007`（`◆` = 该时刻多机位同时可见、已择一）；机位头新增「负责 N」 |
| 面板 | `src/components/OwnerPanel.vue` | KPI（264 人 / 35.8% / 423 段 / **0 冲突**）、身份级vs边级覆盖条、当前时刻「人→责任机位」表、各机位负责数 |
| 入口 | `src/views/VisualizeView.vue` | 工具条「去重视图」pill（开启同时把着色切到「全局人」）+ 右栏面板（含窄屏抽屉） |
| 验收脚本 | `webapp/deploy/verify_dedup.cjs` | 16 项断言 |

**验收结果（远程 `http://192.168.11.16:8099/`，Playwright/Edge）**：

```
结果: OK 16 / FAIL 0      控制台错误: 0 ✅
  · KPI 264 全局人 / 0 owner 冲突 / 身份级覆盖 / C0 边级对照 62.6%(77/123) 均正确渲染
  · 「去重视图」按钮进入按下态；6 个机位头出现「负责 N」；着色自动切到「全局人」
  · ★ 去重在 t≈58s 生效：未去重 18 框 → 去重后 17 框（省掉 320 侧同一人的重复框）
  · 移动端底部标签栏仍 6 项（未新增导航页，符合 DESIGN.md §五）
标准回归 deploy/shot_ui.cjs（4 视口 × 6 页 × 双主题）：ERRORS 0
```

**部署与回滚**：仅替换远程 `webapp/frontend/dist`（旧版备份 `dist.bak.1014-1520`），后端 `main.py` **未改**
（远程 md5 仍为部署前值），服务**未重启**。

★ **一个必须说清的量级事实**：离线扫描（`tools/_find_dedup_moment.py`）显示**全片只有 8 个 2s 采样点存在"同人被多机位同时画出"**，
每次至多省 1 个框。所以「去重视图」的可见收益很小；它的真实价值是
**① 人次计数从 411 实体 → 264 人；② 每段通行有唯一责任机位（owner），使"谁跟着谁"不再各自为政**。

---

## 6. 未做 / 下一步

- **已完成（S4-b，消费端）**：控制台「**去重视图**」——
  `colorBy: 'gperson'` ✅；同一 gperson 只保留责任机位的框（t≈58s 实测 18→17）✅；
  owner 面板（KPI + 当前时刻「人→责任机位」+ 身份级/边级覆盖对比 + 各机位负责数）✅；
  机位头「负责 N」✅；移动端标签栏仍 6 项 ✅。详见 §5。
- **未做**：`visit` 的门户语义（entry/exit 配对成"通行事件"并接入 relay 图状态机）；
- **未做**：计数写进渲染链路（`render.py` 导出视频仍按 tid 画框；去重视图目前只在浏览器叠加层生效）；
- **未做**：319×320 之外的**时钟偏移复核** —— 本次互斥审计 0 例，反向支持现有 `TIME_OFFS` 可用；
- **待人工**：真值事件账本填 `human_person_id` → 才能做人次级对账（设计文档验收②）。
- **下一步候选**：**S5 倒影指派**（Q4，把倒影做成"被真人解释"的指派问题）。

## 7. 复现

```bash
# ① 数据层：指标 + JSON + 控制台载荷
python tools/gil_visit_owner.py            # 同时写 webapp/frontend/public/gil_owner.json
python tools/gil_visit_owner.py --apply    # 写 GIL visit/owner（幂等）
python tools/gil_smoke.py                  # 9/9
python tools/_find_dedup_moment.py         # 找「同人被多机位同时画出」的时刻（默认 8 个）

# ② 前端：构建 → 只替换远程 dist（无需重启后端）
cd webapp/frontend && npm run build
tar -czf ../deploy/dist.gilowner.tgz -C . dist
MSYS_NO_PATHCONV=1 python tools/remote_run.py --put webapp/deploy/dist.gilowner.tgz \
    /home/r4900g3/human-count/dist.gilowner.tgz
MSYS_NO_PATHCONV=1 python tools/remote_run.py \
  'cd ~/human-count && rm -rf _newdist && mkdir -p _newdist && tar -xzf dist.gilowner.tgz -C _newdist \
   && cp -r webapp/frontend/dist webapp/frontend/dist.bak.$(date +%m%d-%H%M) \
   && rm -rf webapp/frontend/dist && mv _newdist/dist webapp/frontend/dist'

# ③ 验收
NODE_PATH=<managed-node>/node_modules node webapp/deploy/verify_dedup.cjs http://192.168.11.16:8099
NODE_PATH=<managed-node>/node_modules node webapp/deploy/shot_ui.cjs    http://192.168.11.16:8099
```
