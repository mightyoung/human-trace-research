# human-count Web 控制台

多机位人员追踪的可视化 / 标注 / 算法配置一体化 Web 应用。
后端 FastAPI（跑在远程 GPU 机的 `rfdetr` 环境，**零额外安装**），前端 Vue3 + Vite，纯静态托管、无 CDN、离线可用。

```
webapp/
├── dataprep/build_web_data.py   # 把分析产物归一化成前端数据包（一次性 / 数据更新后重跑）
├── backend/
│   ├── main.py                  # FastAPI 应用：静态托管 + REST API + 视频 Range
│   ├── render.py                # supervision 单帧渲染 / 标注视频导出
│   ├── evaluate.py              # 跨机判定阈值重算（基于已存分数）
│   └── data/                    # 数据包（cams/tracks/entities/chains/metrics/masks/mirror/pairs/labels/suspects）
├── frontend/                    # Vue3 + Vite + TS 源码；npm run build → dist/
└── deploy/
    ├── make_proxies.sh          # 用 ffmpeg 生成 H.264 网页代理视频
    ├── serve.sh                 # 启动/停止/状态（pid 文件管理）
    └── smoke.cjs                # Playwright 端到端冒烟（截图 + 控制台错误）
```

## 一、功能页面（11 个）

| 页面 | 路由 | 能力 |
|---|---|---|
| **总览** | `#/ops` | 算法中台台账：待判条目/完成率、最近运行、风险与待办（stale run / 失败 run / 待判积压）；台账未部署时给出明确 503 说明 |
| **可视化** | `#/` | 6 路同步播放器（共享 OSD 时间轴）、帧级轨迹框 + ID + 尾迹、**长程全轨迹**、**轨迹预测**（几何外推虚线 + 意图模型）、点击目标跨机联动接力链 + **跨镜头旅程时间线**、图层（全部/去镜保留/倒影/掩膜区）、着色（接力链/清洗状态/同机实体/轨迹ID/**全局人**）、玻璃掩膜叠加、实体存活甘特、落脚点迷你图、C0 指标盘、**★ 去重视图 / 责任划分面板**（Q3：同一人跨机位只保留责任机位一个框 + owner 时间线，数据 `public/gil_owner.json`） |
| **素材标注** | `#/annotate` | ① 跨机候选对复核（318×320，确认/未决/否决，实时写回）；② **轨迹同人复核**（同机跨时段两段轨迹判同人，A 测精度 / B 测召回 / C·D 位置对照，117 对，落 `trackpair_labels.json`）；③ **簇纯度核验**（v2545 走廊机位，逐簇判断是否混人、点出外人 / 看不清，15 簇 / 129 轨，落 `clusterver_labels.json`）；④ 倒影 / 源 标注（按类别筛选）；⑤ 玻璃区域多边形圈选；⑥ 人员属性标注。迷你轨迹图标出**出生/消失端点**（绿/红） |
| **ReID · 语义** | `#/reid` | 全局 ReID 库（539 轨 × 768d × 6 机位）、跨机身份分组（32 组）、以图搜人、**语义查人**（中文描述 → 属性概念 → 检索 → 聚焦跳转） |
| **聚类工作台** | `#/cluster` | 训练情况（12 组无监督实验指标 / 相似度分布 / 意图逐类 P·R·F1 / 训练曲线 / 混淆矩阵）、簇的**重命名 / 删簇 / 成员挪簇 / 移出 / 新建 / 重置**、**★ 图片放大预览**（hover ⤢ 或双击卡片 → 384px 高清 lightbox，←/→ 翻簇内成员，可直接加选区，Esc 关闭） |
| **人脸库** | `#/faces` | P0 人脸证据墙（45 人卡 + 融合/增强对照 + 候选合并证据）、**★ 人工标注**：人卡三态判定（真人/非人/看不清）+ 命名、候选对三态裁决（确认/否决/存疑）、进度统计、只看未标注、CSV 导出（落 `faces_labels.json`，独立审计线，不改 GIL） |
| **真值标注** | `#/truth` | 343×316 出入口事件真值：241 对配对判定（Top-K 30 + 随机审计 20 + 其余）、事件人物编号、互斥冲突标记、实时指标条；保存 = primary 按钮 + Ctrl/⌘+S + 8s 安全网，未保存时拦截离开 |
| **算法配置** | `#/config` | 跨机判定参数在线重算、阈值扫描曲线、分层与人工标签对照、检测级参数只读展示（标注"需 GPU 重跑"）、配置存取 |
| **算法运行** | `#/runs` | run 列表（状态/耗时/类型筛选）→ 详情：参数快照、环境、产物表（uri / 大小 / sha1）、指标、日志尾部；数据来自本机台账 `ops/ops.db` |
| **训练过程** | `#/train` | 训练批次 → 变体表（macro_f1 排序 + 迷你曲线）→ 选中变体的大图（val_f1 / val_acc / best-epoch 标记 / loss 细带）、训练配置、终值 per-cam；数据由 `tools/ops_ingest_train.py` 从历史训练日志回填，新训练走 `ops run` |
| **任务派发** | `#/jobs` | 四列看板（待办/进行中/待验收/已完成）；label 卡显示**实时进度**（done/items + 进度条 + 剩余），队列清零标「队列已清零」；「同步状态」把清零卡推到待验收；**关闭校验队列是否真的判完**，判完则证据由事实自动生成；支持按标注待判量派生任务 |

> **界面设计系统**：见 [`DESIGN.md`](./DESIGN.md)（监控台 · Mission Control 风格，OKLCH 双主题、离线无 CDN、四档响应式）。设计上下文在仓库根 `.impeccable.md`。

## 二、本地运行（开发）

```bash
# 0) 台账（可选；只影响 #/ops 与 #/runs 两页。本机为主库）
python tools/ops_db.py init && python tools/ops_ingest.py

# 1) 生成数据包（数据更新后需重跑）
python webapp/dataprep/build_web_data.py

# 2) 生成网页代理视频（本地预览用）
bash webapp/deploy/make_proxies.sh media-resource/multi-cemra/synced-10min \
     webapp/backend/data/media

# 3) 起后端
cd webapp/backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000
#    ★ 本机 8000 常被 C-Lodop 打印服务占用（0.0.0.0:8000，任何路径都返回它的欢迎页）
#      → 冲突时换端口，例如 --port 8010，访问 http://127.0.0.1:8010/#/annotate

# 4) 前端（开发模式，代理到 8000）
cd webapp/frontend && npm install && npm run dev
# 或构建静态包由后端托管：npm run build
```

## 三、远程部署

```bash
# 打包（排除体积大的代理视频与依赖）
tar -czf webapp/deploy/hc-webapp.tgz \
  --exclude='backend/data/media' --exclude='frontend/node_modules' \
  webapp/backend/*.py webapp/backend/data webapp/frontend/dist webapp/deploy/*.sh

# 上传解包
MSYS_NO_PATHCONV=1 python tools/remote_run.py --put <本地tgz> /home/r4900g3/human-count/hc-webapp.tgz
MSYS_NO_PATHCONV=1 python tools/remote_run.py 'cd ~/human-count && tar -xzf hc-webapp.tgz'

# 远程生成代理视频（源视频已在远程）
MSYS_NO_PATHCONV=1 python tools/remote_run.py \
  'cd ~/human-count; nohup bash webapp/deploy/make_proxies.sh \
   /share/data4/yhh/data/human-count/video-6way \
   /home/r4900g3/human-count/webapp/backend/data/media > webapp/deploy/proxy.log 2>&1 &'

# 起服务
MSYS_NO_PATHCONV=1 python tools/remote_run.py 'cd ~/human-count && bash webapp/deploy/serve.sh start 8099'
```

访问 `http://192.168.11.16:8099/`。

## 四、接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 依赖自检（supervision / cv2 / media / static） |
| GET | `/api/cams` `/api/tracks/{cam}` `/api/entities` `/api/chains` `/api/metrics` `/api/masks` `/api/mirror` `/api/pairs` | 数据包 |
| GET | `/api/suspects` | 疑似同人（换号）候选：**只用于页面标注**，不参与实体图/指标；缺文件时返回空 |
| GET/POST | `/api/labels` | 标注库读写（增量合并） |
| GET/POST | `/api/config` | 算法配置读写 |
| POST | `/api/evaluate` `/api/sweep` | 分层重算 / 阈值扫描 |
| GET | `/api/frame?cam&f&view&trail&masks` | supervision 渲染单帧 JPEG |
| POST | `/api/export` + GET `/api/export/status/{id}` | 后台烘焙标注视频（H.264） |
| GET | `/media/{cam}.mp4` | 视频流（支持 Range，浏览器可拖动） |
| GET | `/api/truth/state` `/api/truth/metrics` | 真值标注（配对/事件/冲突/预算指标） |
| POST | `/api/truth/labels`；GET `/api/truth/export?kind=pair\|event` | 真值保存（增量合并）/ 导出 CSV |
| GET | `/truth/thumbs/*` | 真值配对缩略图 |
| GET | `/api/gil` | GIL 统一存储概况（obs/frag/entity/edge/annotation 计数、边构成、源哈希） |

### 算法中台（P0，本机为主库）

> 权威库 `ops/ops.db` 在本机；远程只做算力与预览。远程找不到库时以下接口返回 **503** 并说明原因，不影响其它页面。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/ops/status` | 台账自检（`ok` / `db` / `authority`），前端据此决定是否显示入口 |
| GET | `/api/ops/overview` | 总览：标注进度（按任务）、运行健康、派发待办、表行数 |
| GET | `/api/ops/tasks` | 8 个标注任务的条目数 / 已判 / 待判 / 百分比 |
| GET | `/api/ops/tasks/{task_id}/items` | 条目列表（含最新判定与作者），支持 `status` / `limit` / `offset` |
| GET | `/api/ops/queue` | **标注页读入口**：`task_id` 可逗号分隔；返回任务进度 + 条目 + 最新判定（`label.data` 为原始对象，`by_source`/`seeded` 区分 seed 与真人） |
| POST | `/api/ops/label` | 写一条判定：`{item_id, value, annotator?}`，`value=null` 表示清除；**先写台账，再投影回原始 json** |
| POST | `/api/ops/verify` | 比对台账与 4 个 json 的差异（missing / conflict） |
| GET | `/api/ops/runs` `/api/ops/runs/{id}` | 运行列表（可按 kind/status 过滤）/ 详情（参数、环境、产物 sha1、指标、`series` 曲线、所属 experiment） |
| GET | `/api/ops/runs/{id}/log` | 日志尾部（`tail` 行） |
| GET | `/api/ops/train` | 训练批次 + 变体 + 每组曲线（实验组来自 `experiment` 表；未挂 experiment 的训练 run 归入「未归入批次」） |
| GET | `/api/ops/jobs` | 派发任务卡列表（可按 status 过滤） |
| POST | `/api/ops/jobs` | 新建任务卡：`{title, kind, goal?, inputs?, acceptance?, priority?, authorization?}` |
| POST | `/api/ops/jobs/{id}/status` | 状态流转：`{status, evidence?}`；越级流转与无证据关闭均返回 400 |
| POST | `/api/ops/jobs/derive` | 按标注待判量派生 label 任务（幂等：已有未关闭的同类卡则跳过） |
| POST | `/api/ops/jobs/sync` | 把队列已清零的 label 卡推进到「待验收」（不自动关闭，关闭仍需人确认） |

**新训练**：走 `tools/ops_run.py --kind train` 包起来，训练脚本里用 `tools/ops_metric.py` 的 `Reporter` 每 epoch 上报；包装器会注入 `OPS_RUN_ID` 并记录执行脚本 sha1（本仓无 git，代码身份靠它）。

**远程训练**（算力线在另一台机器）：远程也部署了同一套 `tools/ops_{db,run,metric,bundle}.py` + `ops/schema.sql`，在那里 `ops_run` 包一层跑，跑完 `ops_bundle.py export --run <id>` 导出单条 run 的闭包（run + 曲线 + 产物 + 所属实验），拉回本机 `ops_bundle.py import` 合并。本机始终是唯一权威库，远程那份只是过程账本。操作细节见 `docs/ops/leader-run-handoff.md`。

**指标文件名约定**：`ops_run` 依次找 `metrics_full.json` → `metrics.json` → `summary.json`，写入前统一归一（`ops_db.normalize_metrics` 认 `{test}` / `{overall}` / `{rep:{...}}` / 顶层四种形态，原始结构保留在 `metrics._raw`）。

**口径**：`label` 经 item→task→dataset 链携带 `dialect`（现为 `clean2`）与 `snapshot_hash`；改判追加新行并 `supersedes` 指向旧行，不覆盖。

### 算法工作台（v1.1 新增）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/track/thumb?cam&tid&f&w` | 任意轨迹的人物缩略图（从代理视频现裁） |
| GET | `/api/reid/store` | 全局 ReID 库（539 轨 / 768d / 6 机位 + 属性短语表） |
| GET | `/api/reid/similar?cam&tid&k` | 以图搜人（跨机位余弦检索） |
| GET | `/api/reid/text?q&k` | **语义查人**：中文/英文 query → 属性概念 → 词表向量组合 → 检索 |
| GET | `/api/reid/entity?cam&tid` | 该轨迹的接力链 / 同机实体反查 |
| GET | `/api/reid/report` `/api/reid/text-eval` | 离线聚类/意图指标、文本检索自评 |
| GET | `/api/cluster/state` | 聚类工作台状态（簇 / 成员 / 未分配 / 操作日志） |
| POST | `/api/cluster/rename` `delete` `move` `remove` `reset` | 重命名 / 删簇 / 挪成员 / 移出到未分配 / 重置 |
| GET | `/api/predict/future?cam&tid&f&horizon` | 几何外推的未来落脚点 |
| GET | `/api/predict/intent?cam&tid&f` | 意图模型（5 类）预测 + 置信度 |
| GET | `/api/predict/all?cam&tid&f` | 两者合并 |
| GET | `/api/faces/labels` | 人脸库人工标注（persons / pairs，落 `faces_labels.json`） |
| POST | `/api/faces/label` | 人卡标注：`{label, verdict? 真人\|非人\|看不清, name? 命名}`（传 null 清除，非法值 400） |
| POST | `/api/faces/pair` | 候选合并对裁决：`{a, b, decision? CONF\|REJ\|MAYBE}`（null 清除） |
| GET | `/api/faces/export` | 导出标注 CSV（UTF-8 BOM，防公式注入） |
| GET | `/api/trackpair/pairs` | 轨迹同人复核：117 对（A/B/C/D）+ 已有判定 + 分类统计 |
| POST | `/api/trackpair/label` | 逐对判定：`{id, label: same\|diff\|unsure\|null}`（null 清除；id 不在 pairs.json → 404） |
| GET | `/api/trackpair/stats` | 按类别（A/B/C/D）的对数 / 已标 / 同人率 |
| GET | `/api/trackpair/export` | 导出全部对 + 判定（JSON 下载） |
| GET | `/trackpair/strips/{name}` | 裁图（3 帧条带 + `__full.jpg` 全景，目标框已内嵌） |
| GET | `/api/clusterver/items` | 簇纯度核验：15 簇 / 129 轨（成员按时间排序）+ 判定 + 标记 |
| POST | `/api/clusterver/label` | 逐簇判定：`{id, verdict: pure\|mixed\|multi\|unsure\|null, marks:{track:"out"\|"unread"}}`（只收本簇成员） |
| GET | `/api/clusterver/stats` | 计数 + **簇纯度 / 簇内对精度**（按轨、按对数加权；`multi` 与 `unsure` 不计入） |
| GET | `/api/clusterver/export` | 导出全部簇 + 判定 + 标记（JSON 下载） |
| GET | `/clusterver/strips/{name}` | 裁图（每条轨 3 帧，红框已内嵌） |
| GET | `/faces/face_data.json` | 人脸证据数据（静态，随 dist 部署） |
| GET | `/figs/*` | 训练曲线 / 混淆矩阵 / 相似度分布图 |

## 五、数据链与口径

- 数据包由 `output_crosscam/` 的分析产物构建：**clean2 口径**（`cleanset_v2` 掩膜+镜像清洗）+ `stitch_clean2` 实体图 + `relay_clean2/cradio_v5` 接力链 + `c0_baseline_v5` 基线。
- 会话时间轴：`t = 相机偏移 + frame/fps − min_offset`，t=0 ↔ 08:54:55（各机位首帧 OSD 相对 08:55:00 的偏移）。
- 轨迹状态由数据层计算：`keep` / `reflection`（`floorref_labels.json` 28 条）/ `masked`（脚点 >50% 落在玻璃掩膜内）。
- 坐标系：检测框为**原始分辨率像素**，前端 Canvas 按显示尺寸缩放；服务端渲染按代理视频实际尺寸缩放。

## 六、诚实边界

- **检测/跟踪级参数**（`box_threshold` / `containment` / `track_activation` / `stride`）改动必须在远程 GPU 重跑推理，本页**不伪造**结果，仅展示当前生效值。
- **算法配置页的"重算"**只重算跨机候选对的判定分层（输入是已存的 `reid_cos` / `sp_median` 等分数），不是全链路重跑。
- 标注结果写入 `backend/data/labels.json`，**不会自动回写**到分析流水线；如需回流需另接导出步骤。
- **「疑似同人」是标注不是判定**：`data/suspects.json` 从 stitch 候选里挑「外观很像 + 时间紧邻 + 位置接得上、但拼接没连上」的相邻实体段（当前 53 组 / 66 段），页面只在画布画**虚线** + 标签后缀 `≈`/`?`，并在检查器列出可跳转的各段；**实体图、接力链、指标都不受影响**，关掉「同人候选」开关即完全消失。硬约束是「同帧共现必为两人」，由 `tools/check_suspects.py` 断言（组内两两共现必须为 0）。
- **台账（P0）的边界**：`#/ops` 只读聚合与进度，`#/runs` 只读运行记录；标注**判读交互仍在原三页签**，但三页的**判定已统一读台账**（`/api/ops/queue`，B1① 读切换），写入仍走各页老接口并由 `_ops_mirror` 同步台账；`stale` 判定见 `tools/ops_stale.py`。

### 算法工作台的能力边界（v1.1）

- **语义查人是"属性词表组合"，不是任意句子的 open-vocab**。远程无 EmbeddingGemma 模型且无外网，故文本塔向量**本地预计算**成 109 条属性短语（13 色 × 6 服饰 + 独立概念）随包下发；运行时把中文 query 解析成概念后取均值。离线自评（539 库、已知人物正例集）top-5 命中 **6/8**；失败硬例为「黑卫衣」「毛皮领」。
- **全局 ReID 库**是 `cradio_gemma2_bridge_v4`（与文本塔同 768d 空间，实测同源 cos≈0.99）。它的 539 条轨迹 **100% 落在** `video-6way` 的检测轨迹 tid 上，因此缩略图与时间轴可无缝对齐。
- **聚类**在 539 轨上以 mutual-kNN（k=3, thr=0.95）重算：219 簇 / 90 个多成员簇 / 最大 16。参数从"身份阈值 ~0.95"标定而来（k=5,thr=0.3 会得到 159 项的糊簇）。工作台的操作**只改 `clusters_state.json`**，不影响 `selector.npz` 与应用侧。
- **轨迹预测**有两层：前端**几何外推**（等速平滑，人人可用，虚线绘制）+ 后端**意图模型**（BERT 范式 Transformer，acc 0.668 / macro-F1 0.477，仅当历史窗 ≥12 采样点时有输出）。意图模型输出的是"未来 1s 的运动类别"，不是精确落点。
- 时间轴点击定位此前用容器宽度换算（含 118px 标签列）导致跳转偏移，v1.1 改为以轨道元素为准。

## 七、界面重设计（v2.0）

- **设计系统**：OKLCH 双主题（深/浅，跟随系统 + 手动切换）、系统字体栈（离线无 Web Font）、4pt 间距、三级文字 + 等宽数字、状态色只编码状态。完整规范见 [`DESIGN.md`](./DESIGN.md)。
- **响应式四档**：`<768px` 单机位 + 底部标签栏；`768–1023px` 顶栏导航（永不换行）+ 抽屉；`1024–1439px` 三栏紧凑（210/288）；`≥1440px` 三栏（232/316）。视频瓦片用**容器查询**决定 2/3 列，与视口解耦。
- **交互**：键盘（空格播放、←/→ 逐帧、1–6 机位、Esc 关抽屉）、`focus-visible` 焦点环、触控目标 ≥44px、`prefers-reduced-motion` 降级。
- **空状态**：可视化右栏在未选中目标时显示四步操作指引，而非三个空面板。
- **验证方式**：Playwright 四视口 × 六页面 × 双主题自动巡检（`deploy/shot_ui.cjs`）+ 交互断言（`deploy/verify_ia.cjs` / `verify_chain.cjs` / `verify_fixes.cjs`），本地与远程各跑一遍，控制台零报错。
- **数据代际可见**：配置页「GIL 统一存储」卡展示 obs/frag/entity/边构成/标注数 + 源哈希（由 `build_web_data.py` 从 `analysis/gil/store_v1` 并入 `gil.json`）；接力链面板在 `--relay v2` 数据源下显示每条链的证据等级（`audit ×N` / `active`），v5 数据源零变化。
- **标注页脚点轨迹**：16:9「画面归一化」迷你地图（画框边界 + 三分网格 + 起止点 + 近静止轨迹的簇范围环 + 采样点数），替代原来的大块纯黑面板。
