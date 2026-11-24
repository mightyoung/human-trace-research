# human-trace-research

**一个已装好的摄像头网络，到底能交付多少身份识别能力？**

本仓库是一份实证研究的完整可复现材料：面对固定的摄像头阵列与冻结的 ReID 模型，
用**三探针 + 五档分级**判断「哪些身份能力可以承诺、哪些必须转人工」。
论文稿（IEEE T-BIOM 格式，10 页）与全部派生数据、复算脚本一并收录。

> 仓库只含**代码、文档与不含影像的实验数据**。原始视频、帧、crop、人脸图一律不入库
> ——它们体积大、含真实人员生物特征，且可由脚本从原始视频再生。详见 [§3](#3-什么不在仓库里)。

---

## 1. 研究问题与主要结论

已装好的场所要回答的不是「能不能做 ReID」，而是「**这套机位组合能做到哪一档、需要多少人工**」。
本项目用三个免标注探针（人脸像素预算 P1、区域共视 P2、服装多样性 P3）把交付能力分为 L0–L3+ 五档，
并在 6 机位 / 3 采集日 / 74 机位小时的现场数据上逐档验收。

| 结论 | 数值 | 出处 |
|---|---|---|
| 固定模型下的同机位基线 | 精度 **0.974** / 召回 **0.345**（阈值 0.78） | `docs/paper/tbiom-r3/main.tex` §V |
| 身份完整度的结构上限 | 同段 53.8% / 同机位跨时段 42.3% / **跨机位 3.7%** | 同上 |
| L2+ 选择器（同机位、固定 τ=0.80） | **56.4% → 68.2%**（122/179），叠人脸上界 83.2% | §VI-C、`deliverable/derived-results-r3-2026-11-24/` |
| 跨机位外观 | AUC **0.490**（近乎随机）；最好的一对机位 0.904 | §V-B、表 IV |
| 共视是否预测可匹配性 | **不预测**（r=0.18；限制到共享区域 ΔAUC 中位 +0.042） | 补充材料 S8 |
| 姿态核的真实机制 | 同机位 +12（留一法、等误受），主要抬**分数刻度**而非排序 | `docs/exp-003a-template-decomposition-2026-11-24.md` §10 |

**最重要的两条负结果**：跨机位外观不可用于合并（只配排复核顺序）；步态（GEI）已被否证。
**最实用的一条正结果**：把「绝对余弦 ≥ τ」换成「顶一减顶二 ≥ m」的间隔规则，
在同等误受下召回显著更高，且零模型成本。

## 2. 仓库结构

```
src/                      Rust 检测服务（RTSP → GroundingDINO ONNX，零原生依赖）
tools/                    研究脚本：建图、聚类、评估、台账核验（Python）
webapp/                   控制台：FastAPI 后端 + Vue 前端（选择器/标注/工作台）
  backend/                选择器、自动人脸、E21 标注、科研工作台 API
  frontend/src/           Vue 3 视图（含 #/selector、#/e21-label、#/workbench）
  data_snap_10min/        10 分钟数据快照，供控制台无视频启动
ops/                      台账：schema.sql、claims/、views/、reports/（证据链）
docs/                     研究记录（含 paper/tbiom-r3 定稿与 FULL-ACCOUNT 完整说明）
deliverable/              派生数据包（CSV/JSON，无影像）
  derived-results-r3-…/   审稿用派生结果包：179 查询逐项、76 次合并重判、21 对机位
  paper-package-v3/       论文表格的原始 CSV
  gt_long35/ e21-allday-*/ 人工标注真值（仅 JSON）
validation/               CHIRLA / MARS 外部集验证的 CSV 汇总
```

## 3. 什么不在仓库里

| 类别 | 具体内容 | 为什么不入库 |
|---|---|---|
| 影像 | `*.mp4/*.avi/*.mkv`、帧、crop、人脸图、缩略图 | 体积（数十 GB）+ 真实人员生物特征 |
| 内嵌图像的标注页 | `deliverable/*/annotate_*.html` 等 | 单个文件 5–23 MB，本质是 base64 影像 |
| 模型权重 | `models/`、`*.onnx/*.pt` | 可由脚本/官方源重新获取 |
| 原始与中间数据 | `datasets/`、`media-resource/`、`remote-results/`、`*.npz/*.jsonl` | GB 级；嵌入与检测转储可由 `tools/` 重算 |
| 本地凭据 | `.env`（含服务器口令） | 安全 |
| 打包快照 | `research_package/`、`handoff_out/`、`docs/commit-015/` | 与源码重复的历史副本 |

完整排除规则见 [`.gitignore`](.gitignore)。**保留的**是「重新计算所需的最小集」：
代码、配置、标注、逐查询/逐对的 CSV/JSON、台账与文档。

## 4. 复算

```bash
# 环境：Python 3.11+（numpy / opencv / PyMuPDF）；Rust 1.80+（MSVC）

# 1) 选择器 179 查询（表 V）
python tools/exp_pose_pool_real.py --emb-dir tmp/full/emb12 --face tmp/full/face8 \
  --det output_crosscam_local/1012_yolo8_dd --gt <gt_8seg.json> \
  --segs "305_test_544 305_train_529 318_test_534 318_train_525 \
          320_test_518 320_train_514 323_test_527 323_train_510" \
  --dump-queries out/queries.csv

# 2) 判定规则与质量门（EXP-003d/e，只用仓库内的派生物）
python tmp/derived/exp003d_margin_regimes.py
python tmp/derived/exp003e_gate_factorial.py

# 3) 台账证据链一致性
python tools/ledger_flow.py && cat ops/reports/ledger/FLOW.md

# 4) 控制台
cd webapp/backend && uvicorn main:app --host 127.0.0.1 --port 8010
```

脚本所需的嵌入与检测文件按 §3 已排除，需先按 `docs/` 中对应实验卡的命令重跑上游步骤。

## 5. 论文

| 文件 | 说明 |
|---|---|
| `docs/paper/tbiom-r3/main.pdf` | 正文 10 页（限 10 页），0 overfull |
| `docs/paper/tbiom-r3/supplement.pdf` | 补充材料 5 页 |
| `docs/paper/tbiom-r3/FULL-ACCOUNT.md` | 不限篇幅的完整说明：方法、贡献、全部实验结论与诚实账本 |
| `docs/paper/tbiom-r3/CHANGES.md` | r2→r3 逐条改动与待决定项 |

## 6. 数据与伦理

现场数据来自**已获书面知情同意**的办公场所采集，仅用于本研究的身份识别评估。
原始影像不随仓库分发；如需复核，请在数据使用协议下联系作者。
仓库内保留的标注与派生数据均为**不含影像的数值/标识**（轨迹号、框坐标、相似度）。

## 7. 约定

- 提交前确认没有把影像或凭据带进索引：`git diff --cached --name-only | grep -Ei '\.(mp4|jpg|png|onnx)$'`
- 文档命名：`docs/<主题>-YYYY-MM-DD.md`；实验卡与结论同文件，结论须与复算命令一一对应。
- `main.tex` 页数余量约为 0：任何新增段落都要同步删减，改完必须编译确认仍是 10 页。

---

原 Rust 检测服务的架构说明已移至 [`docs/runtime/README-detector-service.md`](docs/runtime/README-detector-service.md)。
