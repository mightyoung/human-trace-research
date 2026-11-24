# switch_fix_v2 代码包复核（2026-10-23）

> 对象：`docs/code/switch_fix_v2.zip`（3 个脚本 + 2 份说明，共 48 KB）
> 复核方式：静态读代码 + 在现有仓库上实测（自比对、台账溯源、冒烟）｜未跑完整 10 组网格
> 结论：**代码能跑、评估逻辑是对的；但有 2 个 P0 问题会让结果静默错掉，必须先修再跑。**

---

## 1 包里是什么

| 文件 | 作用 | 与现有仓库的关系 |
|---|---|---|
| `yolo_track_det_v2.py` | `yolo_track_det.py` 的可替换版：跟踪前去重、按秒设 buffer、阈值可调、可选 OSNet 外观否决、检测缓存 | 新增文件，不覆盖旧脚本；输出 4 个字段与原版同名 |
| `eval_switch_fix.py` | 用 46 条串人标注判断新配置有没有在串人点断开，并用「无串人」轨迹算误切率 | 新增，但**与 M1-A 的 `gt/switch_windows.json` 是两套切点口径** |
| `run_switch_grid.ps1` | 323_train_510 上建 1 次缓存 + 跑 10 组配置 + 出 `summary.md` | 一键脚本，默认参数有误（见 §3 P0-1） |
| `README_switch_fix.md` / `switch_fix_v2_readme.txt` | 说明与交接 | — |

安全检查：全包无网络请求、无 `subprocess`/`eval`/`exec`/`base64`，zip 内路径正常（`tools/` 4 项），无越界写入。

---

## 2 实测验证（4 项）

### 2.1 台账到底由哪份 jsonl 生成 —— `1012_yolo8_dd`，不是 `1012_final`

按台账 `tracks_all.csv` 逐行比 `n_frames`：

| 段 | 台账行 | `1012_yolo8_dd` 一致 | `1012_final` 一致 |
|---|---|---|---|
| 305_train_529 | 272 | **272（100%）** | 1（0%） |
| 318_train_525 | 251 | **251（100%）** | 3（1%） |
| 320_train_514 | 51 | **51（100%）** | 0（0%） |
| 323_train_510 | 309 | **309（100%）** | 3（1%） |
| 305_test_544 / 318_test_534 / 320_test_518 / 323_test_527 | 102/78/40/75 | **全部 100%** | 0–2% |

`1012_final` 是另一套跑法（stride 5、305_train 有 810 条轨迹），与台账（stride 10、272 条）**轨迹编号体系不同**，不能混用。

### 2.2 评估逻辑自比对 —— 通过

`--old` 与 `--new` 都指向同一份 `1012_yolo8_dd`，结果应为「0 断开、0 误切」：

```
| self | 49 | 0 | 42 | 0 | 0% | 0/25 | 0/17 | 576 | 0 | 0% |
```

即：49 个可解析事件 → 42 个可判（7 个 NO_WINDOW）→ 全判 SAME；576 个对照点 → 0 误切。
**逻辑正确，可以用。** 但同时说明：**可判分母只有 42**（窗口放宽到 3/4s 只到 43，6s 到 45，说明 4 个事件是结构性不可判——串人点在轨迹首尾）。

### 2.3 串人标注盘点

- `id_switch=1` 的轨迹 **46 条**（323：27｜305：10｜318：7｜320：2），其中 6 条一行写多个帧号。
- 全部可解析事件点 **51 个**（323：29｜305：12｜318：7｜320：3），**2 条只有文字没有帧号**（`318_train_525#1041`、`323_train_510#2872`，写着 "change to female3/8"）。
- README 说的「49 个事件」对得上：用真实 obs 边界过滤后剩 49，再扣 7 个 NO_WINDOW → 42 可判。

### 2.4 能跑起来 —— 通过

```
--max-frames 3（323_train_510 开头，空场）→ 正常退出，boxes=0（与旧结果一致，非 bug）
--start-sec 800 --max-frames 16 → boxes=16 / tracks=1 / median_len 2.99s，输出 schema 与旧版一致
速度 3.3 f/s（CPU）→ 30 min 段约 45 min 建缓存，与 README 的「1 小时 1.5~2 小时」吻合
```

---

## 3 必须先修的问题

### P0-1 `-OLD` 默认值是错的，且失败时**静默**

`run_switch_grid.ps1` 第 7 行默认 `output_crosscam/1012_final/{seg}.jsonl`。实测该文件与台账 0~3% 一致（见 2.1），必须改为：

```powershell
-OLD "output_crosscam_local/1012_yolo8_dd/{seg}/{seg}_detections.jsonl"
```

失败形态很危险（已实测）：323 段因为 tid 不重合，直接 **0 个事件**、表格显示 `断开率 0%`；305 段 tid 部分重合，会产出**看起来正常但基于错误轨迹**的 SPLIT/误切率。脚本里那行「一致率」是打印，不是断言，跳过它就会把垃圾当真结果。

**改法**：默认值改掉，并把一致率 <90% 改成 `raise SystemExit`。

### P0-2 换跟踪器后没有真值，`precision / C229` 算不出来

README 的下一步写「接 a/b 拆分 → 编码 → 聚类，看 precision / C-229」。但这两张表的真值全部绑在**旧 tid** 上：
`gt_gallery_members_FILLED.csv` 257 条 `track_id=seg#tid`、`gt_query_gt_FILLED.csv` 229 条 `query_id`。
v2 的 tid 是新建的（1 起、按创建顺序），与旧 tid 无对应关系 → `eval_precision.py` 会全部落 U。

**包内没有这个环节的脚本**，必须补一个「旧标 → 新轨」迁移（复用 `eval_switch_fix.map_box` 即可，约 40 行）：对新轨迹逐帧取旧框 IoU≥0.5 的多数 tid → 查该 tid 的 `person_id`；覆盖率 <80% 记 UNK。

### P1-1 10 组配置归因不成立

同时动了三件事，且 10 组都共享：

| 环节 | 旧基线（`1012_yolo8_dd`） | v2 |
|---|---|---|
| 去重 | 跟踪**后**对输出框做 box-IoU≤0.5 NMS（`dedup_detections.py`） | 跟踪**前**做掩膜包含率≥0.9 |
| 跟踪器 | ultralytics ByteTrack（`high .10/low .05/new .12/match .85/buffer 60`） | 自实现门控 ByteTrack（默认 `.35/.10/.45/.80`，buffer 按秒） |
| 输出框 | 卡尔曼平滑后的 `track` 框 | 原始检测框（`--box-source det`） |

所以「v2 变好」分不清是哪一项的功劳；网格里的 `oldthr_s10` 也只是「旧阈值 + 新去重 + 新跟踪器」，不是旧基线。
**补一臂即可归因**：v2 关掉跟踪前去重（`--dedup-mask 0`）、输出框用 `--box-source track`，外面再套 `dedup_detections.py --iou 0.5` → 与旧基线只差跟踪器。

### P1-2 两套切点口径，且没过滤「不可判」

- M1-A 已把切点固定为 `gt/switch_windows.json`（46 轨，`old_usable` 只有 **21 条 true**，23 条 tail、2 条 slide），我记忆里写着「窗口唯一来源」。
- `eval_switch_fix.py` 自己从 `id_switch_frame/unk_reason/note` 里正则抓数字，得到 49 个事件，**不看 `old_usable`**。落在 `old_usable=true` 轨迹上的事件只有 **27 个**。

两套数字（46 窗口 vs 49 事件）写进不同文档会互相打架。建议：以 `switch_windows.json` 为准，评估时 join 它，把 `old_usable=false` 的单独统计一列，别混进主分母。

### P2 其它

- `$PKG` 默认 `D:\data-request-v2_2026-10-23` 本机不存在，应为 `deliverable/data-request-v2`。
- 网格只跑 323_train_510（最差段）。305 有 10 条串人、覆盖最好，只调 323 可能把 305 调坏 → 至少 305+323 两段一起扫（缓存翻倍到约 1.5 h）。
- `--window-sec 2.0` 没必要动（扫到 6 s 也只多 3 个事件）。
- 在 `tmp/` 等非 `tools/` 位置直接运行会因 `ROOT=Path(__file__).parent.parent` 找不到模型和视频（属预期，装到 `tools/` 即正常）。

---

## 4 数字上的现实预期

1. **只有 42 个可判事件**（约 40 条轨迹）。串人率从 100% 降到 50% 看得见；50%→40% 这种量级测不出来。别指望这个指标给出 p 值。
2. **断开率上升 = 轨迹变短 = 图库变少**。图库 883 检出 → 621（≥6 帧）→ 579（去小图）→ 513 进簇，每一道都在分母上。误切率 20% 的代价可能直接把 cov 打下去，而 cov 才是 C/229 的瓶颈。
3. 因此这个包的定位应该是「**跟踪层诊断**」，不是结论。真正的裁决仍是 `cov / same / fm` + 逐条 `C/229`。配置在 TRAIN 段选、在 TEST 段验，时间隔离是成立的，这点没问题。

---

## 5 建议动作

**推荐（约 1 天）**
1. 改 `-OLD`/`$PKG` 默认值 + 一致率改成硬失败（5 分钟）。
2. 补 `tools/transfer_gt_to_new_tracks.py`（旧标→新轨，覆盖率阈值 0.8）—— 没有它，后面两步都是空跑。
3. 网格改成 2 段（323+305）×（旧阈值 / v2 默认 / v2+同框集 / veto 0.4）≈ 6 组，先拿到「归因」再谈「最优」。
4. 选定 1–2 个配置后走完整链路，只看 cov / C229 是否 ≥ 0.95 / 0.515。

**替代**：不修跟踪，直接做 M0 的「图库标注覆盖率」——按已有记忆，制约项是「有身份标注的轨迹数 257/579=44%」，这一项的收益上限比调跟踪更确定。

> 复核用的临时产物都在 `tmp/review_switch_fix_v2/`（自比对 events/controls、冒烟 jsonl）。**复核阶段**未改动 `tools/` 与 `deliverable/`；安装见 §6。

---

## 6 安装与改动（2026-10-23 20:50 执行）

### 6.1 装进 `tools/` 的 4 个文件

| 文件 | 来源 | 改动 |
|---|---|---|
| `tools/yolo_track_det_v2.py` | 包内原样 | 未改 |
| `tools/eval_switch_fix.py` | 包内 | ①一致率 <90% 改硬失败（`--allow-weak-ledger` 可放行）②新增 `--windows`，summary 多一组 `old_usable=true` 子集 ③events.csv 增加 `old_usable` 列 |
| `tools/run_switch_grid.ps1` | 包内 | ①`-OLD` 默认改 `1012_yolo8_dd` ②`-PKG` 默认改 `deliverable/data-request-v2` ③新增 `-PY`/自动设 `PYTHONPATH`/`TMPDIR` ④第 0 步加预检 ⑤新增 `-ATTR 1` 归因臂 ⑥**存为 UTF-8 with BOM**（PS 5.1 否则把中文注释按 GBK 解，语法直接报错——已实测踩到） |
| `tools/transfer_gt_to_new_tracks.py` | **新增** | 旧标→新轨迁移 + 丢失/碎片/仍串三项诊断 |

`tools/eval_precision.py` 另加 3 个可选参数（默认值不变）：`--deliv`、`--gt-dir`、`--c-denom`（默认 229）。

### 6.2 验证证据

| 检查 | 结果 |
|---|---|
| `eval_precision.py` 基线回归 | 用 `--deliv tmp/precision_check --run results` 复跑 42 配置，`precision.csv` 与原文件 **逐字节一致**（A_baseline C=118、精度 0.952、C/229 0.515 全部复现） |
| `--c-denom 196` | A_baseline 由 0.515 → **0.602**，与 `118/196` 手算一致 |
| `eval_switch_fix.py` 自比对（old=new=1012_yolo8_dd） | 49 事件 / SPLIT 0 / SAME 42 / UNMATCHED 0 / 对照 576 点 / 误切 0%；`old_usable` 子集 22 个可判事件 |
| 错误 `--old` | 一致率 0% → `SystemExit`，**退出码 1**（ps1 的 `$LASTEXITCODE` 能拦住）；加 `--allow-weak-ledger` 才放行 |
| `transfer_gt_to_new_tracks.py` 恒等自检（old=new） | 图库 257/883 条、查询 295 条，**逐条 person_id 不一致 0 条**；37 条空身份仍为空；丢失 0 / 碎片 0 / 仍串 0 |
| 跨编号自检（old=1012_yolo8_dd、new=冒烟输出） | 1 条新轨迹正确认领 `323_train_510#287 → male7`，coverage 0.5、share 1.0，未崩 |
| `run_switch_grid.ps1` 语法 | `Parser::ParseFile` 通过（加 BOM 后） |

### 6.3 跑法

```powershell
# 只扫配置（323 段，建缓存约 45–60 min，之后 10 组各几秒）
powershell -ExecutionPolicy Bypass -File tools\run_switch_grid.ps1

# 加"只换跟踪器"归因臂（多建一份不去重缓存，再 +45–60 min）
powershell -ExecutionPolicy Bypass -File tools\run_switch_grid.ps1 -ATTR 1

# 有结论后迁真值 → 跑完整链路
python tools/transfer_gt_to_new_tracks.py `
  --old "output_crosscam_local/1012_yolo8_dd/{seg}/{seg}_detections.jsonl" `
  --new "output_crosscam_local/1012_yolo_v2/v2_s10/{seg}/{seg}_detections.jsonl" `
  --out-dir output_crosscam_local/gt_transfer/v2_s10
```

### 6.4 遗留

- **网格尚未实跑**（需 45–60 min CPU）；`summary.md` 出来后才能选配置。
- **归因臂要第二份缓存**：`--det-cache` 把去重烘进缓存，换 `--dedup-mask` 必须重建；想省时间就只在选定配置上跑归因臂。
- **换跟踪器后 `C/229` 不可直接比**：查询总体从 295 变成新数，分母含义变了。用 `--c-denom <新可答数>` 并同时报「可答数 + 比例」，不要只报绝对条数。
- `switch_windows.json` 里 46 条只有 21 条 `old_usable=true`，可判子集只剩 **22 个事件**——主表用全量 42、结论表用这 22，两个都要写。

---

## 7 v2.1 复核（2026-10-23 21:05）

`docs/code/switch_fix_v2.1.zip` = v2 三件套 + `encode_1012_osnet_seq.rgb.patch`。代码差异只有一处：
`yolo_track_det_v2.py` 的 Encoder 改成 `OSNetEnc(device=device, rgb=True)`（用 `__code__.co_varnames` 做兼容判断），
其余 `eval_switch_fix.py` / `run_switch_grid.ps1` 与 v2 逐字节相同。txt 里另列了 7 项问题，只有第 1 项给了补丁。

### 7.1 颜色通道 bug —— 属实，但收益比云端说的小

**代码事实（我自己核的）**：torchreid 0.2.5 的 `FeatureExtractor.__call__` 对 numpy 输入走 `self.to_pil(element)`
（`torchvision.transforms.ToPILImage`），中间**没有任何 BGR→RGB**；只对 `str` 路径做 `Image.open(...).convert('RGB')`。
我们的 `OSNetEnc.encode` 传的是 `cv2` 读出的 BGR 裁图 → PIL 按 RGB 解读 → 红蓝对调。**主链路 `1012_yolo8_dd_emb` 全程如此。**

**本项目实测**（`tmp/rgb_probe.py`，257 张已标裁图 / 29 身份，单裁级，不是轨迹级）：

| 口径 | 同人均值 | 异人均值 | 差值 | AUC | 留一 top-1 |
|---|---|---|---|---|---|
| BGR（历史） | 0.791 | 0.664 | 0.128 | 0.759 | 0.716 |
| RGB（修正） | 0.802 | 0.664 | 0.138 | **0.770** | **0.724** |

只取**跨机位**配对（646 同人 / 21986 异人，这才是本系统的用途）：AUC **0.550 → 0.583（+3.3pp）**，
同人均值 0.682→0.695、异人 0.659→0.658。方向与云端一致，跨机位上的相对增益比整体更大。

**但别过度期待**：这是单裁、含大量 head/partial 框的困难子集（AUC 只有 0.55–0.58，而轨迹级 body 是 0.981）。
换算成 C/229 是多少，只能重编码 8 段 + 重扫 42 配置才知道。

**补丁安全性已验证**：`--rgb` 默认关，改后默认路径与原文件编码结果**逐位一致（max|Δ|=0.0）**，冻结基线不受影响。

### 7.2 另外 7 项逐条核验

| # | 说法 | 我的核验 |
|---|---|---|
| 2 | 部署配置仍开 veto_dmax 0.30，与结论相反 | **属实**：`configs/selector_deploy.json` 里 `veto_dmax: 0.30`。但它 `authorized_by` 明写 10-19 15:13 你授权；真正的问题是 `expected` 块还在引用**已作废的簇级继承数字**（`labeled_precision 0.806`、`correct_labels 87/229`），与 D4 真值（C 118→90、C/229 0.515→0.393）直接冲突。建议加 `d4_truth` 块并把旧块标 deprecated；是否改 `null` 要你拍板（改了就等于换部署基线） |
| 3 | `run_yolo_u1.sh` 重跑复现不了基线 | **属实**：第 25 行只传 `--stride/--force`，没写 `--iou`；`yolo_track_det.py` 默认 0.5，存档是 0.7+事后去重。配置里的 `caveat` 说两者逐帧≈一致，影响可能很小，但「复现」二字不能这么写 |
| 4 | U 被当成错，评估偏向基线 | **口径要分清**：`precision = C/(C+W)` 不含 U；但 `C/229` 的分母含 U。所以「U 当错」对 C/229 成立、对 precision 不成立。且我们已知 14 条 U 全落在 4 个特定簇，不是随机 |
| 5 | 参数在测试集上选 | **属实**：θ 用 self-cal 在 test 的无标签指标上选，精度又在同一 test 上报。这不是标签泄漏，但必须披露；6 机位可做留出 |
| 6 | fm 被头/身重复框抬高 | **与已有审计一致**：`fm_audit` 写着 14 对里 5 对是同人头肩/全身重复框。v2 的跟踪前掩膜去重正是解法——这也是修串人之外的一个附加收益 |
| 7 | `--save-crops` 置信度回查会对错轨迹 | **低风险**：`run_1012_crops_all.sh` 用同源目录，当前未触发 |
| 8 | 网页控制台无鉴权裸奔 | **属实**：`webapp/deploy/serve.sh:18` → `uvicorn --host 0.0.0.0`，`backend/main.py:38` → `allow_origins=["*"]`，无登录，且 `/media` 直出原始监控视频、POST 可改标注。改 `127.0.0.1` 一行即可，但会改变你现在的访问方式（192.168.11.16），需要你点头 |

### 7.3 已安装

- `tools/encode_1012_osnet_seq.py` 已加 `--rgb`（默认关；等效补丁，因原 patch 与本地 CRLF 不兼容，手工应用后逐位回归通过）
- `tools/yolo_track_det_v2.py` 已换成 v2.1（`rgb=True`）
- 注意：现在**跟踪器的外观特征走 RGB、主链路聚类仍走 BGR**。跟踪器只用外观做否决门，不影响聚类结果；但两边口径不同这件事要写进方法说明。

### 7.4 网格第一次启动失败（已修）

`run_switch_grid.ps1` 在 PS 5.1 下，python 往 stderr 写一行 torchreid warning 就被 `$ErrorActionPreference="Stop"`
判成 `NativeCommandError` 并终止（实测 24 秒退出）。已改成统一走 `Run()` 包装：调用期间临时 `Continue`、
只按 `$LASTEXITCODE` 判失败。修后重跑，预检 100%、OSNet 正常加载、检测 2.8 f/s，正在跑。

---

## 8 网格实跑结果（2026-10-23 23:16）：**串人没被修掉**

10 个配置全部跑完（323_train_510 一段，9000 采样帧建缓存 + 10 组重跟踪，墙钟约 2h06m）。
结果在 `output_crosscam_local/switch_eval/323_train_510/summary.md`。

### 8.1 主表（29 个事件，24 个可判）

| 配置 | 断开 SPLIT | 仍串 SAME | 断开率 | 误切 FALSE_SPLIT |
|---|---|---|---|---|
| oldthr_s10（旧阈值基线） | 1 | 25 | 4% | 4（3%） |
| v2_s10（默认） | 2 | 22 | **8%** | **0** |
| v2_buf1_s10 | 2 | 21 | 9% | 0 |
| v2_buf4_s10 | 1 | 23 | 4% | 0 |
| v2_m70_s10 | 3 | 19 | 14% | 4（4%） |
| v2_m85_s10 | 1 | 24 | 4% | 0 |
| v2_veto40/50_s10 | 3 | 20 | 13% | 0–1 |
| v2_s5 / v2_veto40_s5 | 5 | 19 | **21%** | 3（2%） |

高置信子集（`old_usable=true`，16 个事件）：最好 12%（veto40/50，2/16），默认 v2_s10 只有 6%（1/16）。

### 8.2 判读

1. **效应量小到不能叫"修好了"**：最好 21% 断开，也就是 **79% 的串人点仍然串**。29 个事件里断开 1–5 个，配置间差异全在噪声内。这不是"样本不够测不出"，是**没有效应**。
2. **代价确实很低**：`v2_s10` / `v2_m85_s10` / `v2_buf4_s10` 的误切都是 0（150 个对照点）。所以「宁短勿脏」这条不成立——它既不脏也不短，就是**没断**。
3. **与 V0.4 的 M1-A 结论一致**：切点可定位的只有 16/29，且这 16 个里也断不开。**结论：串人不是跟踪器阈值问题**。
4. 头框事件断开更差（1/16），与已知的「头/后脑勺被检成 person、落进别人身体的簇」盲区一致——这类事件不是"跟丢了"，是**检测框本身就把别人的头当成本人**，跟踪层再准也断不开。

### 8.3 这条否证的价值

- 它把「串人」从"待修的工程问题"降级为"表征/检测层的固有限制"，与已有否证链（掩膜门控、纯 detect、更大检测器、第二骨干、扩图库）同一个位置。
- **省掉后续可能投入的大半天**：除非改成像（更高机位分辨率 / 头肩检测头），否则跟踪层没有可挤的收益。
- 也因此，`veto_dmax` 那条"用 d_max 清洗混人簇"的路线更没理由继续——它要清的混人簇，源头不在跟踪。
