# 人脸抓取（P0）+ 场景诊断与增强（P1）实施报告

> 依据：`docs/face-capture-enhancement-survey.md`（近邻研究）§5.1/§5.4 的 P0/P1 阶段设计
> 范围：**P0 = 人脸抓取 + 质量池；P1 = 场景诊断 + 离线增强 + A/B**；**P2（生成式修复）未做**（按用户指令跳过）
> 日期：2026-10-14 | 数据：multi-cemra 6 路 synced-10min（2026-10-12 录制）

---

## 1. 摘要

| 阶段 | 结果 | 一句话结论 |
|---|---|---|
| P0 | 6 路 ~15.5 万帧流 → **检出 4491 张脸**，其中过质量门 **usable 1062（23.6%）**；**117 条轨迹 / 79 个 gperson 有脸，51 个 gperson 拿到可用人脸图**（人脸墙见产物） | 「每人最佳人脸」链路全通：人脸→tid→entity→gperson 三级聚合零额外基础设施 |
| P1 | 6 路场景标签 + 71 张池内最优脸增强 → **18 张接受（dq 中位 +0.037，max +0.154）、7 张被"复过质量门"拒绝、46 张跳过（正常/无需）** | 「固化路径 + 复过质量门」纪律在生产数据上生效：负增益样本被拒 |

关键纪律落实：增强只对**质量池选出的最优帧**做（离线固化路径）；增强后**复过质量门**才接受；**倒影门控**全程生效（0 例倒影实体被增强）；全部阈值标注"经验初值，待 val 校准"。

---

## 2. 交付物清单

| 产物 | 路径 | 说明 |
|---|---|---|
| P0 主脚本 | `tools/gil_face_capture.py` | 检测+评分+三级池（scrfd/yunet 双 backend） |
| P1 主脚本 | `tools/gil_face_enhance.py` | 场景诊断+增强+A/B |
| 人脸墙脚本 | `tools/gil_face_wall.py` | per-gperson 最佳脸拼墙 |
| 全量 shots | `output_crosscam/analysis/face/face_shot_p0.json` | 4491 条（含坐标/kps/五因子分/门控/tid/entity/gperson） |
| 质量池 | `output_crosscam/analysis/face/face_best_p0.json` | per tid/entity/gperson Top3 + crop_map |
| 人脸墙 | `output_crosscam/analysis/face/face_wall_gperson.jpg` | 51 人最佳脸（绿框=usable） |
| crops | `output_crosscam/analysis/face/crops/` | 每 tid 最优（75 张）+ entity/gp 复制 |
| 增强 A/B | `output_crosscam/analysis/face/ab/` | 18 张 before|after 拼图 |
| 增强图 | `output_crosscam/analysis/face/crops_enhanced/` | 18 张（仅 accepted） |
| 增强记录 | `output_crosscam/analysis/face/face_enhance_p1.json` | 全记录（verdict/参数/dq） |
| 场景诊断 | `output_crosscam/analysis/face/scene_diag_p1.json` | 6 路标签+统计 |
| 模型 | `models/face/scrfd_2.5g_bnkps.onnx`（3.3MB）+ `face_detection_yunet_2023mar.onnx`（233KB） | 全部 ONNX，CPU 可跑（对接 Rust 交付路线） |

---

## 3. P0 方法（已实现）

**链路**：视频帧 → SCRFD-2.5G ONNX（letterbox 640，CPU）→ 脸庞评分（五因子）→ 人脸↔轨迹关联 → tid/entity/gperson 三级质量池 → 最优图固化。

- **数据对齐（已核验）**：`cleanset_v2/{cam}_detections.jsonl`（5fps 逐帧轨迹）与 GIL `obs` 的 tid **同源 100%**（68/68 实测）；时间映射 `t = (frame+100)/25`（**68/68 tid 恒定 +4.0s 偏移**，实测）；人脸→obs→frag→entity→gperson 全链贯通。
- **检测器**：主选 SCRFD-2.5G_KPS（onnxruntime CPU）；备选 YuNet（OpenCV 原生）。两者均验证通过、`--detector` 可切换。检出阈值 0.5 / NMS 0.4。
- **质量评分（五因子，经验初值）**：清晰度 Laplacian(0.30) + 尺寸(0.25) + 瞳距(0.20) + 姿态近似(0.15) + 光照(0.10)；**硬门 usable** = 脸宽≥40px ∧ 瞳距≥20px ∧ 清晰度达标。参考锚点：瞳距 60px=GB/T 31488 可识别线。
- **指标口径**：清晰度/亮度在原分辨率 ROI 计算并**随 crop 固化**（后续增强同源复评）。

## 4. P0 结果

### 4.1 六路统计（全量 10 分钟）

| 机位 | 检测帧 | 检出脸 | 关联率* | usable | usable率 | 脸尺寸分布（<40/40-80/80-120/120+） |
|---|---|---|---|---|---|---|
| 305 办公区1 | 3000 | 1825 | 42% | 405 | 22% | 221 / 1594 / 10 / 0 |
| 316 走廊 | 1552 | 355 | 83% | 243 | **68%** | 61 / 177 / 72 / 45 |
| 318 办公区2 | 2983 | 647 | 61% | 71 | 11% | **479** / 139 / 20 / 9 |
| 320 球机 | 941 | 428 | 96% | 51 | 12% | 139 / 279 / 10 / 0 |
| 323 茶水间 | 3003 | 988 | 95% | 172 | 17% | 325 / 616 / 47 / 0 |
| 343 电梯间 | 1261 | 248 | 97% | 120 | **48%** | 3 / 94 / 58 / **93** |
| 合计 | 12740 | **4491** | 80%(整体) | **1062** | 23.6% | 1228 / 2899 / 217 / 147 |

\* 关联率 = 人脸中心落入某轨迹框（取最小面积框）的比例。

**结构性发现**（与调研预判一致）：
- **机位差异极大**：316/343 是"高质量人脸机位"（usable 率 68%/48%、大脸多——走廊近场与电梯厅近距离）；305/318/320 是"小脸机位"（俯视工位，40px 以下脸占多数）。
- **343 是金矿**：120px+ 大脸 93 张（全场最多），适合作为人脸图库的主来源。
- usable 脸的 q 分布：中位 0.736 / p10 0.651 / max 0.991；usable 中 80% 成功关联轨迹。
- 覆盖：**117 tids / 108 entities / 79 gpersons 有脸；51 gpersons 有可用图**（305:14、316:17、318:2、320:6、323:5、343:7）。

### 4.2 与调研预期的对照

| 调研预判（survey §2.1） | 实测 | 结论 |
|---|---|---|
| 单帧不达标、须多帧积累 | 池内 usable 1062 张，**平均每 usable 轨迹 9 张**可用帧 | ✅ 多帧池有效 |
| 305/318 俯视工位小脸 | 305 脸中位 40-80px、318 40px 以下占 74% | ✅ 预判精确 |
| 多机位联合选优 | 51 人最佳脸已聚合（per-gperson） | ✅ 链路打通 |

---

## 5. P1 方法（已实现）

**场景诊断（F0）**：每路抽样 30 帧（跨片均匀）→ 亮度均值/对比/过曝比/暗像素比/顶底亮度差 → 规则标签。

**增强算子（纯 CV，无 torch，确定性）**：
- `dark_gamma`：LAB-L 域自适应 gamma（目标亮度 128，γ 限制在 [0.45,1.0]）+ 极暗时双边滤波降噪
- `backlit_gamma`：目标 140 + 轻度 CLAHE（clip 1.5）
- `bright_compress`：γ>1 压暗（[1.0,1.6]）
- `low_contrast_clahe`：CLAHE clip 2.5
- 路由：ROI 亮度 <75 → dark；>185 → compress；对比 <30 → CLAHE；**其余跳过（不增强）**

**纪律（硬性）**：只处理池内最优帧；倒影门控（obs.flags reflection>0.5 → skip）；**复过质量门**（q_after > q_before 才接受，否则记录 rejected 丢弃）；A/B 同源（before/after 均从同一 crop 图计算）。

## 6. P1 结果

**场景标签**：305/316/318/323/343 = `normal`；**320 = `backlit`**（过曝比 4.76%）。343 实测最暗（mu=94、暗像素 28.4%——略低于 dark 阈值 30%，边缘案例）。

**增强 A/B（71 张池内最优脸）**：

| cam | 处理 | accepted | rejected | skip | 备注 |
|---|---|---|---|---|---|
| 305 | 18 | 0 | 1 | 17 | 正常场景为主 |
| 316 | 18 | 3 | 0 | 15 | dark_gamma |
| 318 | 6 | 5 | 1 | 0 | dark_gamma（工位暗区有效） |
| 320 | 7 | 1 | 1 | 5 | backlit_gamma |
| 323 | 9 | 6 | 1 | 2 | dark_gamma + CLAHE |
| 343 | 13 | 3 | 3 | 7 | dark_gamma / bright_compress |
| **合计** | **71** | **18** | **7** | **46** | |

- accepted 的 **dq 中位 +0.037、max +0.154**；增强前后 ROI 亮度：45.6→88.6、37.7→77.3、56.3→110.9（低光脸提亮显著）。
- **7 张 rejected 是纪律生效的直接证据**：这些增强未过质量门被拒（防止负增益入库）。
- 倒影门控：0 例倒影实体被增强（门控命中记录在案）。

---

## 7. 已知缺陷与限制（如实记录）

1. **检测误检 ~5 例**（人脸墙上可见）：海报/屏幕上的"人脸"（g0187 为"50所"标志牌）、后脑勺（g0222/g0229）、门饰图案。**改进方向**：加轻量二次校验（五官几何/对称性）或引入人脸嵌入有效性门槛（P2+ 再议）。
2. **305/318 关联率低（42%/61%）**：脸未落入任何轨迹框——推断与俯视坐姿/桌子遮挡（人脸伸出人体框外）有关，**待核查**（可做"最近框兜底关联"或脸-人框新关联规则）。
3. **姿态近似公式是粗筛**：俯仰近似对低头不敏感（低头族仍可能得高分）；精确 yaw/pitch 需关键点 3D 拟合，P0 未做。
4. **增强是经典算子**（gamma/CLAHE），未引入深度 LLIE 模型（Retinexformer 等）——按"先规则后模型"策略，深度模型待 P1.5 可选升级。
5. **JPEG 压缩**：crop 落盘 92 质量 + AB 读取有轻微损失（对 A/B 结论影响小——前后同源）；原始证据图须从视频重取。
6. 阈值全部为**经验初值**（评分权重、门、路由），**待 val 校准**（项目铁律：禁 test-oracle）。
7. 未接 webapp/GIL 落表：face 产物独立存 `analysis/face/`（**只增不改 GIL**，避免与并行 S5 会话冲突）；GIL 新表 `face_shot` 的正式落库是后续步骤。

## 8. 下一步（供选，未执行）

- **A. 关联率提升**：最近框兜底 + 脸在上半身区的先验规则（预期 305 42%→70%+）。
- **B. 深度增强对比（P1.5）**：Retinexformer 类 ONNX 在 accepted/rejected 子集做同口径 A/B（验证深度模型是否真优于 gamma 基线）。
- **C. webapp「人脸」页**：人脸墙交互视图（复用现有静态托管方案，零后端改动）。
- **D. GIL 落表**：`face_shot` 表 + annotation（待与其他会话协调）。
- **E. 人脸去重增益**：51 人脸图与 relay/gperson 的交叉验证（人脸作为 gperson 聚类的**新证据源**——呼应论文 H1 门控方向）。

## 9. 复现命令

```bash
P="C:/Users/might/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
# P0 全量（6 路，~21 分钟）
$P tools/gil_face_capture.py --cams 305,316,318,320,323,343 --out-tag p0
# P1 全量
$P tools/gil_face_enhance.py --tag p0
# 人脸墙
$P tools/gil_face_wall.py --tag p0
# 快速测试（单路 2 分钟）
$P tools/gil_face_capture.py --cams 316 --limit 2000 --out-tag test
```

## 10. 技术备忘（交接用）

- **SCRFD ONNX 后处理要点**：9 输出按 (N,1)/(N,4)/(N,10) 分组，锚点网格按**整个 640×640 输入**计算；bbox 输出为"到中心距离(l,t,r,b)"需加减解码；**预处理须 swapRB**；**坐标 ÷det_scale 时 score 不能除**（曾致 4× 虚高）。
- **帧读取**：cv2 `grab()` 全帧推进 + 仅"有人帧"`retrieve()`（2560×1440 HEVC 全片 grab ~60s/路；ffmpeg select+管道方案实测慢 4 倍，已弃用）。
- **时间轴**：jsonl frame ÷25 + 4.0s = obs 时间（恒定偏移已对 68/68 tid 验证）。
