# 16 路实时处理能力评估 + EmbeddingGemma2 量化可行性研究

**日期**：2026-10-14 ｜ **性质**：实测研究（远程 GPU 基准 + 外部调研）｜ **实验产物**：`output_crosscam/analysis/bench_{det_bench,det_bench2,emb_quant_probe,gemm_probe}.json`
**复现**：`tools/det_batch_bench.py`、`tools/det_batch_bench2.py`、`tools/emb_quant_probe.py`、`tools/gemm_probe.py`（远程 `~/yhh/human-count/bench/`）

---

## 0. 结论先行

### 问题 1：最多 16 路实时视频并行，当前系统处理能力如何？

**检测环节是唯一硬瓶颈，且是量级差距；其余环节接近就绪。**

| 口径 | 数值 |
|---|---|
| 16 路需求（保持现有 5 fps 分析率，stride=5） | **80 帧/s** |
| 现状能力（fp32 逐帧，实测） | 5.2–5.5 fps/卡 → 需 **15 卡** |
| 简单优化后（fp16 + batch8，实测） | **8.0 fps/卡**（+55%）→ 需 **10 卡** |
| 外推"再理想优化"（TRT/编译，假设再 2×） | ~16 fps/卡 → 仍需 **5 卡** |

**结论：以 GroundingDINO-base 为检测器的现状架构，无法支撑 16 路实时（服务器仅 2×L20，且要共享给嵌入服务）。** 关键实测发现：检测时间与输入分辨率几乎无关（640 vs 800 完全一致）、批处理收益弱（b1→b8 仅 +17%）、fp16 是最大单项收益（+24%）——说明瓶颈是每帧固定开销（文本分支/deformable 结构/kernel launch），**继续微调参数救不回来，换轻量检测器（YOLO11/RF-DETR 级，closed-set person）是唯一现实路径**。

其余环节（16 路需求 vs 能力）：嵌入 24 图/s < 33–50 图/s ✅；跟踪/接力/事件 ✅（事件级轻量）；渲染写盘 ⚠️（16 路 CPU 编码压力大，建议关）。

### 问题 2：能否把 EmbeddingGemma2 替换为量化版本？

**结论：不建议做——没有"即插即用的量化版"可换；即便本地量化改造，全链收益上限也只有 ~2%（实测 fp16 为 -1%，fp8 路径实测不可用）；真正值得做的是"死重剥离"（省 1.89 GB 显存，cos 1.000000，零风险）。**

三层证据：
1. **可替换性**：官方 EmbeddingGemma 2（2026-10-06 发布）的量化走设备端 QAT（INT4/INT8，LiteRT/MediaPipe 格式），**未发布 PyTorch safetensors 的 QAT 检查点**；且本项目用的是"RADIO→W→embed_vision→语言塔"私有桥接路径，换任何外部权重都会改变嵌入空间 → 选择器/relay/truth 全下游需重算。
2. **收益上限**：语言塔净计算仅占全链 **23.6%**（4.55/19.27 ms），塔内 GEMM 又仅占塔时间 ~11%（其余是 attention/elementwise/launch）→ 全部量化理想化收益 ≈ 1.5–2 ms/图（~8%），实测 fp16 收益 -1%（19.07 vs 19.27 ms，噪声级）。
3. **软件栈**：`torch._scaled_mm` 在 L20 + cuBLAS 13 实测抛 `CUBLAS_STATUS_NOT_SUPPORTED`；远程无 torchao/bitsandbytes/onnxruntime（无外网），标准量化工具链全部不可用。

---

## 1. 问题 1：16 路实时处理能力（实测）

### 1.1 负载需求分解（16 路 1080p@25fps）

| 环节 | 需求 | 依据 |
|---|---|---|
| 检测 | **80 帧/s**（16 路 × 25fps ÷ stride5） | 沿用现状 5fps 分析率 |
| 嵌入 | **~24 图/s**（高密度场景） | long35 实测：318 型 88 图/分钟/路 × 16 |
| 跟踪（ByteTrack） | 80 帧/s，<1 ms/帧 → ~0.08 核 | 轻量 |
| 接力/事件 | 事件级（每秒 3–4 个新轨迹） | 轻量 |
| 渲染/写盘 | 16 路 × 1080p 实时编码 ≈ 32–64 核（x264） | ⚠️ 建议生产关闭渲染 |

### 1.2 检测端实测（GroundingDINO-base，L20 单卡，1920×1080 源）

| 配置 | fwd(ms/帧) | 端到端 fps | 相对现状 |
|---|---|---|---|
| fp32 batch1（生产现状） | 155.7 | **5.2–5.5** | 基准 |
| fp32 batch8 | 129.9 | 6.2–6.5 | +17% |
| fp16 batch8 | **93.1** | **8.0** | **+55%** |
| fp16 + edge640 | 93.0 | 8.0 | 无差异 |
| fp16 + edge640 + b4 | 94.1 | 8.0 | 无差异 |

**三个反常现象（都指向同一结论）**：
1. **分辨率不敏感**：640 vs 800 的 fwd 时间 744.2 vs 744.8 ms（batch8），完全一致 → 视觉 token 数不是瓶颈；
2. **批处理收益弱**：b1→b8 仅 +17%（理想应 2–4×）→ 计算利用率低、固定开销占比高；
3. **fp16 收益有限**：+24%（b8 口径），说明剩下的是非 GEMM 开销。

> 参考：`fwd` 为模型前向，`proc` 为 CPU 预处理（24–30 ms/帧，占端到端 15%，为大 batch 时的附加瓶颈）、`post` 为后处理（~1 ms）。

### 1.3 逐环节核算（16 路）

| 环节 | 需求 | 能力 | 判定 |
|---|---|---|---|
| 检测 | 80 fps | 8.0 fps/卡（优化后） | ❌ 差 10×（需 10 卡） |
| 嵌入 | ~24 图/s | 33 图/s（服务实测）/ 50 图/s（理论上限） | ✅ 余量 40% |
| 跟踪 | 80 帧/s | 可忽略（CPU <1ms/帧） | ✅ |
| 视频解码 | 400 fps 解码 | NVDEC 硬解（容量待实测） | ✅（预期） |
| 输出渲染 | — | 建议生产关闭 | ✅（调整） |

### 1.4 路径建议（按性价比排序）

1. **换轻量检测器（推荐，唯一能达成 16 路的路径）**：closed-set person 检测（YOLO11m/s、RF-DETR 级）+ TensorRT。公开基准外推（**未在本机实测**）：L20 上 1080p 可达 100+ fps → **1–2 卡即可覆盖 80 fps 需求**。代价：失去 open-vocab（本项目 prompt 固定 "person"，无损失）；需要微调/校准（1012 数据可用）；保持 jsonl（boxes/track_ids/scores）输出格式不变即可无缝对接下游。
   - 附带发现：机内已有 RF-DETR person 专用优化引擎等资产（来自其他项目/用户目录，如 `/home/r4900g3/yjk/onnx2trt/rfdetr_xxl_person.trt`）与 YOLO11 预训练权重（`/share/data4/yhh/models/`，属 sop-compliance-demo 项目资产）——**复用前需先确认归属与可用性**。
2. **折中级联**：轻量模型常开做连续跟踪 + GroundingDINO 低频（每 N 秒）校准/补检 → 检测需求可降 5–10×。
3. **多卡堆叠 GroundingDINO**：即使最优优化（16 fps/卡）也需 5 卡，不推荐。
4. **降低分析率**：0.5 fps/路（单卡）对跟踪连续性过大损失，不可取。

> 延迟口径说明：本评估以"吞吐跟上 16 路"为标准；系统端到端实时性（L6 在线闭环）尚未建设，不在此报告范围内。

---

## 2. 问题 2：EmbeddingGemma2 量化可行性（实测 + 调研）

### 2.1 模型现状解剖（实测）

- 总参数 **744.4 M**：language_model **271.0**（在用）+ embed_vision 0.4（在用）+ **audio_tower 304.8 + vision_tower 167.4（死重，不参与推理）**。
- 链路总耗时 **19.27 ms/图**（batch16、与生产 20ms 口径一致）：RADIO 桥 **12.75（66%）** + 语言塔净 **4.55（23.6%）** + embed_vision 0.009（~0）+ 预处理/池化 ~1.96（10%）。

### 2.2 外部调研（官方路径核实）

| 方案 | 现状 | 可用性判定 |
|---|---|---|
| 官方 EmbeddingGemma 2 QAT（INT4/INT8） | 2026-10-06 发布，设备端格式（LiteRT/MediaPipe/Edge） | ❌ 无 PyTorch safetensors QAT 检查点；且会改变嵌入空间 |
| EmbeddingGemma 1 QAT GGUF（Q8_0） | 生态存在（cos 0.9998） | ❌ text-only 模型，与图像嵌入链不兼容 |
| 社区 GGUF/MLX 量化 | 0.983–0.992 cos | ❌ 同上，且无 llama.cpp/MLX 运行时 |
| 本地 PTQ（torchao/bitsandbytes/ONNX） | 标准工具链 | ❌ 远程无外网、包未安装 |
| torch 原生（fp8 via `_scaled_mm`） | 实测 | ❌ `CUBLAS_STATUS_NOT_SUPPORTED`（cuBLAS 13.0） |
| **死重剥离（模块化加载）** | **实测** | ✅ **-1.89 GB、cos 1.000000** |

> 官方文档明确支持"按需加载"：`vision_config`/`audio_config` 置 None 即可省略对应编码器——与实测的剥离收益一致。

### 2.3 量化收益上限实测（关键数据）

1. **GEMM 微基准**（语言塔形状：hidden 512 / inter 2048 / seq 260）：
   - 单图（M=260）：fp32 16.9–30.7 µs vs fp16 18.1–20.8 µs——**基本持平（fp32 甚至更快）**，小形状 = 固定开销主导（理论计算仅 ~1 µs）；
   - 批 16（M=4160）：fp16 快 2.3–3.0×——但语言塔实际只有 ~11% 时间在 GEMM（推算），收益被 attention/elementwise 开销掩盖。
2. **端到端 fp16 复验**：19.07 vs 19.27 ms/图（**-1%，噪声级**），cos 1.0。
3. **fp8**：`_scaled_mm` 在本机直接不可用（CUBLAS_STATUS_NOT_SUPPORTED）。
4. **综合**：语言塔全部量化理想化收益 ≈ 全链 8% 上限（4.55ms 中可优化的部分），实测为 0–1% → **不值得**。

### 2.4 建议

1. **不做量化替换**（三无：无现成权重、无工具链、无实测收益）。
2. **落地"死重剥离"**（推荐）：省 1.89 GB 显存（-38%），cos 逐位一致零风险；落地方式二选一：① 加载后 `audio_tower/vision_tower.to('meta')`；② 加载时 `config_kwargs` 置 None（官方路径）。
3. 若未来需要进一步压缩：priority 在 RADIO（66% 大头）与预处理（10%），而非语言塔。

---

## 3. 附：实验环境与复现

- 环境：L20 单卡基准（cuda:0）；rfdetr env（torch 2.11.0+cu130 / transformers 5.14）+ embed_venv（sentence-transformers 6.1.0）
- 脚本：`tools/det_batch_bench.py`（检测 b1/2/4/8 × fp32）、`tools/det_batch_bench2.py`（× fp16 × edge 800/640）、`tools/emb_quant_probe.py`（分解/复验/剥离）、`tools/gemm_probe.py`（GEMM 微基准）
- 素材：`video-6way/318.mp4`（1920×1080）；嵌入实验 crops=128（316/318 真实轨迹裁剪）
- 已知限制：① fp16 检测未测更激进优化（TRT/compile）；② 第三方轻量检测器未实测（资产归属/包缺失）；③ 嵌入服务 3-worker 并发的极限吞吐未单独压测。

### 3.1 口径自证：实验模型 = 生产模型（2026-10-14 核验）

> 背景：本项目有"外部模型口径混用"的历史教训（d3b 快算事件），凡模型相关结论必须显式自证口径。

| 环节 | 生产组件 | 实验脚本 | 加载路径 / 参数 | 与生产一致性 |
|---|---|---|---|---|
| 检测 | **GroundingDINO-base**（目录名 `dino-model`） | `det_batch_bench{,2}.py` | `/share/data4/yhh/models/dino-model` | 同路径（生产检测即此模型） |
| 嵌入 | **优化版 C-RADIOv4 链路** = FastEncoder（模板法）＋ C-RADIOv4-SO400M ＋ EmbeddingGemma2 | `emb_quant_probe.py` | root=crosscam_run（默认）、`--w-dir cradio_scale_v4` | 与生产 embed_server 启动参数完全一致 |

**文件指纹**（同机同文件，实验与生产加载同一份，sha1 前 20 位）：
- `cradio_scale_v4/radio_W_scale.npy` = `c1d491bdc40a661345fa`
- `cradio/C-RADIOv4-SO400M/model.safetensors` = `29736541aaf0b91ebf49`
- `embgemma/model/master/model.safetensors` = `d4c753b9c508e4847459`
- 检测 `dino-model/model.safetensors` = `f9ce46b65eda3420734c`

**生产 embed_server 启动参数**：`--device cuda:1 --w-dir cradio_scale_v4 --workers 3`；实验用 `cuda:0`（不干扰生产），同 root、同 w_dir、同 batch=16 默认。（2026-10-14 起生产另有 `--strip-dead` 死重剥离，见 `docs/embgemma-strip-feasibility.md`。）

**澄清**：
- 检测环节的 `dino-model` 是 **GroundingDINO**（生产检测器），与 **DINOv3** 无关；DINOv3（d3b）仅在项目早期 ReID 对比实验出现过且已弃用。四份实验脚本 grep 无任何 `dinov3/d3b` 引用。
- cradio 是**嵌入编码器**，不参与检测；16 路评估对"检测（GroundingDINO）"与"嵌入（cradio v4 链路）"两个生产组件分别实测。
