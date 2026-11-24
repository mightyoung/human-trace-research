# cradio_v3 编解码链路并行化研究

**日期**：2026-10-13 ｜ **性质**：推理系统优化（近邻调研 + 现状剖析 + 改造方案 + 实测数据）
**目标**：把 cradio_v3 的编码链路改造为「可批量并行、高效执行」的形态，提高整体执行效率。
**近邻对象**：MiniCPM 系列（高刷视频/端侧实时）、π0/π0.5（VLA 实时控制）；工具脚本：`tools/cradio_parallel_probe.py`（v1）/ `cradio_parallel_probe2.py`（v2）。

---

## 0. 结论先行（Phase 2 完成，2026-10-13）

1. **两阶段改造完成，总提速 9.2–9.4×**：单图逐图 185 ms/图 → 批路径（Phase 1，46.6）→ **模板法快速路径（Phase 2）
   19.7–20.1 ms/图（~50 图/s）**；与现有链路嵌入 cos=**0.99995（min 0.99962）**，唯一差异来源为图像缩放插值路径不同（§2.5）。
2. **Phase 2 的三个本质发现**：
   - **"处理器往返"纯属浪费**：ST 链路 = Gemma processor patchify（9.4ms/图）+ rebuild（8.2）+ RADIO 前处理（7.3）
     ≈ 25 ms/图；而 EmbeddingGemma2Model.forward 的官方装配只需 **input_ids 模板（含 256 图像占位）+ attention_mask
     + 非 None 的 pixel_values 占位**——图像特征直供（masked_scatter），全绕开 ST。
   - **模板恒定**：224² 方图 → L=260、256 图像占位，**与图像内容/尺寸无关**（128²/320² 实测同样成立）→ 捕获一次全局复用。
   - **新瓶颈 = RADIO GPU 算力墙**：12.7 ms/图（fp16 已饱和；这也解释了 probe2 的 13ms/图历史数据）；语言塔仅 4.4 ms/图。
3. **"显式流水"无必要（反直觉）**：PyTorch 异步提交已自带隐式流水（背靠背提交 + 延迟读值）——实测 prefetch 线程
   反而略慢（21.3 vs 19.7 ms/图）；跨批双缓冲（pipelined）与简单串行同价。**简单即最优**。
4. **语言塔低精度无收益**：bf16/fp16 均 21.6–21.9 vs fp32 21.3 ms/图（cos≈0.99995）——维持 fp32（省去复杂度与风险）。
5. **交付**：`tools/embed_fast.py`（`FastEncoder`，模板法、~250 行自含模块，接口 `enc.encode(crops) -> [N,768]`）
   + 冒烟复验（legacy 对照/开关对照）；探针脚本与 JSON 数据在 `tools/cradio_parallel_probe6.py` /
   `output_crosscam/analysis/cradio_parallel/`。
6. **近邻启示的最终映射**：批处理摊销（π0/realtime-vla）✅ 已落地；异步流水（RTC）✅ 由 PyTorch 异步提交天然实现
   （无需自建线程）；跨帧冗余压缩（MiniCPM 3D-Resampler）仍可挖掘（轨迹帧级，非本阶段）。

---

## 1. 近邻研究：实时系统怎么做

### 1.1 MiniCPM-V 4.5（高刷视频理解 / 端侧实时）

**技术事实**：
- **3D-Resampler**：把连续 N 帧（最大 6 帧）作为一组做三维压缩，**一组输出 64 个 token（与单图相同）**，实现 96× 视频 token 压缩率、10 fps 高刷理解，**推理开销与单图持平**。
- 核心洞察：相邻帧之间视觉信息高度冗余（"大部分不变、少部分变化"）——把冗余当压缩空间。
- 可控混合思考（fast/deep thinking 一键切换）：**运行时按需在"速度档"和"精度档"之间切换**。
- 端侧部署友好（llama.cpp/ollama/iOS），显存与平均推理时间在同类中领先。

**对我们的启示**：
- 我们每条轨迹采样 8 帧去编码——帧间冗余极大（同一个人两三秒内的裁剪）。可借鉴：
  a) **帧组联合编码**（对一组帧做一次前向，共享计算，如 3D 结构或批量→池化后合并）；
  b) **关键帧增量**：轨迹内差异小于阈值时复用上一关键帧嵌入（自适应抽帧）；
  c) **精度档切换**："全量精编" vs "快速粗筛"双档（对应 MiniCPM 快/慢思考）——粗筛用 summary 特征（RADIO summary 45ms 那时数据)/小分辨率，精编用全链路。

### 1.2 π0 / π0.5 + RTC（VLA 实时控制）

**技术事实**：
- **Action Chunking**：一次推理生成 50 步动作序列（~70 ms/chunk），机器人按块执行（50Hz）——**把单次推理成本摊销到多次执行上**。
- **RTC（Real-Time Chunking）**：推理在机器人**执行上一块动作的同时异步进行**；新块用"前缀引导"（inpainting）与已执行部分平滑衔接。解决"Action Waiting（等推理暂停）"与"Action Jumping（块间跳变）"。
- 结构側：MoE 路由（VLM 3B 主干 + 300M 动作专家）、flow matching 少步去噪（K=10）。
- 工程側（来自 on-device 综述）：vla.cpp 的 **KV 复用**（跨去噪步复用 vision-language 前缀，避免重复计算）；realtime-vla 的**流式调度**（分解为 24 个 GEMM 逐级 kernel 优化、BF16 无损）；FlashRT 低精度（FP8/NVFP4/INT8/INT4）。
- **关键洞察（原话意译）**："单样本、小 batch（机器人无法批处理）场景下，**memory access 与 kernel launch 开销是主要瓶颈**，模型计算利用率仅 7.6%–13.2%。"

**对我们的启示**：
- **摊销式批处理**：向"批"要效率——把来自多路视频/多个轨迹的裁剪合并成统一 batch（把 π0 的"时间换空间"反过来用："空间换时间"）。
- **异步流水线（RTC 思想）**：编码批次不必等齐再走——"边编码边消费"：批次 A 的嵌入已可服务下游（入库/检索），批次 B 在后台编码；关键路径（新出现的轨迹）优先处理。
- **前缀缓存**：同一条轨迹相邻帧的特征高度相似 → KV/特征复用思路（帧组编码时共享前期计算）。
- **低精度**：BF16/FP16 无损级（之前实测 C-RADIOv4 fp16 1.88× 且 cos=1.00000）；低精度是免费午餐。

### 1.3 借鉴映射表（→ cradio_v3 的具体环节）

| 近邻技术 | 出处 | → 我们的环节 | 预期收益 |
|---|---|---|---|
| 批处理摊销（chunking/流式批） | π0 / realtime-vla | 多图合并 batch 编码（`for im in crops` → `batch`） | GPU 13→0.2ms/图 |
| 异步流水线（RTC） | π0/RTC | CPU 预处理 ↔ GPU 前向双缓冲；编码批次与缓存消费解耦 | 消除串行等待 |
| 跨帧冗余压缩（3D-Resampler） | MiniCPM-V 4.5 | 轨迹 8 帧的冗余利用（帧组/关键帧） | 减少 30-70% 编码量 |
| 快/慢双档 | MiniCPM-V 4.5 | summary 粗筛（快）→ 全链路精编（慢） | 检索/检索场景 5-10× |
| KV/前缀复用 | vla.cpp | 帧组编码共享前段计算（远期） | 批内加速 |
| 低精度（FP8/INT8） | FlashRT | fp16 已证无损 1.88×；FP8 可再进一档 | 1.5-2× |
| 单样本瓶颈=访存/launch | on-device 综述 | 直接对应我们的逐图串行损耗 | 佐证批处理方向 |

---

## 2. 现状剖析：cradio_v3 链路拆解与实测

### 2.1 链路分解（现状，逐图串行）

```
视频帧 --cv2--> crop(224)                                    [CPU]
  └─ Gemma processor: resize→16×16 patchify→position_ids      [CPU，重]
       └─ rebuild_image: patch 序列重建回图（~896×720）        [CPU]
            └─ RADIO CLIPImageProcessor: resize→512           [CPU，重]
                 └─ C-RADIOv4 SO400M forward（512 输入）       [GPU]
                      └─ 2×2 池化 → [256,1152]                 [GPU 微]
                           └─ W 桥接 [1152→768]                [GPU 微]
                                └─ embed_vision [768→512]      [GPU？待确认是否被消费]
                                     └─ ST pooling → normalize [CPU/微]
```

**结构性问题**：
1. `for im in crops:` 逐图串行调用 `st.encode`——GPU 每图只忙 13ms，其余 150ms 在等待/封装；
2. **两趟处理器**（Gemma patchify → 重建 → RADIO resize）是历史兼容产物：重建只为了把图"还原"出来再喂给 RADIO，**可以直接跳过**（crop 直接给 RADIO）；
3. `embed_vision` 的调用疑为无效计算（v2 实测判定）；
4. 无跨轨/跨图 batch；无 CPU-GPU 流水线重叠。

### 2.2 实测数据（探针 v1 + v2，2026-10-13，L20）

| 测量项 | 数值 | 备注 |
|---|---|---|
| 现状 E2E（逐图 st.encode，含封装） | **162.8–201.6 ms/图**（两轮实测） | 16 图 / 8 图均值 |
| `rebuild_image`（CPU） | 1.3 ms/图 | 意外地小 |
| 批处理段（proc→RADIO→池化→W→embed_vision）**fp32 饱和** | **44.0–47.4 ms/图**（bs=8/16/32/64） | 21–22 图/s，与独立微基准 46.3ms 一致 |
| 批处理段 **fp16 饱和** | **13.05–13.66 ms/图（73–77 图/s）** | bs=8/16/32/64 均在 13–14ms；**3.4× vs fp32** |
| 显存峰值 | fp32 5.3–8.7 GB；fp16 6.0–8.3 GB | 随 batch 增大缓增 |
| CPU 预处理（CLIP 处理器） | **单图 1.8 ms**；cv2 快速版 0.92 ms/图 | v2 干净口径 |
| 现状与饱和之差（≈119 ms） | 逐图空转 + Gemma processor/rebuild 往返 + ST 封装 | 批处理路径天然消除 |

> **口径说明**：v1 的 cpu/gpu 拆分（cpu 33–47ms / gpu 0.2–13ms）受 GPU 队列交叠污染
> （`.to(DEV)` 阻塞等待前批 kernel 完成），绝对值不可用；**饱和吞吐口径以 v2 为准**。
> 独立微基准（cradio_parts，背靠背）：fp32 单图 46.3 ms / fp16 单图 24.6 ms（1.88×，cos=1.00000）——
> 与 v2 的 fp32 饱和值互相印证。
> 另：**批处理价值 = 让 GPU 饱和**（bs≥8 后 GPU 摊薄项 <2ms），fp16 价值 = 翻越算力墙（3.4×）。

### 2.2b 分段画像（probe6c 修正版，权威口径）★

> 修正说明：早期探针的 `torch.cuda.synchronize()` 未指定设备 → 默认同步 cuda:0，而计算在 cuda:1，
> 导致 GPU 段计时失真（radio 曾读到 1.02ms/图）。probe6c 全部改为 `synchronize(DEV)` 后，数据自洽。

**legacy 批路径（st.encode list 模式，48.0–49.0 ms/图）分段**（L20、batch16、32 图）：

| 段 | ms/图 | 性质 | 改造后 |
|---|---|---|---|
| Gemma processor（patchify+tokenize） | 9.4 | CPU | **跳过（模板法）** |
| rebuild（patch→图） | 8.2 | CPU | **跳过** |
| rproc（RADIO CLIPImageProcessor） | 7.3 | CPU | 保留（224²→512²） |
| **RADIO 前向+池化（fp16）** | **12.7** | GPU | 保留（**新瓶颈=算力墙**） |
| W 桥接 + embed_vision | 0.03 | GPU | 保留 |
| language_model | 4.3 | GPU | 保留（低精度无收益，见 §2.5） |
| tail（ST 装配/池化/杂项） | 3.4 | CPU | 跳过 |
| **合计** | **45.3–49.0** | | **→ 21.3 / 19.7（§2.5）** |

**RADIO 12.7ms/图 与 probe2 的 fp16 饱和 13ms/图吻合**——此前"RADIO 不是瓶颈"的判断被修正：
它是当前链路的**单一最大计算项**；batch 摊销已完成，进一步的余量只剩低精度（FP8）或算子级优化。

### 2.3 链路解剖（源码级 + 探针 v2–v4）

**完整链路（确认版，含此前未知的语言塔段）**：

```
crop → [Gemma processor: resize→16×16 patchify→input_ids(图像占位 token)+position_ids]
     → rebuild_image（重建回图）→ RADIO CLIPImageProcessor(resize→512)
     → C-RADIOv4 SO400M（fp32 单图46ms / fp16 批处理 13ms/图）
     → 2×2 池化 [256,1152] → W 桥接 [256,768] → embed_vision [256,512]
     → masked_scatter 填入语言序列 → language_model（24 层，真实参与！）
     → 池化 → 最终嵌入 e_A（768 维，L2 归一）
```

**关键事实（均为实测/源码确认）**：
1. **语言塔真实参与**：`EmbeddingGemma2Model.forward` 内部把 `get_image_features(...).pooler_output`
   散入语言序列后跑 `language_model`（源码 L785 处）。因此 e_A ≠ W 输出的简单池化
   （v4 实测 cos(e_A, z.mean)=0.038——两者本质不同）。
2. **embed_vision 输出被消费**（v2 去除后报错），不可省略——但可纳入批量路径一起摊薄。
3. **CPU 预处理不是瓶颈**：CLIP 处理器单图 1.8 ms、cv2 快速版 0.92 ms/图。
4. **逐图串行的真实浪费**：现状 185 ms/图 vs 同等计算批处理饱和 46 ms/图（下述 v5）——
   约 139 ms/图 消耗在"串行空转 + 历史封装（Gemma processor 往返 + ST 逐图包装）"。
5. **模板法三件套（probe6 新增，跳过 processor 的钥匙）**：
   - forward 的装配需求 = **input_ids（含 256 个 `config.image_token_id` 占位）+ attention_mask + 非 None 的
     pixel_values**；图像特征经 `masked_scatter` 填入（`get_image_features.pooler_output`，我们替换为 RADIO 桥接）。
   - **input_ids 模板与图像内容/尺寸无关**：224²/128²/320² 方图均为 `L=260`（head `[2, 255999, 258880…]`、
     尾 `[…258880, 258882, 1]`、图像占位 256、无 position_ids）——捕获一次即可 expand 复用；
     256 占位与 embed_vision 输出 256 token 严格对应（forward 内有数量校验）。
   - **非方图现状直接崩**（`min_resolution_step` 检查，如 300×200 → 请求 512×787 非法）——生产必须保持方图
     crop（现 `prep_crop` 已统一 224²）；模板法反而没有此限制。
   - dummy pixel_values 形状任意（`[B,1,1]` 实测可跑）——它只用于非 None 判断，进入 patched 后被完全忽略。

### 2.4 批量原型（探针 v5）：patched 批处理化 → 全链路 batch 实测 ★

**做法**：把 `get_image_features` 的替换实现从"仅单图"升级为"批处理版"（逐样本 rebuild →
合并 batch RADIO(fp16) → 池化/W/embed_vision 批量），然后直接调用
`st.encode(list_of_images, batch_size=N)`（ST 内部原生批处理语言塔与装配）。

| 口径 | ms/图 | 图/s | 相对现状 | 嵌入一致性 |
|---|---|---|---|---|
| 单图逐图（现状复现） | 184.9 | 5.4 | 1× | 基准 |
| **batch=8** | **49.0** | 20.4 | **3.8×** | **cos=1.0000（min=1.0000）** |
| batch=16 | 46.6 | 21.4 | 4.0× | — |
| batch=32 | 45.2 | 22.1 | **4.1×** | — |
| batch=64 | 47.7 | 21.0 | 3.9× | — |

**结论**：
1. **批处理化可行且无代价**：`st.encode(list, batch_size=N)` 全链路（含语言塔）批量运行，
   与单图逐图的嵌入**数值完全一致（cos=1.0000）**——改造不影响任何下游口径。
2. **实测 4× 提速**（185 → 46.6 ms/图）；batch≥16 后进入平台区。
3. **剩余可挖空间（→ 已在 §2.5 完成）**：跳过 Gemma processor 往返、language_model 低精度、CPU-GPU 重叠。
4. 批处理组装方式已确认：ST 在 batch 模式下向视觉侧传 `pixel_values [B,2520,768]`、
   `image_position_ids [B,2520,2]`（逐样本等长，crop 尺寸一致即可 tensor 化）。

### 2.5 Phase 2 实测（probe6c + embed_fast smoke，2026-10-13）★

**A. F2 模板法快速路径（核心成果）**

做法：捕获 input_ids/attention_mask 模板（一次）→ 后续批次直接
`am(input_ids=模板.expand(B), attention_mask=模板, pixel_values=zeros[B,1,1], image_position_ids=None)`，
图像走我们的批处理 RADIO 桥接（fp16）→ masked_scatter → language_model → mean-pool（`mean_all`）→ L2 归一。

| 口径 | ms/图 | 图/s | vs 单图 185ms | 嵌入对照 |
|---|---|---|---|---|
| legacy 批路径（probe6c 复测） | 49.0 | 20.4 | 3.8× | 基准 |
| legacy（embed_fast smoke 复测） | 48.0 | 20.8 | 3.9× | 基准 |
| **F2（probe6c）** | **21.3** | **46.9** | **8.7×** | cos=0.99995 / min 0.99962 |
| **F2（embed_fast smoke①，prefetch=True 版）** | 21.3 | 46.9 | 8.7× | cos=0.99995 / min 0.99962 |
| **F2 默认路径（prefetch=False，最终交付口径）** | **19.7–20.1** | **49.8–50.8** | **9.2–9.4×** | 同上（同数值路径，两次独立复验） |

- **池化变体对照**（决定装配正确性）：`mean_all` **0.99995** ≫ `mean_text` 0.90049 ≫ `last` 0.7248
  → 确认 ST 侧为 **include_prompt 的 mask-mean 池化**，手动复刻正确。
- **F2 分段**：rproc 4.1 + RADIO 12.7 + bridge_ev 0.03 + lm 4.4 + tail 0.1 ms/图——**tail≈0**，
  证明 ST 封装（processor/rebuild/装配/pool）已被完全绕开；剩余 = GPU 计算 + 一次 resize。
- cos min=0.99962 的偏差来源：legacy 路径对图像存在"224²→720×896 细化→512²"两次重采样，
  F2 为"224²→512²"一次重采样——同一信息、不同插值路径。对下游判定影响 ~1e-4 量级
  （门限 0.9433 的接力判定不可能受影响）。

**B. 语言塔低精度（无收益，维持 fp32）**

| 配置 | ms/图 | cos vs fp32 基准 |
|---|---|---|
| lm fp32 | 21.3 | 基准 |
| lm bf16 | 21.6 | 0.99995 |
| lm fp16 | 21.9 | 0.99995 |

language_model 只占 4.4 ms/图（GPU），低精度省不出足量时间，反而引入数值变化与复杂度 → **不做**。

**C. CPU-GPU 重叠（三变体，结论：无需显式流水）**

| 变体 | ms/图 | vs serial |
|---|---|---|
| serial（rproc 在关键路径 + 即时 pool） | 21.0 | — |
| prefetch（rproc 预取线程） | 20.0 | −4.7% |
| pipelined（预取 + 延迟 pool，完全背靠背） | 19.7 | −6.1% |

反向验证：embed_fast 里把 prefetch 默认打开时 **反而不如关闭**（21.3 vs 19.7）——PyTorch 的异步提交
（kernel 入队后立即返回、延迟到读值时同步）已天然形成"背靠背流水"；显式线程只增加 GIL/调度开销。
**最终实现：默认 prefetch=False、不延迟 pool**（简单版本即达 19.7 ms/图）。

**D. cv2 预处理替代 rproc（不采用）**

cos(vs rproc)=0.9998（min 0.99944）、耗时 21.5 vs 21.3（无差异——rproc 已被隐式流水掩盖）。
收益不足、精度略降 → **保留 rproc**。

**E. 复验与交付**

- `tools/embed_fast.py`：`FastEncoder` 模块（含模板自动捕获 + 端到端自检；多次复验：
  legacy 46.4–48.0 / fast 20.1–21.3 / cos 0.99995，模板 L=260）。
- 冒烟耗时：加载 7.9s + 32 图编码 ~1.5s（含全部对照实验共 ~40s）。
- 数据：`output_crosscam/analysis/cradio_parallel/cradio_parallel_probe6{a,b,}.json`、`embed_fast_smoke.json`。

---

## 3. 改造设计（目标架构）

### 3.1 目标链路（批量并行 + 流水线）

```
[多路视频]
   │ 解码+裁剪（CPU，生产侧数据管道）
   ▼
224² 方形 crop 列表
   │
   ▼
FastEncoder.encode(crops, batch=16)
   ├─ rproc（224²→512²；已被隐式流水掩盖成本）
   ├─ RADIO fp16 批前向（12.7ms/图，算力墙）→ 2×2 池化 → W → embed_vision
   └─ am.forward(模板 input_ids + dummy pixel_values)  ← 无 processor、无 rebuild、无 ST 封装
        → masked_scatter → language_model(4.4ms/图) → mean-pool → L2 归一
   │
   ▼
嵌入 [N,768] → 缓存层（store）→ 下游（接力/检索/gallery）
```

**改造分两阶段（均已完成）**：
- **Phase 1（✅ 实测）**：批处理版 patched + `st.encode(list, batch_size)` → **4×（46.6 ms/图）、cos=1.0000**；
- **Phase 2（✅ 实测）**：模板法跳过 Gemma processor/rebuild → **21.3 ms/图（8.7×）**；
  默认简单路径（无显式流水）**19.7 ms/图（9.4× vs 单图）、cos=0.99995**。
  language_model 低精度与显式流水经验证无收益（详见 §2.5），不纳入交付。

**五个层次改造清单（最终版）**：

| 层 | 最终做法 | 实测结果 |
|---|---|---|
| L1 数据层 | 顺序扫视频、crop 预生成（生产数据管道） | — |
| L2 前处理 | **模板法**：捕获 input_ids/attention_mask 一次、全局复用 | 省 Gemma processor 9.4 + rebuild 8.2 ms/图 |
| L3 模型段 | **统一 batch=16 + RADIO fp16**（语言塔保持 fp32） | GPU 饱和；RADIO 12.7ms/图=算力墙 |
| L4 流水线 | **不显式做**——PyTorch 异步提交天然背靠背 | prefetch 反而 +1.6ms（§2.5C） |
| L5 调度 | 关键路径优先、帧级冗余利用（留待 R3.2/R1 阶段） | 待做 |

### 3.2 验收口径（更新）

- 单元吞吐：批编码器 **≥20 图/s → 实测 46.9–50.8 图/s ✅**（超验收线 2.4×）；
- 一致性：改造 vs 现状嵌入 **cos ≥ 0.999 → 实测 0.99995（min 0.99962）✅**；
- 端到端：6 路 10 分钟数据全量编码时间对比（现状 ≈5.4 图/s 口径）——留待生产接入时按实际数据量测；
- 回归：接力/318×320 评估在**不重训 W** 的前提下 AUC 不回退——嵌入差异 1e-4 量级，
  门限 0.9433 的判定不受影响（理论推断；建议生产接入时抽 1 条链实测复核）。

### 3.3 风险与对策（更新）

| 风险 | 对策 |
|---|---|
| 模板与 ST/模型版本耦合（input_ids 结构变化） | 模板自动捕获（每次加载时从真实 processor 输出学）+ 断言（L=260、占位=256）+ 首次编码自检 |
| 非方 crop 输入（模板法无 patchify 保护） | 生产 `prep_crop` 已统一 224²；encode 入口可加尺寸断言 |
| 嵌入微漂移（cos 0.99962 min） | 已量化；下游判定门限余量充足；生产接入时抽链复核 |
| GPU0 共享（他方服务） | 编码服务固定 GPU1；或与 asr/locate 协商错峰 |
| 磁盘 94% | 编码产物走缓存层（1.5KB/轨），不落大文件 |

---

## 4. 下一步

本方向并入四方向方案的 **R3.2（流水线设计）**——Phase 2 已提前完成其核心目标：
1. **✅ 可复用组件已交付**：`tools/embed_fast.py`（`FastEncoder`，替代探针脚本；`embed_batch.py` 为 Phase 1 版本，保留作对照）；
2. **✅ Phase 2 三项深挖完成**：模板跳过（省 ~18ms/图）、lm 低精度（无收益，不做）、流水线（无必要，不做）；
3. **下一步（生产接入）**：与 R2（全局 ReID 缓存层）对接：编码批次 → store 增量入库；
   与 R1（多目标）对接：按轨迹分批编码的调度策略（关键路径优先）；
   编码服务化（常驻进程 + GPU1 固定 + 队列接口）。
4. **剩余可挖（低优先）**：RADIO 算子级优化（FP8/算子融合，预期 <30%）；MiniCPM 式跨帧冗余压缩（帧组编码，预期减 30–70% 编码量——属数据侧策略）。
