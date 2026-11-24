# 本地推理链路组装 + 与远程历史产物对账（2026-10-17）

> 承接 `docs/local-env-gpu-2026-10-16.md`「六、未解 / 下一步 #1」：跨机位实验管线本地化实跑。
> 目标：不再用远程 r4900g3，在本机完整跑「检测 + 嵌入」6 路 synced-10min，并与远程历史产物对账。

---

## 一、结论先行

| 项 | 状态 | 证据 |
|---|---|---|
| **基座模型纠偏** | ✅ 实锤 | 远程 `dino-model` = `IDEA-Research/grounding-dino-base`（sha1 前20位 `f9ce46b65eda3420734c` 逐字节一致） |
| 本地检测（base ONNX + DirectML） | ✅ 6 路全量 | 17941 帧实测 ~1.6 fps（0.59–0.90 s/帧，lat_p50≈600ms） |
| 本地嵌入（DINOv3-B + torch-directml） | ✅ 6 路全量 | 732 轨迹编码（0 crop-fail），~1.2 s/轨迹 |
| 检测对账（6 路全量） | ✅ **通过** | 框 IoU mean 0.976–0.9995、count within-1 99.2–100%（见 §4.1） |
| 嵌入对账（6 路全量） | ✅ **通过** | 693 配对 cos mean 0.9670 / median **0.9961**；高重叠子集 **0.9951**（见 §4.2） |
| 产物目录 | ✅ | `output_crosscam_local/{detect,embed}/`；对账 JSON + 总览图同目录 |

**核心结论**：本机（无 CUDA 的 AMD RX 5700 XT）已端到端复现远程研究链路的核心推理
（检测 + 嵌入），6 路 synced-10min 全量实跑对账通过。差异分布：
- 检测层面——框 IoU 中位 0.99+，差异集中为 box_threshold=0.30 边缘框的 ±1 级取舍波动
  （跨设备浮点噪声），帧级计数 99.2–100% 在 ±1 内；
- 嵌入层面——同输入环节级 cos 0.99919（千分位）；端到端高重叠子集 cos 0.9951；
  低 cos 尾部已归因（track 切分差异 + 贪心匹配错配，非链路缺陷，见 §4.2 审计）。

---

## 二、重大纠偏：远程基座模型是本机原 ONNX 的「孪生不同体」

### 发现过程
1. 首次用本机 ONNX（`models/onnx/groundingdino.onnx`）跑 305，与远程产物**对不上**：
   框位置接近但分数差 ±0.2、框集合结构性不同（一人拆成上下两框 vs 合并大框）。
2. 排除法：
   - 预处理/后处理口径（HF image processor / post_process）已逐行复刻 → 非此因；
   - ONNX 导出损失 → 实测 ONNX ≈ PyTorch（conf Δ 1e-4 级）→ 排除；
   - HF 转换损失 → 官方转换版（本机转）与原生 HF 版输出一致 → 排除。
   - **结论指向：权重本身不同**。
3. 证据链锁定：
   - `docs/16ch-capacity-and-embgemma-quant-report.md:137`：
     「检测 **GroundingDINO-base**（目录名 `dino-model`）……生产检测即此模型」
   - 该报告记录远程 `dino-model/model.safetensors` sha1 前 20 位 = `f9ce46b65eda3420734c`；
   - 从 ModelScope 下载同源权重（`IDEA-Research/grounding-dino-base`），**sha1 完全一致**；
   - 行为验证：该权重跑 305 frame0 → 与远程 jsonl 吻合（框差 <1px）。

### 结论
- **本机原 ONNX = GroundingDINO SwinT (tiny)**（10-10 从 AI-ModelScope 快照导出，用于 Rust 服务）
- **远程研究链路 = GroundingDINO SwinB (base)** —— 两者输出分布本就不同。
- 本机实验链路自此统一使用 **base**：
  `models/IDEA-Research--grounding-dino-base/`（HF 格式，sha1 已校验）
  → `models/onnx/groundingdino_base.onnx`（`tools/export_onnx_hf.py` 导出）。
- 注：Rust 服务线与研究链路的模型代际差异（tiny vs base）为项目现状事实，
  是否统一为 base 属独立决策，不在本次范围。

---

## 三、本地链路组成

```
视频 (2560×1440 / 1920×1080, 16:9)
  │
  ├─ tools/local_detect.py ── ONNX(base) + onnxruntime-directml ──→ {cam}_detections.jsonl
  │     · 预处理 = HF GroundingDINOImageProcessor 口径（800/1333 → 恒 1333×750, BILINEAR, ImageNet 归一化）
  │     · 后处理 = HF 口径（sigmoid → max over 256 text tokens → box_threshold 0.30）
  │     · Containment-NMS(0.85, larger) + ByteTrack(supervision 0.24.0, activation 0.10)
  │
  └─ tools/local_encode.py ── DINOv3-B + torch-directml ──→ reid_emb_{cam}_d3b_local.npz
        · prep_crop(pad 0.08, h=224) / samples=8 / 均值聚合 + L2 归一（与远程一致）

对账: tools/local_check.py detect | embed（含"内容级匹配"模式，处理 track id 错位）
深度分析: tools/analyze_embed_match.py（cos×overlap 分桶 + 低cos审计）
总览图: tools/plot_reconcile.py（-> reconcile_overview.png）
模型导出: tools/export_onnx_hf.py（HF → ONNX，含 2 个必要补丁，见 §六）
批量执行: tools/run_local_detect_rest.sh（其余 5 路串行）/ tools/run_local_encode_all.sh（6 路嵌入）
```

环境：
- 检测：`envs/dml`（onnxruntime-directml 1.24.4 + opencv + supervision==0.24.0）
- 嵌入：`envs/dml_torch`（torch 2.4.1 + torch-directml 0.2.5 + timm）
- 导出：`envs/default`（transformers 5.20.1 + torch 2.14.1 + onnxscript）

---

## 四、对账结果

### 4.1 检测（本地 ONNX-DML vs 远程 HF-CUDA）——6 路全量 17941 帧

| 路 | 帧数 | IoU mean | IoU p10 | IoU≥0.9 | count within-1 | scores \|Δ\| mean | count_mean 本/远 |
|---|---|---|---|---|---|---|---|
| 305 办公区1 | 3000 | 0.9844 | 0.9804 | 97.1% | 99.9% | 0.0087 | 6.45 / 6.44 |
| 316 行政走廊 | 2985 | **0.9995** | 0.9987 | 99.9% | **100%** | **0.0011** | 1.12 / 1.12 |
| 318 办公区2 | 2983 | 0.9814 | 0.9784 | 96.7% | 99.8% | 0.0063 | 4.18 / 4.17 |
| 320 办公区球机 | 2981 | 0.9945 | 0.9887 | 99.6% | **100%** | 0.0035 | 0.55 / 0.44 |
| 323 茶水间 | 3005 | 0.9763 | 0.9569 | 95.6% | 99.2% | 0.0076 | 3.40 / 3.47 |
| 343 电梯间 | 2987 | 0.9936 | 0.9883 | 99.7% | 99.9% | 0.0049 | 0.52 / 0.52 |

- **全部 6 路 rows 帧级完全对齐**（帧数、处理序一致，frame_mismatch=0）；
- 合计框对 >4.3 万（6 路求和 18233+3311+11059+1309+8538+1430 = 43880 对）；
- 唯一 >10% 相对差的指标是 320 的 count_mean（0.55 vs 0.44）——该路场景极稀疏
  （多数帧 0-1 人），绝对差仅 0.11/帧、within-1=100%，属边缘框级别噪声的放大呈现；
- `boxes_removed_pct`（去重率）六路均在 ±1.8pp 内一致（例：318 24.22% vs 24.04%）。

汇总 JSON：`output_crosscam_local/reconcile_detect_all.json`

**根因说明**：两侧差异集中在 box_threshold=0.30 的边缘框（分数 0.28–0.32），
CUDA↔DML 浮点差异导致个别边缘框在一侧保留、另一侧丢弃 → 计数 ±1 级波动，
并经 tracker 传导为少量新轨/丢轨 → track_id 序列局部错位（编号分叉，frame-exact 0.5%–63% 视场景而定）。

### 4.2 嵌入（本地 DML vs 远程 CUDA）——6 路全量 732 轨迹

**口径 A：同输入环节级**（用远程 jsonl 的同定义轨迹，隔离检测差异）：107/107 轨迹匹配，
cos mean **0.99919**、min 0.99634、100% ≥0.99 —— 即「相同 track 定义下，DML vs CUDA
的嵌入差异在千分位」。

**口径 B：端到端内容级**（本地 jsonl → 本地嵌入 vs 远程；按帧重叠 Jaccard≥0.2 匹配轨迹，
处理 track id 错位）：

| 路 | matched | cos mean | ≥0.95 | ≥0.99 |
|---|---|---|---|---|
| 305 | 101 | 0.9531 | 85.2% | 68.3% |
| 316 | 143 | **0.9977** | 99.3% | 96.5% |
| 318 | 146 | 0.9415 | 84.9% | 57.5% |
| 320 | 25 | 0.9926 | 96.0% | 84.0% |
| 323 | 223 | 0.9622 | 88.3% | 56.1% |
| 343 | 55 | 0.9889 | 96.4% | 72.7% |
| **合计** | **693** | **0.9670**（median 0.9961） | 90.3% | 68.8% |

**分桶归因（关键）**——按 track 帧重叠度分桶后，交叉设备差异与切分差异被分离：

| overlap 桶 | n | cos mean | ≥0.99 |
|---|---|---|---|
| [0.95, 1.0] | 449 | **0.9951** | 91% |
| [0.80, 0.95) | 102 | 0.9628 | 50% |
| [0.50, 0.80) | 91 | 0.9156 | 18% |
| [0.20, 0.50) | 51 | 0.8202 | 2% |

→ **切分一致时（overlap≥0.95，占比 65%），嵌入几乎相同（cos 0.9951）**；低 cos 全部
集中在低重叠/碎片化轨迹区（采样帧不同 → 均值向量天然不同，属对账方法学效应）。

**低 cos 尾部审计**（cos<0.9 共 43/693）：其中 **20 对存在替代高 cos（≥0.95）配对**
——为贪心匹配在拥挤场景的错配/轨迹碎片化（例：318 `v318_142` 与 `a42` cos 仅 0.23，
但同窗口的 `a39` cos **0.9845**）；其余 23 对特征为：overlap 全部 ≤0.67（14 对 <0.5）、
5 对短轨迹（≤5 帧）——均为两侧 track 切分粒度不同所致，非编码链路缺陷。

注：远程 npz 键前缀因路而异（305/316/323/343 为 `v{cam}_*`，318 为 `a*`、320 为 `b*`）；
对账器按"尾数数字"提取 track id，不受前缀影响（`same_id` 标记仅在两侧前缀相同时有效，
对 cos 统计无影响）。

### 4.3 单帧抽查（可复现的定点）
- 305 frame0：本地 vs 远程 5 框集合一致（4 个主框 IoU>0.99），分数差 ≤0.04；
- 318 frame0（1080p 路径）：4 框位置差 <1px、分数差 ≤0.023。

---

## 五、复现命令

```bash
# 0) 准备（一次性）
#   权重已就位：models/IDEA-Research--grounding-dino-base/（sha1 f9ce46b65eda3420734c…）
#   ONNX 已导出：models/onnx/groundingdino_base.onnx
#   （重导）python tools/export_onnx_hf.py --legacy

# 1) 检测（单路 / 全 6 路）
python tools/local_detect.py --video media-resource/.../305_*.mp4 --cam 305
python tools/local_detect.py --all           # 或 bash tools/run_local_detect_rest.sh

# 2) 嵌入（单路）
python tools/local_encode.py --jsonl output_crosscam_local/detect/305_detections.jsonl \
    --video media-resource/.../305_*.mp4 --prefix v305_ \
    --out output_crosscam_local/embed/reid_emb_305_d3b_local.npz

# 3) 对账（检测 6 路 + 嵌入 6 路）
python tools/local_check.py detect --local-dir output_crosscam_local/detect \
    --remote-dir output_crosscam --json output_crosscam_local/reconcile_detect_all.json
python tools/local_check.py embed \
    --local-dir output_crosscam_local/embed \
    --remote-dir output_crosscam/analysis/relay/remote_emb \
    --local-jsonl output_crosscam_local/detect --remote-jsonl output_crosscam \
    --json output_crosscam_local/reconcile_embed_all.json

# 4) 深度分析 + 总览图
python tools/analyze_embed_match.py      # cos×overlap 分桶 + 低cos审计
python tools/plot_reconcile.py           # -> output_crosscam_local/reconcile_overview.png
```

对账产物清单（`output_crosscam_local/`）：
- `detect/{cam}_detections.jsonl` + `{cam}_summary.json`（6 路，与远程同构）
- `detect/{cam}_raw.jsonl`（去重前原始输出，排障用）
- `embed/reid_emb_{cam}_d3b_local.npz`（6 路，L2 归一 768 维）
- `reconcile_detect_all.json` / `reconcile_embed_all.json`（对账明细，含逐帧序列与全量配对）
- `reconcile_overview.png`（4 联总览图）

---

## 六、未解 / 边界（诚实清单）

1. **track_id 编号分叉**：edge-frame 浮点差异经 ByteTrack 传导，本地与远程的 track_id
   序列局部错位（frame-exact 0.5%–63%，视场景而定；内容级匹配后一致）。做"轨迹集合级"
   研究时建议用内容匹配（`local_check.py embed` 已内置），或对快照化比对（冻结检测结果）后重放。
2. **ONNX 导出的两处补丁**（`export_onnx_hf.py` 内注释）：
   ① `generate_masks_with_special_tokens_and_transfer_map` 预计算为常量
      （绕过 isin/cummax/cummin 的 opset-17 不可导出）；
   ② legacy trace 导出器（dynamo 对 Swin reshape 不可追踪）。
   —— prompt 固定 "person." 前提下等价；换 prompt 需重导 ONNX 并更新 `--token-ids`。
3. transformers 版本行为差（4.46.3 vs 5.20.1）：本机导出用 5.20.1（贴近远程 5.14）；
   嵌入链路用 timm 不受影响。
4. torch-directml 直跑 HF 模型仍不可行（DML 内核崩）——ONNX 是唯一 GPU 路。
5. 实测速度（RX 5700 XT / DML）：检测 base ≈ 600ms/帧（6 路 × ~3000 帧实测总 3h05m，
   1.6 fps）；嵌入 ≈ 1.2 s/轨迹（732 轨迹 ≈ 14 min）。如需提速：DML batch/并发策略未探索；
   CPU 多线程为兜底（~20 s/帧级，不推荐）。
6. C 盘空间仍紧（多 venv）。
