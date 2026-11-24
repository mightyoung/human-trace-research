# OSNet 嵌入替换评估 + 低光增强检测实验（2026-10-17）

> 承接指令：「embedding 替换为 OSNet（确认替换对象）」「低光下先实验看看效果」。
> 纪律：只新增产物、不覆盖旧产物；推理优先走 GPU（本机 DirectML）。
> 结论先行，事实与推断分开标注。

---

## 零、先回答：替换的到底是什么

项目里的「embedding」不是单个模型，而是三层：

| 层 | 当前实现 | 作用 |
|---|---|---|
| 视觉 ReID 编码器 A | C-RADIOv4-SO400M → 桥接 W → EmbeddingGemma2 embed_vision（768d） | 全局 ReID 库（`cradio_gemma2_bridge_v4`，539 轨）、语义查人 |
| 视觉 ReID 编码器 B | DINOv3-B/16（768d） | 跨机配对默认后端；本地批量编码（d3b） |
| 文本塔 | EmbeddingGemma2（text） | 中文/英文描述 → 768d 语义向量 |

**结论（事实）**：OSNet 替换的是**视觉 ReID 编码器**（A/B 两套 768d 空间中的视觉侧），
不是 EmbeddingGemma 文本塔本身。OSNet 输出 512d、且没有配套文本塔 →
替换后「文本语义查人」无法保留（除非保留 Gemma 文本塔 + 双索引）。

本报告仅把 OSNet 作为「视觉嵌入的替代候选」验证，**未改动任何生产库/选择器**。

---

## 一、GPU 环境修正（回应「为什么都不用 GPU」）

问题两个，根因不同：

| 环节 | 现象 | 根因 | 修正 |
|---|---|---|---|
| OSNet 编码 | 169 ms/裁剪（CPU） | `envs/default` 的 torch 是 CPU 版；首版脚本默认 `device=cpu`（为对齐 exp8 基准） | 切 `envs/dml_torch`（torch 2.4.1 + torch-directml + torchreid），`device=privateuseone:0` |
| 低光检测 | ~7.9 s/帧（CPU） | `envs/default` 的 onnxruntime 1.31 **无 DmlExecutionProvider** | 切 `envs/dml`（onnxruntime-directml 1.24.4） |

实测（本机 AMD RX 5700 XT）：

| 任务 | CPU | DirectML | 加速 |
|---|---|---|---|
| OSNet 编码（单裁剪，纯推理） | 169 ms | **12.2 ms** | **13.9×** |
| GroundingDINO 检测（1440p→1333×750 单帧） | ~7.9 s | **668 ms** | **11.8×** |

数值一致性：343 路 57 轨，CPU 版 vs DML 版嵌入 **cos = 1.000（全部 57 条）**。

> 批量编码实测 112–229 ms/裁剪（含视频 seek/解码，且与低光实验并行争 GPU/IO），
> 12.2 ms 是纯前向口径。两类指标不要混用。

**★ 后续约定**：OSNet/嵌入类 → `envs/dml_torch`；ORT 检测类 → `envs/dml`；
`envs/default` 只做纯 CPU 分析/导出，不再跑推理。

---

## 二、OSNet 替换实验

### 2.1 产物

- 6 路 **715 轨 × 512d**：`output_crosscam_local/embed_osnet/reid_emb_{cam}_osnet_local.npz`
- 旧 d3b（732 轨 × 768d）：`output_crosscam_local/embed/reid_emb_{cam}_d3b_local.npz`（**未改动**，可回滚）
- 轨迹数差异：305 105 vs 109、318 140 vs 153 —— 裁剪尺寸下限不同导致的 crop-fail，非链路故障

### 2.2 冻结集对照（`frozen_eval_v1`，109 对人工判定）

有效 69 对（same 31 / diff 38）；同裁片、同采样（6 框/轨）、同聚合（per-track 均值 → per-entity 均值）：

| 编码器 | 维度 | AUC | same 中位 | diff 中位 | margin | crops | 设备 |
|---|---|---|---|---|---|---|---|
| OSNet（MSMT17 combineall） | 512 | **0.9270** | 0.7443 | 0.6144 | -0.2013 | 673 | DML |
| DINOv3-B（CLS token） | 768 | 0.6282 | 0.7343 | 0.7077 | -0.2144 | 673 | DML |

- 与 exp8（CPU 版）一致：OSNet 0.9219 / dinov3b 0.6282 → 本次 GPU 版 0.9270，差异在浮点/采样噪声内
- **OSNet 在跨机位外观判别上显著更强（+0.30 AUC）**
- 两者 margin 均为负 → 单阈值不可完美分离，仍需分层/人工复核

### 2.3 关键 caveat（不能忽略）

1. 此处 DINOv3-B 是 **CLS token 原生口径**，不是生产链的 `cradio_gemma2_bridge` 口径
   （本地不可复现）→ **0.6282 不代表生产 DINOv3 的真实水平**。
2. 样本 69 对、单时段（08:55–09:05）、6 固定机位；95% CI 宽。
3. OSNet 是 MSMT17 有监督微调的「行人 ReID」，现场域差异（光照/遮挡/机位）仍是待测问题。
4. 切换成本：按 CQ-12 版本铁律，需重编码全局库（539 轨）+ 重训选择器（412 子原型）
   + 重跑跨机评估；且文本语义能力会丢失。

### 2.4 结论与建议

- **事实**：OSNet 在本项目冻结集上明显优于 DINOv3-B 通用特征。
- **建议**：先做「OSNet 版选择器」本地对照（用 1012 训练集重训），确认端到端收益后再切生产；
  保留 d3b 产物可回滚。不要直接覆盖 `cradio_gemma2_bridge` 全局库。

---

## 三、低光增强检测实验

### 3.1 方法

- 机位：343 电梯间（最暗，luma≈98）/ 320 办公区球机（逆光，luma≈128.7）
- 帧：343 从 frame 1600 起 240 帧；320 从 14100 起 161 帧（至视频尾）；stride 5
- 条件：`orig` / `lab_gamma`（自适应 gamma，目标 128，γ∈[0.45,1]，算法同 P1 人脸增强）/ `clahe`（LAB-L clip 2.5）
- 检测：同一 ONNX（GroundingDINO-base）+ DML + `box_threshold=0.30`；不做跟踪/去重

### 3.2 结果

**343 电梯间（低光）**

| 条件 | 框数 | 帧均 | 检出帧率 | conf mean | conf median | 新增/丢失（IoU>0.5） |
|---|---|---|---|---|---|---|
| orig | 258 | 1.075 | 0.717 | 0.6064 | 0.6516 | — |
| lab_gamma | 250 | 1.042 | 0.717 | 0.6155 | 0.6758 | +5 / -12（conf≈0.31） |
| clahe | 262 | 1.092 | 0.717 | 0.6027 | 0.6375 | +12 / -5（conf≈0.32） |

亮度：orig 97.98 → gamma 118.66 / clahe 110.36。

**320 办公区球机（逆光）**

| 条件 | 框数 | 帧均 | 检出帧率 | conf mean | conf median | 新增/丢失 |
|---|---|---|---|---|---|---|
| orig | 391 | 2.429 | 1.000 | 0.6412 | 0.6626 | — |
| lab_gamma | 392 | 2.435 | 1.000 | 0.6404 | 0.6624 | +1 / 0 |
| clahe | 392 | 2.435 | 1.000 | 0.6449 | 0.6643 | +5 / 0 |

亮度：orig 128.7 → gamma 128.73（γ≈1.0，几乎不动作）/ clahe 135.7。

### 3.3 结论

1. **检出帧率三条件完全一致**（343=0.717；320=1.000）→ 全图增强不改变「有没有人」。
2. 框数差异 ≤3%，新增/丢失框全部落在 0.30–0.36 的阈值边缘 → 噪声级抖动，不是真实召回提升。
3. 343 拼图显示：主要损失在**窗/门强光过曝区**（像素饱和、信息已丢失，增强无法恢复）；
   暗部目标本来就能检出。
4. 320 全图均值已接近目标 128，自适应 gamma≈1.0；CLAHE 只提对比，检出不变。

### 3.4 边界（诚实清单）

- 单时段、单参数、检测代理指标；未做人工全量真值。
- 只测了检测，未测人脸可用率/嵌入判别力。
- 未试深度 LLIE（Retinexformer 等）——本次按「先规则后模型」只验证规则增强。

### 3.5 建议

- **不要把全图 gamma/CLAHE 接进检测主链**（无收益、且改变输入分布）。
- 真正的低光损失在过曝区：应从**曝光/摄像头侧**解决（宽动态、曝光策略），
  或 ROI 局部增强 + 更强检测器。
- 若继续，下一步做 ROI 局部增强（人体框内）与人脸可用率口径的 A/B。

---

## 四、产物与复现

新增脚本：

| 脚本 | 用途 |
|---|---|
| `tools/local_encode_osnet.py` | OSNet 轨迹编码（`--device auto/dml/cpu`，DML 12.2ms/裁剪） |
| `tools/run_local_encode_osnet_all.sh` | 6 路批量编码（`PY=` 可指定 dml_torch venv） |
| `tools/osnet_vs_d3b_eval.py` | 冻结集对照（`--device dml`，同裁片同口径） |
| `tools/lowlight_detect_ab.py` | 低光 A/B（`--start-frame` 避开空场景） |

产物：

- `output_crosscam_local/embed_osnet/reid_emb_{cam}_osnet_local.npz`（6 路 / 715 轨 / 512d）
- `output_crosscam_local/osnet_eval/osnet_vs_d3b_frozen_eval.json`
- `output_crosscam_local/lowlight_ab/{343,320}_ab.json` + `{343,320}_montage.jpg`
- CPU 版对照备份：`.workbuddy/_scratch/evidence/reid_emb_343_osnet_cpu.npz`

复现：

```bash
PY=C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe
bash tools/run_local_encode_osnet_all.sh

$PY tools/osnet_vs_d3b_eval.py --device dml

C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml/Scripts/python.exe \
  tools/lowlight_detect_ab.py --cam 343 --frames 240 --start-frame 1600
```

## 五、本机推理环境约定

| 任务 | venv | 设备 |
|---|---|---|
| ORT 检测（GroundingDINO / ONNX） | `envs/dml` | DmlExecutionProvider（668ms/帧） |
| OSNet 编码 | `envs/dml_torch` | privateuseone:0（12.2ms/裁剪） |
| DINOv3 编码 | `envs/dml_torch` | privateuseone:0（22ms/裁剪） |
| 纯 CPU 分析/导出 | `envs/default` | CPU |

---

## 六、污染审计（用户指正后补做）

### 6.1 五条独立检查

| 检查 | 结果 | 判定 |
|---|---|---|
| splits_v5 内部 tid 交集 | train∩test = 0；train∩val = 0；val∩test = 0 | 干净 |
| splits_v5 内部实体交集 | train 304 / val 59 / test 48 实体，两两交集 **0** | 干净 |
| frozen_eval_v1 vs splits_v5.train | 109 实体中 **81 个在 train**（val 17 / test 11） | **不得用于本地训练模型的评估** |
| 1012 train 段 vs synced-10min | 318_train 08:22–09:22 / 320_train 08:03–09:03 / 323_train 08:35–09:35，均覆盖 08:55–09:05 | **时间重叠，跨数据集评估须隔离** |
| 本地 track_id vs clean2 tid | 本地 npz 对 splits 的 tid 命中率仅 96/349（27%） | **编号分叉，须用 cleanset_v2 轨迹重编码** |

### 6.2 对既有结论的影响

- **OSNet vs DINOv3 的 frozen_eval 对照（0.927 vs 0.628）不受污染影响**：
  两个编码器均为零样本迁移（OSNet 在 MSMT17 训练、DINOv3 为通用预训练），未接触本项目实体。
- **但该集不能用于选择器评估**（81/109 实体在 train 中出现过）。
- 选择器评估改用 splits_v5（实体零交集）+ cleanset_v2 轨迹重新编码。

---

## 七、干净口径选择器评估（splits_v5，实体零交集）

编码：cleanset_v2 轨迹 + 视频重编码（GPU），两编码器**公共轨迹** train 268 / test 68
（部分轨迹因裁剪门槛或 cleanset 缺失被剔除，覆盖率 268/349≈77%、68/98≈69%）。

| 编码器 | AUC（POS30 8 正 vs NEG30 18 负） | POS30 同簇 | NEG30 同簇 | pairs_same 同簇 |
|---|---|---|---|---|
| OSNet | **0.9931** | 1–2/8 | 0/18 | 27/44 |
| DINOv3-B | 0.9722 | 0/8 | 0/18 | 27–30/44 |

- 余弦判别力：OSNet 仍优（+0.021 AUC）；差距比 frozen_eval 的跨机位场景（+0.30）小——
  318×320 是真实重叠机位，本身较易。
- **选择器同簇率极低**：268 轨聚成 178–252 簇（平均 1–1.5 轨/簇）→ 原型库过碎，
  test 轨迹几乎都判 unknown。**样本量不足，需用 1012 的 1395 轨重训**。
- 结论：余弦层结论成立（OSNet 更优）；选择器层需扩量重训后再判定。

### 7.1 1012 重训（进行中）

- 仅用 1012 的 train（上午 60min）训练、test（下午 15min）评估 → **时间隔离**；
  **不**在 synced-10min / frozen_eval 上评估（6.1 已证时间重叠）。
- 编码脚本：`tools/run_osnet_1012.sh`（a/b 分片，8 段 = 4 train + 4 test）。
- 状态：编码已启动，未完成（截至本文档更新）。

---

## 八、跨镜 ReID 稳定跟踪的证据汇总（当前）

| 场景 | 集合 | OSNet | DINOv3-B |
|---|---|---|---|
| 6 路跨机位（含非重叠/跨时段） | frozen_eval_v1（69 对，零样本） | **0.927** | 0.628（CLS 参照） |
| 318×320 重叠对（轨迹级，干净） | splits_v5.test（8 正 18 负） | **0.9931** | 0.9722 |
| 同机跨片段（干净，44 对可见） | splits_v5.test pairs_same | 27/44 | 27–30/44 |

→ OSNet 的优势集中在**困难跨机位**（+0.30），重叠机位提升较小（+0.02）。
语义查询按用户决策放弃，OSNet 替换无文本能力损失问题。
