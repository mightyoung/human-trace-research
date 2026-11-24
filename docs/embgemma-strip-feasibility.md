# EmbeddingGemma2 死重剥离：可行性与下游影响研究报告

- 日期：2026-10-14（与检测头训练实验并行，实验走 GPU1）
- 环境：L20 / cuda:1 / torch 2.11.0+cu130 / transformers 5.19.0 / sentence-transformers 6.1
- 脚本：`tools/emb_strip_verify.py`（主验证）、`tools/emb_strip_flagtest.py`（补丁验证）
- 产物：`output_crosscam/analysis/det_exp/bench_emb_strip.json`、`bench_emb_strip_flagtest.json`
- 远程产物：`/share/data4/yhh/data/human-count/bench_emb_strip.json`（另有留档向量 `emb_strip_flagtest_vectors.npy`）

---

## 一、结论（TL;DR）

**可行，且是零风险优化。** 剥离三个推理期死重塔（`audio_tower` 304.8M + `vision_tower` 167.4M + `embed_audio` 0.8M，合计 472.9M 参数）后：

| 维度 | 实测结果 |
|---|---|
| 图像嵌入（向量化） | **逐位不变**（256 张真实裁剪，bitwise=0 diff，max_abs=0.0） |
| 文本嵌入（查询） | **逐位不变**（117 条生产短语，同上） |
| 显存 | 4.744 GB → **2.899 GB（省 1.845 GB，-38.9%）** |
| 编码速度 | 22.28 → 21.24 ms/图（持平，噪声级） |
| 对四项下游 | **全部零影响**（详见第四节影响矩阵），无需重建任何库/阈值/存储 |

> 逐位一致是压倒性的证据：系统里最紧的决策间隔是 truth 判定的 cos 间隔 **0.0002**（比质变）；剥离带来的差异是 **0**，比该间隔小无穷倍——不存在任何"阈值余量"问题。

---

## 二、为什么可以剥离（结构依据，源码级）

FastEncoder 的推理路径（`crosscam_run/embed_fast.py`）只使用模型的三个组件：
`language_model`（语言塔）+ `embed_vision`（RADIO 桥接投影）+ 我们自训的 W 矩阵。

对 `transformers/models/embedding_gemma2/modeling_embedding_gemma2.py` 逐行核实：

1. **图片路径**：`forward`（L785）调用的是 `self.get_image_features(...)` —— 该方法已被 FastEncoder 替换为 RADIO 桥接（`_patched`），**原生 `vision_tower` 从不被触碰**（docstring 亦明示 capture 不依赖它）。
2. **`embed_vision` 必须保留**：桥接直接调用 `am.embed_vision(inputs_embeds=z2)`；其构造条件是 `config.vision_config is not None`（与 vision_tower 独立）。它只用到 vision_config 的几个标量字段（output_proj_dims/rms_norm_eps），**不需要塔本体**。
3. **`audio_tower` / `embed_audio`**：我们的路径不涉及任何音频。官方 `_keys_to_ignore_on_load_unexpected` 也已把四个塔的权重列为可忽略 pattern（L638-641）——官方设计本就允许它们缺省。
4. **文本路径**：纯文本 forward 不经过任何塔（towers 仅在 image/audio/video 特征请求时被调用），因此查询路径天然不受影响。

## 三、验证设计与实测数据

### 3.1 验证设计（把"cos≈1"升级为"逐位"）

| 项 | 设计 |
|---|---|
| 样本 | **256 crops**（316×128 + 318×128，真实轨迹裁剪，与生产同口径）+ **117 文本**（`webapp/dataprep/build_text_vocab.py` 的生产词表 109 条**逐字复制** + 8 条 demo query） |
| 对比 | float32 的 **uint32 视图逐位比较**（非只看 cos）+ max_abs + 差异元素计数 |
| 确定性底线 | 同一实例连续编码两次 → **逐位一致**（排除浮点非确定性干扰） |
| 机制对比 | 后置剥离（A）vs 载入期 config 传参（B/C）→ 见 3.3 |

### 3.2 主结果

| 对比 | bitwise | max_abs | cos_min |
|---|---|---|---|
| 确定性底线（同实例 ×2） | ✅ | 0.0 | 0.99999988 |
| **后置剥离后 crops（256）** | ✅ | **0.0** | 0.99999988 |
| **后置剥离后 texts（117）** | ✅ | **0.0** | 0.99999988 |
| 补丁 helper 路径 vs 默认（64 crops） | ✅ | 0.0 | — |
| 补丁 init 路径 vs helper（64 crops） | ✅ | 0.0 | — |

参数分布（实测）：`language_model` 271.0M + `embed_vision` **0.4M** 保留；剥离 `audio_tower` 304.8M + `vision_tower` 167.4M + `embed_audio` 0.8M = **472.9M**。

> 注：cos 列显示 0.99999988 是因为两个"不同存储数组"的逐位比较链路上 cos 用 float 累加；关键指标是 **bitwise=true / max_abs=0.0**。

### 3.3 剥离机制对比（三个方案的裁决）

| 机制 | 结果 | 结论 |
|---|---|---|
| A. 后置剥离（加载后置 None + gc） | ✅ 已验证，逐位一致 | **采用** |
| B. 载入期 `config.audio_config=None` 经 ST `model_kwargs` 传 config 对象 | ❌ `TypeError: multiple values for argument 'config'`（ST 6.1 自行传入 config，冲突） | 否决 |
| C. 载入期 `audio_config=None` 直传 kwarg | ❌ TypeError（kwarg 被误传给 model `__init__`） | 否决 |

**结论：单一路径——后置剥离**（更简单：10 行代码、零配置手术、零产物）。载入期的理论收益（省约 0.9GB 瞬时加载峰值、少读几百 MB 磁盘）在 11.6s 加载/48GB 卡的场景下不值得引入 ST 兼容风险。

### 3.4 持久化（剥离 ckpt 落盘）——**决定不做**

探索性测试因 B 失败而未执行；评估后明确**不采纳**：后置剥离已达成全部收益，落盘方案只省 ~0.5GB 磁盘和几秒加载，却要引入新模型产物（路径管理、版本漂移、"哪份是正版"的审计负担）。**保持"官方权重 + 运行时代码剥离"单一事实源**。

---

## 四、下游影响矩阵（用户四问逐一回答）

### 4.1 向量化（图像嵌入生产）——零影响

- 消费链：`embed_fast`（离线批量 20ms/图）→ `embed_server`（在线 33 图/s，GPU1 常驻 7.4GB）→ 全部分析产物。
- 系统内最紧决策间隔 = truth 判定 cos 间隔 **0.0002**；剥离差异 = **0**（逐位）→ 对齐余量无穷大。
- 其他相关阈值：stitch `link-cos≥0.80`、relay 分箱 `0.943/0.95`、选择器 margin `0.003–0.026` —— 均远大于任何可设想噪声，且实际噪声为 0。

### 4.2 查询（文本嵌入）——零影响，且无需重算已有产物

- 生产路径拆解：**离线**（本地 embgemma venv）用 `model.encode(texts, prompt_name="SearchQuery")` 预计算概念短语向量 → `text_vocab.npz` 下发 → **运行时** webapp 只做"中文关键词 → 概念键 → 查表组合"，不再跑文本塔。
- 109 条生产短语（词表全量复制）+ 8 条 demo query 的剥离前后逐位一致 → `text_vocab.npz` **无需重建**（重建也逐位相同）；`embgemma2_text_search.py`（demo 的实时文本编码）同样不受影响。

### 4.3 全局一致性记忆——零影响，且关键风险被消除

- 项目里的"全局一致性记忆"载体：ReID 全局库（768d 桥接嵌入）、GIL store（obs/frag/entity 向量 + gallery）、relay v2/v5 嵌入、truth 判定 cos、链构建 —— 全部消费同一套图像嵌入。
- **唯一能造成影响的路径是"混合版本嵌入"**：若新旧编码器输出有差异，则旧存储（全量版向量）与新推理（剥离版向量）混用时，全局一致性被破坏——这正是 cradio v4/v5 口径事故（cos 仅差 0.0015，就足以破坏 truth 阈值判定）的教训。
- 本次结论：**逐位一致 ⇒ 该风险不存在**。剥离后重启服务、继续追加编码，与历史存储无缝兼容；**不需要重建任何 store**。

### 4.4 聚类选择器训练集选择——零影响

- `selector_v2.npz`（multi8：412 子原型 + per-cam 两档阈值 + margin 拒识 `{305:0.0030, 318:0.0124, 320:0.0224, 323:0.0045}`，r80 规则）由该嵌入训练/校准。
- 逐位一致 ⇒ 既有库、阈值、选择结果不变；**未来即使重建选择器**，剥离版与全量版产出逐位相同，无需保留全量版做"训练集选择"对照。
- 选择器相关一切下游（三档工作点 0.972/0.977/1.000、跨段指标、Top-K 富集）均无需重新验证。

---

## 五、落地建议（补丁已备，待授权启用）

### 5.1 补丁内容（`tools/embed_fast.py`，已改好并副本验证）

```python
DEFAULTS = dict(..., strip_dead=False)   # 默认关闭：不传即全量行为（字节级不变）

class FastEncoder:
    def __init__(self, ..., strip_dead=None):
        ...
        strip_dead = d["strip_dead"] if strip_dead is None else strip_dead
        ...
        if strip_dead:
            self.strip_dead_towers()

    def strip_dead_towers(self):
        """剥离 audio_tower / embed_audio / vision_tower（嵌入逐位不变，省 ~1.85GB）。
        注意：embed_vision 是桥接依赖，必须保留。"""
        for name in ("audio_tower", "embed_audio", "vision_tower"):
            sub = getattr(self.am, name, None)
            if sub is not None:
                setattr(self.am, name, None); del sub
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 5.2 启用方式（三选一，按场景）

| 场景 | 方式 | 效果 |
|---|---|---|
| 在线服务 `embed_server` | 构造处传 `strip_dead=True` → 重启服务 | GPU1 常驻显存 7.4GB → **约 5.5GB** |
| 离线批量（embed_fast 直接调用） | `FastEncoder(strip_dead=True)` | 每次运行省 1.85GB |
| 暂不启用 | 不传（默认） | 行为与现在**完全一致** |

**回退**：删除参数即恢复原行为；补丁本体默认关闭，可先行安全合入。

### 5.3 注意事项（边界纪律）

- **只作用于 FastEncoder 装载点**：`cradio_scale_remote.py` / `cradio_fusion_remote.py` / `dinov3_bridge_remote.py` 等历史脚本**直接使用 `vision_tower`**，它们不经过 FastEncoder，不受影响、也不要给它们加剥离。
- 剥离后 `am.get_video_features()` / 原生 `get_image_features()` 会报错——本项目均不使用（桥接已替换图片路径）；如需恢复，重新加载即可（无需重新下载权重）。

### 5.4 生产执行记录（2026-10-14 15:15–15:22，已授权执行 ✅）

| 步骤 | 证据 |
|---|---|
| 替换前备份 | `embed_fast.py.bak_20261014`、`embed_server.py.bak_20261014`（原文件 byte 级保留） |
| 文件替换 | `embed_fast.py` md5 `db9720c4…`、`embed_server.py` md5 `c75c5381…`，本地↔远程 md5 一致 + 远程 py_compile 通过 + 补丁标记 grep 双确认 |
| 重启前基线 | 旧服务 `POST /encode` 编码 4 张测试裁剪 → `before.json`（sha `1cc248cf…`）；GPU1 显存 7414 MiB |
| 优雅停机 | `POST /shutdown` → `{"bye": true}`；端口关闭、进程退出、GPU1 显存降至 17 MiB |
| 新参数启动 | `--strip-dead`；日志 `[server] 加载 FastEncoder（cuda:1）strip_dead=True`；`/health` ok |
| **生产 A/B 逐位对照** | 重启后同 4 图编码 → **`bitwise_equal(JSON): True`，max_abs=0.0，n_diff=0** |
| 显存 | GPU1 7414 → **4380 MiB**（-3034 MiB；其中剥离本体 ≈1803–1845 MiB，其余为 21h 运行碎片的重启回收） |
| 旁证 | 训练进程（GPU0）不受影响（96% 利用率持续）；`/health` 队列 0、workers 3 |

**回退方法**（任一）：
1. 轻回退：重启服务时去掉 `--strip-dead`（新文件默认关闭，秒级回退，无产物变化）；
2. 全回退：`cp embed_fast.py.bak_20261014 embed_fast.py; cp embed_server.py.bak_20261014 embed_server.py` 后重启。

---

## 六、限制与未验证项（如实标注）

1. **跨进程确定性未单独实测**：本次为同进程内"全量实例 vs 剥离实例"逐位一致 + 同实例重复逐位一致（确定性底线）。跨进程（重启后）理论上同卡同 shape 的 cuBLAS 确定性成立；且生产重启后全部向量来自单一版本，不存在混合一致性风险。
2. **载入期机制被否决**（ST 6.1 接口冲突），非"不可行"而是"不值得"：后置剥离语义等价且更稳。
3. **未测 bf16 源权重 dtype 的影响**：加载统一走 `torch_dtype=float32`（与生产一致），剥离不改变 dtype 路径。
4. 剥离收益数字基于 cuda:1 单实例实测（4.744→2.899GB）；多进程/多实例场景按实例数线性叠加。

## 七、复现

```bash
# 主验证（GPU1，~4 分钟）
~/yhh/human-count/embed_venv/bin/python ~/yhh/human-count/bench/emb_strip_verify.py \
    --device cuda:1 --out /share/data4/yhh/data/human-count/bench_emb_strip.json

# 补丁副本验证（~1 分钟）
~/yhh/human-count/embed_venv/bin/python ~/yhh/human-count/bench/emb_strip_flagtest.py --device cuda:1
```

本地脚本：`tools/emb_strip_verify.py`、`tools/emb_strip_flagtest.py`；补丁：`tools/embed_fast.py`（strip_dead 开关）。
