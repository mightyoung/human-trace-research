# 本机环境完善 + GPU 加速验证（2026-10-16）

> 阶段交接文档。目标：本机（不再用远程 r4900g3）完整可跑；显卡环境就绪；Rust 服务真正可用。
> 全部结论均有实测证据，命令可复现。

---

## 一、结论先行

| 项 | 状态 | 证据 |
|---|---|---|
| Rust 服务编译 | ✅ | MSVC 升至 14.44 后 `cargo build --release` 通过（约 1m10s） |
| Rust 服务 E2E | ✅ | 1440p 真实帧 → 检出 6 人（与 Python 路交叉一致） |
| **Rust 服务 GPU（DirectML）** | ✅ **22×** | 9.5 s/帧 → **0.42 s/帧**（稳态），数值与 CPU 等价 |
| Python 检测 GPU（ORT-DML） | ✅ 18× | 0.43s vs 7.9s；检出集合一致、conf Δ<2e-3、框 Δ<1.3e-5 |
| Python 嵌入 GPU（torch-directml） | ✅ 12.5× | DINOv3-B 22ms vs 275ms |
| GroundingDINO on torch-directml | ❌ 不可用 | DML 内核硬崩（详见 §四），检测走 ORT-DML |
| 本机实验链路 | 🟡 地基就绪 | venv / 模型 / 工具齐备；跨机位管线脚本尚未本地实跑 |

**核心结论**：AMD RX 5700 XT（无 CUDA）在本机现在**可以**加速项目主链路——
检测走 ONNX + ORT-DML，嵌入/ReID 走 torch-directml 或 ONNX。

---

## 二、两个关键认知修正（推翻了早前判断）

### 1. "DML 跑不了 GroundingDINO" 是误判
早前失败的两个叠加原因：① 喂了错误尺寸（模型只接受导出尺寸）；② 报错信息是
GBK 编码的 Windows 消息，被 Python 当 UTF-8 解码失败掩盖（可用
`UnicodeDecodeError.object` 还原原文）。

**修正**：`onnxruntime-directml 1.24.4` 在 1333×750 下完整跑通 GroundingDINO，
稳态 0.43s（CPU 7.9s），任务级数值等价。

### 2. ONNX 模型的 750×1333 约束对实际使用**无影响**
导出时 dummy=750(H)×1333(W) 且开启常量折叠，Swin 的形状常量被固化——模型只能在
**1333 宽 × 750 高**下运行。

**但 1333:750 = 16:9**。官方预处理对任意 16:9 源（720p/1080p/1440p）输出恰好都是
1333×750。项目全部实验视频（6 路）是 **2560×1440 16:9** → 完美命中。
仅非 16:9 输入会失败（如 `testdata/bus.jpg` 810×1080；海康 704×576 子码流）。

> 若未来要支持非 16:9 输入：在导出尺寸下重导 ONNX（对数尺寸各一份），或改预处理
> 做 letterbox 到 1333×750。当前无需求。

---

## 三、Rust 服务改动清单（DirectML 落地）

| 文件 | 改动 |
|---|---|
| `Cargo.toml` | ort features 增加 `"directml"`（ort-sys 该 feature 为空，无下载/链接副作用） |
| `src/config.rs` | `ModelConfig` 新增 `execution_provider`（`"auto"`默认 / `"cpu"` / `"dml"`） |
| `src/detector.rs` | DML EP 注册；auto=先试 DML→失败回退 CPU；**启动自检** DirectML.dll 新鲜度 |
| `config.toml` | 新参数文档 + 实测数字 |

**运行前提**（唯一部署动作）：

```bash
cp vendor/ort/onnxruntime/lib/DirectML.dll target/release/DirectML.dll
```

原因：exe 静态链接的 ORT 已含 DML EP（导入表可见 DirectML.dll/dxgi/d3d12/dxcore），
但 System32 自带的是 2020 旧版（1.2MB）。**DLL 搜索顺序中 exe 目录优先**——
放了新 DLL（18MB）才真正启用 GPU；没放则 DML"注册成功但静默回退 CPU"（实测 9.4s/帧）。
服务启动自检会就此打印明确告警（防"以为在用 GPU"）。

### 实测矩阵（2560×1440 视频，预处理 1333×750）

| 场景 | 结果 |
|---|---|
| `cpu` | 9.5–9.9 s/帧 |
| `auto` + 新 DLL | **0.42 s/帧稳态**；首帧 1.0s；冷启动最坏 3.5s |
| `auto` + 无新 DLL | 告警 + 直选 CPU（9.6–9.9s），不假装 GPU |
| `dml`（强制） | 正常，1026ms 首帧 / 426ms 稳态 |
| DML vs CPU 数值 | 同一帧检出集合一致；框差 <0.3px；conf 差 <1e-3 |

---

## 四、torch-directml 的适用边界（实测）

| 模型 | 结果 |
|---|---|
| matmul 2048² | 69→3 ms（23×） |
| ViT-tiny / DINOv3-B | 19→11ms / 275→**22ms**（12.5×） |
| **GroundingDINO 视觉路径** | ❌ `roll` 回退 CPU 后，DML 内核硬崩（`dml_tensor_desc.cc CHECK failed`，进程中止） |

→ **检测模型不要走 torch-directml**；嵌入类（DINOv3/ReID）可走，或统一走 ONNX+ORT-DML。

环境备忘（dml_torch venv，py3.11）：
- torch 2.4.1 + torch-directml 0.2.5.dev240914（装时需 pip==23.3.2 绕元数据 bug）
- 官方仓库格式 GroundingDINO 本机可直接跑（需 transformers **4.46.3**；5.x 删了
  `get_extended_attention_mask`；另需 yapf/addict/pycocotools/cv2）
- 坑：prompt 必须 `"person ."`（带中点）；只传 `"person"` 会在掩码构建处
  `torch.stack([])` 崩溃

---

## 五、环境清单（本机）

| 项 | 值 |
|---|---|
| GPU | AMD RX 5700 XT（RDNA1, 8GB），DirectX 12 + DirectML |
| Rust | 1.99.1 stable；`cargo build --release --offline`（D:/human-count） |
| MSVC | VS2022 BuildTools 14.44.35207（原 14.29 缺 `__std_*` 符号） |
| ffmpeg | `D:/dev/ffmpeg/bin/ffmpeg.exe`（7.1 essentials） |
| venv `default` | 全 ML 栈：torch 2.14.1+cpu / transformers 5.20.1 / timm / supervision / onnxruntime 1.31 等 |
| venv `dml` | `onnxruntime-directml 1.24.4`（检测 GPU 路线） |
| venv `dml_torch` | torch 2.4.1 + torch-directml 0.2.5（嵌入 GPU 路线）+ transformers 4.46.3 |
| 模型 | 6 路视频（2560×1440）；ONNX 检测模型；DINOv3-B/S、C-RADIOv4、人脸模型等齐备 |

复现命令（关键三条）：

```bash
# 1) Rust 服务 GPU 推理
cp vendor/ort/onnxruntime/lib/DirectML.dll target/release/
./target/release/human-count.exe --config .workbuddy/_scratch/config_video1440.toml \
    --video media-resource/multi-cemra/synced-10min/01_*/316_*.mp4 --max-frames 6

# 2) Python ORT-DML 检测（1333×750）
python .workbuddy/_scratch/cross_check_frame.py <图片>

# 3) torch-directml 嵌入测速
python .workbuddy/_scratch/dml_torch_test.py   # dml_torch venv
```

---

## 六、未解 / 下一步

1. **跨机位实验管线本地化**：`tools/infer_supervision.py` 之前依赖远程 L20；
   本地替代路径已就绪（ONNX-DML 检测 + DINOv3 torch-directml 嵌入），
   但**尚未组装成端到端本地脚本**——这是"实验在本机做"的实际下一步。
2. 非 16:9 输入的检测支持（当前不支持，无需求时可不管）。
3. 服务 RTSP 实流未验证（本机无摄像头；视频文件链路已验证）。
4. C 盘余量偏紧（多 venv 占用）。
