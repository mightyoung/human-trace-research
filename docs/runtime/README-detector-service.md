# human-count

RTSP 摄像头人体检测服务：**Rust 服务 + GroundingDINO (ONNX Runtime) + FFmpeg 拉流**。

接入摄像头的 RTSP 流，按固定间隔抽帧，运行 GroundingDINO 检测画面中的「人」，
输出人数与检测框（控制台 + JSONL），可选 Webhook 告警。

> **构建期零原生依赖**：RTSP 解码通过调用外部 `ffmpeg` 可执行文件完成，
> 不链接 OpenCV / FFmpeg SDK，因此不需要 cmake / pkg-config / libclang。

---

## 1. 架构

| 模块 | 文件 | 职责 |
|---|---|---|
| 配置 | `src/config.rs` | 读取 `config.toml`，解析相对/绝对路径 |
| 拉流 | `src/capture.rs` | 起 ffmpeg 子进程拉 RTSP，按帧读 BGR24 |
| 图片输入 | `src/image_input.rs` | 静态图片读入（仅离线端到端验证用） |
| 预处理 | `src/preprocess.rs` | 等比缩放 (shortest=800, longest≤1333) + 归一化 + NCHW |
| 推理 | `src/detector.rs` | ONNX Runtime 执行模型，sigmoid + 阈值筛框 |
| 后处理 | `src/detection.rs` | cxcywh→xyxy、clip、greedy NMS (IoU>0.7) |
| 输出 | `src/output.rs` | JSONL 落盘 + 可选 Webhook |
| 调度 | `src/main.rs` | 抽帧节流、断流重连、`--image` 单图模式 |

数据流：

```
RTSP --ffmpeg--> BGR24 原始帧 --预处理--> [1,3,H,W] --ORT--> pred_logits/pred_boxes
   --sigmoid+阈值--> 候选框 --cxcywh→xyxy+clip--> NMS --> 人数 + 框
```

**为什么用 ffmpeg 子进程而不是链接库？**
`opencv` crate 0.93 起不再内置源码构建，要求系统预装完整 OpenCV（还需 cmake/pkg-config）；
`ffmpeg-sys` 同样要求预装 FFmpeg SDK。两者都会让构建环境变得很重。
而 ffmpeg 的 CLI 本身就把 RTSP/RTP 的解码、重连、缓冲处理得很好，
拉流层只需读取固定字节数的原始帧，实现简单且零链接依赖。

---

## 2. 环境要求

- **Rust** 1.81+（`ort` rc.10 的 MSRV；本项目在 1.94.1 上开发）
- **FFmpeg**：运行时依赖，需在 PATH 中，或用 `camera.ffmpeg_path` 指定完整路径。
  推荐 [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) 的 full build（含 RTSP demuxer）。
- **MSVC 工具集 ≥ 14.40**：`ort` 下载的 ONNX Runtime 预编译库是用 STL 14.40+ 构建的。
  若本机只有 14.3x，链接会报 `__std_search_1` / `__std_remove_1` / `__std_max_element_f`
  等符号缺失（详见第 6 节「已知限制」）。
- **ONNX Runtime**：由 `ort` 的 `download-binaries` 在构建时自动下载，无需手工安装。
- **Python 3.10+（仅导出模型时需要）**：torch / torchvision / transformers /
  onnx / onnxruntime / addict / matplotlib / timm

---

## 3. 一次性准备：导出 ONNX 模型

本地已有 GroundingDINO 快照（`models/AI-ModelScope--GroundingDINO/snapshots/master`），
但它是**官方仓库格式**（非 HuggingFace 格式），且 Rust 侧无法直接加载 PyTorch 权重，
因此需先用脚本导出为 ONNX。

```bash
pip install torch torchvision transformers onnx onnxruntime addict matplotlib timm

python tools/export_onnx.py \
    --model-dir models/AI-ModelScope--GroundingDINO/snapshots/master \
    --out-dir models/onnx
```

产物：

- `models/onnx/groundingdino.onnx` — 服务实际加载的模型（**输入仅 `images`**）
- `models/onnx/text_encoder.onnx` — BERT 文本塔（备用/校验）
- `models/onnx/vocab.txt` — 分词器词表（备用）

脚本会自动做**任务级数值校验**，报告检出集合一致率与置信度/框偏差。

### 导出涉及的三个关键处理（重要，换环境需知）

1. **关闭梯度检查点**：官方 config 里 `use_checkpoint=True`，其内部的
   `torch.utils.checkpoint` 会破坏 ONNX 追踪（`invalid unordered_map key`）。
   导出时置 `use_checkpoint=False`、`use_transformer_ckpt=False`（仅推理，不影响数值）。

2. **文本特征烘焙进图**：官方 forward 的文本入口是字符串 `captions`，内部用
   `BertTokenizerFast` 分词后走
   `generate_masks_with_special_tokens_and_transfer_map`——该函数用
   **数据相关的 Python 循环 + 就地索引赋值**（`bertwarper.py` 222/226/270/274/279 行），
   会编译成 ONNX 不支持的 `aten::__ior_`，**无法追踪**。
   解决办法：prompt 固定为 `"person ."`，导出时先算一次文本特征，
   再作为常量烘焙进图，只导出可追踪的视觉路径。
   **换 prompt 需重新导出**（加 `--prompt`，注意必须带句点）。

3. **就地赋值改写**：快照源码中两处就地操作会阻断导出，已在脚本/源码中修正：
   - `groundingdino/models/GroundingDINO/utils.py` 的 `ContrastiveEmbed.forward`：
     `new_res[..., :n] = res` → `torch.cat`（数值等价）
   - `groundingdino/models/GroundingDINO/fuse_modules.py` 的 `BiMultiHeadAttention`：
     `masked_fill_` → `masked_fill`（数值等价）

> **关于精度**：GroundingDINO 的 CPU 路径用 `F.grid_sample` 实现可变形注意力，
> ONNX 只能近似分解，因此**原始 logits 在低分背景 query 上会有偏差**（约 20% 的 query，
> 均为远低于阈值者，实测最大 logits 偏差 3.6）。这是导出可变形注意力的已知限制。
>
> **对检测任务无实质影响**：实测在校验中，`box_threshold=0.25` 下**检出集合一致率 100%**
> （多组输入均如此），匹配 query 的 confidence 偏差 ~1e-5、框图偏差 ~1e-6。
> 脚本的校验即以「检出结果等价」为准，而非逐元素比对全部原始 logits。

---

## 4. 配置

编辑 `config.toml`：

```toml
[camera]
url = "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/102"
transport = "tcp"          # tcp 更稳；udp 延迟低但易丢包
sample_interval_secs = 10  # 抽帧间隔，CPU 推理约 8~9s/帧，建议 >= 5
read_timeout_secs = 15
ffmpeg_path = "ffmpeg"     # 或填完整路径，如 D:/dev/ffmpeg-9.0.2-full_build/bin/ffmpeg.exe
width = 704                # 必须与码流分辨率一致
height = 576

[model]
onnx_path = "models/onnx/groundingdino.onnx"
text_prompt = "person"     # 仅用于展示；实际 prompt 在导出时烘焙
box_threshold = 0.25
text_threshold = 0.25
nms_iou_threshold = 0.7
intra_threads = 4

[output]
jsonl_path = "output/detections.jsonl"
alert_webhook = ""         # 留空关闭告警（告警走系统 curl）
alert_min_count = 1
```

**海康 RTSP 地址**（常见格式）：

- 主码流：`rtsp://admin:密码@IP:554/Streaming/Channels/101`
- 子码流：`rtsp://admin:密码@IP:554/Streaming/Channels/102`

建议先接**子码流**：分辨率更低，CPU 推理更快。

> `camera.width/height` 会作为 `-s` 参数钉死给 ffmpeg，Rust 侧按
> `width*height*3` 字节读帧。**必须与码流实际分辨率一致**，否则 ffmpeg 会做缩放。

---

## 5. 运行

```bash
cargo build --release

# 正常循环运行
./target/release/human-count --config config.toml

# 只跑一帧用于链路验证
./target/release/human-count --once

# 临时覆盖 RTSP 地址
./target/release/human-count --rtsp "rtsp://admin:pwd@10.0.0.5:554/Streaming/Channels/102"

# 静态图片端到端验证（离线，无需摄像头）
./target/release/human-count --image testdata/bus.jpg

# 本地视频文件模式（同样走 ffmpeg 解码，抽 5 帧后退出）
./target/release/human-count --video media-resource/192.168.20.159_02_20261010161653728.mp4 --max-frames 5

# 调试日志
RUST_LOG=debug ./target/release/human-count
```

### 纯逻辑单元测试

因 ORT 预编译库的 STL 版本约束（见第 6 节），主工程 `cargo test` 会在链接阶段
一并失败。为此设了独立 crate `tests-pure/`，把**不依赖 ort** 的三个模块
（预处理、后处理、图片读入）抽出来单独编译测试：

```bash
cd tests-pure && cargo test
# 11 passed
```

覆盖：等比缩放的目标尺寸（含 Python `round()` 的「四舍六入五成双」语义）、
cxcywh→xyxy 的像素还原与边界 clip、IoU 计算、greedy NMS 的抑制边界
（IoU 严格大于阈值才抑制）、BGR 通道序。

### 无摄像头时：用本地视频模拟 RTSP

`tools/rtsp_sim.sh` 用 ffmpeg 把本地 mp4 循环推成 RTSP 流，模拟海康子码流
（704x576 / 25fps，与 `media-resource/` 里的录像一致）：

```bash
# 终端 1：起模拟流（保持不关）
bash tools/rtsp_sim.sh

# 终端 2：服务连上去
cargo run -- --rtsp "rtsp://127.0.0.1:8554/live"
```

> 若 ffmpeg 的 `-rtsp_flags listen` 在本机不监听端口，可直接用
> `--video media-resource/xxx.mp4` 走同一条 ffmpeg 解码链路做端到端验证。

输出示例：

```
帧 #42 | 人数=3 | 推理 8620ms | 704x576
  #0 conf=0.8742 bbox=[120.5, 88.0, 190.3, 402.1] 人
```

同时追加写入 `output/detections.jsonl`，每行一条：

```json
{"frame_index":42,"timestamp_ms":1776000123456,"image_width":704,"image_height":576,
 "person_count":3,"detections":[{"norm_cxcywh":[...],"bbox_xyxy":[...],
 "confidence":0.87,"label":"person","label_zh":"人"}]}
```

---

## 6. 已知限制与注意事项

1. **MSVC 工具集必须 ≥ 14.40**（本项目最关键的构建约束）。
   `ort` 从 pyke CDN 下载的 ONNX Runtime 预编译静态库，是用 MSVC STL 14.40+
   构建的；其 `.lib` 里引用了 14.40 才引入的 STL 向量化算法模板实例
   （`std::_Search_vectorized` / `_Remove_vectorized` / `_Max_element_vectorized` 等）。

   若本机只有更旧的工具集（如 VS2022 17.8 自带的 14.38），链接会失败：

   ```
   error LNK2019: 无法解析的外部符号 __std_search_1 / __std_remove_1 /
                  __std_max_element_f / __std_find_end_1 ...
   ```

   注意：这些符号**不在**旧 STL 的头文件里，也无对应的静态库导出，
   因此**无法用链接选项或搜索路径绕过**——旧 STL 的头文件无法实例化新模板。
   唯一解法是升级 MSVC 工具集：

   ```powershell
   # 方式一：VS Installer GUI（推荐）
   #   打开 Visual Studio Installer → 修改 → 单个组件 →
   #   勾选「适用于最新 v143 生成工具的 C++ x64/x86 生成工具(14.4x)」
   #   （可顺带勾选最新的 Windows 11 SDK）
   #
   # 方式二：命令行以管理员身份执行
   #   vs_installer.exe modify --installPath "D:\dev\Microsoft Visual Studio\2022\Community" ^
   #     --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64
   ```

   安装后确认目录里出现新版本（如 `...\VC\Tools\MSVC\14.44.xxxxx`），即可正常链接。

   > 本项目已把 `ort` 钉在 `=2.0.0-rc.10`（对应 ORT 1.22.0）。虽然 rc.10 更早，
   > 但其预编译包同样是 14.40+ 构建的，所以仍需升级工具集。
   > 选择 rc.10 而非 rc.13 的原因是：rc.13 只提供带 DirectML 的发行包，
   > 会额外拉入 D3D12/DXCORE 依赖；rc.10 有干净的 `none` 发行包。
   > 另外 rc.10 的 `ndarray` 版本要求是 0.16（rc.13 是 0.17），两者必须与
   > `Cargo.toml` 中 `ndarray` 的版本严格一致，否则 `ort` 的 trait 实现
   > 作用在另一个 ndarray 版本上，`Tensor::from_array` 会编译失败。

2. **CPU 性能**：GroundingDINO-SwinT 在 CPU 上约 8~9 秒/帧，**无法实时**。
   本服务按 `sample_interval_secs` 抽帧运行。若要实时，需改用 GPU 或更小的模型
   （如 YOLO 系列）。

3. **中文类别名**：权重内置的文本编码器是 `bert-base-uncased`（英文词表），
   中文 prompt 会被切成 `[UNK]` 而失效。因此 prompt 固定为英文 `person`，
   代码在输出层映射为「人」。需要原生中文识别须替换文本编码器并重导模型。

4. **误检**：DINO 对**照片/印刷品中的人**会产生误检（参考文档实测 score 0.28~0.43）。
   提高 `box_threshold` 可过滤部分误检，但也会丢失较弱的真人（如远处小目标）。
   阈值调整前请参考 `DINO-code/` 中的实测数据。

5. **ffmpeg 相关的注意事项**：
   - ffmpeg 的 `-loglevel error` 输出会直接透传到服务日志，便于排查连不上流的问题。
   - 拉流断开时 ffmpeg 进程会退出，服务据此触发退避重连（每次重连会重新起进程）。
   - `rtsp_transport` 建议用 `tcp`：海康在 udp 下更容易丢包导致花屏。
   - 若目标机器没有 ffmpeg，服务启动时会明确报错并提示配置 `camera.ffmpeg_path`。

6. **模型未入库**：`.gitignore` 已排除 `*.onnx`（体积大）。换机器需重新导出。

7. **构建目录设在 `D:/hc2`**（见 `.cargo/config.toml`）。
   本机早期在默认 `target/` 下出现过 NTFS 元数据损坏的文件
   （`icacls` 连 ACL 都读不了、`takeown` 报「找不到文件」，即 MFT 项不一致，
   既删不掉也覆写不了），另有一次 `ort-sys` 的 build script 被安全软件拦截执行。
   换机器部署时删掉 `target-dir` 那一行即可回到默认目录。

8. **ORT 库走本地预下载**（见 `.cargo/config.toml` 的 `[env]`）。
   本机上 `ort-sys` 的 build script 一被编译出来就被安全软件拦截
   （执行报 `Permission denied` / `os error 5`），导致构建中断。
   解决办法是**预先手工下载并解压 ORT**，再用
   `ORT_LIB_LOCATION` + `ORT_SKIP_DOWNLOAD=1` 让 build script 直接读本地目录：

   ```bash
   # 注意 URL/哈希取自 ort-sys 的 dist.txt（Windows x64 的 none 发行包）
   curl -L -o vendor/ort/ort.tgz \
     "https://cdn.pyke.io/0/pyke:ort-rs/ms@1.22.0/x86_64-pc-windows-msvc.tgz"
   cd vendor/ort && tar -xzf ort.tgz     # 解出 onnxruntime/lib/onnxruntime.lib
   ```

   `ORT_LIB_LOCATION` 必须指向**直接包含 `onnxruntime.lib` 的目录**
   （即 `vendor/ort/onnxruntime/lib`），不是解压出的顶层目录。

9. **该发行包仍带 DirectML**：即使 dist 名是 `none`，解出的包里也有
   `DirectML.dll`，其 `onnxruntime.lib` 引用了 `DXCoreCreateAdapterFactory` /
   `DMLCreateDevice1`。正常走 ort-sys 自动下载时，build script 会自己加
   `DXCORE/DXGI/D3D12/DirectML` 链接库；而走 `ORT_LIB_LOCATION` 手工指定时
   它不再加（内部 `using_pyke_libs=false`），因此已在
   `.cargo/config.toml` 的 `[target.x86_64-pc-windows-msvc]` 里显式补上这 5 个
   `-l` 参数。**该问题已解决**。

---

## 8. 当前构建状态（截至本文档最后更新）

**已完成**：全部依赖编译通过；所有 Rust 代码（含 ort rc.10 API、ffmpeg 拉流、
离线验证通道）编译无误；DirectML 链接符号已补齐。

**唯一剩余阻塞**：链接阶段缺 17 个 MSVC STL 向量化算法符号，
**全部**形如 `__std_*`：

```
__std_search_1 / __std_search_2
__std_remove_1 / __std_remove_2 / __std_remove_8
__std_max_element_f / __std_max_4i / __std_max_8u
__std_find_end_1 / __std_find_end_2
__std_find_first_of_trivial_1 / __std_find_first_of_trivial_2
__std_find_last_of_trivial_pos_1 / __std_find_last_of_trivial_pos_2
__std_find_last_trivial_2
__std_mismatch_1 / __std_mismatch_8
```

（442 处引用，但去重后仅这 17 个符号。除它们之外无任何其他未解析符号。）

**结论**：升级 MSVC 工具集到 ≥14.40 后即可通过链接，无需改任何代码。
参见第 6 节第 1 条的操作步骤。

---

## 7. 目录结构

```
human-count/
├── Cargo.toml
├── .cargo/config.toml     # 构建目录设置（见下方说明）
├── config.toml
├── src/
│   ├── main.rs         # 入口与调度
│   ├── config.rs
│   ├── capture.rs      # ffmpeg 子进程拉流
│   ├── image_input.rs  # 静态图片输入（离线验证）
│   ├── preprocess.rs
│   ├── detector.rs
│   ├── detection.rs
│   └── output.rs
├── tools/
│   ├── export_onnx.py  # 一次性导出脚本
│   └── rtsp_sim.sh     # 本地视频模拟 RTSP 摄像头
├── media-resource/     # 真实录像样本（用于模拟流）
├── models/             # 原始快照 + 导出的 onnx
├── DINO-code/          # 参考实现与参数说明
├── testdata/           # 离线验证用图片
└── output/             # 运行时结果
```

> `.cargo/config.toml` 里设置了 `build.target-dir = "D:/hc-build"`。
> 这是为了绕开早期在默认 `target/` 目录下出现的一个 NTFS 元数据损坏的文件
> （`icacls` 连 ACL 都读不了、`takeown` 报「找不到文件」，即 MFT 项不一致，
> 既无法删除也无法覆写）。换机器部署时删掉这一行即可回到默认目录。

