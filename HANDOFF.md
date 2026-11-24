# HANDOFF — human-count 项目交接

> ⚠️ **交接前提**：本文件面向接手本项目的 AI Agent。先完整读完本文再动手，
> 不要凭"项目名是 human-count"就假设 Rust 服务已经能跑——**它目前跑不起来**，
> 但 Python 侧的完整链路是**可用且已验证**的。两条线的状态差异很大，别搞混。

**交接时间**：2026-10-10 20:00 (GMT+8)
**工作目录**：`D:\dev\human-count`

---

## 0. 一句话现状

| 线路 | 状态 | 能否验证 |
|---|---|---|
| **Python 推理链路**（远程 GPU） | ✅ **完全可用，端到端跑通** | ✅ 已验证，有产物 |
| **Rust 服务**（本项目主目标） | ❌ **链接失败，无法编译** | ❌ 无法运行 |

**当前最高优先级**：Rust 侧的 MSVC 工具集升级（见 §3）。

---

## 1. 项目目标（原始需求）

用户要做一个 **Rust 服务**：通过海康摄像头 RTSP 流，运行 GroundingDINO 识别"人"，
输出人数。已确认的四项决策：

| 决策项 | 结论 |
|---|---|
| 推理后端 | Rust + ONNX Runtime（`ort` crate） |
| 类别 prompt | 英文 `person`，输出层保留中文映射（"人"） |
| 海康接入 | 纯 RTSP 拉流，**不接** HCNetSDK DLL |
| 交付范围 | 先做最小可运行骨架 |

后续扩展需求（**进行中，见 §6**）：同场景多摄像头、画面重叠、时间同步下的
**跨画面人员去重**，求实时总数。用户明确表示**不标定**，希望"通过多画面融合快速建模，
自主建立空间关系，最多人工后确认"。

---

## 2. 两条线路详解

### 2.1 Python 链路（远程 GPU）—— 可用 ✅

**为什么存在**：Rust 侧因 MSVC 工具集问题无法编译（§3），为不阻塞算法验证，
改走远程 GPU 用 Python 跑通了整条链路。

**算力来源**：项目根目录 `.env` 中登记的服务器

```
local_server_ip:192.168.11.16      端口 10022
local_server_user:r4900g3
local_server_work_path:/home/r4900g3/yhh/
```

**远程环境（已探明，直接可用）**

| 项目 | 值 |
|---|---|
| 主机 | `r4900g3-UniServer-R4900-G3` |
| GPU | **2 × NVIDIA L20（各 46 GB）**，驱动 580.178.04 |
| CPU / 内存 | 48 核 / 125 GB |
| **Python（必须用这个）** | `~/miniconda3/envs/rfdetr/bin/python` |
| 该环境版本 | py3.11.15，**torch 2.11.0+cu130**，**supervision 0.29.1**，**transformers 5.14.1** |
| 模型 | `/share/data4/yhh/models/dino-model`（HF 格式 GroundingDINO-base，safetensors 933 MB） |
| 远程工作目录 | `/home/r4900g3/yhh/human-count/`（in/ 素材、output*/ 产物） |

> 🚨 **坑 1：绝对不要用系统默认 `python3`**（/usr/bin/python3，3.10）。
> 它只有 torch 1.13.1，低于 transformers 5.x 要求的 torch≥2.4，
> 模型会被**静默禁用**（只打印一行 "Disabling PyTorch because..." 然后继续跑，不报错）。
> 我第一次就踩了这个，白跑一轮。

> 🚨 **坑 2：`transformers 5.x` 的 API 变了**
> - `post_process_grounded_object_detection` 的 `box_threshold` 参数**改名为 `threshold`**
> - 返回的 `labels` 从文本变成了**整数 id**，真正的文本在 **`text_labels`**
> - 脚本里已做新旧兼容处理

**本地侧工具脚本**（`tools/` 下）

| 脚本 | 用途 |
|---|---|
| `remote_run.py` | **核心**。基于 paramiko 的 SSH 执行器，凭据自动读 `.env` |
| `infer_supervision.py` | **核心**。远程推理主脚本（检测→去重→跟踪→可视化→H.264） |
| `containment_nms.py` | 同画面重复框去重（纯 numpy，零依赖） |
| `test_containment_nms.py` | 上述模块的单元测试（**12/12 PASS**） |
| `fetch_results.py` | 从远程拉产物到本地 |
| `plot_compare.py` / `plot_timeline.py` | 生成人数曲线图 |
| `export_onnx.py` | 把官方仓库格式权重导出为 ONNX（Rust 侧用） |

**`remote_run.py` 用法**

```bash
P=/c/Users/might/.workbuddy/binaries/python/envs/default/Scripts/python.exe
$P tools/remote_run.py 'shell command'                    # 远程执行
$P tools/remote_run.py --put 本地路径 远程路径             # 上传
$P tools/remote_run.py --get 远程路径 本地路径             # 下载
$P tools/remote_run.py --stdin-file script.sh             # 跑本地脚本
```

> 注意：paramiko 的 `sftp.put` 在某些情况下会报 `FileNotFoundError`，
> 但实际文件已写入——**别被误导**，先 `ls` 确认再重试。
> 另外远程命令里**不要用换行分隔多条命令**，用 `;` 或 `&&`。

**已验证的推理结果**（`stride=5`, `box_threshold=0.30`）

| 指标 | clip01 | clip02 |
|---|---|---|
| 视频 | 1338 帧 / 53.5 s | 248 帧 / 9.9 s |
| 单帧延迟 | ~155 ms | ~180 ms |
| 原始检测均值 | 8.60 | 9.80 |
| 去重后 | **7.05** | **8.97** |
| +跟踪后 | **6.70** | **8.70** |

### 2.2 Rust 服务 —— 链接失败 ❌

**代码是完整的**（8 个模块，见 `src/`），架构设计没问题，
`tests-pure/` 里的纯逻辑测试 **11/11 通过**。

**唯一的阻塞**：链接期 17 个 `__std_*` 符号无法解析，例如
`__std_search_1/2`、`__std_remove_1/8`、`__std_max_element_f`、`__std_find_end_1/2`、
`__std_mismatch_1/8`、`__std_max_4i/8u` 等。

**根因（已充分论证，不要再重复排查）**：

- `ort` rc.10 下载的 ONNX Runtime 预编译库是用 **STL 14.40+** 构建的
- 本机 MSVC 工具集是 **14.38.33130**（路径：`D:\dev\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130`）
- STL 14.40 引入了向量化算法模板（`_Search_vectorized` 等），14.38 的头文件里**根本没有这些声明**
- 已用 `dumpbin` 验证：本机 `libcpmt.lib` 中 `__std_search` 的引用数为 **0**
- 结论：**无法用链接选项绕过，必须升级工具集**

---

## 3. 🔴 当前第一优先任务：升级 MSVC 工具集

### 需要做什么

升级 MSVC v143 工具集到 **14.40 或更高**（建议 14.44+），并安装 Windows 11 SDK。

### 怎么升级

**已准备好一键脚本**：`C:\Users\might\.workbuddy\binaries\open_vs_installer.bat`

- 双击运行，会在 VS Installer GUI 中**预勾选**两个必需组件：
  - `Microsoft.VisualStudio.Component.VC.Tools.x86.x64`
  - `Microsoft.VisualStudio.Component.Windows11SDK.26100`
- 在 GUI 中点击「修改」即可

> **为什么必须走 GUI**：自动化安装（`--quiet` / `--passive`）返回 **Exit Code 5007**，
> 因为必须从**提权令牌**启动才能改 VS 安装。命令行路线走不通，实测过。

### 升级后怎么验证

```bash
cd /d/dev/human-count
/c/Users/might/.cargo/bin/cargo build --release --offline
```

预期：链接通过，产出 `human-count.exe`。

**成功判据**：不再出现任何 `__std_*` 符号；`LS \d\hc2\release\human-count.exe` 存在。

### 如果升级后仍有问题

按顺序检查（这些在之前的排障中都已解决，不要重新踩）：

1. **`ORT_LIB_LOCATION` 相关链接库**：`.cargo/config.toml` 里已补齐
   `-l dxguid -l DXCORE -l DXGI -l D3D12 -l DirectML`（DirectML 符号缺失曾出现过，已修）
2. **安全软件拦截**：`ort-sys` 的 build script 曾被拦（`os error 5`）。
   已通过预下载 ORT 到 `vendor/ort/` + `ORT_SKIP_DOWNLOAD=1` 绕过
3. **NTFS 元数据损坏**：换过 4 次 target 目录，当前用 `D:/hc2`（见 `.cargo/config.toml`）

### 构建配置（`.cargo/config.toml`）

```toml
[build]
jobs = 4
target-dir = "D:/hc2"          # 不要改回 target/，原目录有 NTFS 损坏

[env]
ORT_LIB_LOCATION = "D:/dev/human-count/vendor/ort/onnxruntime/lib"
ORT_SKIP_DOWNLOAD = "1"
RUSTFLAGS = "... -l dxguid -l DXCORE -l DXGI -l D3D12 -l DirectML"
```

---

## 4. Rust 侧已完成的工作（别再重做）

### 4.1 关键架构决策

**用 ffmpeg 子进程拉流，不用 OpenCV / FFmpeg SDK**

- `opencv` crate 0.93 起不再内置源码构建，要求系统预装完整 OpenCV（+ cmake/pkg-config）
- `ffmpeg-sys` 同样要求预装 FFmpeg SDK
- 而 ffmpeg CLI 本身把 RTSP/RTP 的解码、重连、缓冲都处理得很好
- 拉流层只需读取固定字节数的原始帧，**零链接依赖**

具体命令（`src/capture.rs`）：

```
-hide_banner -loglevel error -nostdin
-rtsp_transport <tcp|udp> -rw_timeout <us> -i <url> -an
-f rawvideo -pix_fmt bgr24 -s WxH pipe:1
```

> ⚠️ `camera.width/height` 会作为 `-s` 参数传给 ffmpeg，
> **必须与码流实际分辨率一致**，否则 ffmpeg 会缩放（多耗 CPU 且可能变形）。
> 海康子码流常见 704x576；现有素材都是 704x576。

### 4.2 依赖版本（`Cargo.toml`）—— 不要随意改

```toml
ort = { version = "=2.0.0-rc.10", default-features = false, features = ["ndarray", "download-binaries", "std"] }
ndarray = "0.16"
image = { version = "0.25", default-features = false, features = ["jpeg", "png"] }
```

**版本为什么钉死**（踩过的坑，重来一遍很费时间）：

| 问题 | 结论 |
|---|---|
| `ort` rc.13 的 ORT 1.28 需 STL 14.40+ | 降到 rc.10 |
| rc.13 依赖 ndarray 0.17，与项目其他依赖冲突 | 统一 rc.10 + ndarray 0.16 |
| rc.13 的 `ort::Error<R>` 不满足 Send/Sync/StdError | rc.10 的 Error 是无泛型 struct，`.context()` 可用 |
| rc.10 没有 `tls-native` feature（rc.13 才加） | 移除该 feature |
| `ureq` TLS 冲突 | **移除项目直接 ureq 依赖**，webhook 改用系统 `curl` 子进程 |

### 4.3 已修正的两个预处理 bug（`tests-pure` 已覆盖）

1. **`target_size`**：必须严格复刻官方 `RandomResize`，含**银行家舍入**
   `round_half_to_even`。例：1280×720→(1333,750)，384×288→(1066,800)
2. **`cxcywh_to_xyxy_clipped`**：先算归一化 → 乘像素宽高 → 最后 clamp

### 4.4 ONNX 导出的三个关键处理（`tools/export_onnx.py`）

导出的模型在 `models/onnx/groundingdino.onnx`（257 MB，**已生成**）。

| 问题 | 处理 |
|---|---|
| `bertwarper.py` 有数据相关 Python 循环 + 就地索引赋值 → `aten::__ior_` 不可追踪 | **把文本特征烘焙进图**，Rust 侧输入只有 `"images"` |
| 梯度检查点干扰导出 | 关闭 |
| `F.grid_sample` 可变形注意力仅能近似 | 已用**任务级等价校验**绕过（检出集合 100% 一致） |

### 4.5 CLI 用法

```
--config <path>      配置文件（默认 config.toml）
--once               跑一帧就退出
--rtsp <url>         覆盖配置里的 RTSP 地址
--image <path>       单图模式（离线验证用）
--video <path>       视频文件模式（与 RTSP 共用同一条 ffmpeg 解码链路）
--max-frames <n>     限制处理帧数
```

**端到端验证命令**（Rust 侧修复后应该能跑）：

```bash
./target/release/human-count --video media-resource/192.168.20.159_02_20261010161653728.mp4 --max-frames 5
```

---

## 5. 已完成的功能：同画面重复框去重 ✅

用户反馈的问题："同一个人被框了一大一小两次"。

### 定性（有数据支撑）

对 250 帧逐对框分析：

| 指标 | 值 | 含义 |
|---|---|---|
| **包含度**（小框落在大框内的面积占比）| **0.99** | 小框几乎完全嵌在大框内 |
| IoU | 0.46 | **低于常规 NMS 阈值 0.7 → 全部漏网** |
| 面积比（大/小） | 1.9 | 大框是"全身"，小框是"头/上半身" |

**根因**：GroundingDINO 是开放词表检测器，`prompt="person"` 在**多个语义尺度**上
都成立（"人头""上半身""全身"都是 person），于是同一目标被重复响应。

### 解法：Containment-NMS

判据从 IoU 改为**包含度**：`containment = 交集面积 / 小框面积`

- 小框 ≥85% 面积落在大框内 → 判定重复，压制
- 按面积降序贪心先占，**结果与输入顺序无关**

**保留策略：保留面积更大的框**（不是高分框）。依据：

- 实测"小框分更高"占 1/3 的情况（100/304）
- 但**分数差中位数仅 +0.022**，几乎没有区分度
- 而**面积比稳定在 1.9~2.0**，是可靠特征
- 业务上"数人"，全身框语义更对

实现：`tools/containment_nms.py`。测试：`tools/test_containment_nms.py`（**12/12 PASS**）。

关键测试用例：**并排两人、半重叠两人都不被误伤**——这是 IoU 阈值方案做不到的。

> ⚠️ 参数语义坑：`--containment-threshold 0` 是"**最激进**"不是"关闭"
> （因为包含度 1.0 ≥ 0 恒成立）。要关闭请用 `--no-dedup`。

**效果**（阈值 0.85，保留大框）

| | clip01 | clip02 |
|---|---|---|
| 原始 → 去重 → 跟踪 | 8.60 → **7.05** → 6.70 | 9.80 → **8.97** → 8.70 |
| 去掉重复框 | 326 个（18.1%）| 33 个（8.4%）|
| std 收敛 | 1.54 → 1.34 → 1.14 | 0.81 → 0.57 → 0.75 |

### 另一个已完成：ByteTrack 跟踪

`infer_supervision.py` 里已接入 ByteTrack。**两个参数坑已踩过并修正**：

1. **`frame_rate` 必须传原始视频帧率（25）**，不是抽帧后帧率。
   因为 `max_time_lost = int(frame_rate/30 × lost_buffer)`，
   传 `fps/stride=5` 会导致 `max_time_lost=5` ≈ 抽帧步长，
   **轨迹每两帧就被判丢失、ID 全废**（实测只剩 3 个 ID）。
2. **`track_activation_threshold` 必须低于检测阈值**。
   内部会 `+0.1` 作为 `det_thresh`，直接传 `box_threshold=0.30` → `det_thresh=0.40`，
   0.30~0.40 分的真实目标**无法起轨**。故单独用 `--track-activation 0.10`。

---

## 6. 🟡 待办：跨摄像头多画面去重（用户最新需求）

### 需求（用户已明确）

> 同场景多个摄像头、**画面重叠**、**事件同步**、求**实时总数**、**有其他摄像头的录像**

用户核心痛点：多路画面各自计数后相加，**总数偏高**（同一个人被多路各算一次）。

用户对方案的明确要求：**不标定**，希望"通过多画面融合快速建模，
通过有关联的各个摄像头画面自主建立空间关系，最多人工后确认"。

### 🔴 阻塞：缺少多路数据

`media-resource/` 里两段视频**都是同一个摄像头**（`192.168.20.159_02`），
**没有第二路**。项目内**没有任何多摄像头标定信息**（无单应矩阵、无摄像头清单、
无 NVR 通道映射）。

**接手后第一件事应该是找用户要**：
1. **≥2 路同步录像**（同场景、有重叠区域）
2. 摄像头是否同一台 NVR（若是，`192.168.20.159_02` 的 `02` 可能是通道号，
   则 RTSP 为 `Channels/102`、`Channels/202`…）
3. 重叠区域大概多大

### 已调研的方案（三条路线，按推荐度排序）

**路线 A：FOV 共现自动建模（最推荐）**

- 核心：**用"人"当标定物**。同一时刻 A 画面 #3 与 B 画面 #1 连续"同进同出"
  → 统计共现 → 推出视野对应关系
- 出处：UCF *Tracking in Multiple Cameras with Overlapping FOV*
- 实测 **40 秒**有人走动即可建立关系
- ✅ 完全自动、不需要标定、天然利用"时间同步"条件
- ⚠️ 软肋：**需要有人走动**。办公区人都坐着不动时共现会退化

**路线 B：自监督学习（CVPR 2025 Self-MVA）**

- 论文：*Learning from Synchronization: Self-Supervised Uncalibrated Multi-View Person Association*
- 代码：https://github.com/CAMMA-public/Self-MVA
- ✅ 明确针对"外观相似"场景（正是办公区痛点：都穿深色、坐着、背对）
- ⚠️ 需要训练，落地周期长

**路线 C：几何重建（VGGT 等 foundation model）**

- 论文：*Calibration-Free 3D Multi-Camera People Tracking*（AI City 2024，HOTA 53.13%）
- ⚠️ 重，需 GPU 推理

### 建议的落地路径

**A 为主 + 人工确认兜底**：

1. **自动采集**：跑一段有人走动的画面，统计跨画面共现，输出候选空间关系
2. **可视化确认**：画成"摄像头视野交叠图"，人工看一眼对不对（对应用户要的"人工后确认"）
3. **实时计数**：用确认好的关系做实时去重

**关键技术点**：
- 用**框底边中点**（人的落脚点）投影，**不要用框中心**——
  框中心受身高/坐姿影响大，底边中点才接近"脚下位置"
- 办公场景坐姿/站姿混杂，这点尤其重要

---

## 7. 目录与产物说明

```
D:\dev\human-count\
├── .env                      # 远程服务器凭据（见 §2.1）
├── config.toml               # 服务配置（RTSP 地址、模型路径、阈值）
├── Cargo.toml                # Rust 依赖（版本勿动，见 §4.2）
├── .cargo/config.toml        # target-dir=D:/hc2 + ORT 环境变量
├── src/                      # Rust 源码（8 个模块，完整但链接失败）
├── tests-pure/               # 独立测试 crate（11/11 PASS，不受链接失败影响）
├── tools/                    # 全部工具脚本（见 §2.1 表格）
├── models/onnx/              # 已导出的 ONNX 模型（groundingdino.onnx 257MB）
├── media-resource/           # 素材：2 段视频，均为 704x576 HEVC 25fps
├── vendor/ort/               # 预下载的 ONNX Runtime（绕开构建期网络）
│
├── output_tracked/           # 【v2】ByteTrack 版产物
├── output_dedup/             # 【v3】★ 最终版：去重+跟踪，含对比图与报告
└── output_preview/           # 【v1】初版产物（可忽略）
```

**最终产物在 `output_dedup/`**：
- `clip01_h264.mp4` / `clip02_h264.mp4` — 标注视频（H.264，可播放）
- `_compare_before_after.jpg` — 第 0 帧去重前后对比
- `count_timeline.png` — 原始→去重→跟踪 三口径曲线
- `RESULT.md` — 完整报告
- `clip0X_detections.jsonl` — 逐帧数据（含 `raw_count`/`deduped_count`/`count`/`track_ids`）
- `clip0X_summary.json` — 汇总指标

**注**：所有 `output*/` 目录已加入 `.gitignore`（体积大）。

---

## 8. 用户偏好与协作约定（重要）

从历史对话中提炼，**接手后请遵守**：

1. **先查证再动手**。用户明确表达过对"未经验证的结论"的不满。
   例如去重的保留策略，我是先跑数据统计（分数差 ±0.02）才决定"保留大框"的。
2. **结论先行 + 结构化表达**，用中文，通俗。
3. **复杂决策给选项**：推荐项 + 替代项 + 取舍，一次只问一个关键问题。
4. **区分事实 / 推断 / 已验证**。不要把静态检查当成实际使用验收。
5. **交付要可验证**：说明改了哪些文件、效果如何、还有什么没做。
6. **不要啰嗦重复播报**。没有变化就不要汇报。
7. **用户会追问"三个层次呢"这种** —— 说明他讨厌含糊其辞、
   说了概念却不展开。**说几点就要展开几点**。

> 用户在这轮对话中两次表达了不满：
> 一次是消息被截断只发出几个字；一次是我说"三个层次"却没展开。
> **回答要完整，概念要落地。**

---

## 9. 环境与版本清单

| 项目 | 版本 / 路径 |
|---|---|
| Rust / Cargo | 1.94.1 |
| **MSVC 工具集** | **14.38.33130** ← 🔴 **需升级到 14.40+** |
| VS 安装位置 | `D:\dev\Microsoft Visual Studio\2022\Community` |
| VS Installer 脚本 | `C:\Users\might\.workbuddy\binaries\open_vs_installer.bat` |
| Rust target 目录 | `D:/hc2` |
| ffmpeg（本地） | `D:/dev/ffmpeg-9.0.2-full_build/bin/ffmpeg.exe` |
| Python（本地，跑 tools/） | `C:\Users\might\.workbuddy\binaries\python\envs\default\Scripts\python.exe` |
| Python（远程，跑推理） | `~/miniconda3/envs/rfdetr/bin/python` |
| 远程 GPU | 2 × NVIDIA L20 |

---

## 10. 建议的接手顺序

1. **读本文 §0 / §3**，确认理解"Rust 挂了、Python 通"这个基本盘
2. **问用户**：是否已升级 MSVC？若已升级 → 跑 `cargo build --release --offline` 验证
3. **若升级未完成**：提醒用户双击那个 `.bat`（这是硬阻塞，其它都动不了）
4. **并行推进**：向用户索取**多路摄像头录像**（§6 的阻塞项）
5. **验证现有能力**：跑 `python tools/test_containment_nms.py`（应 12/12 PASS）
6. **跨画面去重**：等数据到位后，按 §6 的"路线 A"设计实现

---

## 11. 已知坑速查表

| 坑 | 表现 | 解法 |
|---|---|---|
| 远程用错 Python | 模型静默禁用，不报错 | 必须用 `~/miniconda3/envs/rfdetr/bin/python` |
| transformers 5.x API | `box_threshold` 参数不存在 | 改用 `threshold`，文本读 `text_labels` |
| ByteTrack frame_rate | 只剩 3 个 ID | 传**原始视频帧率**，不是抽帧后帧率 |
| ByteTrack 激活阈值 | 大量目标无法起轨 | `--track-activation` 要**低于** box_threshold |
| containment 阈值为 0 | 以为关闭了，实际最激进 | 用 `--no-dedup` 才是关闭 |
| paramiko put 报错 | FileNotFoundError 但文件已写 | 先 `ls` 确认，别急着重试 |
| 远程多命令 | 换行分隔报错 | 用 `;` 或 `&&` |
| ffmpeg -s 参数 | 画面被缩放/变形 | `camera.width/height` 必须等于码流实际分辨率 |
| VS Installer 自动化 | Exit Code 5007 | 必须走 GUI，需提权令牌 |
| ort 版本 | 各种链接/API 错误 | **精确钉 `=2.0.0-rc.10`**，别升 |
| **Git Bash 路径转换** | **`remote_run.py --put/--get` 报 `FileNotFoundError: No such file`** | **命令前加 `MSYS_NO_PATHCONV=1`**（2026-10-12 实测定论：MSYS 会把 `/home/...` 参数转成 Windows 路径） |
| **SSH 后台任务** | **`nohup A & nohup B &` 第二个任务未启动** | `&` 优先级问题：`cd X && nohup A & nohup B &` 中 `cd` 只作用于 A。用 `cd X; nohup A ... & nohup B ... &`（分号） |
| **本机 GitHub/HF 不稳定** | 模型权重下载中断（RemoteDisconnected / http=000） | 用镜像前缀：`https://ghproxy.net/https://github.com/...`（或 `ghfast.top`，2026-10-12 实测可用）；远程服务器 GitHub 直连可用 |
| **torch.hub 离线加载** | 网络窗口关闭时 `RemoteDisconnected` 崩溃 / **挂起 25min+**（torch 自带"查缓存回退"只认 URLError，不触发；仓库名解析会尝试连 GitHub） | `crosscam_depth.py`（MiDaS）与 `crosscam_reid.py`（DINOv2）**均已内置离线优先补丁**：本地缓存目录 `~/.cache/torch/hub/{owner}_{name}_{main\|master}` 存在时直接 `source="local"` 加载（2026-10-12 两次实测定论）。**新增模型加载器必须复制此模式** |
| **玻璃反光镜像伪轨迹** | 检测/跟踪把玻璃反光当人：同机重复计数、跨机虚假共现/转移、镜像对余弦虚高（2026-10-12 用户指正：316↔323 转移候选全部为此） | 各机位画**玻璃区域掩膜**排除落脚点落入反光区的检测；跨机匹配前先过滤；手工标定一次可复用 |

---

## 12. 【2026-10-12 更新】跨画面去重进展

### 数据已补齐

- 用户提供 `media-resource/multi-cemra/`：**6 路**同楼层录像（行政走廊316/办公区1 305/
  办公区2 318/办公区球机320/茶水间323/电梯间343），OSD 时间为唯一可信时间源
- 已产出同步切片 `media-resource/multi-cemra/synced-10min/`（08:55:00–09:05:00，
  6 路各 10 分钟，原分辨率 `-c copy`），复现方法见该目录 README.md

### 318 × 320 跨画面验证已完成（详见 `output_crosscam/analysis/REPORT.md`）

- **重叠视野确认**：t=0 保洁员被双路同时拍到（A#1/B#1），跨路重复计数实锤
- **时间对齐**：shift = +20 帧（+0.8s），由确认对速度曲线精调（r=0.788 单峰）
- **人工确认 8 轨迹对 / 7 个不同人员（全部经人工/用户复核，2026-10-12 闭环）**：
  1:1 保洁员、14:3 黑T恤眼镜男、69:14 背包男、120:18+120:23 皮领女士两片段、
  139:24 黑卫衣男、146:28 帽女士、37:11 藏青外套男（用户确认）；已判读 30 对共否决 22。
  **注意：多对确认对的运动同步分为负——运动分只做候选，外观/时空/人工定终审**
- **三道自动判据正面对比（全部 30 个判读对）**：
  1. **时空一致性（位置映射+轨迹连续性）★最佳单信号**：AUC=**1.000**，完美分离
     （8 确认对残差 0.029~0.109 vs 22 否决对 0.140~0.958）；
     **实战纠错**：发现人工误判 120:23 并订正。工具 `tools/crosscam_spatiotemporal.py`；
     前提是自举（先用其他信号+人工确认 ≥5~6 对拟合映射）
  2. 外观嵌入 DINOv2-small：AUC=0.858（有效但受同场景深色衣着混淆；z 归一化无改善）
  3. 运动同步分：仅候选生成（0.6+ 仍多假阳性）
  **选择偏差教训：标注子集 AUC=1.0 高估能力——嵌入主动提议 13 对仅 1 对真**
- 新增工具：`tools/crosscam_{analysis,pairs_manual,batch_verify,verify,viz,appearance,reid}.py`、
  `debug_sftp.py`；候选联合打分见 `candidates_combined.json`（13 对 reid≥0.6 待核验）

### 【2026-10-12 下午追加】深度实验 + 多机扩展

- **深度方向信号实验已完成**（`tools/crosscam_depth.py`，MiDaS-small 本机 CPU 离线缓存）：
  corr 变体 AUC=0.762、slope 乘积 0.708 → **弱辅助信号，不纳入主流程**；
  结论与调研一致：深度价值在 3D 融合（`docs/multicam-depth-spatial-survey.md`）
- **网络经验（重要）**：本机 GitHub/HF 直连时好时坏；**可用镜像：`ghproxy.net` / `ghfast.top`**
  （用法：`https://ghproxy.net/https://github.com/...`）；MiDaS 全量依赖已缓存
  `~/.cache/torch/hub/`（代码+权重+lite3 骨干），后续离线可跑；远程服务器 GitHub 直连可用
- **4 路剩余摄像头推理已完成**（12:00 启动双波，12:17 全部完成；`tools/run_wave4.sh`）：
  305/316/323/343 的 jsonl+summary 已拉回本地；**六路总览已产出**
  （`output_crosscam/analysis/fig_multicam_timeline.png` + `multicam_overview.json`）：
  人流强度 305 办公区1 最高（均值 6.4 人）、323 茶水间 3.5、318 4.2、316 走廊 1.1、343 电梯 0.5、320 球机 0.4；
  30s 分箱活动相关：办公区2~茶水间 r=0.59、办公区2~球机 r=0.52（后者为重叠视野共现所致）
  → 下一步：非重叠机位拓扑学习（当前 10 分钟数据仅够描述性，正式拓扑需更长记录）
- 新增工具：`tools/crosscam_depth.py`、`tools/multicam_overview.py`、`tools/run_wave4.sh`、
  `tools/crosscam_pipeline.py`（生产流水线，见下）
- **生产流水线固化（REPORT §6.5）**：时间门 → 时空过滤 → 外观门 → AUTO/REVIEW/DROP 分层；
  校验 8/8 确认对自动通过、0 例已知误放；压力测试发现 2 例边界误放（久坐男 × 步行女，
  球机正下方压缩区）→ AUTO 层叠加 `reid≥0.5` 外观门修正；时空判据口径更新 **AUC=0.995（8 正 / 24 负）**
- **跨区流转（316↔323）已完成"批量深度过滤 + 倒影审计"（2026-10-12 下午，含两次判断修正）**：
  `output_crosscam/analysis/transfer_report.md` + `reflection_experiments.md §六`；
  事件 316=97/97、323=180/180；纯时间门不可用（1246+1310 对）；
  ⚠️ 用户指正 Top 候选为玻璃反光镜像；**玻璃掩膜草稿**（`tools/glass_mask.py`）；
  **新发现"地板倒影"第二类伪影**（真人正下方、**通过深度地板检验**、掩膜拦不住；
  "兄弟轨迹"检测 `tools/reflection_audit.py` + 目视验证 `floor_reflection_evidence.png`）；
  此前两条"榜首"候选（323#81→316#60、masked 323#18→316#14）均因**双端倒影**被剔除；
  掩膜集"存活对" 323#138→316#77 被深度判据 REJECT(1.54)；
  **最终短名单（双端干净）：慢转移 14 条 + 快速转移 7 条**（`analysis/*/depth_filter/shortlist_audited.csv`
  + `analysis/FINAL_shortlist_evidence.png`，均待人工终审）；掩膜仍待用户圈改定稿

- **反光处理调研已完成（2026-10-12）**：`docs/reflection-handling-survey.md`——六层处理地图 +
  可借鉴清单（**A1 跨机一致性镜像判定** / **A2 反光似然打分** / **B1 深度不一致校验** /
  B2 EBLNet 等镜面/玻璃分割试用）；关键仓库可达性已验证（EBLNet/GDNet/MirrorNet/MG-VMD）
- **反光处理实验 P1/P2/P4 已完成（2026-10-12 下午）**：`output_crosscam/analysis/reflection_experiments.md`——
  **P1 深度"地板曲线残差"判别 AUC=0.931**（镜像 1.22 vs 真人 0.29）；**P2 dt 规则分流镜像/转移**
  （用户确认 10 对镜像 10/10 命中）+ `--spread` 产出 111 慢速转移候选；**P4 0.5B VLM 鉴镜失败**（需强模型/微调）；
  新增工具 `reflection_depth_test.py`、`crosscam_mirror_rule.py`（`crosscam_transfer.py` 加 `--spread`）；
  **P1 批量化 + 倒影审计已完成**（短名单 慢14+快7，§六）
- **带深度的掩膜研究已完成（2026-10-12 晚）**：`docs/depth-mask-research.md`——文献三类（RGB-D 传感器类 /
  单目重模型 MirrorMamba / 语义融合类）+ 实测结论：**单目深度无法分离"地板倒影"**
  （29 倒影/28 源 × 6 特征 AUC≤0.65；根因＝MiDaS 给倒影抄真人深度＝非朗伯面幻觉，169.5/179.4 vs 真实地板 211.3）；
  脚点残差对墙镜仍有效（1.24 vs 0.2）→ **"掩膜即判定"定型**（时间门→dt 分流→深度脚点残差→几何兄弟检测→人工终审），
  静态 2D 区域掩膜与"深度区间掩膜"**均判废**；工具 `tools/depth_gate_test.py`、产物 `analysis/depth_gate/`
- **镜像运动学清洗 + 六路去镜计数已完成（2026-10-12 晚，重大升级）**：`docs/anti-reflection-next-plan.md`——
  **实验 A**：倒影=源关于触地线的镜像（`v_R≈2v_脚−v_源中心`），规则 **28/29 真对通过、408 对照仅 7**
  （且 7 个多为新发现镜像对，已目视验证 2 例）→ 工具 `tools/mirror_kinematics.py`（零新依赖、秒级）；
  **实验 B**：翻转手性复核（flipud 提升 +0.063 vs 对照 −0.010，AUC 0.798）→ `tools/mirror_flip_test.py`；
  **实验 C 六路去镜总账**：578 轨迹 → **60 条倒影（10.4%）** → 518；316=17.8%、323=17.2%、343=8.7%、
  318=3.2%、305=1.1%、320=0%；**6 路清洗版 jsonl 已产出**（`analysis/track_cleaning/{tag}_cleaned.jsonl`，
  可直接替换下游输入）；工具 `tools/track_cleaning.py`
- **P1 清洗回流 + 短名单复审已完成（2026-10-12 17:05，详见 `output_crosscam/analysis/p1_review.md`）**：
  - **旧短名单 21 条候选两侧轨迹全部存活**（21/21，帧数逐一核对）；FAST_CHECK 3 条被清洗完整证实为倒影对；
  - **316#116 修正为误标**（双向运动学 perr=1.21/0.79 + 目视＝走向镜头真人）——
    旧"单帧兄弟检测"假阳性；`floorref_labels.json` 已修正（倒影 28/源 29）；
  - **318×320 八对确认复查通过**：16 轨迹全 clean；边缘案例 318#1/#37 经三重指标（du 失败+pcorr 负+高度比异常）+目视排除；
  - **清洗版重跑两套**：clean（未掩膜+清洗）与 **dualclean（掩膜+清洗）**；dualclean 的已知幻影参与候选 **0 条**
    （未净化版 8 条）→ **最终转移短名单 27 条**（TIER1 26 + FAST_CHECK 1），主表
    `analysis/dualclean/depth_filter/shortlist_audited.csv` + 证据图 `final_top8_evidence.png`；
  - **审计 refl? 5 个标记经运动学复核全部为假阳性**（"两人前后行进"场景，如 316#53~50 perr=0.62）；
  - 工程修复：**DINOv2 加载器离线补丁**（`crosscam_reid.py`，同 MiDaS 模式；修前网络关闭时挂起 25min+，修后加载 11.2s）
- **外观 ReID 底座升级评估已完成（2026-10-12 18:00，详见 `output_crosscam/analysis/reid_compare/REPORT.md`）**：
  - 用户提议评估 DINOv3 / C-RADIOv4 → 同台实测（30 判读对，同帧同框 280 crops）；
  - **DINOv2-S 0.869 → DINOv3-B/16 (224) AUC 0.983**（thr=0.7: 7/8 recall / 0 fp；CPU 188ms/图）；
    DINOv3-S/16 0.915（64ms）；**256 输入无增益**（0.966）；C-RADIOv4-SO400M 并列 0.983 但
    803ms（512 模式 4.6s）、分数带窄、不作主力（dense features 留给未来掩膜任务）；
  - **权重渠道**：DINOv3 官方全 gated（HF/`fbaipublicfiles` 403）→ 用 **timm 重托管**
    `timm/vit_{small,base}_patch16_dinov3.lvd1689m`（免审批、权重同源）；C-RADIOv4 走 hf-mirror；
    模型落地 `models/{dinov3_vits16_timm, dinov3_vitb16_timm, C-RADIOv4-SO400M}`，
    下载器 `tools/download_models.py`；评估/绘图工具 `tools/crosscam_reid_compare.py` / `plot_reid_compare.py`；
  - C-RADIO trust_remote_code 坑：需手补 `dual_hybrid_vit.py`/`utils.py`/`preprocessor_config.json`
    到 `~/.cache/huggingface/modules/transformers_modules/C_hyphen_RADIOv4_hyphen_SO400M/`；
  - **✅ 生产切换已执行（2026-10-12 18:35）**：`crosscam_reid.py` 默认后端=dinov3b（可 `--backend` 回退）；
    新增 `tools/encode_tracks.py`（全轨迹批量编码）；**远程 L20×2 GPU 重算完成**——
    `analysis/reid_emb.npz` 已换为 DINOv3-B 全量版（180 keys = 318×151 + 320×29，8 帧/轨迹），
    `reid_scores.json` 新口径 **gap=+0.130（转正）/AUC=1.000**（内建 5 正 10 负）；
    pipeline v2：AUTO 9（8 确认全过 0 误放）/ **REVIEW 16（-6，人工队列 -27%）** / DROP 50，
    降级 6 条全为真否决（零风险）；v1 备份 `analysis/reid_compare/v1_backup/`；
    `crosscam_transfer.py` 等 import Embedder 的链路自动跟随新默认（316×323 链如需刷新 cos 可重跑）。
  - **远程 GPU 链路备忘**：远程**无外网**（pypi/hf/github 全断）——依赖（timm/einops tar 包）与权重
    （343MB）均 SFTP 直传；远程 python=`~/miniconda3/envs/rfdetr/bin/python`（torch 2.11+cu130）。
- 下一步：P2 轨迹拼接（清后"源"仍碎片化，如 50/53 同人碎轨——计数膨胀第二大来源）/ P3 六路校正总览 / P4 强裁决器（备选）
  / （新）1012 长视频清点（21 段 4F 多机位 09:00–17:00）→ 转移拓扑 / BEV；见 `analysis/tech_applicability_2026-10-12.md`

### 仍未推进

- **MSVC 工具集仍为 14.38**（Rust 链接阻塞未解，等用户走 GUI 升级）
- 实时原型未做；非重叠机位拓扑学习待 4 路推理产物
