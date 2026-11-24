# human-count 数据目录

集中存放 human-count 项目的样本数据与推理产物。
**迁移日期**：2026-10-12（原位于 `/home/r4900g3/yhh/human-count/`，原路径已改为软链接指向此处）。

## 目录结构

| 目录 | 内容 | 大小 |
|---|---|---|
| `video-6way/` | 六路同步切片视频 305/316/318/320/323/343.mp4 | 547M |
| `inference-out/` | 六路推理产物：`{cam}_detections.jsonl` / `_summary.json` / `_annotated.mp4` / `_first_frame.jpg` | 1.1G |
| `crosscam-run/` | 跨机位分析工作区：`tools/`（分析脚本）、`models/dinov3_vitb16_timm/`（ReID 权重）、`out/`（轨迹编码产物） | 334M |
| `reflection-test/` | 反光/镜像判定测试集 | 2.5M |

## 视频规格（video-6way/）

- 时间窗：2026-10-10 08:55:00 – 09:05:00（OSD 为准），25 fps，约 596s / 14900 帧
- 分辨率：318=1920×1080；320=2560×1440；其余见推理 summary
- 点位：305 办公区1 / 316 行政走廊（鱼眼）/ 318 办公区2 / 320 办公区球机（鱼眼）/ 323 茶水间 / 343 电梯间

## 关键产物

- 六路检测：`inference-out/{cam}_detections.jsonl`（帧级：frame/boxes/scores/track_ids）
- DINOv3-B 轨迹 embedding（8 帧均值、L2 归一化）：
  - `crosscam-run/out/reid_emb_318_d3b.npz`（151 轨迹，key=`a{tid}`）
  - `crosscam-run/out/reid_emb_320_d3b.npz`（29 轨迹，key=`b{tid}`）
- ReID 权重：`crosscam-run/models/dinov3_vitb16_timm/model.safetensors`（343MB）

## 旧路径软链接（保持兼容）

```
/home/r4900g3/yhh/human-count/crosscam/in             -> video-6way/
/home/r4900g3/yhh/human-count/crosscam/out            -> inference-out/
/home/r4900g3/yhh/human-count/crosscam/reflection_test -> reflection-test/
/home/r4900g3/yhh/human-count/crosscam_run            -> crosscam-run/
```

## 约定

- 后续算法任务（推理、编码、拼接、分析）均在本机（GPU 服务器）执行，数据读写以此目录为根。
- 新增分析产物建议放 `crosscam-run/out/` 或按任务新建子目录。
