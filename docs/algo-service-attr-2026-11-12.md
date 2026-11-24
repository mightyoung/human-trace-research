# 单图人物属性服务（Docker，2026-11-12）

> 目标：输入一张图片，输出图中**每个人**的框、框高档位与 9 个属性字段（值 + 置信度 + 可承诺档位）。
> 代码 `webapp/algo_service/{app.py,Dockerfile,pack.sh,README.md}`；镜像 `hc-attr:1.0`（远程，13.9 GB）。

## 结论先行

远程 `192.168.11.16` 上镜像已构建并跑通：加载 4.9 s，热机后 **1080p 单人 224 ms / 1440p 十人 777 ms**。
属性输出带三条约束：**按框高档位闸门**（S/M/L）、**不可判给 null 不硬猜**、**is_person 单独给分**。

但它**还不是一个可以对外承诺精度的服务**：精度只在原始素材分布上验过，
且本轮目视核对已经抓到两个典型失败模式（见 §5）。上线前必须做真实图片验收。

## 一、架构

```
图片字节 ──► YOLO11m(person, conf .35, imgsz 1280)
           └─► 裁图 bbox+10%（与训练同口径）
                ├─► EG2 编码（全图 768 维）
                └─► 下半身 62% 编码（仅 L 档，拎东西用）→ concat 1536 维
                     └─► 9 个 LogisticRegression 头 + is_person 头
                          └─► 按 band 做能力闸门 → JSON
```

- 权重：`attr_heads.joblib`（3.5 MB，本轮训练落盘，见 §2）。
- 单次推理只跑一次检测 + 两批编码；下半身嵌入只对 L 档算，省掉约 40% 编码时间。

## 二、权重的来源：训练一次并序列化

原 `build_attr_index.py` **只写预测结果，不存模型**——服务化必须补这一步。
新增 `tmp/attr/export_heads.py`（远程 `~/work/eg2/export_heads.py`）：

- 复用缓存嵌入 `eg2_emb_457.npz`，只补算下半身；
- 每条 pipeline（StandardScaler + PCA + LogisticRegression）整条 joblib 序列化，
  连同 `capability`、`pad`、`lower_from`、`band` 一起写进 bundle —— 保证线上与线下指标是同一套东西；
- 产出 `attr_heads.joblib` 3.5 MB。

留出指标（实体分组 CV，与历史口径一致）：性别 .965 / 上衣色 .888 / 上衣型 .946 /
下装色 .913 / 下装长度 .993 / 眼镜 .946 / 帽 .995 / 背包 .939（Δ −0.011）/ 拎东西 .935；
is_person AUC **.9858**（457 正例 + 144 负例）。

★ 背包的 CV 仍低于「全猜否」基线 —— 与既有结论一致，只标 `ref`。

## 三、离线构建：三个必须知道的坑

远程**没有外网**（pypi 与 docker registry 均不可达），所以镜像不能 `pip install`：

1. **组装而非安装**：`FROM ubuntu:22.04`（本地已有，与宿主 22.04.5 glibc 一致）
   + COPY 宿主 conda 环境（7.0 G）+ venv 覆盖层（192 M）+ EG2 模型（1.5 G）+ yolo11m + heads。
   build context 用 `cp -al` 硬链接生成，不占额外空间。
2. **覆盖 PATH 会弄丢 /sbin**：Dockerfile 里把 PATH 写成 `...:/usr/bin:/bin` 后，
   `ldconfig` 找不到（exit 127）→ 必须写 `/sbin/ldconfig`。
3. **不要 `chmod -R /opt`**：它会把 9 GB 环境整个重写一层，导出阶段多花十几分钟；
   本次第一次构建就卡在 `exporting layers` 上，去掉后才正常。

缺的系统库（cv2 需要 GL/X/glib）共 11 个，`pack.sh` 从宿主 `/lib` 补到 `/usr/local/lib`，
并把 `/usr/local/lib` 加进 `LD_LIBRARY_PATH`。

## 四、接口

```
POST /v1/analyze        请求体=图片字节（不用 multipart：镜像里没有 python-multipart）
POST /v1/analyze_json   {"image_b64": "..."}
GET  /healthz           加载状态、字段列表
GET  /v1/spec           能力矩阵 + 各字段留出指标
```

参数：`?crops=1` 返回每人裁图 base64；`?drop_non_person=1` 按 is_person≥.50 过滤。

## 五、验证：跑通 + 两个真实失败模式

四张原片帧（316/305/323/318）实测：

| 图 | 分辨率 | 人数 | 耗时 | 备注 |
|---|---|---|---|---|
| 316_f1660 | 2560×1440 | 1 | 1757 ms（冷） | 首次含预热 |
| 305_f1600 | 2560×1440 | 10 | **777 ms** | det 21 / emb 713 / head 18 |
| 323_f600 | 1920×1080 | 1 | 224 ms | M 档 → 背包/拎东西不可判 |
| 318_f900 | 1920×1080 | 4 | 255 ms | S 档 2 人 → 下装/眼镜不可判 |

档位闸门行为正确：L 档 9 字段齐全（背包/拎东西标 ref）；M 档下装/眼镜转 `ref`、背包/拎东西返回 `null`；
S 档下装/眼镜/背包/拎东西全 `null`，上衣类型标 `ref`。

**目视核对抓到的两个失败模式（重要）**：

| 案例 | 输出 | 实际 | 判断 |
|---|---|---|---|
| 305 第一位（女/白短袖/黑裤，背面站立） | 女、白、短袖、黑、长裤 | 一致 | ✅ 上衣/下装/性别都对 |
| 同上，眼镜字段 | `是 1.00` | 背面看不到脸 | ❌ **L 档「可承诺」的眼镜在背面失效** |
| 305 第二位（男，浅灰 Polo，椅背搭深蓝外套） | 上衣 `蓝 0.66`、`短袖 0.53` | 浅灰 Polo | ❌ 被椅背外套带偏，但**置信度同时掉到 0.53–0.66** |

→ 两条可操作结论：
1. **背面/侧面时 `glasses` 不可信**，即使 band=L。能力矩阵是按框高分的，没有按「朝向」分。
2. **低置信度是有效信号**（0.53/0.66 那两条确实错），调用方按 `conf ≥ .7` 过滤能挡掉这类；
   但要挡干净，得加「朝向」判据——这需要额外标注，不在本轮范围。

## 六、交付物与运维

```bash
# 镜像（远程）
docker images | grep hc-attr          # hc-attr:1.0  13.9GB
docker run -d --name hc-attr -p 18080:8080 --gpus all hc-attr:1.0
docker logs -f hc-attr                # [startup] 加载完成 4.9s device=cuda:0
curl --data-binary @x.jpg http://127.0.0.1:18080/v1/analyze

# 重建（源码改动后）
bash /share/data4/yhh/data/human-count/algo_image/pack.sh
```

- 源码与 Dockerfile：`/share/data4/yhh/data/human-count/algo_image/`
- 构建 context（硬链接）：`/tmp/hc_img/`（重启后需 `pack.sh` 重建）
- 镜像导出给其他机器：`docker save hc-attr:1.0 | gzip > hc-attr-1.0.tar.gz`（约 5–6 GB）

## 七、没过关的部分（不要当成已完成）

- **精度没有在真实图片上验收**：以上四张仍是本项目机位的帧。合影、远景、俯拍、手机照全部未测。
- **朝向问题**：背面/侧面的眼镜、背包判读不可靠，能力矩阵里没有这一维。
- **只支持单帧**：视频需要调用方自己抽帧；跨帧投票在本项目里已被证伪（多帧反而更差），不做。
- **13.9 GB 偏大**：conda 环境里带着 ultralytics/torch 的全套依赖；
  真要瘦身可换成 `python:3.11-slim` 基础层 + 只拷必需 site-packages，但无外网时要补更多系统库。
