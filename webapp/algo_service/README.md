# 单图人物属性服务

输入一张图片，输出图中每个人的框、框高档位与 9 个属性字段（值 + 置信度 + 可承诺档位）。

```
POST /v1/analyze        请求体 = 图片字节（image/jpeg、image/png、octet-stream 均可）
POST /v1/analyze_json   请求体 = {"image_b64": "..."}
GET  /healthz           加载状态与字段列表
GET  /v1/spec           能力矩阵（哪些字段在哪个档位可承诺）+ 各字段留出指标
```

## 调用

```bash
# 二进制
curl --data-binary @x.jpg "http://HOST:18080/v1/analyze" | python -m json.tool

# 带裁图、过滤掉像人分低的框
curl --data-binary @x.jpg "http://HOST:18080/v1/analyze?crops=1&drop_non_person=1"
```

### 自备框（客户已有检测器，只要属性）

`boxes` 参数可跳过内置检测，两种写法：

```bash
# 整图当一个人（适合已经裁好的行人图）
curl --data-binary @crop.jpg "http://HOST:18080/v1/analyze?boxes=full"

# 自己检测出来的框，坐标用原图像素
curl --data-binary @frame.jpg \
  'http://HOST:18080/v1/analyze?boxes=[[812,344,998,1000],[120,300,260,900]]'
```

自带框时 `det_conf` 固定返回 1.0（这不是模型的检测置信度，别拿它当质量分）。
框会被裁剪到图像范围内，小于 4 px 的框直接忽略；一个有效框都没有则返回 400。

## 返回

```json
{
  "image": {"w": 2560, "h": 1440, "name": "x.jpg"},
  "n_persons": 2,
  "persons": [
    {"idx": 3, "box": [812.0, 344.1, 998.6, 1000.0], "h": 655.9, "band": "L",
     "det_conf": 0.91, "is_person": 0.9665,
     "attrs": {
        "gender":    {"v": "男", "conf": 0.9997, "cap": "ok"},
        "top_color": {"v": "黑", "conf": 0.9974, "cap": "ok"},
        "bottom_color": {"v": null, "conf": null, "cap": null,
                         "reason": "S 档不可判（框高 62px，该字段没有可靠证据）"},
        "backpack":  {"v": "否", "conf": 0.83, "cap": "ref"}
     },
     "not_judgeable": ["bottom_color", "bottom_type", "glasses"]}
  ],
  "timing_ms": {"det": 22.1, "crop": 23.2, "emb": 318.9, "head": 8.7, "total": 350.9}
}
```

## 三条不能改的口径

1. **裁图 bbox + 10%**：属性头是在这个口径的 crop 上训练的（`build_ent_crops.py`），
   换口径等于换分布，指标不再成立。
2. **按框高档位闸门**：S < 80 px / M 80–240 / L ≥ 240。某字段在该档不可判时返回
   `null` + `reason`，不硬猜；只到「参考」档的字段照常返回但标 `cap="ref"`。
3. **`is_person` 单独给分**：模型对假框（倒影、海报、货架）照样会输出属性，
   调用方需要自己决定要不要用（建议 ≥ 0.5）。

## 能力边界（已用真实数据验收，别只看 CV）

验收报告：`docs/algo-service-acceptance-2026-11-12.md`。分两层，**结论完全不同**。

### 域内（long35 六机位，按身份留出的 5 折实测，v1.3 权重）

| 字段 | acc | 备注 |
|---|---|---|
| 性别 | **.974** | 全部档位可判 |
| 上衣颜色 | .888 | 沿用旧头（PA 无颜色标注） |
| 上衣类型 | .941 | 只有长袖/短袖两类 |
| 下装颜色 | .913 | 只有黑/白/蓝三类 |
| 下装类型 | .954 | 长裤/短裤/**裙**（v1.3 新增裙类） |
| 戴眼镜 | .924 | 比 v1.2 低 2.5 点，换来域外 FPR .756→.072 |
| 戴帽子 | .989 | v1.3 起域外也能判（TPR 0→.52） |
| 背包 | .948（参考） | 只在 L 档给，S/M 档弃权 |
| 拎东西 | .915（参考） | 只在 L 档给，S/M 档弃权 |

### 域外（PA-100K 800 张，室外监控，从没参与训练）

| 字段 | v1.2 | **v1.3** | 说明 |
|---|---|---|---|
| 性别 | .855 | **.894** | |
| 上衣类型 | .856 | **.899** | |
| 下装类型 | .806 | **.882** | 且现在能输出"裙" |
| 戴眼镜 | .414 | **.838** | FPR .756→.072，TPR .879→.589（更保守） |
| 戴帽子 | .860（TPR 0） | **.895**（TPR .52） | 从"从不判是"到能判 |
| 背包 | .717 | .728 | 仍是参考级 |
| 拎东西 | .761 | .712 | 唯一变差的字段 |

v1.3 = long35 + PA-100K_train（8000 张）混合重训、域内块 ×30，颜色头沿用旧权重。
`/v1/spec` 的 `spec.dataset` 会写明这一来源，换权重时能追溯。

`is_person` 在域外把 **26% 的真人**判到 0.5 以下 —— `drop_non_person=1` 在陌生场景会误删真人。
这一项**没有重训**（PA 全是真人，没有负例），仍是已知缺陷。

**掉点主要不是分辨率**：域内 M 档（80–240 px）眼镜 .984，域外中位 190 px 只有 .375；
按清晰度二分只差 5–7 个点。是场景/外观分布问题，与主线"跨机位外观不可用"同源 ——
这也是"加 8000 张域外数据能涨 40 个点、但仍然到不了 .95"的原因：剩下的差距在 EG2 特征层。

## 构建与运行（远程无外网，镜像靠 COPY 组装）

```bash
cd /share/data4/yhh/data/human-count/algo_image
docker build -t hc-attr:1.3 .
docker run -d --name hc-attr -p 18080:8080 --gpus all hc-attr:1.3
docker logs -f hc-attr
```

若 `--gpus all` 报错，改用 CDI：`--device nvidia.com/gpu=all`。

## 控制台入口

控制台「素材标注 → 图片试跑」页签会把上传的图经本地后端 `/api/attr/infer` 转发到本服务，
是端到端联调最省事的入口（也顺带验证了容器、GPU、档位闸门）。转发地址由后端环境变量
`HC_ATTR_URL` 指定，默认 `http://192.168.11.16:18080`。注意走 `Session(trust_env=False)`，
不要让局域网请求落到系统 HTTP 代理上。
