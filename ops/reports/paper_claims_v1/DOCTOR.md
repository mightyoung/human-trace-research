# 台账体检（doctor）

> 只读检查：登记过的文件是否还在、内容是否变、run 的输入是否漂移、代码版本是否过期。
> 大文件（>200 MB）跳过哈希，只验存在性。

| 检查项 | 结果 |
|---|---|
| `asset_version` 行数 | 129 行 / 115 个路径（当前一致 111、缺失 0、内容变化 0、跳过 4；历史版本 14） |
| run 引用的唯一路径 | 87（缺失 0、内容变化 0、登记时已缺失 0） |
| `code_version` 过期 | 0 / 1 |

## 资产跳过大文件（4）

| 路径 | 类型/条数 | 详情 |
|---|---|---|
| `media-resource/multi-cemra/long-35min/305_long35.mp4` | media | 326236253 B |
| `media-resource/multi-cemra/long-35min/316_long35.mp4` | media | 315451048 B |
| `media-resource/multi-cemra/long-35min/320_long35.mp4` | media | 491501235 B |
| `media-resource/multi-cemra/long-35min/323_long35.mp4` | media | 527425705 B |
