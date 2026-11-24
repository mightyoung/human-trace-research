# 运行配置（全部取自 supplementary S2，未做站点微调）

| 组件 | 配置 |
|---|---|
| 检测 | YOLO8，person 单类，官方预训练权重；三个域同一检测器，差异在采样步长与跟踪器 |
| 跟踪 | ByteTrack：track_high 0.10 / track_low 0.05 / new_track 0.12 / match 0.85 / fuse_score on；track_buffer 60@stride5（建库段）、100@stride3（查询段），均约 12 s |
| 外观 | OSNet-x1.0，MSMT17 权重，512 维，8 帧时间均匀均值，冻结 |
| 人脸 | SCRFD-2.5G 检测 + 五点对齐 + ArcFace 512 维；D1 用 w600k_r50，D2 选择器池用 w600k_mbf |
| 第二编码器 E2 | EmbeddingGemma-2，768 维图像嵌入 |
| 建图 | 同机位 average-linkage 余弦 0.15；跨机位贪心 ≥0.75，受同帧共现、机位重叠、属性 ≥8/9 三条约束 |
| 属性头 | 9 个服装字段；5 项一致率 0.88–0.99，拎东西 0.656、背包 0.458 |
