# 10-selector — L2+ 模板核与因果池

| 文件 | 内容 |
|---|---|
| `selector_queries_179.csv` | 179 个查询逐项：机位、GT 身份、查询框高、库轨道数、人脸相似度/余量/选中身份，以及 S0 / S1 / S0+实测池 / S1+实测池 四套方案的 top1、分数、是否命中 |
| `selector_pools.json` | 实测人脸通道写池结果：按 (机位, 身份) 分组的池条目，共 19 次写入、18 次正确 |
| `pose_pool_summary.json` | 原脚本 `--out` 的完整汇总（含 4 个 τ 的召回表与写入统计） |
| `selector_config.json` | 口径：σ=0.6 的来源、τ=0.80、池 k/max、人脸门 0.35 与 margin 0.02、质量门 |
| `exp003a*.json` | EXP-003a 核拆解：分层、200 次安慰剂置换、σ=0.3/1.0 两次敏感性。**注意：其中的「同机位归一化」已作废**（库与查询同机位，除同一常数是恒等式，不构成景深证据） |
| `exp003b*.json/csv` | EXP-003b 跨机位移植性：λ 扫描、分机位对、随机惩罚对照；逐对 cos 与两侧框高 |
| `exp003c*.json` | EXP-003c 几何分解（h / w / 宽高比）、判别 vs 刻度、留一法四配置、决策规则对照（绝对余弦 vs 相对间隔）、召回-误受曲线 |
| `exp003d.json` | EXP-003d 五协议（论文原口径 / 部署口径 / 反向 / 同段 / 全部留一）下的判定规则曲线与配对 bootstrap |
| `exp003e.json` | EXP-003e 2×2 质量门阶乘：核的增益在加库门后由 +24 降到 +8 |
| `openset.json` | **开放集**：29 条「本人不在图库」的查询在 τ=0.80 下被接受 5（S0）/6（S5）条，即 17.2% / 20.7% 的误认；质量门挡不住 |

## 复算

```bash
python tools/exp_pose_pool_real.py \
  --emb-dir tmp/full/emb12 --face tmp/full/face8 --det output_crosscam_local/1012_yolo8_dd \
  --gt deliverable/paper-package-2026-11-10/20-data/e4-full-corpus/gt_8seg.json \
  --segs "305_test_544 305_train_529 318_test_534 318_train_525 320_test_518 320_train_514 323_test_527 323_train_510" \
  --dump-queries selector_queries_179.csv --dump-pools selector_pools.json \
  --out pose_pool_summary.json
```

## 与论文的对应

- S0 / S1 / S1+实测池 = 101/179、122/179、125/179，即论文 §VI-C 的 56.4% / 68.2% / 69.8%；
- 79.3%（S0+oracle 池）与 83.2%（S1+oracle 池）是**机制上界**，对应 `pose_pool_summary.json` 中的 `s0+gt` / `s5+gt`；
- 池写入 19 条、其中 18 条身份正确（`w_real` / `w_real_ok`）。

**不含**：嵌入向量（`.npz`）与人脸图像。逐查询分数足以复核排名与命中，重建嵌入需原始视频。
