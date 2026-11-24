# 20-graph — 身份图、真值与时间校正

| 文件 | 内容 | 论文位置 |
|---|---|---|
| `judged_merges_76.csv` | 76 次合并操作的逐条重判：初判、复核判定、是否进入 63（13 条不决被排除） | §III-D、表 II |
| `T1_gt_pairs.csv` / `T1_gt_summary.json` | T1 的 126 对（34 同 / 92 异）与计数 | 表 III、§III-B |
| `T2_items.json` / `T2_labels_20261112.json` | T2 的 27 对物理矛盾项与裁定 | §V、表 IV |
| `attr_gate_runs.json` | 属性闸门两次运行（base 434/76、gate 443/67）的全部计数 | 表 II（tab:gate） |
| `graph_443_v0.json` / `graph_450_v1.json` | 参考图 v0 与约束着色后的 v1（443→450） | 表 VI |
| `merge_strategies.json` | 帧号 / OSD / OSD+共视门(60 s, 300 s) 四种合并策略 | 表 VII |
| `verifier_split.json` | 成对验证器的划分级切分扫描 | 表 VI |
| `spacetime_audit.json` | OSD 时钟上的时间审计（27 项队列的来源） | §VI-B、L3 |
| `osd_offsets.csv` | 六路首帧 OSD 与相对 318 的偏移（0/1/2/16/25/121 s **按机位**，非升序） | 表 I、§IV-A |

## 读这几张表要注意

1. **两次闸门运行不是嵌套的**：有门 67 次合并里只有 43 次带判读，无门是 76 次里 63 次，所以「少 9 次操作」却是「少 20 条判读」。
2. **T1 与 T2 不可合并**：T1 单独算精度是 5/5，judged 是 46/63，抽样机制不同。
3. 组级口径：T2 的 27 对里有 12 对同人（0.444），但 15 个组里 7 个是混的。
