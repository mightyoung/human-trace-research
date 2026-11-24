# 对 r2 读者的主动说明（response letter 素材，2026-11-24 20:00）

> 作者已在 11-24 19:35 确认三项处置：**(1) 接受收回 0.904 的共视归因；(2) 摘要 gate 改 inform；(3) 补充材料 5 页可接受、S8 两表不合并。**
> 本文是给 response letter 用的**英文可粘贴段**，未写入 `main.tex`（正文改动只到摘要那一句）。

## 1. 主动说明：0.904 的归因已收回（必须写进 response letter）

事实链：r2 的 §V-B 用 318–320 的 AUC 0.904 支持「P2 共享区域 ⇒ L1 可用」，并在 §V-B 末段自认「in-region AUC 是最缺的实验」。r3 补做了这个实验（WILDTRACK，21 对），结果是零相关，且区域级限制不带来增益。因此 r2 的那条归因在 r3 被收回，0.904 只保留为「一对视角也最相似」的例子。

可直接粘贴：

```
One claim from the previous version is withdrawn, and we would rather flag it
than let it pass silently. The earlier draft read the 0.904 AUC of camera pair
318-320 as evidence that appearance becomes matchable inside a region both
cameras observe (our probe P2). We have now run that experiment on a public
benchmark (WILDTRACK, seven synchronised views, global person IDs). Across 21
view pairs, co-visibility and cross-view AUC are unrelated (r = 0.18, Spearman
0.16; r = -0.05 after removing pairs with fewer than 30 shared identities), the
null survives equalising the shared identity set, and restricting pairs to the
ground region both cameras observe moves AUC by a median of +0.04 (11 of 17
pairs positive). We therefore withdraw the causal reading of 0.904. P2 is
retained with a narrower claim: it certifies that both cameras observe a region
at the same time, which is what makes a temporal anchor and an adjudication
entry point possible; it does not certify that appearance will match across the
pair. Section V-B now states this, the L1 row of Table IV carries the negative
external check, and the limitations state that the check rests on one public
dataset and one encoder.
```

## 2. 摘要与 P1 的范围

- 摘要原句 `... gate five delivery tiers.` → `... inform rather than gate five delivery tiers, each with a stated scope.` 理由：P2 已不再是「共享区域 ⇒ L1 自动可用」的闸门，用 gate 会被读成保证。
- P1 的范围（只约束人脸通道、不约束身体外观）**没有改正文**：§P1 原文只谈人脸，未越界；限定句写在补充材料 S8 的像素扫描段（`P1 is a face-channel constraint rather than a general one`）。正文加这句会溢到 11 页（实测），故不重复。

若审稿人追问 P1 与「人太小导致跨镜失败」的关系，用这条：

```
The pixel sweep was run to test the most natural explanation of our cross-camera
failure, namely that persons are too small. It does not hold: AUC stays at 0.671
even when persons are reduced to 16 px (native 0.795). We therefore scope P1 to
the face channel and do not invoke resolution to explain body-appearance failure.
```

## 3. 补充材料页数

r3 补充材料 5 页（r2 为 4，新增 S8、S9）。作者确认可接受，**S8 的区域级 A/B 表与像素扫描表不合并**。若期刊对补充材料也有页数上限，只需把 S8 两张表并成一张（同一批 17 对的两个切面），内容不损失。

## 3b. L2+ 选择器的两个敏感性（新增 S9，主动写明）

这两条不改变任何数字，改变的是"这个增益在部署条件下还剩多少"。写进 supplement §S9 而不是正文，是因为 §VI-C 的原句（「τ is the deployed selector's operating point」）字面为真，问题只在读者可能把表格当成部署成绩。

```
Two sensitivities of the L2+ row are reported in the supplement (Section S9).
First, the 179 queries and their morning library are not filtered by person-box
height. Applying a 60 px floor to both -- the operating point of the selector we
deploy -- leaves 157 queries; the plain gallery mean reaches 100 of 157 and the
box-height kernel adds 8, not the 24 it adds without the floor. The reason is
that filtering the library already raises the plain mean on the surviving
queries from 89 to 100, i.e. most of what the kernel was doing was suppressing
low-quality templates. We therefore describe the L2+ gain as conditional on an
unfiltered library. Second, replacing the absolute threshold with a relative
margin changes recall at a fixed false-accept count (one false accept: 116 vs
56; four: 118 vs 112), and the two remedies do not combine (115 at four false
accepts), which suggests both address the same score-scale defect rather than
two independent weaknesses.
```

## 4. 改完后的状态

| 文件 | 页数 | Overfull | 未定义引用 |
|---|---:|---:|---:|
| `main.pdf` | **10**（限 10） | 0 | 0 |
| `supplement.pdf` | 5 | 1（r2 既有，表 S3，4.07 pt） | 0 |

正文差异只有两处：摘要的 gate→inform；以及 r3 早前那九处 P2 收窄改动（见 `CHANGES.md` §1）。补充材料在 S8 之外新增 S9（L2+ 的两个敏感性），**页数仍为 5**。
剩下三处红字仍是作者事实（AQ1 资助 / AQ2 伦理 / AQ3 商业关系），与本轮三项决定无关。
