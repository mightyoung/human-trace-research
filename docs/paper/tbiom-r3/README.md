# tbiom-r3 — 投稿件（P2 收窄版，2026-11-24）

> 由 r2 改出：**九处 P2 收窄 + 新增 WILDTRACK 外推检验 + 补充材料 S8**。页数仍为 10（期刊上限）。
> **本目录自带 IEEEtran.cls 与两张图，`pdflatex main` 两次即可复现 PDF。**

## 文件

| 文件 | 说明 |
|---|---|
| `main.tex` / `main.pdf` | 主稿，IEEEtran `journal`，**10 页**（含 3 处红字 AQ） |
| `supplement.tex` / `supplement.pdf` | 补充材料，独立文件，**5 页**（新增 S8） |
| `CHANGES.md` | r2→r3 的逐处改动与作者决定 |
| `RESPONSE-NOTE.md` | response letter 素材（含 0.904 归因收回的主动说明） |
| `FULL-ACCOUNT.md` / `.docx` / `.pdf` | 不受篇幅限制的完整方法-贡献-结论说明（26 页） |
| `AQ-RESOLUTION.md` | r1 的 33 条 AQ 逐条处置（历史记录） |
| `IEEEtran.cls`、`Figure_1.pdf`、`Figure_2.pdf` | 编译所需 |

## 编译

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # 第二遍定交叉引用
pdflatex -interaction=nonstopmode supplement.tex
```

环境：MiKTeX 25.12 / pdfTeX 4.23，`journal` 类，**main 0 overfull / 0 未定义引用**；supplement 仅 1 处 r2 既有 4.07 pt。

## 三处红字（作者事实，不是数据）

`main.tex` 第 12 行 `\newif\ifAQ \AQtrue` 控制显隐：

1. **AQ1 资助**（Acknowledgment）——二选一填一句。
2. **AQ2 伦理**（首页脚注）——机构 + 委员会 + 批号 + 日期，或豁免依据；并补 consent 表述。需与投稿表单逐字一致。
3. **AQ3 商业关系**（Acknowledgment）——声明或明确否认。

**正确做法**：先把 `\AQ{…}` 花括号里的占位符整体替换为真实声明，再把 `\AQtrue` 改成 `\AQfalse`。
宏的定义是 `\ifAQ <红字提示> \else #1 \fi`——关闭开关后**输出花括号里的内容**，而不是删除它。
（r2 及更早的宏在关闭时会连内容一起删掉，这是已修的缺陷；若沿用旧文件请先同步该宏。）

关掉后必须再编译一次确认仍是 10 页：第 10 页已排到参考文献收尾，余量很薄。

## 未解决

- 新增 9 条参考文献的**页码需按 DOI 逐条核**（r1 遗留）。
- 单站点、单判读人、小样本（11 / 5 / 3）是「实验数据不足」的直接对象，改稿解决不了；需要第二站点。
- 派生结果包（逐查询结果、合并记录、WILDTRACK 复算脚本）按需提供，不随稿。
