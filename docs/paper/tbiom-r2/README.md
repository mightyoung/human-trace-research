# tbiom-r2 — 投稿件（改动版）

> 由 r1（`tbiom-r1-source.zip`）改出：把 33 处作者查询压到 3 处，页数仍为 10。
> **本目录自带 IEEEtran.cls 与两张图，`pdflatex main` 两次即可复现 PDF。**

## 文件

| 文件 | 说明 |
|---|---|
| `main.tex` / `main.pdf` | 主稿，IEEEtran `journal`，**10 页**（含 3 处红字） |
| `supplement.tex` / `supplement.pdf` | 补充材料，独立文件，4 页 |
| `AQ-RESOLUTION.md` | 33 条 AQ 的逐条处置与依据文件 |
| `IEEEtran.cls`、`Figure_1.pdf`、`Figure_2.pdf` | 编译所需 |

## 编译

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # 第二遍定交叉引用
pdflatex -interaction=nonstopmode supplement.tex
```

本次编译环境：MiKTeX 25.12 / pdfTeX 4.23，`journal` 类，**0 overfull hbox、0 未定义引用**。

## 还剩三处红字

`main.tex` 第 12 行 `\newif\ifAQ \AQtrue` 控制显隐。剩下三条都是作者事实，不是数据：

1. **AQ1 资助**（在 Acknowledgment）——二选一填一句。
2. **AQ2 伦理**（首页脚注）——机构 + 委员会 + 批号 + 日期，或豁免依据；并补 consent 表述。需与投稿表单逐字一致。
3. **AQ3 商业关系**（Acknowledgment）——声明或明确否认。

填完把 `\AQtrue` 改成 `\AQfalse`，红字全消失。**改完必须再编译一次确认仍是 10 页**：第 10 页已排到参考文献 [33] 收尾，余量很薄；三条替换文本若明显长于占位符，需要同步删一句（`AQ-RESOLUTION.md` 末节给了候选）。

## 未解决（r1 遗留，本轮未动）

- 新增 9 条参考文献的**页码是凭记忆写的**，需按 DOI 逐条核。
- Table III 的区间是 Hanley–McNeil 近似（假设配对独立），不是按身份重采样的 bootstrap；正文已明写这个限制。
- 单站点、单判读人、小样本（11 / 5 / 3）仍是「实验数据不足」的直接对象，改稿解决不了。
