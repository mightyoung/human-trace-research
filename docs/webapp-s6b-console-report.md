# 控制台 S6-b：待裁决判定页 + 倒影降权开关视图

> 需求：① 需要判定的放到前端相关模块（没有就新增）让我判定；② 加一个开关视图。
> 交付方式：**纯前端 + 静态载荷**（不碰后端 → 免重启、不影响正在用的真值标注）。

---

## 1. 待裁决模块（真值标注页新增标签页）

**入口**：`http://192.168.11.16:8099/#/truth` → 标签 **「S6-b 待裁决」**。

| 分组 | 条数 | 问法 | 选项 |
|---|---|---|---|
| ① relay 冲突边 | **16** | 这两段观测是同一人吗？ | 同人 / 不是 / 看不清 |
| ② 倒影「搭桥」合并 | **3** | 左右两组是同一人吗？ | 同人（该留）/ 不同人（该拆）/ 看不清 |
| ③ 倒影实体 | **11** | 这条轨迹是玻璃倒影还是真人？ | 是倒影 / 是真人 / 看不清 |

每张卡片给出**并排缩略图**（③ 还会给出「源」参考图）、`cos / dt / 成员 / gid`、
以及判定依据（如"倒影实体 `323-E051` 重定向到源后：邻居 g0105 vs 源 g0201 → 身份层判不同人"）。
支持 `只看未判`、逐组进度、以及 `置信 high/mid/low` + 备注。

**保存**：走既有 `POST /api/truth/labels`，`pair` 命名空间、id 前缀 `S6B_`。
→ 与用户已有的 241 对/107 事件**共用同一个 labels.json**，`grep '^S6B_'` 即可取出。
⚠ 因为走同一 `pair` 命名空间，`stats.labeled` 会被这些新增项累加（对既有 241 对的指标无影响）。

**数据来源**：`tools/gil_pending_web.py` → `public/s6b_pending.json` + `public/s6b_thumbs/*.jpg`
（50 张实体裁剪图，从 **全分辨率** `media-resource/multi-cemra/synced-10min/` 抽帧，外扩 45% 裁切）。

---

## 2. 倒影降权开关视图（可视化页）

**入口**：`http://192.168.11.16:8099/#/` → 打开 **「去重视图」** → 工具条出现 **「倒影降权」**。

- 关：**现状臂**（A）—— 264 人、505 owner 区间、423 通行段；
- 开：**降权臂**（B）—— **262 人**、493 owner 区间、413 通行段（倒影不作为独立身份节点）。

开关同时切换**框标签 / 着色 / 责任机位**所用的 gperson 划分，右栏面板给出两臂对照：
> 现状 `264` 人 → 降权 `262` 人 · 通行 `423→413` · owner `505→493`

**数据来源**：`tools/gil_visit_owner.py --gperson-map …/gperson_ab_B.json --web-out public/gil_owner_deduprefl.json`
—— ★ 直接用 **S6-b A/B 里 B 臂的簇划分**，保证"开关视图"与"A/B 结论"**完全同源**，不是两套口径。

---

## 3. 验收（远程 Edge / Playwright，`deploy/verify_s6b.cjs`）

**19/19 通过，控制台 0 错误**：

```
[OK] 真值页有「S6-b 待裁决」标签        [OK] 三类清单分组按钮 = 3
[OK] 分组含 relay 冲突 / 桥接 / 倒影实体 [OK] 第一组卡片数 = 16
[OK] 缩略图全部加载成功 (43/43)          [OK] 第三组卡片数 = 11
[OK] 「只看未判」过滤器可用              [OK] 点击「同人」进入选中态
[OK] 保存成功（已保存 1 项）             [OK] labels.json 已含 S6B_ 前缀判定
[OK] 验收写入已清理（S6B 残留 = 0）      [OK] 工具条有「倒影降权」按钮
[OK] 面板给出两臂人数对照 264 → 262      [OK] ★ 264 → 262
[OK] 开关进入按下态 / 文案「降权臂」      [OK] KPI 全局人变为 262
[OK] 可切回现状臂（264）                 [OK] 控制台零错误
```

另跑项目标准回归 `deploy/shot_ui.cjs`（4 视口 × 6 页 × 双主题）：**ERRORS 0**。
旧 dist 已备份为 `dist.bak.s6b.<时间戳>`。

★ 验收脚本对真值文件**自带清理**：写完一条判定后立即用 `null` 回滚并断言 `S6B_ 残留 = 0`，
不会污染用户正在标注的数据。

---

## 4. 复现

```bash
# ① 待裁决载荷 + 缩略图（需全分辨率视频在 media-resource/multi-cemra/synced-10min/）
python tools/gil_pending_web.py

# ② 两臂 owner 载荷
python tools/gil_visit_owner.py                                              # A 臂 → gil_owner.json
python tools/gil_visit_owner.py --gperson-map output_crosscam/analysis/gil/gperson_ab_B.json \
    --out output_crosscam/analysis/gil/visit_owner_deduprefl.json \
    --web-out webapp/frontend/public/gil_owner_deduprefl.json                # B 臂

# ③ 构建 + 部署（只在远程替换 dist，不重启后端）
cd webapp/frontend && npm run build
tar -czf ../deploy/dist.s6b.tgz -C . dist
MSYS_NO_PATHCONV=1 python tools/remote_run.py --put webapp/deploy/dist.s6b.tgz /home/r4900g3/human-count/dist.s6b.tgz
# 远程解包替换 webapp/frontend/dist

# ④ 验收
NODE_PATH=<managed>/node_modules node webapp/deploy/verify_s6b.cjs http://192.168.11.16:8099
NODE_PATH=<managed>/node_modules node webapp/deploy/shot_ui.cjs  http://192.168.11.16:8099
```

---

## 5. 未做

- 待裁决结果是**证据层**：尚未回灌到 GIL（`annotation` / `edge.status`）——
  需人工判完再一次性回灌，避免半成品污染。
- `render.py` 导出视频仍按 tid 画框（去重视图只在浏览器叠加层生效）。
