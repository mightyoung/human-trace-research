# 训练/切片结果 → 台账：交接单（路线 B 已就绪）

> 面向正在算力线上跑切片与训练的同事。**本机是主库**，算力线只负责算。
> 本文档对应 2026-11-09 的部署：远程 `~/human-count` 上已放好 ops 工具，训练脚本已接上报口。

---

## 0. 边界先说清

- 本机台账 `ops/ops.db` 是**唯一权威**。远程也有一份 `~/human-count/ops/ops.db`，它只是**跑的时候记账**用的临时本，跑完用 bundle 搬回来。
- 远程工具不读数据盘、不占 GPU；只在训练进程里写几行 SQLite。
- 同一个 `--out` 目录**不要两个任务同时写**：`ops_run` 会拒绝启动（返回码 3）并打印占用者 run_id。
- 训练脚本被改过（见 §2 的备份），只加了指标上报，**没有改训练逻辑与 stdout 格式**。

---

## 1. 切片产物（e21 合并后的 mp4）

切片不产生训练指标，但它是数据版本。跑完登记一条 dataset，后续训练用 `--dataset` 指向它：

```bash
cd ~/human-count
~/miniconda3/envs/rfdetr/bin/python tools/ops_run.py \
    --name e21-slice-v1 --kind build --record-only \
    --dataset ds_e21_slice_v1 \
    --dataset-root /share/data4/yhh/data/human-count/e21 \
    --dataset-dialect e21-joined \
    --note "每日 joined 片段"
```

`--record-only` 只登记 dataset + 一条 run，不执行命令、不扫产物。

---

## 2. 路线 B：在远程包一层跑（推荐，证据完整）

### 2.1 远程已部署的东西

```
~/human-count/
├── ops/
│   ├── schema.sql          台账结构
│   ├── ops.db              跑的时候自动建
│   ├── logs/               每次 run 的完整日志
│   └── bundles/            导出的 run 包
└── tools/
    ├── ops_db.py           访问层
    ├── ops_run.py          运行包装器
    ├── ops_metric.py       指标上报口
    └── ops_bundle.py       run 包导出/导入
```

训练脚本 `intent_cc/intent_model_cc.py`、`intent_v2/intent_model_social.py` 已加入 `Reporter`（原文件备份为 `*.bak_pre_ops_metric`）。补丁只在 `OPS_RUN_ID` 存在时生效，直接跑脚本时完全 no-op。

### 2.2 标准命令

```bash
cd ~/human-count
~/miniconda3/envs/rfdetr/bin/python tools/ops_run.py \
    --name e21-intent-v1 --kind train \
    --dataset ds_e21_slice_v1 \
    --out intent_cc/out_e21_v1 \
    --tag lr=1e-3 --tag d_model=256 --tag epochs=40 \
    -- ~/miniconda3/envs/rfdetr/bin/python intent_cc/intent_model_cc.py \
         --train <train.npz> --val <val.npz> --test <test.npz> \
         --out-dir intent_cc/out_e21_v1 --epochs 40 --device cuda
```

`--` 之后的命令原样透传，`ops_run` 只负责记账。跑完会打印：

```json
{ "run_id": "run_20261109_115354_36cfa3", "status": "succeeded",
  "exit_code": 0, "seconds": 7.5, "artifacts": {"n": 2} }
```

记下 `run_id`。**一个变体一个 `--out` 目录**，不要复用。

### 2.3 多个变体批量跑

用 shell 循环即可，每个变体独立 `--out` 和 `--name`：

```bash
for cfg in "cam305 1e-3" "global 5e-4"; do
  set -- $cfg
  ~/miniconda3/envs/rfdetr/bin/python tools/ops_run.py \
      --name "e21-$1" --kind train --dataset ds_e21_slice_v1 \
      --out intent_cc/out_e21_$1 --tag variant=$1 --tag lr=$2 \
      -- ~/miniconda3/envs/rfdetr/bin/python intent_cc/intent_model_cc.py \
           --train ... --val ... --test ... --out-dir intent_cc/out_e21_$1 \
           --cams ${1#cam} --lr $2 --epochs 40 --device cuda
done
```

批次跑完，`--experiment` 归组在导出时不用管；回本机后用 `ops_ingest_train.py --experiment` 或直接由我在导入时挂上去。

### 2.4 跑完导出 run 包

```bash
# 单个
~/miniconda3/envs/rfdetr/bin/python tools/ops_bundle.py export \
    --run run_20261109_115354_36cfa3 --out ops/bundles/

# 或批量：把本次所有 run 都导出
for rid in $(~/miniconda3/envs/rfdetr/bin/python tools/ops_db.py sql \
             "select id from run where status='succeeded'" | grep -o 'run_[0-9_a-z]*'); do
  ~/miniconda3/envs/rfdetr/bin/python tools/ops_bundle.py export --run $rid --out ops/bundles/
done
```

> `export` 默认把包写到 `--out` 目录（同名 json）；`running` 状态会被拒绝，跑完再导。

### 2.5 回本机入账

```bash
# 本机执行
python tools/remote_run.py --get \
    /home/r4900g3/human-count/ops/bundles/run_20261109_115354_36cfa3.json \
    ops/imports/bundles/run_20261109_115354_36cfa3.json

python tools/ops_bundle.py inspect ops/imports/bundles/<file>.json   # 先看一眼
python tools/ops_bundle.py import ops/imports/bundles/<file>.json
```

导入是**幂等**的：同一条 run 导两次不会翻倍（artifact 按快照替换，曲线按 `(step,name)` 覆盖）。

### 2.6 这台机器上已经验过一遍

```
run_20261109_115354_36cfa3   intent-cc-e2e   succeeded   7.5s
  host   r4900g3-UniServer-R4900-G3   GPU  L20 ×2
  代码   intent_cc/intent_model_cc.py · dc9903d60066
  环境   py3.11.15 · torch2.11.0+cu130
  曲线   9 点（3 epoch × loss/val_acc/val_f1）
  指标   acc 0.7017 · macro_f1 0.3658 · n 1294 · per-cam 4 机位
  产物   best.pt 144.6KB sha 7ff596e96d / metrics_full.json 8.4KB
  说明   这是 3 epoch / d-model 64 的链路验证，不是正式结果
```

---

## 3. 路线 A：事后入账（备用，零改代码）

如果某次训练没走 `ops_run`，把这三样带回来即可入账：

1. `out` 目录绝对路径
2. `metrics.json` 或 `metrics_full.json`
3. 训练日志（含 `[ep NN] loss=… val_acc=… val_f1=…` 行）

```bash
# 远程打包（只要指标和日志，别把 .pt 打进去）
cd <out_dir> && tar czf /tmp/run_import.tgz metrics*.json *.log
# 本机拉回
python tools/remote_run.py --get /tmp/run_import.tgz ops/imports/<slug>.tgz
tar xzf ops/imports/<slug>.tgz -C ops/imports/<slug>/
python tools/ops_ingest_train.py --root ops/imports/<slug> \
    --experiment "intent · e21 切片"
```

能拿到：run、指标、产物 sha1、日志里的逐 epoch 曲线。
拿不到：参数快照、**代码 sha1**、环境快照 —— 所以路线 B 优先。

---

## 4. 已知坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 训练页指标是 `—` | 指标文件叫 `metrics_full.json` 但包装器只认 `metrics.json` | **已修**（`find_metrics` 现在优先认 `metrics_full.json`）；远程工具已更新 |
| 曲线为空 | 日志无 `[ep NN]` 行，或脚本没接 `Reporter` | 路线 B 会写 `run_metric`，不依赖日志 |
| 同一 `out` 目录两次入账 | 目录被复用，产物混在一起 | 一个 run 一个目录；`ops_run` 会拒绝并发占用 |
| 训练页看不到 | run 没挂 `experiment` | 落到「未归入批次」；或用 `--experiment` 指定 |
| `--get` 传目录失败 | sftp 单文件，不支持递归 | 远端 `tar czf` 再取单包 |
| 导入后 `out_dir` 是远程路径 | 本机没有那个目录 | 属正常：台账只记指针。需要本地化时用 `--rewrite-out OLD=NEW` |
| `best epoch` 是 `—` | `metrics_full.json` 的 `rep` 结构不带该字段 | **已修**：前端从 `val_f1` 曲线取峰值点 |

---

## 5. 一条硬约定

**每个变体一个 `--out` 目录，跑完立刻 export。** 目录复用会导致产物覆盖，事后无法判断某个文件属于谁；不 export 就跑下一批，`ops.db` 里的曲线会被下一次的覆盖（同一个 run_id 不会，但目录会被判为"已用过"）。
