# -*- coding: utf-8 -*-
"""现状盘点：标注页 / 产物 / 实验 / 训练痕迹。只读。"""
import json, os, glob, collections

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def jl(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


print("=== 1. 标注页数据盘点 ===")

# 1.1 素材标注：候选对
pairs = jl(f"{ROOT}/webapp/backend/data/pairs.json", [])
lab = jl(f"{ROOT}/webapp/backend/data/labels.json", {})
pd = lab.get("pair_decisions", {})
md = lab.get("mirror_decisions", {})
print(f"素材标注-候选对: 总 {len(pairs)} / 已判 {len(pd)} "
      f"({collections.Counter(pd.values())})")
mirror = jl(f"{ROOT}/webapp/backend/data/mirror.json", {})
n_refl = len(mirror.get("reflection", []))
n_src = len(mirror.get("source", []))
print(f"素材标注-倒影/源: reflection {n_refl} / source {n_src} / 已判 {len(md)} "
      f"({collections.Counter(md.values()).most_common(4)})")

# 1.2 真值标注
st = jl(f"{ROOT}/webapp/backend/data/truth/state.json", {})
if isinstance(st, dict):
    print("真值标注 state keys:", list(st.keys()))
    for k, v in st.items():
        if isinstance(v, (list, dict)):
            print(f"   {k}: {len(v)}")

# 1.3 人脸库
for cand in [f"{ROOT}/webapp/frontend/public/faces/face_data.json",
             f"{ROOT}/webapp/backend/data/faces_labels.json",
             f"{ROOT}/output_crosscam/analysis/face/faces_labels.json"]:
    if os.path.exists(cand):
        d = jl(cand, {})
        print(f"人脸: {os.path.basename(cand)} "
              f"{ {k: len(v) if isinstance(v,(list,dict)) else v for k,v in d.items()} if isinstance(d,dict) else len(d) }")

# 1.4 静态待裁载荷
sp = jl(f"{ROOT}/webapp/frontend/public/s6b_pending.json", {})
if isinstance(sp, dict):
    print("S6-b 待裁载荷:", {k: len(v) if isinstance(v, (list, dict)) else v
                             for k, v in sp.items() if k != "meta"})

# 1.5 cluster 状态
cs = jl(f"{ROOT}/webapp/backend/data/clusters_state.json", {})
print("cluster state:", list(cs.keys())[:6] if cs else "MISSING")

print("\n=== 2. 分析产物规模 ===")
base = f"{ROOT}/output_crosscam/analysis"
dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
print(f"analysis 子目录 {len(dirs)} 个")
for d in sorted(dirs):
    p = os.path.join(base, d)
    nf = sum(len(f) for _, _, f in os.walk(p))
    print(f"   {d:34s} {nf:>5d} files")

print("\n=== 3. 图表/指标文件 ===")
figs = glob.glob(f"{ROOT}/**/figs/**/*.png", recursive=True) + \
       glob.glob(f"{ROOT}/**/figs/**/*.jpg", recursive=True)
print(f"figs 图片 {len(figs)} 张；上周新增 {sum(1 for f in figs if os.path.getmtime(f) > 1759000000)}")

print("\n=== 4. 近期产物（按修改时间） ===")
items = []
for pat in ["output_crosscam/analysis/**/*.json", "webapp/backend/data/*.json"]:
    for f in glob.glob(f"{ROOT}/{pat}", recursive=True):
        items.append((os.path.getmtime(f), f.replace(ROOT + "/", "")))
items.sort(reverse=True)
for t, f in items[:15]:
    import datetime
    print("  ", datetime.datetime.fromtimestamp(t).strftime("%m-%d %H:%M"), f)
