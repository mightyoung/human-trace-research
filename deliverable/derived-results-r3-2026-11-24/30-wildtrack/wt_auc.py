"""WILDTRACK · 阶段 C：区域共视 → 外观判别力（P2 外推验证）。

口径（与 CHIRLA 版一致，便于对比）：
  · 共视率 = Σ(两视角同时可见的人帧) / Σ(任一可见的人帧)，用**全部 400 帧**算
  · AUC 只用下载到的 20 帧/视角（1855 crops），且分两种配对：
      cross —— 同一 personID 的**不同时刻**裁剪（跨时刻，排除同时刻信息）
      same  —— 同一 personID 的**同一帧**裁剪（同时刻，仅在共视区非零时存在）
  · 负例 = 不同 personID 的随机配对，数量与正例相等
  · AUC 用 Mann-Whitney 秩公式；置信区间按**身份重采样** bootstrap（1000 次）

产物：tmp/wt/wildtrack_p2.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("tmp/wt")
ROOT = OUT / "data/Wildtrack_dataset"
ANN = ROOT / "annotations_positions"
NV = 7


def load_ann(path):
    arr = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for rec in arr if isinstance(arr, list) else []:
        pid = rec.get("personID")
        if pid is None:
            continue
        vis = {}
        for v in rec.get("views") or []:
            k = v.get("viewNum")
            if not isinstance(k, int) or not (0 <= k < NV):
                continue
            x1, y1, x2, y2 = (v.get("xmin", -1), v.get("ymin", -1),
                              v.get("xmax", -1), v.get("ymax", -1))
            if min(x1, y1, x2, y2) < 0 or x2 - x1 < 10 or y2 - y1 < 20:
                continue
            vis[k] = (x1, y1, x2, y2)
        if vis:
            out.append((pid, vis))
    return out


def auc_of(pos, neg):
    """Mann-Whitney：P(同人得分 > 异人得分)。秩从 1 起算，并列取平均秩（向量化）。

    2026-11-25 修正：原实现用 argsort().argsort() 得到 0 起秩，却代入 1 起公式，
    结果恒低 1/n_neg（8 正 8 负完全分离时返回 0.875 而非 1.0），且未处理并列。
    """
    if len(pos) < 5 or len(neg) < 5:
        return None
    allv = np.concatenate([pos, neg])
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))

def main():
    meta = json.loads((OUT / "crops_wt/meta.json").read_text(encoding="utf-8"))
    E = np.load(OUT / "wt_emb.npy").astype(np.float32)
    assert E.shape[0] == len(meta), f"E {E.shape} vs meta {len(meta)}"
    print(f"crops {len(meta)}  emb {E.shape}")

    # 1) 共视（全部帧）
    both = np.zeros((NV, NV), dtype=int)
    either = np.zeros((NV, NV), dtype=int)
    seen = np.zeros(NV, dtype=int)
    n_pf = 0
    for f in sorted(ANN.glob("*.json")):
        for _pid, vis in load_ann(f):
            n_pf += 1
            for k in vis:
                seen[k] += 1
            for i in range(NV):
                for j in range(i + 1, NV):
                    a, b = i in vis, j in vis
                    if a and b:
                        both[i, j] += 1
                    if a or b:
                        either[i, j] += 1
    cov = {}
    print(f"\n人帧 {n_pf}；各视角可见 {seen.tolist()}")
    print("\n共视矩阵（行=列对称）")
    print("      " + "".join(f"{f'C{j+1}':>8}" for j in range(NV)))
    for i in range(NV):
        row = ""
        for j in range(NV):
            if j == i or not either[i, j]:
                row += f"{'—':>8}"
            else:
                cov[(i, j)] = cov[(j, i)] = both[i, j] / either[i, j]
                row += f"{cov[(i,j)]:>8.2f}"
        print(f"C{i+1:<5}{row}")

    # 2) 索引：view -> pid -> [(idx, stem)]
    idx_by = defaultdict(lambda: defaultdict(list))
    for i, m in enumerate(meta):
        idx_by[m["k"]][m["pid"]].append((i, m["stem"]))

    rng = np.random.default_rng(0)

    def boot_ci(pos_pairs, neg_pairs, n_boot=1000):
        """按身份重采样：pos_pairs/neg_pairs 是 {pid: [score,...]} 的字典。"""
        pids = sorted(set(pos_pairs) | set(neg_pairs))
        if len(pids) < 4:
            return (None, None)
        vals = []
        for _ in range(n_boot):
            pick = rng.choice(pids, len(pids), replace=True)
            p = np.concatenate([pos_pairs[x] for x in pick]) if pos_pairs else np.array([])
            q = np.concatenate([neg_pairs[x] for x in pick]) if neg_pairs else np.array([])
            a = auc_of(p, q)
            if a is not None:
                vals.append(a)
        if len(vals) < 50:
            return (None, None)
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    rows = []
    for i in range(NV):
        for j in range(i + 1, NV):
            shared = sorted(set(idx_by[i]) & set(idx_by[j]))
            rec = dict(a=i, b=j, cov=round(cov.get((i, j), 0.0), 3),
                       n_id=len(shared))
            if len(shared) < 2:
                rec.update(auc_cross=None, auc_same=None)
                rows.append(rec)
                continue

            pos_c, neg_c = [], []
            pos_s, neg_s = [], []
            pos_c_by, neg_c_by = defaultdict(list), defaultdict(list)
            pos_s_by, neg_s_by = defaultdict(list), defaultdict(list)

            for pid in shared:
                Ai, Bj = idx_by[i][pid], idx_by[j][pid]
                # 跨时刻同人
                for ia, sa in Ai:
                    for ib, sb in Bj:
                        if sa == sb:
                            continue
                        v = float(E[ia] @ E[ib])
                        pos_c.append(v)
                        pos_c_by[pid].append(v)
                # 同时刻同人
                for ia, sa in Ai:
                    for ib, sb in Bj:
                        if sa != sb:
                            continue
                        v = float(E[ia] @ E[ib])
                        pos_s.append(v)
                        pos_s_by[pid].append(v)
                # 异人（同视角对，随机组合；同帧异人与跨帧异人都算合法负例）
                for p2 in shared:
                    if p2 == pid:
                        continue
                    ia = Ai[rng.integers(len(Ai))][0]
                    ib = idx_by[j][p2][rng.integers(len(idx_by[j][p2]))][0]
                    v = float(E[ia] @ E[ib])
                    neg_c.append(v)
                    neg_c_by[pid].append(v)
                for p2 in shared:
                    if p2 == pid:
                        continue
                    ia = Ai[rng.integers(len(Ai))][0]
                    ib = idx_by[j][p2][rng.integers(len(idx_by[j][p2]))][0]
                    v = float(E[ia] @ E[ib])
                    neg_s.append(v)
                    neg_s_by[pid].append(v)

            pc, nc = np.array(pos_c), np.array(neg_c)
            ps, ns = np.array(pos_s), np.array(neg_s)
            ac = auc_of(pc, nc)
            as_ = auc_of(ps, ns) if len(ps) >= 5 else None
            rec.update(
                auc_cross=None if ac is None else round(ac, 4),
                auc_same=None if as_ is None else round(as_, 4),
                n_pos_cross=len(pc), n_pos_same=len(ps), n_neg=len(nc),
                pos_cross_med=None if not len(pc) else round(float(np.median(pc)), 4),
                neg_med=None if not len(nc) else round(float(np.median(nc)), 4),
            )
            if ac is not None:
                lo, hi = boot_ci(pos_c_by, neg_c_by)
                rec["cross_ci"] = [None if lo is None else round(lo, 3),
                                   None if hi is None else round(hi, 3)]
            if as_ is not None:
                lo, hi = boot_ci(pos_s_by, neg_s_by)
                rec["same_ci"] = [None if lo is None else round(lo, 3),
                                  None if hi is None else round(hi, 3)]
            rows.append(rec)

    rows.sort(key=lambda r: -r["cov"])
    print("\n" + "=" * 108)
    print(f"{'视角对':<9}{'共视':>7}{'共享ID':>7}{'AUC跨时刻':>11}{'95%CI':>16}"
          f"{'AUC同时刻':>10}{'95%CI':>16}{'同人中位':>10}{'异人中位':>10}")
    print("-" * 108)
    for r in rows:
        ci = r.get("cross_ci") or [None, None]
        cis = r.get("same_ci") or [None, None]
        f1 = "—" if r["auc_cross"] is None else f"{r['auc_cross']:.3f}"
        f2 = "—" if r["auc_same"] is None else f"{r['auc_same']:.3f}"
        c1 = f"[{ci[0]},{ci[1]}]" if ci[0] is not None else "—"
        c2 = f"[{cis[0]},{cis[1]}]" if cis[0] is not None else "—"
        print(f"C{r['a']+1}-C{r['b']+1:<6}{r['cov']:>7.2f}{r['n_id']:>7}"
              f"{f1:>11}{c1:>16}{f2:>10}{c2:>16}"
              f"{str(r.get('pos_cross_med')):>10}{str(r.get('neg_med')):>10}")

    # 3) 趋势
    for key, tag in (("auc_cross", "跨时刻"), ("auc_same", "同时刻")):
        v = [r for r in rows if r.get(key) is not None]
        if len(v) < 5:
            continue
        c = np.array([r["cov"] for r in v])
        a = np.array([r[key] for r in v])
        pr = float(np.corrcoef(c, a)[0, 1])
        # Spearman
        rc = np.argsort(np.argsort(c)).astype(float)
        ra = np.argsort(np.argsort(a)).astype(float)
        sr = float(np.corrcoef(rc, ra)[0, 1])
        hi = [r[key] for r in v if r["cov"] >= 0.5]
        mid = [r[key] for r in v if 0.2 <= r["cov"] < 0.5]
        lo = [r[key] for r in v if r["cov"] < 0.2]
        print(f"\n【{tag}】n={len(v)}  Pearson r={pr:.3f}  Spearman ρ={sr:.3f}")
        for name, arr in (("共视≥0.50", hi), ("0.20–0.50", mid), ("<0.20", lo)):
            if arr:
                print(f"  {name:<10} AUC 中位 {np.median(arr):.3f}"
                      f"  范围 [{min(arr):.3f}, {max(arr):.3f}]  n={len(arr)}")

    (OUT / "wildtrack_p2.json").write_text(
        json.dumps(dict(n_crops=len(meta), n_person_frames=n_pf,
                        coverage=dict(seen=seen.tolist(), both=both.tolist(),
                                      either=either.tolist()),
                        pairs=rows), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("\n-> tmp/wt/wildtrack_p2.json")


if __name__ == "__main__":
    main()
