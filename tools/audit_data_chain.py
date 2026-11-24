#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据链一致性审计（评审缺陷 #1/#2/#3 的可复现核验）。

回答三件事：
  A. 逐框契约：每一帧 boxes / track_ids / scores 必须同长度，count == len(boxes)。
     对 raw / cleanset(v1) / cleanset_v2 三口径分别统计违规行数。
  B. 版本链哈希：raw → cleanset → cleanset_v2 的 md5 清单（结果必须记录上游哈希）。
  C. 悬空引用：relay 接力边引用的实体是否都存在于当前 clean 实体集合内。

用法:
  python tools/audit_data_chain.py                 # 全部检查
  python tools/audit_data_chain.py --view clean2   # 指定 clean 口径做 C 检查
输出:
  output_crosscam/analysis/audit/data_chain_audit.json
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
SRC = os.path.join(ROOT, "output_crosscam")
OUT = os.path.join(SRC, "analysis", "audit")
CAMS = ["305", "316", "318", "320", "323", "343"]


def md5(p, bs=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def contract_scan(path):
    """逐框契约检查：返回 (rows, n_bad_count, n_bad_scores, n_bad_tids)"""
    n = bc = bs = bt = 0
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            nb = len(r.get("boxes") or [])
            if r.get("count") != nb:
                bc += 1
            if len(r.get("scores") or []) != nb:
                bs += 1
            if len(r.get("track_ids") or []) != nb:
                bt += 1
    return dict(rows=n, bad_count=bc, bad_scores=bs, bad_tracks=bt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="clean2", choices=["clean", "clean2"],
                    help="做悬空引用检查时采用的 clean 实体口径（v5 需 clean2）")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rep = {"contract": {}, "hashes": {}, "dangling": {}}

    # ---- A + B：三口径逐框契约 + 哈希 ----
    paths = {
        "raw": {c: os.path.join(SRC, f"{c}_detections.jsonl") for c in CAMS},
        "clean": {c: os.path.join(SRC, "cleanset", f"{c}_detections.jsonl") for c in CAMS},
        "clean2": {c: os.path.join(SRC, "cleanset_v2", f"{c}_detections.jsonl") for c in CAMS},
    }
    print("=" * 72)
    print("A/B. 逐框契约（boxes/track_ids/scores 同长；count==len(boxes)）与 md5")
    for view, ps in paths.items():
        tot = dict(rows=0, bad_count=0, bad_scores=0, bad_tracks=0)
        rep["contract"][view], rep["hashes"][view] = {}, {}
        for cam, p in ps.items():
            r = contract_scan(p)
            if r is None:
                print(f"  {view:>6} {cam}: MISSING")
                continue
            rep["contract"][view][cam] = r
            rep["hashes"][view][cam] = md5(p) if os.path.exists(p) else None
            for k in tot:
                tot[k] += r[k]
            flag = "OK" if (r["bad_count"] == r["bad_scores"] == r["bad_tracks"] == 0) else "违规"
            print(f"  {view:>6} {cam}: rows={r['rows']:>5} bad(count/scores/tracks)="
                  f"{r['bad_count']}/{r['bad_scores']}/{r['bad_tracks']}  [{flag}]")
        rep["contract"][view]["_total"] = tot
        print(f"  {view:>6} 合计: rows={tot['rows']} bad_count={tot['bad_count']} "
              f"bad_scores={tot['bad_scores']} bad_tracks={tot['bad_tracks']}")

    # 链式关系：clean2 与 raw 的差异必须只来自“删除框”，其余字段逐行等长
    print("\n  哈希链（结果必须记录上游哈希）:")
    for cam in CAMS:
        print(f"    {cam}: raw={rep['hashes']['raw'][cam][:10]} "
              f"clean={rep['hashes']['clean'][cam][:10]} "
              f"clean2={rep['hashes']['clean2'][cam][:10]}")

    # ---- C：悬空引用（按各自实体口径） ----
    print("\n" + "=" * 72)
    print("C. 悬空引用：relay 接力边 vs 对应实体集合")
    import importlib
    import c0_baseline as C  # noqa: E402

    NET_OLD = os.path.join(SRC, "analysis", "stitch")
    NET_C2 = os.path.join(SRC, "analysis", "stitch_clean2")

    def entity_set(net_dir, view):
        """按 net_dir 构建实体集；foot 视图必须与该实体图口径一致，
        否则被清洗删除/恢复的轨会在 build_entity_tracks 里静默丢帧 → 假悬空。"""
        os.environ["DET_VIEW"] = view
        importlib.reload(C)
        C.NET_DIR = net_dir
        s = set()
        for cam in CAMS:
            try:
                for eid in C.build_entity_tracks(cam):
                    s.add(f"{cam}:{eid}")
            except Exception as ex:  # noqa: BLE001
                print(f"    [warn] 实体重建失败 {cam}: {str(ex)[:80]}")
        return s

    sets = {"stitch(旧)": entity_set(NET_OLD, "clean"),
            "stitch_clean2": entity_set(NET_C2, "clean2")}
    for k, s in sets.items():
        print(f"  实体集[{k}] 总数: {len(s)}")

    # 口径无关的成员判定：直接读 net_tracks json 的 eid 全集（不经视图过滤）
    raw_sets = {}
    for name, d in (("stitch(旧)", NET_OLD), ("stitch_clean2", NET_C2)):
        s = set()
        for cam in CAMS:
            p = os.path.join(d, f"net_tracks_{cam}.json")
            if os.path.exists(p):
                dd = json.load(open(p, encoding="utf-8"))
                s |= {f"{cam}:{e['eid']}" for e in dd["entities"]}
        raw_sets[name] = s
    for k, s in raw_sets.items():
        print(f"  实体图[{k}] 文件实体总数（未过滤）: {len(s)}")

    relay_files = {
        "v3": (os.path.join(SRC, "analysis", "relay", "relay_triage_cradio_v3.json"), "stitch(旧)"),
        "v4": (os.path.join(SRC, "analysis", "relay", "relay_chains_cradio_v4.json"), "stitch(旧)"),
        "v5": (os.path.join(SRC, "analysis", "relay_clean2", "relay_chains_cradio_v5.json"), "stitch_clean2"),
    }
    for tag, (p, which) in relay_files.items():
        if not os.path.exists(p):
            print(f"  {tag}: MISSING")
            continue
        d = json.load(open(p, encoding="utf-8"))
        if tag == "v3":
            edges = [tuple(x.strip() for x in s.split("->")) for s in d["confirmed_edges"]]
        else:
            edges = [(e["a"], e["b"]) for e in d["edges"]]
        ents = sets[which]
        nodes = {x for e in edges for x in e}
        dangling = nodes - ents
        dangling_raw = nodes - raw_sets[which]
        dead = [e for e in edges if e[0] not in ents or e[1] not in ents]
        rep["dangling"][tag] = dict(edges_file=os.path.relpath(p, ROOT), entities=which,
                                    n_edges=len(edges), n_nodes=len(nodes),
                                    n_dangling_nodes=len(dangling), n_dead_edges=len(dead),
                                    n_dangling_vs_raw_entities=len(dangling_raw),
                                    dangling=sorted(dangling),
                                    dangling_vs_raw=sorted(dangling_raw))
        print(f"  {tag} [实体={which}]: edges={len(edges)} nodes={len(nodes)} "
              f"悬空(视图过滤后)={len(dangling)} 悬空(对文件实体全集)={len(dangling_raw)} "
              f"含悬空端点的边={len(dead)}")
        if dangling_raw:
            print(f"       ！真悬空: {sorted(dangling_raw)[:8]}"
                  f"{' ...' if len(dangling_raw) > 8 else ''}")

    # ---- D：版本链闭环 ----
    print("\n" + "=" * 72)
    print("D. 版本链：每级记录上游哈希与口径（目标=每个指标绑定唯一输入）")
    chain = {"L1_detections": {}, "L2_entities": {}, "L3_relay": {}, "L4_eval": {}}
    for v in ("raw", "clean", "clean2"):
        chain["L1_detections"][v] = dict(hashes=rep["hashes"][v],
                                         contract_violations=rep["contract"][v]["_total"])
    for name, d in (("stitch_old", NET_OLD), ("stitch_clean2", NET_C2)):
        man = os.path.join(d, "manifest.json")
        entry = {}
        if os.path.exists(man):
            m = json.load(open(man, encoding="utf-8"))
            entry = dict(manifest=os.path.relpath(man, ROOT), view=m.get("view"),
                         totals=m.get("totals"),
                         cameras={c: {k: v.get(k) for k in ("n_entities", "md5_source")}
                                  for c, v in m.get("cameras", {}).items()})
        else:
            entry = dict(note="无 manifest（旧版未记录上游哈希）",
                         md5={c: md5(os.path.join(d, f"net_tracks_{c}.json")) for c in CAMS
                              if os.path.exists(os.path.join(d, f"net_tracks_{c}.json"))})
        # L2 口径一致性：实体成员轨迹在 clean2 中的存活率
        c2t = {}
        for cam in CAMS:
            p = os.path.join(SRC, "cleanset_v2", f"{cam}_detections.jsonl")
            s = set()
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    for t in r.get("track_ids", []) or []:
                        if t is not None and int(t) >= 0:
                            s.add(int(t))
            c2t[cam] = s
        n_ent = n_bad = 0
        per_cam = {}
        for cam in CAMS:
            p = os.path.join(d, f"net_tracks_{cam}.json")
            if not os.path.exists(p):
                continue
            dd = json.load(open(p, encoding="utf-8"))
            for e in dd["entities"]:
                n_ent += 1
                if any(t not in c2t[cam] for t in e["tracks"]):
                    n_bad += 1
                    per_cam[cam] = per_cam.get(cam, 0) + 1
        entry["members_alive_in_clean2"] = dict(n_entities=n_ent,
                                                n_with_missing_tracks=n_bad, per_cam=per_cam)
        chain["L2_entities"][name] = entry
    for tag, (p, which) in relay_files.items():
        if os.path.exists(p):
            chain["L3_relay"][tag] = dict(file=os.path.relpath(p, ROOT), md5=md5(p),
                                          upstream_entities=which,
                                          dangling_nodes=rep["dangling"][tag]["n_dangling_nodes"])
    evals = {
        "c0_v5": os.path.join(SRC, "analysis", "c0", "c0_baseline_v5.json"),
        "c0_v4": os.path.join(SRC, "analysis", "c0", "c0_baseline_v4_clean2.json"),
        "c0_v3": os.path.join(SRC, "analysis", "c0", "c0_baseline_v3_clean2.json"),
    }
    for tag, p in evals.items():
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            m = d.get("meta", {})
            chain["L4_eval"][tag] = dict(file=os.path.relpath(p, ROOT), md5=md5(p),
                                         edges_source=m.get("edges_source"),
                                         nets_dir=m.get("nets_dir"), view=m.get("view"))
    rep["version_chain"] = chain
    print("  L1 检测层（逐框契约违规 count/scores/tracks）:")
    for v in ("raw", "clean", "clean2"):
        t = chain["L1_detections"][v]["contract_violations"]
        print(f"    {v}: {t['bad_count']}/{t['bad_scores']}/{t['bad_tracks']}")
    print("  L2 实体层:")
    for k, v in chain["L2_entities"].items():
        alive = v.get("members_alive_in_clean2", {})
        print(f"    {k}: view={v.get('view')} totals={v.get('totals') or '(无 manifest)'}")
        print(f"        实体成员在 clean2 中缺失: {alive.get('n_with_missing_tracks')}"
              f"/{alive.get('n_entities')} 个实体 {alive.get('per_cam') or ''}")
    print("  L3 接力层:")
    for k, v in chain["L3_relay"].items():
        print(f"    {k}: 上游实体={v['upstream_entities']} 悬空节点={v['dangling_nodes']}")
    print("  L4 评测层:")
    for k, v in chain["L4_eval"].items():
        print(f"    {k}: 边源={v['edges_source']}")
        print(f"        实体目录={v['nets_dir']} 视图={v['view']}")

    # ---- E：消费端口径一致性（WP0：默认必须指向规范实体图 stitch_clean2） ----
    print("\n" + "=" * 72)
    print("E. 消费端口径一致性：默认实体图目录 / 硬编码残留 / splits 键完整性")
    os.environ["DET_VIEW"] = "clean2"
    os.environ.pop("NET_DIR", None)
    for m in list(sys.modules):
        if m in ("det_view", "crosscam_relay", "c0_baseline", "make_splits_v3",
                 "unsup_reid_experiment"):
            del sys.modules[m]
    import det_view  # noqa: E402
    canonical = os.path.join(SRC, "analysis", "stitch_clean2")
    ok_net, msg_net = det_view.net_consistency()
    rep["consumers"] = {"canonical": os.path.relpath(canonical, ROOT),
                        "det_view_net_dir": os.path.relpath(det_view.net_dir(), ROOT),
                        "consistent": ok_net, "msg": msg_net}
    print(f"  {msg_net}")

    # 动态解析：可导入的消费端实际解析到的目录
    import importlib  # noqa: E402
    resolved = {}
    for mod in ("crosscam_relay", "c0_baseline"):
        try:
            mm = importlib.import_module(mod)
            resolved[mod] = mm.NET_DIR
        except Exception as ex:  # noqa: BLE001
            resolved[mod] = f"<import failed: {str(ex)[:60]}>"
    resolved["make_splits_v3"] = det_view.net_dir()
    for k, v in resolved.items():
        good = os.path.normcase(os.path.abspath(v)) == os.path.normcase(canonical)
        print(f"  {k:<18} NET_DIR = {os.path.relpath(v, ROOT) if v.startswith(ROOT) else v} "
              f"[{'OK' if good else '非规范'}]")
    rep["consumers"]["resolved"] = {k: (os.path.relpath(v, ROOT) if v.startswith(ROOT) else v)
                                    for k, v in resolved.items()}

    # 静态扫描：五个消费端是否仍有硬编码旧实体目录的读取（写 legacy 的除外）
    import re  # noqa: E402
    scan = {
        "tools/make_splits_v3.py": r'"stitch"\s*,\s*(f?"net_tracks|"manifest")',
        "tools/relay_missing_recheck.py": r'"stitch"',
        "tools/unsup_reid_experiment.py": r'"stitch"\s*,\s*(f?"net_tracks|"manifest")',
        "tools/build_handoff_pkg.py": r'"analysis/stitch/net_tracks',
        "tools/crosscam_relay.py": r'"stitch"\)',
    }
    rep["consumers"]["static_scan"] = {}
    for rel, pat in scan.items():
        p = os.path.join(ROOT, rel)
        txt = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
        hits = [ln for ln in txt.splitlines()
                if re.search(pat, ln) and not ln.strip().startswith("#")]
        rep["consumers"]["static_scan"][rel] = hits
        print(f"  {rel:<38} 硬编码残留读取: {len(hits)}"
              + (f"  ！{hits[:2]}" if hits else "  [OK]"))

    # splits 键完整性：test/val/train 轨必须存在于对应视图；实体必须存在于规范实体图
    sp = os.path.join(SRC, "analysis", "splits_v5.json")
    if os.path.exists(sp):
        sd = json.load(open(sp, encoding="utf-8"))
        ents5 = {}
        for cam in CAMS:
            d = json.load(open(os.path.join(canonical, f"net_tracks_{cam}.json"),
                               encoding="utf-8"))
            ents5[cam] = {e["eid"] for e in d["entities"]}
        miss_tracks, miss_ents = {}, {}
        for split in ("test", "val", "train"):
            for cam, tids in sd[split]["tracks"].items():
                bad = sorted(set(int(t) for t in tids) - c2t[cam])
                if bad:
                    miss_tracks[f"{split}:{cam}"] = len(bad)
        for split in ("test", "val"):
            for cam, eids in sd[split]["entities"].items():
                bad = sorted(set(eids) - ents5[cam])
                if bad:
                    miss_ents[f"{split}:{cam}"] = len(bad)
        rep["consumers"]["splits_v5"] = dict(
            file=os.path.relpath(sp, ROOT),
            meta={k: sd["meta"].get(k) for k in
                  ("det_view", "net_dir", "frozen_sha1_16",
                   "frozen_test_tracks_sha1_16", "frozen_splits_sha1_16")},
            n_test=sum(len(v) for v in sd["test"]["tracks"].values()),
            n_val=sum(len(v) for v in sd["val"]["tracks"].values()),
            n_train=sum(len(v) for v in sd["train"]["tracks"].values()),
            missing_tracks_in_view=miss_tracks, missing_entities=miss_ents)
        m = rep["consumers"]["splits_v5"]["meta"]
        print(f"  splits_v5: test/val/train = "
              f"{rep['consumers']['splits_v5']['n_test']}/"
              f"{rep['consumers']['splits_v5']['n_val']}/"
              f"{rep['consumers']['splits_v5']['n_train']}")
        print(f"    口径: DET_VIEW={m['det_view']} net_dir={m['net_dir']}")
        print(f"    冻结哈希: 评估对={m['frozen_sha1_16']} test轨集={m['frozen_test_tracks_sha1_16']}")
        print(f"    轨不存在于 clean2 的键: {miss_tracks or '{}'}  [OK if empty]")
        print(f"    实体不存在于 stitch_clean2 的键: {miss_ents or '{}'}  [OK if empty]")
    else:
        print("  splits_v5.json: MISSING")

    json.dump(rep, open(os.path.join(OUT, "data_chain_audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n[out] {os.path.join(OUT, 'data_chain_audit.json')}")
    bad_total = rep["contract"]["clean2"]["_total"]
    print("\n结论: clean2 逐框契约违规="
          f"{bad_total['bad_count']}/{bad_total['bad_scores']}/{bad_total['bad_tracks']} "
          f"（0=通过）；clean(v1) 违规="
          f"{rep['contract']['clean']['_total']['bad_count']} 行 count 失配")
    print("  悬空引用（对文件实体全集）: "
          + "；".join(f"{k}={v['n_dangling_vs_raw_entities']}"
                      for k, v in rep["dangling"].items())
          + " —— 需以 v5 为主口径")
    cons = rep.get("consumers", {})
    print(f"  消费端口径: {cons.get('msg', '')}")
    sp5 = cons.get("splits_v5")
    if sp5:
        print(f"  splits_v5 键完整性: 缺轨={sp5['missing_tracks_in_view'] or 0} "
              f"缺实体={sp5['missing_entities'] or 0}（0=通过）")


if __name__ == "__main__":
    main()
