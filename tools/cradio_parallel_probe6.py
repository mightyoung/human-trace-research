# -*- coding: utf-8 -*-
"""probe6: cradio_v3 并行化 Phase 2 深挖（L20，GPU1）。

章节：
  S0 现状批路径（legacy patched）分段画像 + 无 sync 基准吞吐
  S1 模板 dump：dump st.encode 内部传给 am 的 input_ids/attention_mask（跳 patchify 的前提）
  S2 F2 快速路径：手动组装（模板 expand + dict 直喂 am）全绕开 ST → cos + 分段 + 吞吐
  S3 语言塔 bf16 / fp16：cos + 吞吐（在 F2 路径上）
  S4 CPU(rproc)-GPU 预取重叠：吞吐对比
  S5 cv2 预处理替代 rproc：cos + 吞吐（可选）
"""
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import cv2
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from sentence_transformers import SentenceTransformer

ROOT = "/home/r4900g3/yhh/human-count/crosscam_run"
RADIO_DIR = os.path.join(ROOT, "cradio", "C-RADIOv4-SO400M")
EMB_PATH = os.path.join(ROOT, "embgemma", "model", "master")
W_P = os.path.join(ROOT, "cradio_scale_v3", "radio_W_scale.npy")
DATA = "/share/data4/yhh/data/human-count"
DEV = os.environ.get("FUSION_DEV", "cuda:1")
OUT_P = os.path.join(ROOT, "cradio_parallel_probe6.json")

pc = time.perf_counter


def log(*a):
    print(*a, flush=True)


def sync():
    torch.cuda.synchronize(DEV)   # 关键：必须同步实际计算设备（cuda:1），默认参数只会同 cuda:0


def r5(x):
    try:
        return round(float(x), 5)
    except Exception:
        return None


# ================= 数据准备（probe5 同款） =================

def prep_crop(frame, box, out_h=224, pad=0.08):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad)); Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad)); Y2 = min(frame.shape[0], int(y2 + h * pad))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0 or c.shape[0] < 12 or c.shape[1] < 8:
        return None
    s = out_h / c.shape[0]
    nw = max(1, int(round(c.shape[1] * s)))
    c = cv2.resize(c, (nw, out_h))
    if nw >= out_h:
        off = (nw - out_h) // 2
        c = c[:, off:off + out_h]
    else:
        padw = out_h - nw
        c = cv2.copyMakeBorder(c, 0, 0, padw // 2, padw - padw // 2, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def load_tracks(jsonl_path):
    t = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            for i, b in enumerate(r.get("boxes", []) or []):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    t.setdefault(int(tids[i]), []).append((int(r["frame"]), b))
    return t


def make_crops(cam="316", n_tracks=4, per_track=8):
    tracks = load_tracks(f"{DATA}/inference-out/{cam}_detections.jsonl")
    cap = cv2.VideoCapture(f"{DATA}/video-6way/{cam}.mp4")
    crops = []
    for tid in sorted(tracks)[:n_tracks * 2]:
        if len(crops) >= n_tracks * per_track:
            break
        rows = tracks[tid]
        if len(rows) < 4:
            continue
        idx = np.linspace(0, len(rows) - 1, min(per_track, len(rows))).astype(int)
        for j in idx:
            fr, box = rows[j]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, frame = cap.read()
            if not ok:
                continue
            c = prep_crop(frame, box)
            if c is not None:
                crops.append(c)
    cap.release()
    return crops[:n_tracks * per_track]


def rebuild_image(pixel_values, position_ids):
    pv = pixel_values[0]; ids = position_ids[0]
    valid = (ids != -1).all(dim=-1)
    pv = pv[valid]; ids = ids[valid]
    xs, ys = ids[:, 0], ids[:, 1]
    gw = int(xs.max()) + 1; gh = int(ys.max()) + 1
    order = torch.argsort(ys * gw + xs)
    patches = pv[order].reshape(gh, gw, 16, 16, 3)
    return patches.permute(0, 2, 1, 3, 4).reshape(gh * 16, gw * 16, 3)


# ================= 计时与状态 =================

PROF = {"on": False}
ACC = {}          # 分段累计（仅 PROF.on 时更新）
ST = {"proc_proxy": None, "am_forward_orig": None, "lm_orig": None, "cap": []}


def acc_add(k, dt):
    if PROF["on"]:
        ACC[k] = ACC.get(k, 0.0) + dt


def main():
    res = {}
    log("[S0] crops ...")
    crops = make_crops("316", 4, 8)
    log(f"    crops n={len(crops)} shape={tuple(crops[0].shape) if crops else None}")

    log("[S0] load models ...")
    m_radio = AutoModel.from_pretrained(RADIO_DIR, trust_remote_code=True).to(DEV).eval()
    rproc = CLIPImageProcessor.from_pretrained(RADIO_DIR)
    st = SentenceTransformer(EMB_PATH, device=DEV, model_kwargs={"torch_dtype": "float32"})
    am = getattr(st[0], "auto_model", None) or getattr(st[0], "model", None)
    Wt = torch.from_numpy(np.load(W_P).astype(np.float32)).to(DEV)
    log(f"    image_token_id={am.config.image_token_id} pad={am.config.text_config.pad_token_id} "
        f"vram={torch.cuda.memory_allocated(DEV)/1e9:.2f}GB")

    BATCH = 16
    N = len(crops)

    # ---------- processor 类级 hook（计时；绕开"引用被缓存"导致 probe6a gemma_proc=0"） ----------
    PROC_ACC = {"n": 0, "t": 0.0}
    try:
        _Pcls = type(st[0].processor)
        _orig_call = _Pcls.__call__
        def _timed_call(self, *a, **k):
            t = pc()
            r = _orig_call(self, *a, **k)
            PROC_ACC["n"] += 1
            PROC_ACC["t"] += pc() - t
            return r
        _Pcls.__call__ = _timed_call
        _attrs = [x for x in vars(st[0]).keys() if "proc" in x.lower() or "feat" in x.lower()]
        log(f"    processor hook: {_Pcls.__name__}  st[0] 相关属性={_attrs}")
        res["processor_cls"] = _Pcls.__name__
    except Exception as ex:
        log(f"    processor hook 失败: {ex}")

    # ---------- radio bridge（共享） ----------
    def radio_bridge(pix):
        t = pc()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = m_radio(pix)
            sp = out[1]
            B2, L, D = sp.shape
            g = sp.reshape(B2, 32, 32, D)
            pp = g.reshape(B2, 16, 2, 16, 2, D).mean(dim=(2, 4)).reshape(B2, 256, D)
        sync()
        acc_add("radio", pc() - t)
        t = pc()
        z2 = (pp.reshape(B2 * 256, D).float() @ Wt)
        po = am.embed_vision(inputs_embeds=z2)
        sync()
        acc_add("bridge_ev", pc() - t)
        per = po.shape[0] // B2
        parts = list(torch.split(po, [per] * B2))
        return BaseModelOutputWithPooling(last_hidden_state=z2, pooler_output=parts)

    # ---------- legacy patched ----------
    SHAPES = []

    def patched_legacy(pixel_values, image_position_ids=None, **kw):
        if len(SHAPES) < 8:
            SHAPES.append((tuple(pixel_values.shape), tuple(image_position_ids.shape)
                           if image_position_ids is not None else None))
        if PROF["on"]:
            sync()                      # 隔离前批 GPU 残留，rebuild 读数=纯 CPU
            t = pc()
        else:
            t = pc()
        pv = pixel_values.detach().cpu()
        ip = image_position_ids.detach().cpu()
        imgs = []
        for b in range(pv.shape[0]):
            img = rebuild_image(pv[b:b + 1], ip[b:b + 1])
            imgs.append((img.numpy() * 255).clip(0, 255).astype(np.uint8))
        acc_add("rebuild", pc() - t)
        t = pc()
        pix = rproc(images=[Image.fromarray(x) for x in imgs], return_tensors="pt",
                    do_resize=True, do_center_crop=False, do_normalize=False).pixel_values.to(DEV)
        acc_add("rproc", pc() - t)
        return radio_bridge(pix)

    # ---------- lm 计时 ----------
    lm_orig = am.language_model.forward
    def lm_timed(*a, **k):
        t = pc()
        r = lm_orig(*a, **k)
        sync()
        acc_add("lm", pc() - t)
        return r
    am.language_model.forward = lm_timed

    # ---------- am.forward 捕获（S1） ----------
    ST["am_forward_orig"] = am.forward
    def am_cap(*a, **k):
        if len(ST["cap"]) < 10 and k.get("input_ids") is not None and k["input_ids"].shape[0] == 1:
            ids = k["input_ids"][0].detach().cpu()
            msk = k.get("attention_mask")
            d = {
                "L": int(ids.shape[0]),
                "ids": ids.clone(),
                "mask": (msk[0].detach().cpu().clone() if msk is not None else None),
                "img_pos_count": int((ids == am.config.image_token_id).sum()),
                "head": ids[:4].tolist(),
                "tail": ids[-4:].tolist(),
                "has_pos_ids": k.get("position_ids") is not None,
            }
            ST["cap"].append(d)
        return ST["am_forward_orig"](*a, **k)
    am.forward = am_cap

    am.get_image_features = patched_legacy

    # ================= S0: 基准 + 分段画像 =================
    log("[S0] legacy baseline ...")
    try:
        st.encode(crops[:BATCH], batch_size=BATCH, show_progress_bar=False)
        sync()
        # 无 sync 轮（真实吞吐）
        t0 = pc()
        e_base = np.asarray(st.encode(crops, batch_size=BATCH, show_progress_bar=False))
        sync()
        t_nosync = pc() - t0
        res["legacy_ms_per_img"] = r5(t_nosync / N * 1000)
        log(f"    无 sync: {t_nosync/N*1000:.1f} ms/图 ({N/t_nosync:.1f} 图/s)")
        # 分段轮（PROF 开）
        ACC.clear(); PROC_ACC["n"] = 0; PROC_ACC["t"] = 0.0; PROF["on"] = True
        t0 = pc()
        e_base2 = np.asarray(st.encode(crops, batch_size=BATCH, show_progress_bar=False))
        sync()
        t_seg = pc() - t0
        PROF["on"] = False
        coss = [float(np.dot(e_base[i], e_base2[i])) for i in range(N)]
        res["legacy_seg_total_ms_per_img"] = r5(t_seg / N * 1000)
        res["legacy_seg_cos_min"] = r5(min(coss))
        res["legacy_shapes"] = [[list(a), list(b) if b else None] for a, b in SHAPES[:4]]
        seg = {k: r5(v / N * 1000) for k, v in ACC.items()}
        seg[f"processor[{PROC_ACC['n']}次]"] = r5(PROC_ACC["t"] / N * 1000)
        seg["tail(装配/池化/杂项)"] = r5(t_seg / N * 1000 - sum(seg.values()))
        res["legacy_segments_ms_per_img"] = seg
        log(f"    分段画像(ms/图): {json.dumps(seg, ensure_ascii=False)}")
    except Exception as ex:
        import traceback
        log(f"    [S0 失败] {type(ex).__name__}: {str(ex)[:300]}")
        log(traceback.format_exc()[:1200])
        res["s0_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
        e_base = None

    # ================= S1: 模板 dump =================
    log("[S1] template dump ...")
    TMPL = None
    try:
        ST["cap"].clear()
        test_imgs = [
            ("224sq", crops[0]),
            ("128sq", cv2.resize(crops[1], (128, 128))),
            ("320sq", cv2.resize(crops[2], (320, 320))),
        ]
        for tag, im in test_imgs:
            st.encode({"image": im}, normalize_embeddings=True, show_progress_bar=False)
        sync()
        # 非方图单独测（probe6a 里 300x200 会让 processor 报非法分辨率）
        try:
            st.encode({"image": cv2.resize(crops[3], (300, 200))},
                      normalize_embeddings=True, show_progress_bar=False)
            res["non_square_ok"] = True
        except Exception as ex2:
            res["non_square_error"] = str(ex2)[:150]
            ST["cap"] = ST["cap"][:len(test_imgs)]   # 丢弃可能写入的半条
        info = [{"L": d["L"], "img_pos": d["img_pos_count"], "head": d["head"], "tail": d["tail"],
                 "has_pos_ids": d["has_pos_ids"]} for d in ST["cap"]]
        res["template_dumps"] = info
        log(f"    dumps: {json.dumps(info, ensure_ascii=False)}")
        log(f"    非方图: {res.get('non_square_ok') or res.get('non_square_error')}")
        # 一致性：L 与 ids 内容是否一致
        d0 = ST["cap"][0]
        same = all(d["L"] == d0["L"] and bool((d["ids"] == d0["ids"]).all()) for d in ST["cap"][1:])
        res["template_invariant"] = same
        log(f"    模板不变性(跨尺寸): {same}")
        TMPL = {"ids": d0["ids"].unsqueeze(0), "mask": d0["mask"].unsqueeze(0) if d0["mask"] is not None else None}
    except Exception as ex:
        import traceback
        log(f"    [S1 失败] {type(ex).__name__}: {str(ex)[:300]}")
        log(traceback.format_exc()[:800])
        res["s1_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"

    # ================= S2: F2 快速路径 =================
    log("[S2] F2 fast path ...")
    e_fast = None
    if TMPL is not None and e_base is not None:
        try:
            FAST = {"crops": None, "pix": None}

            def patched_fast(pixel_values, image_position_ids=None, **kw):
                if FAST.get("pix") is None:
                    t = pc()
                    pix = rproc(images=[Image.fromarray(x) for x in FAST["crops"]], return_tensors="pt",
                                do_resize=True, do_center_crop=False, do_normalize=False).pixel_values.to(DEV)
                    acc_add("rproc", pc() - t)
                else:
                    pix = FAST["pix"]
                    FAST["pix"] = None
                return radio_bridge(pix)

            am.get_image_features = patched_fast

            ids1 = TMPL["ids"].to(DEV)          # [1,L]
            mask1 = (TMPL["mask"].to(DEV) if TMPL["mask"] is not None
                     else torch.ones_like(TMPL["ids"]).to(DEV))
            img1 = (ids1 == am.config.image_token_id)
            pad_id = am.config.text_config.pad_token_id

            def dummy_pv(B):
                return torch.zeros(B, 1, 1, device=DEV)

            def fast_pool(lh, mask):
                m = mask.unsqueeze(-1).to(torch.float32)
                v1 = (lh.float() * m).sum(1) / m.sum(1)                       # mean all
                mt = (~img1).expand_as(mask).unsqueeze(-1).float()
                v2 = (lh.float() * mt).sum(1) / mt.sum(1).clamp(min=1)        # mean text only
                v3 = lh.float()[torch.arange(lh.shape[0]), mask.sum(1) - 1]   # last valid
                nz = lambda x: torch.nn.functional.normalize(x, dim=-1)
                return {"mean_all": nz(v1), "mean_text": nz(v2), "last": nz(v3)}

            def fast_encode(crops_list, bs=BATCH):
                embs = []
                FAST["pix"] = None
                for i in range(0, len(crops_list), bs):
                    chunk = crops_list[i:i + bs]
                    B = len(chunk)
                    FAST["crops"] = chunk
                    ids = ids1.expand(B, -1)
                    mask = mask1.expand(B, -1)
                    with torch.no_grad():
                        out = am(input_ids=ids, attention_mask=mask, pixel_values=dummy_pv(B),
                                 image_position_ids=None, return_dict=True)
                        lh = out.last_hidden_state
                    embs.append(fast_pool(lh, mask))
                return {k: np.concatenate([e[k].cpu().numpy() for e in embs], 0) for k in embs[0]}

            # 预热
            fast_encode(crops[:BATCH])
            sync()
            # 无 sync 轮
            t0 = pc()
            ef = fast_encode(crops)
            sync()
            t_fast_nosync = pc() - t0
            res["fast_ms_per_img"] = r5(t_fast_nosync / N * 1000)
            log(f"    无 sync: {t_fast_nosync/N*1000:.1f} ms/图 ({N/t_fast_nosync:.1f} 图/s)")
            # cos 对照（三种池化变体）
            cos_tab = {}
            for k, arr in ef.items():
                cc = [float(np.dot(arr[i], e_base[i])) for i in range(N)]
                cos_tab[k] = {"mean": r5(np.mean(cc)), "min": r5(np.min(cc))}
            res["fast_cos_vs_base"] = cos_tab
            log(f"    cos vs base: {json.dumps(cos_tab, ensure_ascii=False)}")
            best = max(cos_tab, key=lambda k: cos_tab[k]["mean"])
            res["fast_best_pool"] = best
            e_fast = ef[best]
            # 分段轮
            ACC.clear(); PROF["on"] = True
            t0 = pc()
            ef2 = fast_encode(crops)
            sync()
            t_seg = pc() - t0
            PROF["on"] = False
            seg = {k: r5(v / N * 1000) for k, v in ACC.items()}
            seg["tail(组装/池化/杂项)"] = r5(t_seg / N * 1000 - sum(seg.values()))
            res["fast_segments_ms_per_img"] = seg
            res["fast_seg_total_ms_per_img"] = r5(t_seg / N * 1000)
            log(f"    分段画像(ms/图): {json.dumps(seg, ensure_ascii=False)}")
            # dummy 形状确认
            res["dummy_pv_shape"] = [1, 1, 1]
        except Exception as ex:
            import traceback
            log(f"    [S2 失败] {type(ex).__name__}: {str(ex)[:300]}")
            log(traceback.format_exc()[:1200])
            res["s2_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"

    # ================= S3: 语言塔低精度 =================
    if e_fast is not None:
        for tag, dt in [("bf16", torch.bfloat16), ("fp16", torch.float16)]:
            log(f"[S3] lm {tag} ...")
            try:
                am.language_model.to(dt)
                sync()
                fast_encode(crops[:BATCH]); sync()
                t0 = pc()
                ef = fast_encode(crops)
                sync()
                t = pc() - t0
                cc = [float(np.dot(ef[res["fast_best_pool"]][i], e_base[i])) for i in range(N)]
                res[f"lm_{tag}_ms_per_img"] = r5(t / N * 1000)
                res[f"lm_{tag}_cos"] = {"mean": r5(np.mean(cc)), "min": r5(np.min(cc))}
                log(f"    {t/N*1000:.1f} ms/图  cos mean={np.mean(cc):.5f} min={np.min(cc):.5f}")
            except Exception as ex:
                import traceback
                log(f"    [S3 {tag} 失败] {type(ex).__name__}: {str(ex)[:300]}")
                log(traceback.format_exc()[:800])
                res[f"s3_{tag}_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"

    # ================= S4: CPU(rproc)-GPU 重叠（三变体） =================
    if e_fast is not None:
        log("[S4] overlap variants ...")
        try:
            def prep_pix_cpu(chunk):
                return rproc(images=[Image.fromarray(x) for x in chunk], return_tensors="pt",
                             do_resize=True, do_center_crop=False, do_normalize=False).pixel_values

            chunks = [crops[i:i + BATCH] for i in range(0, N, BATCH)]
            ids1 = TMPL["ids"].to(DEV)
            mask1 = (TMPL["mask"].to(DEV) if TMPL["mask"] is not None
                     else torch.ones_like(TMPL["ids"]).to(DEV))

            def fast_one(chunk, pix_cpu):
                B = len(chunk)
                FAST["crops"] = chunk
                FAST["pix"] = (pix_cpu.to(DEV) if pix_cpu is not None else None)
                ids = ids1.expand(B, -1); mask = mask1.expand(B, -1)
                with torch.no_grad():
                    out = am(input_ids=ids, attention_mask=mask,
                             pixel_values=torch.zeros(B, 1, 1, device=DEV),
                             image_position_ids=None, return_dict=True)
                m = mask.unsqueeze(-1).to(torch.float32)
                return ((out.last_hidden_state.float() * m).sum(1) / m.sum(1))

            def run_serial():
                for ch in chunks:
                    fast_one(ch, None)

            def run_prefetch():
                ex = ThreadPoolExecutor(2)
                futs = [ex.submit(prep_pix_cpu, ch) for ch in chunks]
                for i, ch in enumerate(chunks):
                    fast_one(ch, futs[i].result())
                ex.shutdown()

            def run_pipelined():
                ex = ThreadPoolExecutor(2)
                futs = [ex.submit(prep_pix_cpu, ch) for ch in chunks]
                outs = []
                for i, ch in enumerate(chunks):
                    B = len(ch)
                    FAST["crops"] = ch
                    FAST["pix"] = futs[i].result().to(DEV)
                    ids = ids1.expand(B, -1); mask = mask1.expand(B, -1)
                    with torch.no_grad():
                        o = am(input_ids=ids, attention_mask=mask,
                               pixel_values=torch.zeros(B, 1, 1, device=DEV),
                               image_position_ids=None, return_dict=True)
                    outs.append((o, mask))
                embs = []
                for o, mask in outs:
                    m = mask.unsqueeze(-1).to(torch.float32)
                    embs.append(((o.last_hidden_state.float() * m).sum(1) / m.sum(1)))
                torch.cat(embs, 0)
                ex.shutdown()

            times = {}
            for tag, fn in [("serial", run_serial), ("prefetch", run_prefetch),
                            ("pipelined", run_pipelined)]:
                fn(); sync()                     # 预热
                tt = []
                for _ in range(2):
                    sync(); t0 = pc()
                    fn()
                    sync(); tt.append(pc() - t0)
                times[tag] = min(tt)
                log(f"    {tag:<9} {min(tt)/N*1000:7.2f} ms/图  ({N/min(tt):5.1f} 图/s)")
            base = times["serial"]
            res["s4_ms_per_img"] = {k: r5(v / N * 1000) for k, v in times.items()}
            res["s4_gain_vs_serial_pct"] = {k: r5((base - v) / base * 100) for k, v in times.items()}
        except Exception as ex:
            import traceback
            log(f"    [S4 失败] {type(ex).__name__}: {str(ex)[:300]}")
            log(traceback.format_exc()[:800])
            res["s4_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"

    # ================= S5: cv2 预处理替代 rproc =================
    if e_fast is not None:
        log("[S5] cv2 preprocess ...")
        try:
            def pix_cv2(imgs):
                xs = [cv2.resize(im, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                      for im in imgs]
                return torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()

            ids1 = TMPL["ids"].to(DEV)
            mask1 = (TMPL["mask"].to(DEV) if TMPL["mask"] is not None
                     else torch.ones_like(TMPL["ids"]).to(DEV))

            def pool_of(lh, mask):
                m = mask.unsqueeze(-1).to(torch.float32)
                if res.get("fast_best_pool") == "mean_text":
                    imgmask = (TMPL["ids"].to(DEV) == am.config.image_token_id)
                    mt = (~imgmask).expand_as(mask).unsqueeze(-1).float()
                    return (lh.float() * mt).sum(1) / mt.sum(1).clamp(min=1)
                if res.get("fast_best_pool") == "last":
                    return lh.float()[torch.arange(lh.shape[0]), mask.sum(1) - 1]
                return (lh.float() * m).sum(1) / m.sum(1)

            def fast_encode_cv2(crops_list, bs=BATCH):
                embs = []
                FAST["crops"] = None
                for i in range(0, len(crops_list), bs):
                    chunk = crops_list[i:i + bs]
                    B = len(chunk)
                    FAST["pix"] = pix_cv2(chunk).to(DEV)
                    ids = ids1.expand(B, -1); mask = mask1.expand(B, -1)
                    with torch.no_grad():
                        out = am(input_ids=ids, attention_mask=mask,
                                 pixel_values=torch.zeros(B, 1, 1, device=DEV),
                                 image_position_ids=None, return_dict=True)
                        lh = out.last_hidden_state
                    embs.append(pool_of(lh, mask))
                return torch.cat(embs, 0).cpu().numpy()

            # 同口径对照（fp16 lm 下，rproc 路径）
            fast_encode(crops[:BATCH])  # warmup
            e_ref16 = fast_encode(crops)[res.get("fast_best_pool", "mean_all")]
            e_ref16 = np.asarray(e_ref16)
            fast_encode_cv2(crops[:BATCH]); sync()
            t0 = pc()
            ef_cv2 = fast_encode_cv2(crops)
            sync()
            t = pc() - t0
            cc_base = [float(np.dot(ef_cv2[i], e_base[i])) for i in range(N)]
            cc_ref = [float(np.dot(ef_cv2[i], e_ref16[i])) for i in range(N)]
            res["cv2_ms_per_img"] = r5(t / N * 1000)
            res["cv2_cos_vs_base"] = {"mean": r5(np.mean(cc_base)), "min": r5(np.min(cc_base))}
            res["cv2_cos_vs_rproc16"] = {"mean": r5(np.mean(cc_ref)), "min": r5(np.min(cc_ref))}
            log(f"    {t/N*1000:.1f} ms/图  cos(vs base)={np.mean(cc_base):.5f} "
                f"cos(vs 同口径 rproc)={np.mean(cc_ref):.5f}")
        except Exception as ex:
            import traceback
            log(f"    [S5 失败] {type(ex).__name__}: {str(ex)[:300]}")
            log(traceback.format_exc()[:800])
            res["s5_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"

    with open(OUT_P, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    log(f"[DONE] -> {OUT_P}")
    log(json.dumps(res, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
