# -*- coding: utf-8 -*-
"""train_rfdetr.py — RF-DETR 单类 person 训练（rfdetr 1.9，PyTorch Lightning 栈）。

用法：
  python train_rfdetr.py small --epochs 80
  python train_rfdetr.py small --check                 # 只做 device 补丁/解析自检，不训练
  python train_rfdetr.py small --epochs 1 --batch 2 --device cuda:1 --out-dir .../_fixcheck

兼容说明（2026-10-14）：
  rfdetr 1.9.0.dev 的 `_resolve_trainer_device_kwargs("cuda:0")` 返回 devices=[0]（list），
  而 `training/trainer.py::_requests_multiple_devices()` 只处理 int/str → AttributeError（启动即崩）。
  本脚本在调用前打最小补丁：list/tuple 按“有效元素数 > 1”判定（与 PTL devices=[N] 语义一致）。
  若补丁未能生效，自动退避：去掉设备索引（"cuda:N" → "cuda"），走单卡默认（cuda:0）。
"""
import argparse

D = "/share/data4/yhh/data/human-count/det-bench"


def _patch_device_helper():
    """兼容 rfdetr 1.9.0.dev：devices 为 list[int] 时不再崩溃。返回是否已生效。"""
    _tn = None
    try:
        from rfdetr.training import trainer as _tn  # type: ignore
    except Exception:
        try:
            import rfdetr.training.trainer as _tn  # type: ignore
        except Exception:
            return False
    orig = getattr(_tn, "_requests_multiple_devices", None)
    if orig is None:
        return False
    if getattr(orig, "_list_compat", False):
        return True

    def _patched(devices, accelerator=None):
        if isinstance(devices, (list, tuple)):
            return len([d for d in devices if d is not None and str(d).strip() != ""]) > 1
        return orig(devices, accelerator)

    _patched._list_compat = True  # type: ignore[attr-defined]
    _tn._requests_multiple_devices = _patched
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["nano", "small", "medium", "large"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=str, default="auto", help='int 或 "auto"')
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--check", action="store_true", help="只做补丁/解析自检")
    args = ap.parse_args()

    applied = _patch_device_helper()
    print(f"[rfdetr] device patch applied: {applied}", flush=True)

    device = args.device
    if not applied and ":" in device:
        device = device.split(":")[0]
        print(f"[rfdetr] WARN patch 未生效，device 退避为 {device}", flush=True)

    import rfdetr

    if args.check:
        try:
            resolved = rfdetr.RFDETRSmall._resolve_trainer_device_kwargs(device)
            print(f"[check] resolve({device}) -> {resolved}", flush=True)
            from rfdetr.training import trainer as _t  # type: ignore
            print(f"[check] helper([0], 'gpu') -> {_t._requests_multiple_devices([0], 'gpu')}", flush=True)
            print(f"[check] patched flag: {getattr(_t._requests_multiple_devices, '_list_compat', False)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[check] FAILED: {type(e).__name__}: {e}", flush=True)
        return

    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
           "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}[args.variant]
    m = cls()

    batch = args.batch if args.batch == "auto" else int(args.batch)
    out_dir = args.out_dir or f"{D}/runs/rfdetr_{args.variant}"
    m.train(
        dataset_dir=f"{D}/dataset",
        dataset_file="roboflow",  # 我们的导出目录是 Roboflow 布局（train/valid + _annotations.coco.json）
        epochs=args.epochs,
        batch_size=batch,
        grad_accum_steps=4,
        output_dir=out_dir,
        device=device,
        notes={"experiment": "human-count det-bench pseudo-labels", "variant": args.variant},
    )
    print(f"[rfdetr] {args.variant} training finished", flush=True)


if __name__ == "__main__":
    main()
