# -*- coding: utf-8 -*-
"""probe_rfdetr_cfg.py — 打印 RF-DETR 各类模型的默认权重配置（不实例化、不下载）。"""
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall

for cls in [RFDETRSmall, RFDETRMedium, RFDETRLarge]:
    cfg = getattr(cls, "model_config", None)
    print(cls.__name__,
          "| model_name:", getattr(cfg, "model_name", None),
          "| pretrain:", getattr(cfg, "pretrain_weights", None),
          "| resolution:", getattr(cfg, "resolution", None))
