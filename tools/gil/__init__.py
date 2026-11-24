# -*- coding: utf-8 -*-
"""GIL（Global Identity Layer）统一存储层。

用法：
    from gil import GILStore
    st = GILStore("output_crosscam/analysis/gil/store_v1")
    st.upsert_camera("316", 2560, 1440)
    ...
    st.snapshot()

设计依据见 docs/gil-storage-design.md（含选型实测数据）。
"""
from .store import GILStore, CAMS, TIME_OFFS  # noqa: F401
from .vectors import VectorBank  # noqa: F401

__all__ = ["GILStore", "VectorBank", "CAMS", "TIME_OFFS"]
