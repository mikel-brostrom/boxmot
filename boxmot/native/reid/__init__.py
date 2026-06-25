"""Native ReID Python bindings."""

from boxmot.native.reid.capi import CppOnnxReID, ensure_reid_capi_library

__all__ = (
    "CppOnnxReID",
    "ensure_reid_capi_library",
)
