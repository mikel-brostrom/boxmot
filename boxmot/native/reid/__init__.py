"""Low-level native ReID C-ABI bindings."""

from boxmot.native.reid.capi import ReIDLibrary, ensure_reid_capi_library, get_reid_capi_library

__all__ = (
    "ReIDLibrary",
    "ensure_reid_capi_library",
    "get_reid_capi_library",
)
