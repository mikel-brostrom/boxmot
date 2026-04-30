# Backwards-compatibility shim. Retained so external callers that include
# this file directly continue to work. New code should
# ``include(BoxMOTNative)`` instead, which exposes ``boxmot_add_native_tracker``
# and the up-to-date warning helper.
include(${CMAKE_CURRENT_LIST_DIR}/BoxMOTNative.cmake)
