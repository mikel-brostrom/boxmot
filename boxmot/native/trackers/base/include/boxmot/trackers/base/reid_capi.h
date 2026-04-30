// C ABI for the native ONNX ReID inference path. Used by the Python eval
// pipeline (via ctypes) to populate the embedding cache with the same C++
// inference path that the native trackers use at replay time.
//
// All boxes are AABBs in image coordinates ``[x1, y1, x2, y2]``. Output
// embeddings are L2-normalized row-major floats of shape ``(n_boxes,
// feature_dim)``. The feature dimension is fixed for a given model and can be
// queried up-front via ``boxmot_reid_capi_feature_dim``.

#pragma once

#include <cstdint>

#if defined(_WIN32)
#  if defined(BOXMOT_REID_CAPI_BUILDING_DLL)
#    define BOXMOT_REID_CAPI __declspec(dllexport)
#  else
#    define BOXMOT_REID_CAPI __declspec(dllimport)
#  endif
#else
#  define BOXMOT_REID_CAPI __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Create a native ReID handle from an ONNX model file.
//
// ``model_path``: UTF-8 path to a ``.onnx`` ReID model.
// ``preprocess``: optional preprocess name (``"resize"``, ``"resize_pad"``,
// ...). Pass ``nullptr`` to use the default (``"resize_pad"``).
// ``out_handle``: receives the new opaque handle on success.
// Returns 1 on success, 0 on failure (call ``boxmot_reid_capi_last_error`` for
// the error message).
BOXMOT_REID_CAPI int boxmot_reid_capi_create(
    const char* model_path,
    const char* preprocess,
    void** out_handle);

// Destroy a handle previously created via ``boxmot_reid_capi_create``.
BOXMOT_REID_CAPI void boxmot_reid_capi_destroy(void* handle);

// Probe the model's output feature dimension. Triggers a single dummy forward
// pass on the first invocation, then returns the cached value.
BOXMOT_REID_CAPI int boxmot_reid_capi_feature_dim(void* handle, int* out_feature_dim);

// Compute L2-normalized features for ``n_boxes`` AABB boxes against a single
// image. Output is written contiguously row-major into ``out_features``.
BOXMOT_REID_CAPI int boxmot_reid_capi_compute_features(
    void* handle,
    const float* boxes_xyxy,  // shape (n_boxes, 4) row-major
    int n_boxes,
    const std::uint8_t* image_data,
    int image_rows,
    int image_cols,
    int image_channels,
    float* out_features,
    int out_capacity_floats);

// Returns the last error message produced on the calling thread. The pointer
// remains valid until the next call into the C ABI on the same thread.
BOXMOT_REID_CAPI const char* boxmot_reid_capi_last_error(void);

#ifdef __cplusplus
}
#endif
