// C ABI for the native ONNX ReID inference path. Used by the Python eval
// pipeline (via ctypes) to populate the embedding cache with the same C++
// inference path that the native trackers use at replay time.
//
// AABB and OBB preprocessing entry points accept image-coordinate
// ``[x1,y1,x2,y2]`` and ``[cx,cy,w,h,theta_rad]`` rows respectively. Output
// embeddings are L2-normalized row-major floats of shape
// ``(n_boxes, feature_dim)``.

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

// Read the ONNX graph's NCHW input specification without running inference.
// ``out_batch`` receives 0 for a dynamic batch dimension, otherwise the
// graph's required fixed batch. Height and width are always static positive
// dimensions for accepted native ReID models.
BOXMOT_REID_CAPI int boxmot_reid_capi_input_spec(
    void* handle,
    int* out_batch,
    int* out_channels,
    int* out_height,
    int* out_width);

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

// ---- Staged feature extraction -------------------------------------------
//
// The three calls below split the work performed by
// ``boxmot_reid_capi_compute_features`` into the same preprocess/process/
// postprocess buckets exposed by the Python ``BaseModelBackend`` surface so
// host languages can attribute timing to each stage. They must be invoked in
// order on the same handle and are not thread-safe with respect to other
// calls touching the same handle (intermediate state is stored inside the
// handle to avoid copying the crop blob and raw feature buffer across the
// FFI boundary).

// Stage 1: crop + resize + standardise into the handle's internal blob.
BOXMOT_REID_CAPI int boxmot_reid_capi_preprocess(
    void* handle,
    const float* boxes_xyxy,
    int n_boxes,
    const std::uint8_t* image_data,
    int image_rows,
    int image_cols,
    int image_channels);

// Stage 1 (OBB): rectify each oriented rectangle before resize/standardize.
BOXMOT_REID_CAPI int boxmot_reid_capi_preprocess_obb(
    void* handle,
    const float* boxes_xywha,
    int n_boxes,
    const std::uint8_t* image_data,
    int image_rows,
    int image_cols,
    int image_channels);

// Stage 2: run the model forward pass over the staged blob, writing raw
// (un-normalised) features into the handle's internal output buffer.
BOXMOT_REID_CAPI int boxmot_reid_capi_process(void* handle);

// Stage 3: L2-normalise the staged raw features into ``out_features``.
BOXMOT_REID_CAPI int boxmot_reid_capi_postprocess(
    void* handle,
    float* out_features,
    int out_capacity_floats);

// Returns the last error message produced on the calling thread. The pointer
// remains valid until the next call into the C ABI on the same thread.
BOXMOT_REID_CAPI const char* boxmot_reid_capi_last_error(void);

#ifdef __cplusplus
}
#endif
