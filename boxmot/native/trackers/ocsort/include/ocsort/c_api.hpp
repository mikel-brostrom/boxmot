#pragma once

#include <cstdint>

#if defined(_WIN32)
#  if defined(BOXMOT_OCSORT_BUILDING_DLL)
#    define BOXMOT_OCSORT_API __declspec(dllexport)
#  else
#    define BOXMOT_OCSORT_API __declspec(dllimport)
#  endif
#else
#  define BOXMOT_OCSORT_API __attribute__((visibility("default")))
#endif

extern "C" {

struct BoxMOTOCSORTConfig {
    float min_conf;
    float det_thresh;
    float iou_threshold;
    int max_age;
    int min_hits;
    int delta_t;
    int use_byte;
    float inertia;
    float q_xy_scaling;
    float q_s_scaling;
    int max_obs;
};

struct BoxMOTOCSORTHandle;

BOXMOT_OCSORT_API BoxMOTOCSORTHandle* boxmot_ocsort_create(const BoxMOTOCSORTConfig* config);
BOXMOT_OCSORT_API void boxmot_ocsort_destroy(BoxMOTOCSORTHandle* handle);
BOXMOT_OCSORT_API int boxmot_ocsort_reset(BoxMOTOCSORTHandle* handle);
BOXMOT_OCSORT_API int boxmot_ocsort_update(
    BoxMOTOCSORTHandle* handle,
    const float* dets,
    int det_rows,
    int det_cols,
    const std::uint8_t* image_data,
    int image_rows,
    int image_cols,
    int image_channels,
    float* out_tracks,
    int out_capacity_rows,
    int out_cols,
    int* out_rows,
    int* out_is_obb
);
BOXMOT_OCSORT_API const char* boxmot_ocsort_last_error();

}  // extern "C"
