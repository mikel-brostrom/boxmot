#pragma once

#include <cstdint>

#if defined(_WIN32)
#  if defined(BOXMOT_SFSORT_BUILDING_DLL)
#    define BOXMOT_SFSORT_API __declspec(dllexport)
#  else
#    define BOXMOT_SFSORT_API __declspec(dllimport)
#  endif
#else
#  define BOXMOT_SFSORT_API
#endif

extern "C" {

struct BoxMOTSFSORTConfig {
    float high_th;
    float match_th_first;
    float new_track_th;
    float low_th;
    float match_th_second;
    int dynamic_tuning;
    float cth;
    float high_th_m;
    float new_track_th_m;
    float match_th_first_m;
    float obb_theta_damping;
    int marginal_timeout;
    int central_timeout;
    int frame_width;
    int frame_height;
    int horizontal_margin;
    int vertical_margin;
    int frame_rate;
    int max_obs;
};

struct BoxMOTSFSORTHandle;

BOXMOT_SFSORT_API BoxMOTSFSORTHandle* boxmot_sfsort_create(const BoxMOTSFSORTConfig* config);
BOXMOT_SFSORT_API void boxmot_sfsort_destroy(BoxMOTSFSORTHandle* handle);
BOXMOT_SFSORT_API int boxmot_sfsort_reset(BoxMOTSFSORTHandle* handle);
BOXMOT_SFSORT_API int boxmot_sfsort_update(
    BoxMOTSFSORTHandle* handle,
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
BOXMOT_SFSORT_API const char* boxmot_sfsort_last_error();

}  // extern "C"
