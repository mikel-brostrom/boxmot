#pragma once

#include "boxmot/trackers/base/c_api_v2.hpp"

#if defined(_WIN32)
#if defined(BOXMOT_SFSORT_BUILDING_DLL)
#define BOXMOT_SFSORT_API __declspec(dllexport)
#else
#define BOXMOT_SFSORT_API __declspec(dllimport)
#endif
#else
#define BOXMOT_SFSORT_API __attribute__((visibility("default")))
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
    const char* asso_func;
};

struct BoxMOTSFSORTHandle;

BOXMOT_SFSORT_API BoxMOTSFSORTHandle* boxmot_sfsort_create(const BoxMOTSFSORTConfig* config);
BOXMOT_SFSORT_API void boxmot_sfsort_destroy(BoxMOTSFSORTHandle* handle);
BOXMOT_SFSORT_API int boxmot_sfsort_reset(BoxMOTSFSORTHandle* handle);
BOXMOT_SFSORT_API int boxmot_sfsort_update_v2(BoxMOTSFSORTHandle* handle,
                                              const BoxMOTDetectionBatchV2* detections,
                                              const BoxMOTImageV2* image,
                                              BoxMOTTrackBatchV2** output);
BOXMOT_SFSORT_API void boxmot_sfsort_result_free_v2(BoxMOTTrackBatchV2* output);
BOXMOT_SFSORT_API const char* boxmot_sfsort_last_error();

}  // extern "C"
