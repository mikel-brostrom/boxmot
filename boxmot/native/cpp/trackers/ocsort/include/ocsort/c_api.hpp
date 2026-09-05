#pragma once

#include "boxmot/trackers/base/c_api_v2.hpp"

#if defined(_WIN32)
#if defined(BOXMOT_OCSORT_BUILDING_DLL)
#define BOXMOT_OCSORT_API __declspec(dllexport)
#else
#define BOXMOT_OCSORT_API __declspec(dllimport)
#endif
#else
#define BOXMOT_OCSORT_API __attribute__((visibility("default")))
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
    const char* asso_func;
};

struct BoxMOTOCSORTHandle;

BOXMOT_OCSORT_API BoxMOTOCSORTHandle* boxmot_ocsort_create(const BoxMOTOCSORTConfig* config);
BOXMOT_OCSORT_API void boxmot_ocsort_destroy(BoxMOTOCSORTHandle* handle);
BOXMOT_OCSORT_API int boxmot_ocsort_reset(BoxMOTOCSORTHandle* handle);
BOXMOT_OCSORT_API int boxmot_ocsort_update_v2(BoxMOTOCSORTHandle* handle,
                                              const BoxMOTDetectionBatchV2* detections,
                                              const BoxMOTImageV2* image,
                                              BoxMOTTrackBatchV2** output);
BOXMOT_OCSORT_API void boxmot_ocsort_result_free_v2(BoxMOTTrackBatchV2* output);
BOXMOT_OCSORT_API const char* boxmot_ocsort_last_error();

}  // extern "C"
