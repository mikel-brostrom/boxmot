#pragma once

#include "boxmot/trackers/base/c_api_v2.hpp"

#if defined(_WIN32)
#if defined(BOXMOT_BYTETRACK_BUILDING_DLL)
#define BOXMOT_BYTETRACK_API __declspec(dllexport)
#else
#define BOXMOT_BYTETRACK_API __declspec(dllimport)
#endif
#else
#define BOXMOT_BYTETRACK_API __attribute__((visibility("default")))
#endif

extern "C" {

struct BoxMOTByteTrackConfig {
    float min_conf;
    float track_thresh;
    float match_thresh;
    int track_buffer;
    int frame_rate;
    int max_obs;
    const char* asso_func;
};

struct BoxMOTByteTrackHandle;

BOXMOT_BYTETRACK_API BoxMOTByteTrackHandle* boxmot_bytetrack_create(
    const BoxMOTByteTrackConfig* config);
BOXMOT_BYTETRACK_API void boxmot_bytetrack_destroy(BoxMOTByteTrackHandle* handle);
BOXMOT_BYTETRACK_API int boxmot_bytetrack_reset(BoxMOTByteTrackHandle* handle);
BOXMOT_BYTETRACK_API int boxmot_bytetrack_update_v2(BoxMOTByteTrackHandle* handle,
                                                    const BoxMOTDetectionBatchV2* detections,
                                                    const BoxMOTImageV2* image,
                                                    BoxMOTTrackBatchV2** output);
BOXMOT_BYTETRACK_API void boxmot_bytetrack_result_free_v2(BoxMOTTrackBatchV2* output);
BOXMOT_BYTETRACK_API const char* boxmot_bytetrack_last_error();

}  // extern "C"
