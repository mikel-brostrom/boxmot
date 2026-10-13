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

struct BoxMOTByteTrackConfigV3 {
    float min_conf;
    float track_thresh;
    float match_thresh;
    float second_match_thresh;
    float unconfirmed_match_thresh;
    int max_lost_frames;
    int max_obs;
    const char* asso_func;
    double centroid_max_distance;
    int fuse_first_associate;
    int fuse_unconfirmed_associate;
    double duplicate_distance_thresh;
};

struct BoxMOTByteTrackHandle;

BOXMOT_BYTETRACK_API BoxMOTByteTrackHandle* boxmot_bytetrack_create_v3(
    const BoxMOTByteTrackConfigV3* config);
BOXMOT_BYTETRACK_API void boxmot_bytetrack_destroy(BoxMOTByteTrackHandle* handle);
BOXMOT_BYTETRACK_API int boxmot_bytetrack_reset(BoxMOTByteTrackHandle* handle);
BOXMOT_BYTETRACK_API int boxmot_bytetrack_update_v2(BoxMOTByteTrackHandle* handle,
                                                    const BoxMOTDetectionBatchV2* detections,
                                                    const BoxMOTImageV2* image,
                                                    BoxMOTTrackBatchV2** output);
BOXMOT_BYTETRACK_API void boxmot_bytetrack_result_free_v2(BoxMOTTrackBatchV2* output);
BOXMOT_BYTETRACK_API const char* boxmot_bytetrack_last_error();

}  // extern "C"
