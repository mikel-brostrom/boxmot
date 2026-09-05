#pragma once

#include "boxmot/trackers/base/c_api_v2.hpp"

#if defined(_WIN32)
#if defined(BOXMOT_BOTSORT_BUILDING_DLL)
#define BOXMOT_BOTSORT_API __declspec(dllexport)
#else
#define BOXMOT_BOTSORT_API __declspec(dllimport)
#endif
#else
#define BOXMOT_BOTSORT_API __attribute__((visibility("default")))
#endif

extern "C" {

struct BoxMOTBotSortConfig {
    float track_high_thresh;
    float track_low_thresh;
    float new_track_thresh;
    int track_buffer;
    float match_thresh;
    float proximity_thresh;
    float appearance_thresh;
    float second_match_thresh;
    float unconfirmed_match_thresh;
    float unconfirmed_emb_scale;
    const char* cmc_method;
    int frame_rate;
    int fuse_first_associate;
    int use_embeddings;
    int max_obs;
    const char* asso_func;
};

struct BoxMOTBotSortHandle;

BOXMOT_BOTSORT_API BoxMOTBotSortHandle* boxmot_botsort_create(const BoxMOTBotSortConfig* config);
BOXMOT_BOTSORT_API void boxmot_botsort_destroy(BoxMOTBotSortHandle* handle);
BOXMOT_BOTSORT_API int boxmot_botsort_reset(BoxMOTBotSortHandle* handle);
BOXMOT_BOTSORT_API int boxmot_botsort_update_v2(BoxMOTBotSortHandle* handle,
                                                const BoxMOTDetectionBatchV2* detections,
                                                const BoxMOTImageV2* image,
                                                BoxMOTTrackBatchV2** output);
BOXMOT_BOTSORT_API void boxmot_botsort_result_free_v2(BoxMOTTrackBatchV2* output);
BOXMOT_BOTSORT_API const char* boxmot_botsort_last_error();
}
