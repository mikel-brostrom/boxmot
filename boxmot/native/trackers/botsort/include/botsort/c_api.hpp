#pragma once

#include <cstdint>

#if defined(_WIN32)
#  if defined(BOXMOT_BOTSORT_BUILDING_DLL)
#    define BOXMOT_BOTSORT_API __declspec(dllexport)
#  else
#    define BOXMOT_BOTSORT_API __declspec(dllimport)
#  endif
#else
#  define BOXMOT_BOTSORT_API __attribute__((visibility("default")))
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
    const char* cmc_method;
    int frame_rate;
    int fuse_first_associate;
    int with_reid;
    int max_obs;
    const char* reid_model_path;
    const char* reid_preprocess;
};

struct BoxMOTBotSortHandle;

BOXMOT_BOTSORT_API BoxMOTBotSortHandle* boxmot_botsort_create(const BoxMOTBotSortConfig* config);
BOXMOT_BOTSORT_API void boxmot_botsort_destroy(BoxMOTBotSortHandle* handle);
BOXMOT_BOTSORT_API int boxmot_botsort_reset(BoxMOTBotSortHandle* handle);
BOXMOT_BOTSORT_API int boxmot_botsort_update(
    BoxMOTBotSortHandle* handle,
    const float* dets,
    int det_rows,
    int det_cols,
    const float* embs,
    int emb_rows,
    int emb_cols,
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
BOXMOT_BOTSORT_API int boxmot_botsort_last_reid_time_ms(BoxMOTBotSortHandle* handle, double* out_reid_time_ms);
BOXMOT_BOTSORT_API const char* boxmot_botsort_last_error();

}
