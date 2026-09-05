#pragma once

#include "boxmot/trackers/base/c_api_v2.hpp"

#if defined(_WIN32)
#if defined(BOXMOT_OCCLUBOOST_BUILDING_DLL)
#define BOXMOT_OCCLUBOOST_API __declspec(dllexport)
#else
#define BOXMOT_OCCLUBOOST_API __declspec(dllimport)
#endif
#else
#define BOXMOT_OCCLUBOOST_API __attribute__((visibility("default")))
#endif

extern "C" {

struct BoxMOTOccluBoostConfig {
    // BoostTrack inherited
    int max_age;
    int min_hits;
    float det_thresh;
    float iou_threshold;
    int min_box_area;
    float aspect_ratio_thresh;
    float lambda_iou;
    float lambda_mhd;
    float lambda_shape;
    int use_dlo_boost;
    int use_duo_boost;
    float dlo_boost_coef;
    int s_sim_corr;
    int use_rich_s;
    int use_sb;
    int use_vt;
    int use_embeddings;
    const char* cmc_method;
    int max_obs;

    // OccluBoost
    float recovery_appearance_thresh;
    float recovery_iou_thresh;
    int recovery_max_age;
    float feat_alpha;
    float track_low_thresh;
    float second_iou_thresh;
    float second_appearance_thresh;
    int second_pass_max_age;
    int second_pass_min_hits;
    int use_second_pass;
    float new_track_thresh;
    int confirm_hits;
    float instant_confirm_thresh;
    int tentative_max_age;
    float duplicate_iou_thresh;
    int ams_enabled;
    float ams_alpha0;
    float ams_threshold;
    int ams_buffer_size;
    float ams_shrink_ratio;
    float lambda_emb_multiplier;

    // OBB-specific operating point
    float obb_det_thresh;
    float obb_iou_threshold;
    float obb_new_track_thresh;
    float obb_instant_confirm_thresh;
    int obb_max_age;
    int obb_recovery_max_age;
    float obb_second_iou_thresh;

    // ReID
    const char* asso_func;
};

struct BoxMOTOccluBoostHandle;

BOXMOT_OCCLUBOOST_API BoxMOTOccluBoostHandle* boxmot_occluboost_create(
    const BoxMOTOccluBoostConfig* config);
BOXMOT_OCCLUBOOST_API void boxmot_occluboost_destroy(BoxMOTOccluBoostHandle* handle);
BOXMOT_OCCLUBOOST_API int boxmot_occluboost_reset(BoxMOTOccluBoostHandle* handle);
BOXMOT_OCCLUBOOST_API int boxmot_occluboost_update_v2(BoxMOTOccluBoostHandle* handle,
                                                      const BoxMOTDetectionBatchV2* detections,
                                                      const BoxMOTImageV2* image,
                                                      BoxMOTTrackBatchV2** output);
BOXMOT_OCCLUBOOST_API void boxmot_occluboost_result_free_v2(BoxMOTTrackBatchV2* output);
BOXMOT_OCCLUBOOST_API const char* boxmot_occluboost_last_error();
}
