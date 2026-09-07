#include "occluboost/c_api.hpp"

#include "boxmot/trackers/base/live_c_api.hpp"
#include "occluboost/tracker.hpp"
#include "occluboost/types.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;

occluboost::Config ConvertConfig(const BoxMOTOccluBoostConfig& config) {
    occluboost::Config c;
    c.max_age = config.max_age;
    c.min_hits = config.min_hits;
    c.det_thresh = config.det_thresh;
    c.iou_threshold = config.iou_threshold;
    c.min_box_area = config.min_box_area;
    c.aspect_ratio_thresh = config.aspect_ratio_thresh;
    c.lambda_iou = config.lambda_iou;
    c.lambda_mhd = config.lambda_mhd;
    c.lambda_shape = config.lambda_shape;
    c.use_dlo_boost = config.use_dlo_boost != 0;
    c.use_duo_boost = config.use_duo_boost != 0;
    c.dlo_boost_coef = config.dlo_boost_coef;
    c.s_sim_corr = config.s_sim_corr != 0;
    c.use_rich_s = config.use_rich_s != 0;
    c.use_sb = config.use_sb != 0;
    c.use_vt = config.use_vt != 0;
    c.use_embeddings = config.use_embeddings != 0;
    c.cmc_method = config.cmc_method == nullptr ? "ecc" : std::string(config.cmc_method);
    c.max_obs = config.max_obs;

    c.recovery_appearance_thresh = config.recovery_appearance_thresh;
    c.recovery_iou_thresh = config.recovery_iou_thresh;
    c.recovery_max_age = config.recovery_max_age;
    c.feat_alpha = config.feat_alpha;
    c.track_low_thresh = config.track_low_thresh;
    c.second_iou_thresh = config.second_iou_thresh;
    c.second_appearance_thresh = config.second_appearance_thresh;
    c.second_pass_max_age = config.second_pass_max_age;
    c.second_pass_min_hits = config.second_pass_min_hits;
    c.use_second_pass = config.use_second_pass != 0;
    c.new_track_thresh = config.new_track_thresh;
    c.confirm_hits = config.confirm_hits;
    c.instant_confirm_thresh = config.instant_confirm_thresh;
    c.tentative_max_age = config.tentative_max_age;
    c.duplicate_iou_thresh = config.duplicate_iou_thresh;
    c.ams_enabled = config.ams_enabled != 0;
    c.ams_alpha0 = config.ams_alpha0;
    c.ams_threshold = config.ams_threshold;
    c.ams_buffer_size = config.ams_buffer_size;
    c.ams_shrink_ratio = config.ams_shrink_ratio;
    c.lambda_emb_multiplier = config.lambda_emb_multiplier;
    c.obb_det_thresh = config.obb_det_thresh;
    c.obb_iou_threshold = config.obb_iou_threshold;
    c.obb_new_track_thresh = config.obb_new_track_thresh;
    c.obb_instant_confirm_thresh = config.obb_instant_confirm_thresh;
    c.obb_max_age = config.obb_max_age;
    c.obb_recovery_max_age = config.obb_recovery_max_age;
    c.obb_second_iou_thresh = config.obb_second_iou_thresh;

    c.asso_func = config.asso_func == nullptr ? "iou" : std::string(config.asso_func);
    return c;
}

}  // namespace

struct BoxMOTOccluBoostHandle {
    explicit BoxMOTOccluBoostHandle(occluboost::Config tracker_config)
        : config(std::move(tracker_config)),
          tracker(std::make_unique<occluboost::OccluBoostTracker>(config)) {}

    occluboost::Config config;
    std::unique_ptr<occluboost::OccluBoostTracker> tracker;
};

extern "C" {

BoxMOTOccluBoostHandle* boxmot_occluboost_create(const BoxMOTOccluBoostConfig* config) {
    try {
        if (config == nullptr) {
            throw std::runtime_error("Native OccluBoost config is required.");
        }
        occluboost::Config native_config = ConvertConfig(*config);
        g_last_error.clear();
        return new BoxMOTOccluBoostHandle(std::move(native_config));
    } catch (const std::exception& exc) {
        boxmot::native::SetLastError(g_last_error, exc.what());
        return nullptr;
    } catch (...) {
        boxmot::native::SetLastError(g_last_error,
                                             "Unknown native OccluBoost creation failure");
        return nullptr;
    }
}

void boxmot_occluboost_destroy(BoxMOTOccluBoostHandle* handle) {
    delete handle;
}

int boxmot_occluboost_reset(BoxMOTOccluBoostHandle* handle) {
    return boxmot::native::GuardCall(
        [&]() {
            if (handle == nullptr) {
                throw std::runtime_error("Native OccluBoost handle is null.");
            }
            handle->tracker = std::make_unique<occluboost::OccluBoostTracker>(handle->config);
        },
        g_last_error,
        "Unknown native OccluBoost failure");
}

int boxmot_occluboost_update_v2(BoxMOTOccluBoostHandle* handle,
                                const BoxMOTDetectionBatchV2* detections,
                                const BoxMOTImageV2* image,
                                BoxMOTTrackBatchV2** output) {
    return boxmot::native::GuardCall(
        [&]() {
            if (handle == nullptr || handle->tracker == nullptr) {
                throw std::runtime_error("Native OccluBoost handle is not initialized.");
            }
            if (detections == nullptr || output == nullptr) {
                throw std::runtime_error("Native OccluBoost input/output pointers are null.");
            }
            *output = nullptr;
            const std::vector<occluboost::Detection> converted =
                boxmot::trackers::base::ConvertLiveDetectionsV2<occluboost::Detection, true>(
                    *detections, "OccluBoost");
            const cv::Mat image_mat =
                boxmot::trackers::base::WrapOptionalLiveImageV2(image, "OccluBoost");
            const bool cmc_needs_image =
                !handle->config.cmc_method.empty() && handle->config.cmc_method != "none";
            if (image_mat.empty() && cmc_needs_image) {
                throw std::runtime_error("Native OccluBoost requires an image when CMC is active.");
            }
            if (handle->config.use_embeddings && detections->rows > 0 &&
                detections->embedding_cols <= 0) {
                throw std::runtime_error("Native OccluBoost requires precomputed embeddings.");
            }
            const std::vector<occluboost::TrackOutput> tracks =
                handle->tracker->Update(converted, image_mat);
            *output =
                boxmot::trackers::base::AllocateLiveOutputV2(tracks, detections->geometry_cols);
        },
        g_last_error,
        "Unknown native OccluBoost failure");
}

void boxmot_occluboost_result_free_v2(BoxMOTTrackBatchV2* output) {
    boxmot::trackers::base::FreeLiveOutputV2(output);
}

const char* boxmot_occluboost_last_error() {
    return g_last_error.c_str();
}

}  // extern "C"
