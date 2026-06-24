#include "occluboost/c_api.hpp"

#include "boxmot/trackers/base/live_c_api.hpp"
#include "occluboost/tracker.hpp"
#include "occluboost/types.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
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
    c.with_reid = config.with_reid != 0;
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

    c.reid_model_path = config.reid_model_path == nullptr ? "" : std::string(config.reid_model_path);
    c.reid_preprocess = config.reid_preprocess == nullptr ? "resize_pad" : std::string(config.reid_preprocess);
    c.reid_device = config.reid_device == nullptr ? "auto" : std::string(config.reid_device);
    return c;
}

std::vector<occluboost::Detection> ConvertDetections(
    const float* dets,
    const int det_rows,
    const int det_cols,
    const float* embs,
    const int emb_rows,
    const int emb_cols
) {
    if (det_rows < 0 || det_cols < 0 || emb_rows < 0 || emb_cols < 0) {
        throw std::runtime_error("Negative matrix dimensions are not allowed.");
    }
    if (embs != nullptr && emb_rows != det_rows) {
        throw std::runtime_error("Detection and embedding row counts must match.");
    }

    std::vector<occluboost::Detection> converted =
        boxmot::trackers::base::ConvertLiveDetections<occluboost::Detection>(dets, det_rows, det_cols, "OccluBoost");
    for (std::size_t row = 0; row < converted.size(); ++row) {
        if (embs != nullptr && emb_cols > 0) {
            occluboost::Detection& detection = converted[row];
            detection.embedding.resize(emb_cols);
            const float* emb_row = embs + (row * static_cast<std::size_t>(emb_cols));
            for (int col = 0; col < emb_cols; ++col) {
                detection.embedding(col) = emb_row[col];
            }
        }
    }
    return converted;
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
        boxmot::trackers::base::SetLastError(g_last_error, exc.what());
        return nullptr;
    } catch (...) {
        boxmot::trackers::base::SetLastError(g_last_error, "Unknown native OccluBoost creation failure");
        return nullptr;
    }
}

void boxmot_occluboost_destroy(BoxMOTOccluBoostHandle* handle) {
    delete handle;
}

int boxmot_occluboost_reset(BoxMOTOccluBoostHandle* handle) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr) {
            throw std::runtime_error("Native OccluBoost handle is null.");
        }
        handle->tracker = std::make_unique<occluboost::OccluBoostTracker>(handle->config);
    }, g_last_error, "Unknown native OccluBoost failure");
}

int boxmot_occluboost_update(
    BoxMOTOccluBoostHandle* handle,
    const float* dets,
    const int det_rows,
    const int det_cols,
    const float* embs,
    const int emb_rows,
    const int emb_cols,
    const std::uint8_t* image_data,
    const int image_rows,
    const int image_cols,
    const int image_channels,
    float* out_tracks,
    const int out_capacity_rows,
    const int out_cols,
    int* out_rows,
    int* out_is_obb
) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr || handle->tracker == nullptr) {
            throw std::runtime_error("Native OccluBoost handle is not initialized.");
        }
        if (out_rows == nullptr || out_is_obb == nullptr) {
            throw std::runtime_error("Output pointers are null.");
        }
        const std::vector<occluboost::Detection> detections =
            ConvertDetections(dets, det_rows, det_cols, embs, emb_rows, emb_cols);
        const cv::Mat image =
            boxmot::trackers::base::WrapLiveImage(image_data, image_rows, image_cols, image_channels, "OccluBoost");
        const std::vector<occluboost::TrackOutput> tracks = handle->tracker->Update(detections, image);
        boxmot::trackers::base::WriteLiveOutputs(tracks, out_tracks, out_capacity_rows, out_cols, "OccluBoost");
        *out_rows = static_cast<int>(tracks.size());
        *out_is_obb = boxmot::trackers::base::LiveOutputUsesObb(tracks, det_cols) ? 1 : 0;
    }, g_last_error, "Unknown native OccluBoost failure");
}

int boxmot_occluboost_last_reid_time_ms(BoxMOTOccluBoostHandle* handle, double* out_reid_time_ms) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr || handle->tracker == nullptr) {
            throw std::runtime_error("Native OccluBoost handle is not initialized.");
        }
        if (out_reid_time_ms == nullptr) {
            throw std::runtime_error("Output ReID timing pointer is null.");
        }
        *out_reid_time_ms = handle->tracker->LastReIdTimeMs();
    }, g_last_error, "Unknown native OccluBoost failure");
}

int boxmot_occluboost_last_reid_preprocess_time_ms(BoxMOTOccluBoostHandle* handle, double* out_time_ms) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr || handle->tracker == nullptr) {
            throw std::runtime_error("Native OccluBoost handle is not initialized.");
        }
        if (out_time_ms == nullptr) {
            throw std::runtime_error("Output ReID timing pointer is null.");
        }
        *out_time_ms = handle->tracker->LastReIdPreprocessTimeMs();
    }, g_last_error, "Unknown native OccluBoost failure");
}

int boxmot_occluboost_last_reid_process_time_ms(BoxMOTOccluBoostHandle* handle, double* out_time_ms) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr || handle->tracker == nullptr) {
            throw std::runtime_error("Native OccluBoost handle is not initialized.");
        }
        if (out_time_ms == nullptr) {
            throw std::runtime_error("Output ReID timing pointer is null.");
        }
        *out_time_ms = handle->tracker->LastReIdProcessTimeMs();
    }, g_last_error, "Unknown native OccluBoost failure");
}

int boxmot_occluboost_last_reid_postprocess_time_ms(BoxMOTOccluBoostHandle* handle, double* out_time_ms) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr || handle->tracker == nullptr) {
            throw std::runtime_error("Native OccluBoost handle is not initialized.");
        }
        if (out_time_ms == nullptr) {
            throw std::runtime_error("Output ReID timing pointer is null.");
        }
        *out_time_ms = handle->tracker->LastReIdPostprocessTimeMs();
    }, g_last_error, "Unknown native OccluBoost failure");
}

const char* boxmot_occluboost_last_error() {
    return g_last_error.c_str();
}

}  // extern "C"
